"""RelayWorkflow routing, ordering, failure semantics and persistence."""

from __future__ import annotations

import threading
import time

import pytest

import core.relay as relay_mod
from core.openchamber import (
    ModelRef,
    OpenChamberBadRequestError,
    OpenChamberCancelledError,
    OpenChamberInterruptedError,
    OpenChamberModelRequestRejectedError,
    OpenChamberSessionError,
    OpenChamberTimeoutError,
)
from core.protocol import ProtocolFormat, parse_message
from core.relay import (
    CANCELLED_MARKER,
    MODEL_REJECTION_AGAIN,
    MODEL_REJECTION_PROMPT,
    MODEL_REJECTION_RETRY_PREFIX,
    OPENCHAMBER_CONTINUE_PROMPT,
    OpenChamberContinue,
    RelayWorkflow,
    directory_key,
    resolve_executor_kind,
)
from core.relay_settings import (
    TARGET_EXECUTOR,
    TARGET_OPENCHAMBER,
    TARGET_REASONIX,
    RelaySettings,
)
from core.task_registry import TaskRegistry
from tests.fakes import (
    ScriptedOpenChamber,
    assistant_message,
    make_dispatch,
    question_part,
    text_part,
    user_message,
)


class FakeReasonix:
    def __init__(self, reply: str = "reasonix-reply", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.executed: list[str] = []

    def self_check(self):
        return {"window": True, "composer": True, "composer_writable": True,
                "send_button": True}

    def execute(self, task: str) -> str:
        self.executed.append(task)
        if self.fail:
            raise RuntimeError("Reasonix window was not found")
        return f"{self.reply}: {task}"


def v1_task(
    target: str,
    body: str,
    message_id: str = "task-001",
    workdir: str | None = None,
) -> str:
    headers = [
            "AI_RELAY/1",
            f"MESSAGE_ID: {message_id}",
            "SOURCE: CHATGPT",
            f"TARGET: {target}",
            "TYPE: TASK",
            "ROUND: 1",
            "MAX_ROUNDS: 3",
    ]
    if workdir:
        headers.append(f"WORKDIR: {workdir}")
    return "\n".join((*headers, "", body))


def legacy_task(body: str, task_id: str = "legacy-001") -> str:
    return "\n".join(
        (
            "----- AI_RELAY_BEGIN -----",
            "SOURCE: CHATGPT",
            f"TARGET: {TARGET_REASONIX}",
            "TYPE: TASK",
            f"TASK_ID: {task_id}",
            "ROUND: 1",
            "MAX_ROUNDS: 3",
            "CONTENT:",
            body,
            "----- AI_RELAY_END -----",
        )
    )


def make_workflow(tmp_path, reasonix=None, oc=None, settings=None, default=TARGET_REASONIX):
    settings = settings or RelaySettings(
        default_target=default,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    reasonix = reasonix or FakeReasonix()
    return RelayWorkflow(
        reasonix=reasonix,
        registry=TaskRegistry(tmp_path / "tasks.json"),
        settings=settings,
        openchamber=oc,
        replies_dir=tmp_path / "replies",
    )


def scripted_oc(directory: str = "D:/proj") -> ScriptedOpenChamber:
    oc = ScriptedOpenChamber(directory=directory)
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")],
                model=ModelRef("9router-new", "9auto"),
                parent_id="u_new",
            ),
        ]
    ]
    return oc


class SharedContextOpenChamber:
    """One fixed session whose message history GROWS between tasks, the way
    a reused OpenChamber session does: round 2 sees round 1's messages.

    ``send`` records the pre-send snapshot, appends the new user message AND
    its completed assistant reply immediately (so the wait loop instantly
    sees a finished round), and every round is attributed by the unchanged
    client-side snapshot logic.
    """

    def __init__(self, session_id: str = "ses_test123", directory: str = "D:/proj"):
        self.session_id = session_id
        self.directory = directory
        self.history: list[dict] = []
        self.call_log: list[str] = []
        self.status = "idle"

    def verify(self) -> None:
        self.call_log.append("verify")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        self.call_log.append(f"list:{directory or ''}")
        if directory is not None and directory != self.directory:
            return []
        return [(self.session_id, "Shared test session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.session_id and directory == self.directory

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def send(self, session_id, prompt, directory, agent=None, model=None) -> object:
        self.call_log.append(f"send:{prompt!r}")
        assert session_id == self.session_id, "must reuse the configured session"
        assert directory == self.directory
        pre_ids = frozenset(
            message["info"]["id"]
            for message in self.history
            if message.get("info", {}).get("id")
        )
        n_user = sum(
            1 for m in self.history if m["info"].get("role") == "user"
        )
        n_assistant = sum(
            1 for m in self.history if m["info"].get("role") == "assistant"
        )
        user_id = f"u{n_user + 1}"
        assistant_id = f"a{n_assistant + 1}"
        created = 1000 + 200 * (n_user + n_assistant)
        self.history.append(
            user_message(user_id, prompt, created, session_id=session_id)
        )
        self.history.append(
            assistant_message(
                assistant_id,
                created + 100,
                completed=created + 200,
                finish="stop",
                parts=[text_part(f"answer {n_user + 1}")],
                parent_id=user_id,
                session_id=session_id,
            )
        )
        return make_dispatch(
            session_id=session_id,
            directory=directory,
            pre_ids=pre_ids,
        )

    def session_status(self, session_id: str, directory: str) -> str:
        return self.status

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return list(self.history)

    def close(self) -> None:
        pass


class RecoveryOpenChamber:
    """A shared-identity session whose round FAILS with no assistant reply
    for the first ``recover_at`` sends and completes from the next one on.

    That is exactly the shape of an abnormally interrupted OpenChamber
    round: idle session, prompt recorded, but no usable final answer.  The
    auto-recovery (or a manual continue) re-sends into the SAME session and
    eventually gets a verified completed reply.
    """

    def __init__(
        self,
        session_id: str = "ses_test123",
        directory: str = "D:/proj",
        recover_at: int = 1,
    ):
        self.session_id = session_id
        self.directory = directory
        self.history: list[dict] = []
        self.call_log: list[str] = []
        self.send_count = 0
        self.recover_at = recover_at

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        return self.session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        if directory is not None and directory != self.directory:
            return []
        return [(self.session_id, "Recovery session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.session_id and directory == self.directory

    def session_status(self, session_id: str, directory: str) -> str:
        return "idle"

    def send(self, session_id, prompt, directory, agent=None, model=None) -> object:
        self.call_log.append(f"send:{prompt!r}")
        assert session_id == self.session_id, "must reuse the configured session"
        assert directory == self.directory
        pre_ids = frozenset(
            m["info"]["id"] for m in self.history if m.get("info", {}).get("id")
        )
        n_user = sum(1 for m in self.history if m["info"].get("role") == "user")
        n_assistant = sum(
            1 for m in self.history if m["info"].get("role") == "assistant"
        )
        index = self.send_count
        self.send_count += 1
        user_id = f"u{n_user + 1}"
        created = 1000 + 200 * (n_user + n_assistant)
        self.history.append(
            user_message(user_id, prompt, created, session_id=session_id)
        )
        if index >= self.recover_at:
            assistant_id = f"a{n_assistant + 1}"
            self.history.append(
                assistant_message(
                    assistant_id,
                    created + 100,
                    completed=created + 200,
                    finish="stop",
                    parts=[text_part(f"recovered answer {n_user + 1}")],
                    parent_id=user_id,
                    session_id=session_id,
                )
            )
        return make_dispatch(
            session_id=session_id,
            directory=directory,
            pre_ids=pre_ids,
            user_message_id=user_id,
        )

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return list(self.history)

    def close(self) -> None:
        pass


class BadRequestOpenChamber:
    """A configured session that answers EVERY send with a non-retryable
    HTTP 400 (the Qwen-upstream shape) and must never be auto-retried."""

    def __init__(self, session_id: str = "ses_test123", directory: str = "D:/proj"):
        self.session_id = session_id
        self.directory = directory
        self.history: list[dict] = []
        self.call_log: list[str] = []

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        return self.session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        if directory is not None and directory != self.directory:
            return []
        return [(self.session_id, "Bad requests session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.session_id and directory == self.directory

    def session_status(self, session_id: str, directory: str) -> str:
        return "idle"

    def send(self, session_id, prompt, directory, agent=None, model=None) -> object:
        self.call_log.append(f"send:{prompt!r}")
        raise OpenChamberBadRequestError(
            "OpenChamber API returned HTTP 400: model rejected the request"
        )

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return list(self.history)

    def close(self) -> None:
        pass


class RejectedThenWorkingOpenChamber:
    """The preconfigured session rejects EVERY send with the structured
    non-retryable 400 (isRetryable=false); ``create_session`` returns a FRESH
    session whose sends complete normally.  Models a Qwen-upstream rejection
    that is resolved by the "新会话重试" flow."""

    def __init__(
        self,
        rejected_session_id: str = "ses_rejected",
        directory: str = "D:/proj",
        fresh_session_id: str = "ses_new",
    ):
        self.rejected_session_id = rejected_session_id
        self.fresh_session_id = fresh_session_id
        self.directory = directory
        self.history: dict[str, list] = {
            rejected_session_id: [],
            fresh_session_id: [],
        }
        self.call_log: list[str] = []

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        assert directory == self.directory, "retry must stay in the project dir"
        return self.fresh_session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        if directory is not None and directory != self.directory:
            return []
        return [(self.rejected_session_id, "Rejected session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.rejected_session_id and directory == self.directory

    def session_status(self, session_id: str, directory: str) -> str:
        return "idle"

    def send(self, session_id, prompt, directory, agent=None, model=None) -> object:
        self.call_log.append(f"send:{prompt!r}")
        assert directory == self.directory
        if session_id == self.rejected_session_id:
            raise OpenChamberModelRequestRejectedError(
                "OpenChamber rejected the request (HTTP 400) because the "
                "upstream model service rejected it: APIError\n"
                "statusCode: 400\nisRetryable: false\n"
                "url: http://192.168.100.190:8080/v1/chat/completions",
                status_code=400,
                is_retryable=False,
                request_url="http://192.168.100.190:8080/v1/chat/completions",
            )
        assert session_id == self.fresh_session_id, "must use the brand-new session"
        history = self.history[session_id]
        pre_ids = frozenset(
            m["info"]["id"] for m in history if m.get("info", {}).get("id")
        )
        user_id = f"u{len([m for m in history if m['info'].get('role') == 'user']) + 1}"
        created = 1000 + 200 * len(history)
        history.append(user_message(user_id, prompt, created, session_id=session_id))
        history.append(
            assistant_message(
                f"a{len([m for m in history if m['info'].get('role') == 'assistant']) + 1}",
                created + 100,
                completed=created + 200,
                finish="stop",
                parts=[text_part("fresh session answer")],
                parent_id=user_id,
                session_id=session_id,
            )
        )
        return make_dispatch(
            session_id=session_id,
            directory=directory,
            pre_ids=pre_ids,
            user_message_id=user_id,
        )

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return list(self.history.get(session_id, []))

    def close(self) -> None:
        pass


class AlwaysRejectedOpenChamber:
    """Every existing session AND every created session rejects with the
    structured non-retryable 400: the fresh-session retry must stop after
    exactly ONE creation and never spawn a third session."""

    def __init__(
        self,
        rejected_session_id: str = "ses_rejected",
        directory: str = "D:/proj",
        fresh_session_id: str = "ses_new_too",
    ):
        self.rejected_session_id = rejected_session_id
        self.fresh_session_id = fresh_session_id
        self.directory = directory
        self.call_log: list[str] = []
        self.created: list[str] = []

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        self.created.append(self.fresh_session_id)
        return self.fresh_session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        if directory is not None and directory != self.directory:
            return []
        return [(self.rejected_session_id, "Always rejected session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.rejected_session_id and directory == self.directory

    def session_status(self, session_id: str, directory: str) -> str:
        return "idle"

    def send(self, session_id, prompt, directory, agent=None, model=None) -> object:
        self.call_log.append(f"send:{prompt!r}")
        raise OpenChamberModelRequestRejectedError(
            "OpenChamber rejected the request (HTTP 400) because the "
            "upstream model service rejected it: APIError\n"
            "statusCode: 400\nisRetryable: false\n"
            "url: http://192.168.100.190:8080/v1/chat/completions",
            status_code=400,
            is_retryable=False,
            request_url="http://192.168.100.190:8080/v1/chat/completions",
        )

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return []

    def close(self) -> None:
        pass


class BusyOpenChamber:
    """A configured session that is ALWAYS busy: the wait loop never
    completes on its own and can be stopped only via the cancel event."""

    def __init__(self, session_id: str = "ses_test123", directory: str = "D:/proj"):
        self.session_id = session_id
        self.directory = directory
        self.history: list[dict] = []
        self.call_log: list[str] = []
        self.sessions: list[tuple[str, str]] = [(session_id, "Busy session")]

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        return self.session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        if directory is not None and directory != self.directory:
            return []
        return list(self.sessions)

    def session_exists(self, session_id: str, directory: str) -> bool:
        return (
            session_id == self.session_id
            and directory == self.directory
            and any(sid == session_id for sid, _title in self.sessions)
        )

    def session_status(self, session_id: str, directory: str) -> str:
        return "busy"

    def round_has_pending_user_action(
        self, session_id: str, directory: str, dispatch
    ) -> bool:
        return False

    def send(self, session_id, prompt, directory, agent=None, model=None) -> object:
        self.call_log.append(f"send:{prompt!r}")
        assert session_id == self.session_id
        assert directory == self.directory
        n = len([m for m in self.history if m.get("info", {}).get("role") == "user"])
        user_id = f"u{n + 1}"
        self.history.append(
            user_message(user_id, prompt, 1000 + 100 * (n + 1), session_id=session_id)
        )
        return make_dispatch(
            session_id=session_id,
            directory=directory,
            user_message_id=user_id,
        )

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return list(self.history)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------- #
# routing
# ---------------------------------------------------------------------- #


def test_explicit_targets_map_to_themselves():
    settings = RelaySettings(default_target=TARGET_OPENCHAMBER)
    assert resolve_executor_kind(TARGET_REASONIX, settings) == TARGET_REASONIX
    assert resolve_executor_kind(TARGET_OPENCHAMBER, settings) == TARGET_OPENCHAMBER


def test_executor_target_follows_default():
    assert resolve_executor_kind(TARGET_EXECUTOR, RelaySettings()) == TARGET_REASONIX
    assert (
        resolve_executor_kind(
            TARGET_EXECUTOR, RelaySettings(default_target=TARGET_OPENCHAMBER)
        )
        == TARGET_OPENCHAMBER
    )


def test_explicit_target_not_overridden_by_default(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    response = wf.process(v1_task(TARGET_REASONIX, "hi"))
    assert "reasonix-reply" in response
    assert oc.call_log == []  # OpenChamber must not be touched


def test_route_reasonix_explicit(tmp_path):
    reasonix = FakeReasonix()
    wf = make_workflow(tmp_path, reasonix=reasonix)
    response = wf.process(v1_task(TARGET_REASONIX, "say hi"))
    assert reasonix.executed == ["say hi"]
    message = parse_message(response)
    assert message.protocol_format is ProtocolFormat.V1
    assert message.in_reply_to == "task-001"
    assert message.target == "CHATGPT"
    assert message.message_type.value == "RESPONSE"
    assert "say hi" in message.body


def test_route_reasonix_legacy_format(tmp_path):
    reasonix = FakeReasonix()
    wf = make_workflow(tmp_path, reasonix=reasonix)
    response = wf.process(legacy_task("say hi"))
    message = parse_message(response)
    assert message.protocol_format is ProtocolFormat.LEGACY_WEB
    assert message.message_id == "legacy-001"
    assert "----- AI_RELAY_BEGIN -----" in response


def test_route_openchamber_explicit(tmp_path):
    oc = scripted_oc()
    reasonix = FakeReasonix()
    wf = make_workflow(tmp_path, reasonix=reasonix, oc=oc)
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert reasonix.executed == []
    message = parse_message(response)
    assert message.body == "final answer"
    assert message.in_reply_to == "task-001"


def test_route_executor_default_openchamber(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    response = wf.process(v1_task(TARGET_EXECUTOR, "do it"))
    assert "final answer" in response
    assert any(call.startswith("send:") for call in oc.call_log)


def test_reasonix_offline_does_not_block_openchamber(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(
        tmp_path, reasonix=FakeReasonix(fail=True), oc=oc
    )
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert "final answer" in response


def test_unknown_target_rejected(tmp_path):
    wf = make_workflow(tmp_path)
    with pytest.raises(RuntimeError, match="not executable"):
        wf.process(v1_task("SOMEONE_ELSE", "x"))


def test_response_message_type_rejected(tmp_path):
    wf = make_workflow(tmp_path)
    with pytest.raises(RuntimeError, match="not a TASK"):
        wf.process(
            "AI_RELAY/1\nMESSAGE_ID: t1\nSOURCE: CHATGPT\nTARGET: REASONIX\n"
            "TYPE: RESPONSE\n\nhello"
        )


# ---------------------------------------------------------------------- #
# OpenChamber flow
# ---------------------------------------------------------------------- #


def test_openchamber_flow_order_confirm_open_send(tmp_path):
    """The configured session is existence-checked first; the relay never
    creates a session and only sends into the fixed, verified session."""
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    kinds = [call.split(":")[0] for call in oc.call_log if call != "opened"]
    assert kinds == ["verify", "list", "open", "send"]
    # the deep-link request strictly precedes the send
    send_call = next(call for call in oc.call_log if call.startswith("send:"))
    assert "AI_RELAY_TASK_ID: task-001" in send_call
    assert oc.call_log.index("open:ses_test123") < oc.call_log.index(send_call)
    # the relay must never create a session and never invent a title
    assert not any(call.startswith("create:") for call in oc.call_log)


def test_fixed_session_reused_across_two_rounds(tmp_path):
    """Two consecutive tasks must use the SAME configured session: the
    second continues the first round's context, only the second round's
    final reply is returned, and no new session is created."""
    oc = SharedContextOpenChamber()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    first = wf.process(v1_task(TARGET_OPENCHAMBER, "round 1", "task-r1"))
    second = wf.process(v1_task(TARGET_OPENCHAMBER, "round 2", "task-r2"))

    # round 2 was sent into the SAME configured session (context continues)
    assert wf.registry.record("task-r1")["session_id"] == "ses_test123"
    assert wf.registry.record("task-r2")["session_id"] == "ses_test123"
    assert wf.outcome is not None and wf.outcome.session_id == "ses_test123"
    # round 2 built on round 1's history in the session
    assert len(oc.history) == 4  # u1 a1 u2 a2, one shared session
    # each round returns ONLY its own final answer, never the other round's
    assert "answer 1" in first
    assert "answer 2" in second
    assert "answer 1" not in second
    # exactly two sends, and never a session create
    assert sum(call.startswith("send:") for call in oc.call_log) == 2
    assert not any(call.startswith("create:") for call in oc.call_log)
    # round ids stay independent per task
    assert parse_message(first).in_reply_to == "task-r1"
    assert parse_message(second).in_reply_to == "task-r2"


def test_openchamber_missing_session_id_creates_project_session(tmp_path):
    oc = scripted_oc()
    oc.next_session_id = "ses_created"
    settings = RelaySettings(
        default_target=TARGET_REASONIX, openchamber_directory="D:/proj"
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert any(call.startswith("create:AI Relay - proj:D:/proj") for call in oc.call_log)
    assert settings.openchamber_sessions[directory_key("D:/proj")] == "ses_created"


def test_configured_session_not_found_creates_replacement(tmp_path):
    oc = scripted_oc()  # only knows ses_test123
    oc.next_session_id = "ses_replacement"
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_gone",
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert any(call.startswith("create:") for call in oc.call_log)
    assert any(call.startswith("send:") for call in oc.call_log)
    assert settings.openchamber_sessions[directory_key("D:/proj")] == "ses_replacement"


def test_task_workdir_selects_its_project_session(tmp_path):
    oc = scripted_oc(directory="D:/other")
    oc.sessions.append(("ses_project", "Project", "D:/project"))
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/other",
        openchamber_session_id="ses_test123",
        openchamber_sessions={directory_key("D:/project"): "ses_project"},
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it", workdir="D:/project"))
    assert "list:D:/project" in oc.call_log
    assert "open:ses_project" in oc.call_log
    assert wf.current_session.directory == "D:/project"


def test_protocol_preserves_workdir():
    message = parse_message(
        v1_task(TARGET_OPENCHAMBER, "do it", workdir="D:/project")
    )
    assert message.workdir == "D:/project"


def test_relay_settings_persist_session_id(tmp_path):
    path = tmp_path / "relay_settings.json"
    settings = RelaySettings(openchamber_session_id="ses_x")
    settings.save(path)
    assert RelaySettings.load(path).openchamber_session_id == "ses_x"
    assert (
        RelaySettings.load(tmp_path / "missing").openchamber_session_id == ""
    )


def test_relay_settings_persist_project_sessions(tmp_path):
    path = tmp_path / "settings.json"
    key = directory_key("D:/project")
    settings = RelaySettings(openchamber_sessions={key: "ses_project"})
    settings.save(path)
    assert RelaySettings.load(path).openchamber_sessions == {key: "ses_project"}


def test_openchamber_task_persisted_and_reply_stored(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"
    assert record["executor"] == "OPENCHAMBER"
    assert record["session_id"] == "ses_test123"
    assert record["directory"] == "D:/proj"
    # the reply file is hash-named (untrusted task ids never touch the name)
    reply_files = list((tmp_path / "replies").glob("rel_*.response.txt"))
    assert len(reply_files) == 1
    assert record["reply_file"] == str(reply_files[0])
    assert record["actual_model"] == "9router-new/9auto"
    assert wf.load_reply("task-001") == response
    assert wf.outcome is not None
    assert wf.outcome.session_id == "ses_test123"
    assert wf.outcome.executor == "OPENCHAMBER"


def test_deeplink_failure_prevents_send(tmp_path):
    oc = scripted_oc()
    oc.open_error = OpenChamberSessionError("no protocol handler")
    wf = make_workflow(tmp_path, oc=oc)
    with pytest.raises(RuntimeError, match="no protocol handler"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert not any(call.startswith("send:") for call in oc.call_log)
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_test123"  # kept for manual check
    # invariant 7: a failure is never silent -- side A gets a structured
    # RESPONSE bound to the original TASK_ID.
    failure = wf.load_reply("task-001")
    assert failure is not None
    assert "TYPE: RESPONSE" in failure
    assert "IN_REPLY_TO: task-001" in failure
    assert "任务执行失败" in failure
    assert wf.failure_response == failure


def test_send_failure_keeps_session_and_fails(tmp_path):
    oc = scripted_oc()
    oc.send_error = OpenChamberSessionError("prompt not dispatched")
    wf = make_workflow(tmp_path, oc=oc)
    with pytest.raises(RuntimeError, match="prompt not dispatched"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_test123"


def test_timeout_fails_and_keeps_session(tmp_path):
    oc = scripted_oc()
    oc.status_timeline = ["busy"]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        completion_timeout=0.05,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(OpenChamberTimeoutError, match="ses_test123"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert "timeout" in record["error"].lower() or "did not finish" in record["error"]
    assert record["session_id"] == "ses_test123"


def test_model_mismatch_reported_in_outcome(tmp_path):
    oc = scripted_oc()
    requested = ModelRef("4090", "qwen3.8-27b")
    oc.send_dispatch = make_dispatch(
        session_id="ses_test123",
        directory="D:/proj",
        requested=requested,
        resolved=requested,
    )
    other = ModelRef("9router-new", "9auto")
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")], model=other,
                parent_id="u_new",
            ),
        ]
    ]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="4090/qwen3.8-27b",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf.outcome is not None
    assert wf.outcome.note is not None
    assert "不一致" in wf.outcome.note


def test_request_resolved_actual_mismatch_keeps_note(tmp_path):
    """Request A / resolved B / actual B is its own independent mismatch
    (requested != resolved).  Even when actual equals resolved the note must
    stay, and model_info must carry all three layers.  Choosing the saved
    task in the registry must restore these details."""
    oc = scripted_oc()
    requested = ModelRef("provA", "modelA")
    resolved = ModelRef("provB", "modelB")
    oc.send_dispatch = make_dispatch(
        session_id="ses_test123",
        directory="D:/proj",
        requested=requested,
        resolved=resolved,
    )
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")], model=resolved,
                parent_id="u_new",
            ),
        ]
    ]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="provA/modelA",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))

    assert wf.outcome is not None
    assert wf.outcome.note is not None
    assert "模型不一致" in wf.outcome.note
    assert "provA/modelA" in wf.outcome.note
    assert "provB/modelB" in wf.outcome.note
    assert wf.outcome.model_info is not None
    assert "请求 provA/modelA" in wf.outcome.model_info
    assert "解析 provB/modelB" in wf.outcome.model_info
    assert "实际 provB/modelB" in wf.outcome.model_info

    record = wf.registry.record(wf.outcome.task_id)
    assert record is not None
    assert record["requested_model"] == "provA/modelA"
    assert record["resolved_model"] == "provB/modelB"
    assert record["actual_model"] == "provB/modelB"
    assert record["model_note"] is not None
    assert "不一致" in record["model_note"]


def test_missing_directory_fails_clearly(tmp_path):
    oc = scripted_oc()
    settings = RelaySettings(default_target=TARGET_REASONIX, openchamber_directory="  ")
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(RuntimeError, match="项目目录未配置"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert oc.call_log == []


def test_duplicate_task_rejected(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    with pytest.raises(RuntimeError, match="already processed"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it again"))
    assert sum(call.startswith("send:") for call in oc.call_log) == 1


def test_reasonix_failure_marks_failed(tmp_path):
    wf = make_workflow(tmp_path, reasonix=FakeReasonix(fail=True))
    with pytest.raises(RuntimeError, match="Reasonix window"):
        wf.process(v1_task(TARGET_REASONIX, "hi"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["executor"] == "REASONIX"


# ---------------------------------------------------------------------- #
# OpenChamber questions / permissions / ambiguity at workflow level
# ---------------------------------------------------------------------- #


def test_question_waits_for_user_then_relay_continues(tmp_path):
    oc = scripted_oc()
    statuses = ["busy", "idle", "idle", "busy", "idle"]
    pending = [
        user_message("u_new", "task body", 1000),
        assistant_message(
            "a_new", 1100, parts=[question_part("pending")], parent_id="u_new",
        ),
    ]
    answered = [
        user_message("u_new", "task body", 1000),
        assistant_message(
            "a_new", 1100, completed=1200, finish="tool-calls",
            parts=[question_part("completed")], parent_id="u_new",
        ),
        assistant_message(
            "a_final", 1201, completed=1300, finish="stop",
            parts=[text_part("final answer")], parent_id="u_new",
        ),
    ]
    oc.status_timeline = statuses
    oc.message_timelines = [pending, pending, answered]
    wf = make_workflow(tmp_path, oc=oc)
    statuses_seen: list[str] = []
    response = wf.process(
        v1_task(TARGET_OPENCHAMBER, "do it"), statuses_seen.append
    )
    assert "final answer" in response
    # the relay reported the wait for the user, never failed
    assert any("请在 OpenChamber 中处理" in s for s in statuses_seen)
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"
    assert wf.load_reply("task-001") == response


def test_ambiguous_round_fails_without_guessing(tmp_path):
    oc = scripted_oc()
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            user_message("u_manual", "人工插入", 1001),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")], parent_id="u_new",
            ),
        ]
    ]
    wf = make_workflow(tmp_path, oc=oc)
    with pytest.raises(OpenChamberSessionError, match="ambiguous"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_test123"
    # invariant 7: the unguessable round fails loudly AND still answers
    # side A with a structured failure bound to the original TASK_ID.
    failure = wf.load_reply("task-001")
    assert failure is not None
    assert "TYPE: RESPONSE" in failure
    assert "IN_REPLY_TO: task-001" in failure
    assert wf.failure_response == failure


def test_unanswered_question_times_out_keeps_session(tmp_path):
    oc = scripted_oc()
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, parts=[question_part("pending")], parent_id="u_new",
            ),
        ]
    ]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        completion_timeout=0.3,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(OpenChamberTimeoutError, match="ses_test123"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_test123"
    assert sum(call.startswith("send:") for call in oc.call_log) == 1


# ---------------------------------------------------------------------- #
# current session tracking and model-detail persistence
# ---------------------------------------------------------------------- #


def test_current_session_task_follows_running_task(tmp_path):
    """current_session follows the task now running (same fixed session,
    task identity changes); it is not the last successful reply."""
    oc = SharedContextOpenChamber()
    wf = make_workflow(tmp_path, oc=oc)
    wf.process(v1_task(TARGET_OPENCHAMBER, "task A"))
    assert wf.current_session is not None
    assert wf.current_session.session_id == "ses_test123"
    assert wf.current_session.task_id == "task-001"

    wf.process(v1_task(TARGET_OPENCHAMBER, "task B", "task-002"))
    assert wf.current_session is not None
    assert wf.current_session.session_id == "ses_test123"
    assert wf.current_session.task_id == "task-002"
    assert wf.outcome.task_id == "task-002"


def test_current_session_kept_after_timeout(tmp_path):
    oc = scripted_oc()
    oc.status_timeline = ["busy"]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        completion_timeout=0.05,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(OpenChamberTimeoutError, match="ses_test123"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    # no successful reply yet, but the session must still be openable
    assert wf.outcome is None
    assert wf.current_session is not None
    assert wf.current_session.session_id == "ses_test123"


def test_session_callback_invoked_after_session_created(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    captured: list = []
    wf.process(
        v1_task(TARGET_OPENCHAMBER, "do it"),
        session_callback=captured.append,
    )
    assert len(captured) == 1
    assert captured[0].session_id == "ses_test123"
    assert captured[0].task_id == "task-001"
    assert captured[0].directory == "D:/proj"


def test_model_details_persisted_in_completion_record(tmp_path):
    oc = scripted_oc()
    requested = ModelRef("4090", "qwen3.8-27b")
    other = ModelRef("9router-new", "9auto")
    oc.send_dispatch = make_dispatch(
        session_id="ses_test123",
        directory="D:/proj",
        requested=requested,
        resolved=requested,
    )
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")], model=other, parent_id="u_new",
            ),
        ]
    ]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="4090/qwen3.8-27b",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["requested_model"] == "4090/qwen3.8-27b"
    assert record["resolved_model"] == "4090/qwen3.8-27b"
    assert record["actual_model"] == "9router-new/9auto"
    assert "不一致" in record["model_note"]


def test_model_mismatch_note_persists_even_when_task_fails(tmp_path):
    """The requested != resolved warning must survive later status updates
    and a failure: stored in the registry record at dispatch time."""
    oc = scripted_oc()
    oc.status_timeline = ["busy"]  # never completes -> timeout
    requested = ModelRef("4090", "qwen3.8-27b")
    resolved = ModelRef("9router-new", "9auto")
    oc.send_dispatch = make_dispatch(
        session_id="ses_test123",
        directory="D:/proj",
        requested=requested,
        resolved=resolved,
    )
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="4090/qwen3.8-27b",
        completion_timeout=0.05,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(OpenChamberTimeoutError):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["requested_model"] == "4090/qwen3.8-27b"
    assert record["resolved_model"] == "9router-new/9auto"


# ---------------------------------------------------------------------- #
# abnormal interruption: automatic recovery once, then manual continue/stop
# ---------------------------------------------------------------------- #


def test_auto_recovery_continues_interrupted_task_once(tmp_path, monkeypatch):
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = RecoveryOpenChamber(recover_at=1)
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    statuses: list[str] = []
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"), statuses.append)

    message = parse_message(response)
    assert message.in_reply_to == "task-001"  # original TASK_ID kept
    assert "recovered answer" in message.body
    # EXACTLY one continuation (the auto-recovery) after the original send
    assert [call for call in oc.call_log if call.startswith("send:")] == [
        "send:'do it\\n\\n[AI_RELAY_TASK_ID: task-001]'",
        f"send:{OPENCHAMBER_CONTINUE_PROMPT!r}",
    ]
    assert any("续接中断，10 秒后复查" in s for s in statuses)
    assert any("正在原会话自动续接（1/1）" in s for s in statuses)
    assert any("恢复成功，回复已包装并复制" in s for s in statuses)
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"
    assert record["session_id"] == "ses_test123"
    assert wf.pending_continue is None  # cleared on success
    assert wf.recovery_attempted is True


def test_auto_recovery_fails_without_a_second_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = RecoveryOpenChamber(recover_at=10)  # continue also interrupts
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    statuses: list[str] = []
    with pytest.raises(OpenChamberInterruptedError, match="no_assistant_reply"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"), statuses.append)
    # original send + ONE auto-recovery only: no endless retry loop
    assert sum(call.startswith("send:") for call in oc.call_log) == 2
    assert any("自动恢复失败，请点击“继续当前任务”" in s for s in statuses)
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert wf.recovery_attempted is True
    # the manual continue offer is kept so the UI button can retry
    assert wf.pending_continue is not None
    assert wf.pending_continue.session_id == "ses_test123"
    assert wf.pending_continue.message.message_id == "task-001"


def test_http_400_is_not_auto_recovered(tmp_path):
    """An ordinary HTTP 400 (no upstream isRetryable=false marker) is never
    auto-recovered in the same session and offers no fresh-session retry:
    the original send is the only send and the operator sees a plain note."""
    oc = BadRequestOpenChamber()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    with pytest.raises(OpenChamberBadRequestError, match="已跳过原会话自动恢复"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    # the continuation prompt was NEVER sent into the broken session
    assert [call for call in oc.call_log if call.startswith("send:")] == [
        "send:'do it\\n\\n[AI_RELAY_TASK_ID: task-001]'"
    ]
    assert wf.recovery_attempted is False
    assert wf.pending_continue is None  # continue button stays disabled
    assert wf.pending_rejection is None  # not a model rejection → no retry
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert "已跳过原会话自动恢复" in record["error"]


def test_manual_continue_keeps_original_task_id_and_new_response(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = RecoveryOpenChamber(recover_at=2)  # auto-recovery fails, manual works
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    with pytest.raises(OpenChamberInterruptedError):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf.pending_continue is not None
    assert wf.registry.record("task-001")["state"] == "FAILED"

    statuses: list[str] = []
    response = wf.continue_openchamber_task(statuses.append)
    message = parse_message(response)
    assert message.in_reply_to == "task-001"  # original TASK_ID kept
    assert message.message_id != "task-001"   # fresh RESPONSE_ID
    assert "recovered answer" in message.body
    assert any("正在人工继续当前任务" in s for s in statuses)
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"
    assert record["session_id"] == "ses_test123"
    assert wf.pending_continue is None  # cleared after manual success
    assert len(wf.registry.completed_records()) == 1


def test_continue_without_pending_task_is_rejected(tmp_path):
    wf = make_workflow(tmp_path, oc=scripted_oc(), default=TARGET_OPENCHAMBER)
    with pytest.raises(relay_mod.RelayWorkflowError, match="没有可继续"):
        wf.continue_openchamber_task()


# ---------------------------------------------------------------------- #
# non-retryable model rejection -> "新会话重试" (fresh session, once per task)
# ---------------------------------------------------------------------- #


def test_non_retryable_400_skips_auto_recovery_and_offers_new_session(tmp_path):
    """statusCode=400 + isRetryable=false must NEVER be auto-recovered in the
    original session: only the single original send happens, the continue
    offer stays empty and the fresh-session retry offer is kept."""
    oc = RejectedThenWorkingOpenChamber()
    wf = make_workflow(
        tmp_path, oc=oc,
        settings=RelaySettings(
            default_target=TARGET_OPENCHAMBER,
            openchamber_directory="D:/proj",
            openchamber_session_id="ses_rejected",
            openchamber_sessions={directory_key("D:/proj"): "ses_rejected"},
            completion_timeout=5.0,
            poll_interval=0.01,
        ),
    )
    with pytest.raises(
        OpenChamberModelRequestRejectedError, match="Qwen 拒绝当前会话"
    ) as ei:
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    exc = ei.value
    assert exc.status_code == 400
    assert exc.is_retryable is False
    assert exc.request_url == "http://192.168.100.190:8080/v1/chat/completions"
    # no continue was ever sent into the rejected session
    assert [call for call in oc.call_log if call.startswith("send:")] == [
        "send:'do it\\n\\n[AI_RELAY_TASK_ID: task-001]'"
    ]
    assert wf.recovery_attempted is False
    assert wf.pending_continue is None
    assert wf.pending_rejection is not None
    assert wf.pending_rejection.session_id == "ses_rejected"
    assert wf.pending_rejection.message.message_id == "task-001"
    assert wf.pending_rejection.directory == "D:/proj"
    assert wf.pending_rejection.agent is None
    assert wf.registry.record("task-001")["state"] == "FAILED"
    # NOT terminal yet: the "新会话重试" offer keeps the task resumable, so
    # side A must NOT receive a failure response (the final answer, success
    # or not, comes only once the retry / stop ends).
    assert wf.failure_response is None


def test_new_session_retry_creates_fresh_session_no_fork_and_keeps_task_id(
    tmp_path,
):
    """The retry creates a brand-new session in the SAME directory with the
    SAME agent/model and the ORIGINAL prompt (never a fork or continue), wraps
    under the original TASK_ID with a fresh RESPONSE_ID and persists the new
    session id both into the fixed field and the per-project mapping."""
    oc = RejectedThenWorkingOpenChamber()
    wf = make_workflow(
        tmp_path, oc=oc,
        settings=RelaySettings(
            default_target=TARGET_OPENCHAMBER,
            openchamber_directory="D:/proj",
            openchamber_session_id="ses_rejected",
            openchamber_sessions={directory_key("D:/proj"): "ses_rejected"},
            completion_timeout=5.0,
            poll_interval=0.01,
        ),
    )
    with pytest.raises(OpenChamberModelRequestRejectedError):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))

    statuses: list[str] = []
    response = wf.retry_model_rejected_task(statuses.append)
    message = parse_message(response)
    assert message.in_reply_to == "task-001"  # original TASK_ID kept
    assert message.message_id != "task-001"   # fresh RESPONSE_ID
    assert "fresh session answer" in message.body

    # exactly ONE fresh session was created (no fork, no third one)
    assert sum(call.startswith("create:") for call in oc.call_log) == 1
    creates = [call for call in oc.call_log if call.startswith("create:")]
    assert creates == ["create:AI Relay 重试 - proj:D:/proj"]
    sends = [call for call in oc.call_log if call.startswith("send:")]
    assert len(sends) == 2
    # the second send went into the FRESH session with the retry preface:
    # the ORIGINAL prompt is resent, never a fork or a "继续"
    fresh_users = [
        m for m in oc.history["ses_new"]
        if m.get("info", {}).get("role") == "user"
    ]
    assert len(fresh_users) == 1
    fresh_prompt = fresh_users[0]["parts"][0]["text"]
    assert fresh_prompt.startswith(MODEL_REJECTION_RETRY_PREFIX)
    assert "do it" in fresh_prompt
    assert "[AI_RELAY_TASK_ID: task-001]" in fresh_prompt
    assert "正在创建全新会话" in "\n".join(statuses)

    # the new session was persisted to BOTH the fixed field and the mapping
    assert wf.settings.openchamber_sessions[directory_key("D:/proj")] == "ses_new"
    assert wf.settings.openchamber_session_id == "ses_new"
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"
    assert record["session_id"] == "ses_new"
    assert wf.pending_rejection is None  # one-shot: cleared after the retry
    assert statuses[-1] == "恢复成功，回复已包装并复制"


def test_new_session_retry_stops_after_one_second_rejection(tmp_path):
    """A rejection on the FRESH session stops everything: exactly ONE new
    session is ever created, the offer is consumed and no third session (and
    no model switch) is attempted."""
    oc = AlwaysRejectedOpenChamber()
    wf = make_workflow(
        tmp_path, oc=oc,
        settings=RelaySettings(
            default_target=TARGET_OPENCHAMBER,
            openchamber_directory="D:/proj",
            openchamber_session_id="ses_rejected",
            openchamber_sessions={directory_key("D:/proj"): "ses_rejected"},
            completion_timeout=5.0,
            poll_interval=0.01,
        ),
    )
    with pytest.raises(OpenChamberModelRequestRejectedError, match="Qwen 拒绝"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf.pending_rejection is not None

    statuses: list[str] = []
    with pytest.raises(
        OpenChamberModelRequestRejectedError, match="新会话仍被模型服务拒绝"
    ):
        wf.retry_model_rejected_task(statuses.append)
    assert oc.created == ["ses_new_too"]  # exactly one creation, then stop
    assert wf.pending_rejection is None   # no further retry offer
    assert wf.registry.record("task-001")["state"] == "FAILED"
    assert not any("继续" in s for s in statuses)  # never touched the old session
    # terminal double-rejection: side A gets ONE structured failure bound to
    # the ORIGINAL TASK_ID (no third session, no silent stop).
    failure = wf.load_reply("task-001")
    assert failure is not None
    assert "TYPE: RESPONSE" in failure
    assert "IN_REPLY_TO: task-001" in failure
    assert wf.failure_response == failure


def test_model_rejection_variants_never_trigger_from_ambiguous_length(tmp_path):
    """ambiguous / length rounds are PLAIN session failures: they neither
    auto-recover nor offer a fresh-session retry."""
    oc = ScriptedOpenChamber(directory="D:/proj")
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="length",
                parts=[text_part("truncated")], parent_id="u_new",
            ),
        ]
    ]
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    with pytest.raises(OpenChamberSessionError, match="truncated by the model output"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf.pending_rejection is None
    assert wf.pending_continue is None
    assert wf.recovery_attempted is False

    oc2 = ScriptedOpenChamber(directory="D:/proj")
    oc2.status_timeline = ["idle", "idle", "idle"]
    oc2.message_timelines = [
        [user_message("u_new", "task body", 1000)],
        [assistant_message("a1", 1100, completed=1200, finish="stop",
                           parts=[text_part("x")], parent_id="u_new")],
        [
            user_message("u2", "task body", 2000),
            user_message("u3", "task body", 2100),
            assistant_message("a2", 2200, completed=2300, finish="stop",
                              parts=[text_part("ambiguous")], parent_id="u2"),
        ],
    ]
    wf2 = make_workflow(
        tmp_path / "w2", oc=oc2, default=TARGET_OPENCHAMBER
    )
    with pytest.raises(OpenChamberSessionError, match="ambiguous"):
        wf2.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf2.pending_rejection is None
    assert wf2.pending_continue is None


def test_ambiguous_task_is_not_auto_recovered(tmp_path, monkeypatch):
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = scripted_oc()
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            user_message("u_manual", "人工插入", 1001),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")], parent_id="u_new",
            ),
        ]
    ]
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    with pytest.raises(OpenChamberSessionError, match="ambiguous"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert [call for call in oc.call_log if call.startswith("send:")] \
        == ["send:'do it\\n\\n[AI_RELAY_TASK_ID: task-001]'"]
    assert wf.recovery_attempted is False
    assert wf.pending_continue is None
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"


def test_stop_cancels_wait_keeps_session_marks_stopped_by_user(tmp_path):
    oc = BusyOpenChamber()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    cancel = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            wf.process(
                v1_task(TARGET_OPENCHAMBER, "do it"), cancel_event=cancel
            )
        except BaseException as exc:  # noqa: BLE001 - collected for asserts
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.2)  # let the wait loop actually start
    cancel.set()
    thread.join(timeout=5)
    assert not thread.is_alive(), "stop did not unblock the wait loop"
    assert len(errors) == 1
    assert isinstance(errors[0], OpenChamberCancelledError)
    assert CANCELLED_MARKER in str(errors[0])
    record = wf.registry.record("task-001")
    assert record["state"] == "STOPPED_BY_USER"
    assert record["session_id"] == "ses_test123"
    assert wf.pending_continue is None
    # invariant 7: a user stop is a terminal answer, not a silent stop --
    # side A gets a structured stop response bound to the original TASK_ID.
    failure = wf.load_reply("task-001")
    assert failure is not None
    assert "TYPE: RESPONSE" in failure
    assert "IN_REPLY_TO: task-001" in failure
    assert "手动停止" in failure
    assert wf.failure_response == failure
    # the OpenChamber session is preserved, never deleted/terminated
    assert oc.session_exists("ses_test123", "D:/proj")


def test_stop_during_manual_continue_marks_stopped_by_user(tmp_path):
    oc = BusyOpenChamber()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    message = parse_message(v1_task(TARGET_OPENCHAMBER, "do it"))
    wf.pending_continue = OpenChamberContinue(
        message=message, session_id="ses_test123", directory="D:/proj",
    )
    cancel = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            wf.continue_openchamber_task(cancel_event=cancel)
        except BaseException as exc:  # noqa: BLE001 - collected for asserts
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.2)
    cancel.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(errors[0], OpenChamberCancelledError)
    assert CANCELLED_MARKER in str(errors[0])
    assert wf.registry.record("task-001")["state"] == "STOPPED_BY_USER"
    assert wf.pending_continue is None
    assert oc.session_exists("ses_test123", "D:/proj")


# ---------------------------------------------------------------------- #
# terminal-state protection against late / superseded worker writes
# ---------------------------------------------------------------------- #


def test_late_worker_cannot_move_a_completed_task(tmp_path):
    """Once a task is COMPLETED a late (superseded) worker write is a no-op:
    it can never flip the task back to PROCESSING or to another terminal
    state.  Only an explicit mark_reentry may move it on."""
    from core.task_registry import TERMINAL_STATES

    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf.registry.record("task-001")["state"] == "COMPLETED"

    # simulate a late worker still finishing its round
    assert wf.registry.mark_if_active(
        "task-001", "PROCESSING", executor=TARGET_OPENCHAMBER,
        session_id="ses_test123", directory="D:/proj",
    ) is False
    assert wf.registry.record("task-001")["state"] == "COMPLETED"
    # a late terminal write is blocked too (no COMPLETED -> FAILED flip)
    assert wf.registry.mark_if_active(
        "task-001", "FAILED", error="late", executor=TARGET_OPENCHAMBER
    ) is False
    assert wf.registry.record("task-001")["state"] == "COMPLETED"

    # explicit operator re-entry still works (manual continue of the round)
    assert "COMPLETED" in TERMINAL_STATES
    wf.registry.mark_reentry(
        "task-001", "PROCESSING", executor=TARGET_OPENCHAMBER,
        session_id="ses_test123", directory="D:/proj",
    )
    assert wf.registry.record("task-001")["state"] == "PROCESSING"


def test_late_worker_cannot_move_a_stopped_or_failed_task(tmp_path):
    """STOPPED_BY_USER and FAILED are also terminal for ordinary worker
    writes: a late worker cannot resurrect them to PROCESSING."""
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    message = parse_message(v1_task(TARGET_OPENCHAMBER, "do it"))

    # user stop of an interrupted task writes the terminal state (force)
    wf._finalize_failure(
        message, "STOPPED_BY_USER", "user stopped",
        executor=TARGET_OPENCHAMBER,
        session_id="ses_test123", directory="D:/proj",
        force=True,
    )
    assert wf.registry.record("task-001")["state"] == "STOPPED_BY_USER"
    assert wf.registry.mark_if_active(
        "task-001", "PROCESSING", executor=TARGET_OPENCHAMBER,
        session_id="ses_test123", directory="D:/proj",
    ) is False
    assert wf.registry.record("task-001")["state"] == "STOPPED_BY_USER"

    # a FAILED task (recovery exhausted) is protected the same way
    message2 = parse_message(v1_task(TARGET_OPENCHAMBER, "do it2", "task-002"))
    wf._finalize_failure(
        message2, "FAILED", "openchamber_execute:OpenChamberInterruptedError: 自动恢复失败",
        executor=TARGET_OPENCHAMBER,
        session_id="ses_test123", directory="D:/proj",
    )
    assert wf.registry.record("task-002")["state"] == "FAILED"
    assert wf.registry.mark_if_active(
        "task-002", "PROCESSING", executor=TARGET_OPENCHAMBER
    ) is False
    assert wf.registry.record("task-002")["state"] == "FAILED"
