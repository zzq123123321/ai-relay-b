"""RelayWorkflow routing, ordering, failure semantics and persistence."""

from __future__ import annotations

import pytest

from core.openchamber import (
    ModelRef,
    OpenChamberSessionError,
    OpenChamberTimeoutError,
)
from core.protocol import ProtocolFormat, parse_message
from core.relay import RelayWorkflow, resolve_executor_kind
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


def v1_task(target: str, body: str, message_id: str = "task-001") -> str:
    return "\n".join(
        (
            "AI_RELAY/1",
            f"MESSAGE_ID: {message_id}",
            "SOURCE: CHATGPT",
            f"TARGET: {target}",
            "TYPE: TASK",
            "ROUND: 1",
            "MAX_ROUNDS: 3",
            "",
            body,
        )
    )


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
    assert oc.call_log.index("open:ses_test123") < oc.call_log.index(
        "send:'do it'"
    )
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


def test_openchamber_missing_session_id_fails_clearly(tmp_path):
    """No configured session id: the task fails with a clear prompt and the
    relay neither guesses a session nor creates one."""
    oc = scripted_oc()
    settings = RelaySettings(
        default_target=TARGET_REASONIX, openchamber_directory="D:/proj"
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(RuntimeError, match="会话 ID 未配置"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert oc.call_log == []  # OpenChamber was never touched
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert "missing session id" in record["error"]


def test_configured_session_not_found_does_not_fallback(tmp_path):
    """The configured session does not exist in the directory: clear error,
    no send, and NEVER a fallback to another session."""
    oc = scripted_oc()  # only knows ses_test123
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_gone",
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(RuntimeError, match="不存在"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert not any(call.startswith("send:") for call in oc.call_log)
    assert not any(call.startswith("create:") for call in oc.call_log)
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_gone"
    assert wf.load_reply("task-001") is None


def test_relay_settings_persist_session_id(tmp_path):
    path = tmp_path / "relay_settings.json"
    settings = RelaySettings(openchamber_session_id="ses_x")
    settings.save(path)
    assert RelaySettings.load(path).openchamber_session_id == "ses_x"
    assert (
        RelaySettings.load(tmp_path / "missing").openchamber_session_id == ""
    )


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
    assert wf.load_reply("task-001") is None


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
    assert wf.load_reply("task-001") is None


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