"""Tests for the interrupted-continuation recovery path.

Covers: idle-blip confirmation (3 consecutive abnormal-idle polls), reset on
busy/growth/completion, limited transport-layer reconnect (SSE timeout /
connection reset), auto-recovery self-recovery during the 10s re-check, and
user cancellation during that window.
"""

from __future__ import annotations

import threading

import pytest

import core.openchamber as oc_mod
import core.relay as relay_mod
from core.openchamber import (
    ModelRef,
    OpenChamberCancelledError,
    OpenChamberInterruptedError,
    OpenChamberUnavailableError,
    wait_for_completion,
)
from core.relay import (
    CANCELLED_MARKER,
    RelayWorkflow,
)
from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
from core.task_registry import TaskRegistry
from tests.fakes import (
    assistant_message,
    make_dispatch,
    question_part,
    text_part,
    tool_part,
    user_message,
)

HISTORY = [
    user_message("u1", "old task", 100),
    assistant_message(
        "a1", 200, completed=300, finish="stop",
        parts=[text_part("old reply")], parent_id="u1",
    ),
]
HISTORY_IDS = frozenset({"u1", "a1"})
MODEL = ModelRef("4090", "qwen3.8-27b")


def empty_round() -> list:
    return HISTORY + [
        user_message("u2", "new task", 1000),
        assistant_message(
            "a2", 1100, completed=None, finish=None, parts=[], parent_id="u2",
        ),
    ]


def completed_round(body: str = "final answer") -> list:
    return HISTORY + [
        user_message("u2", "new task", 1000),
        assistant_message(
            "a2", 1100, completed=1300, finish="stop",
            parts=[text_part(body)], parent_id="u2",
        ),
    ]


class StepSession:
    """A configured fixed session driven by an explicit (status, messages)
    step list.  ``session_status`` advances the step; ``messages`` always
    returns that same step's payload, so idle/busy/growth/completion
    transitions are expressed unambiguously (no shared-call interleaving)."""

    def __init__(self, steps, session_id="ses_test123", directory="D:/proj"):
        self.steps = list(steps)
        self.session_id = session_id
        self.directory = directory
        self.call_log: list[str] = []
        self._status_index = 0
        self._current_step = 0
        self.send_dispatch = None

    def _status_type(self, status):
        return status

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        return self.session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list:
        self.call_log.append(f"list:{directory or ''}")
        if directory is not None and directory != self.directory:
            return []
        return [(self.session_id, "step session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.session_id and directory == self.directory

    def session_status(self, session_id: str, directory: str) -> str:
        idx = min(self._status_index, len(self.steps) - 1)
        self._current_step = idx
        self._status_index += 1
        return self.steps[idx][0]

    def messages(self, session_id: str, directory: str) -> list:
        return list(self.steps[self._current_step][1])

    def round_has_pending_user_action(
        self, session_id: str, directory: str, dispatch
    ) -> bool:
        for message in self.messages(session_id, directory):
            for part in message.get("parts") or []:
                if part.get("type") not in ("tool", "permission"):
                    continue
                state = part.get("state")
                if isinstance(state, dict) and state.get("status") == "pending":
                    return True
        return False

    def send(self, session_id, prompt, directory, agent=None, model=None):
        self.call_log.append(f"send:{prompt!r}")
        self.send_dispatch = make_dispatch(
            session_id=session_id,
            directory=directory,
            pre_ids=set(HISTORY_IDS),
            user_message_id="u2",
        )
        return self.send_dispatch

    def close(self) -> None:
        pass


class FlakySession(StepSession):
    """A StepSession that raises transient transport errors (SSE timeout /
    connection reset) for the first ``fail_for`` status polls, then behaves
    normally."""

    def __init__(self, steps, fail_for=2, **kw):
        super().__init__(steps, **kw)
        self.fail_for = fail_for
        self.fail_message = "SSE read timed out"

    def session_status(self, session_id: str, directory: str) -> str:
        if self._status_index < self.fail_for:
            self.call_log.append("transport_fail")
            self._status_index += 1
            raise OpenChamberUnavailableError(self.fail_message)
        return super().session_status(session_id, directory)

    def messages(self, session_id: str, directory: str) -> list:
        if self._status_index < self.fail_for:
            self.call_log.append("transport_fail_messages")
            raise OpenChamberUnavailableError(self.fail_message)
        return super().messages(session_id, directory)


def run(fake, dispatch, timeout=5.0):
    return wait_for_completion(
        fake, dispatch, timeout, poll_interval=0.01, grace_seconds=0.05
    )


def new_dispatch(**kwargs):
    kwargs.setdefault("pre_ids", HISTORY_IDS)
    kwargs.setdefault("requested", MODEL)
    kwargs.setdefault("resolved", MODEL)
    return make_dispatch(directory="D:/proj", **kwargs)


# ---------------------------------------------------------------------- #
# idle-blip confirmation: 3 consecutive abnormal-idle polls required
# ---------------------------------------------------------------------- #


def test_three_consecutive_empty_idle_confirms_interruption():
    """tool-calls -> idle -> empty assistant repeated three times must raise
    an interrupted continuation (never a success)."""
    fake = StepSession([("idle", empty_round())])
    with pytest.raises(OpenChamberInterruptedError, match="not_completed"):
        run(fake, new_dispatch(), timeout=0.0)


def test_single_idle_blip_then_busy_then_complete_is_success():
    """A one-off idle with an empty message must NOT be misread: when the
    session goes busy and then completes, the wait succeeds."""
    fake = StepSession(
        [
            ("idle", empty_round()),
            ("busy", completed_round("working now")),
            ("idle", completed_round("final answer")),
        ]
    )
    result = run(fake, new_dispatch(), timeout=0.0)
    assert result.final_text == "final answer"
    assert result.finish == "stop"


def test_idle_text_growth_then_complete_is_success():
    """Growing assistant text while idle must reset the confirmation counter
    (streaming is not an interruption); an eventual completed reply wins."""
    fake = StepSession(
        [
            ("idle", HISTORY + [
                user_message("u2", "new task", 1000),
                assistant_message(
                    "a2", 1100, completed=None, finish=None,
                    parts=[text_part("g")], parent_id="u2",
                ),
            ]),
            ("idle", HISTORY + [
                user_message("u2", "new task", 1000),
                assistant_message(
                    "a2", 1100, completed=None, finish=None,
                    parts=[text_part("gr")], parent_id="u2",
                ),
            ]),
            ("idle", HISTORY + [
                user_message("u2", "new task", 1000),
                assistant_message(
                    "a2", 1100, completed=1300, finish="stop",
                    parts=[text_part("growing final")], parent_id="u2",
                ),
            ]),
        ]
    )
    result = run(fake, new_dispatch(), timeout=0.0)
    assert result.final_text == "growing final"
    assert result.finish == "stop"


def test_completed_stop_nonempty_is_success():
    """A completed + finish=stop + non-empty body round is a normal success."""
    fake = StepSession([("idle", completed_round("done"))])
    result = run(fake, new_dispatch(), timeout=0.0)
    assert result.final_text == "done"
    assert result.finish == "stop"


# ---------------------------------------------------------------------- #
# transport-layer limited reconnect (SSE timeout / connection reset)
# ---------------------------------------------------------------------- #


def test_transport_retry_recovers_without_raising(monkeypatch):
    """Two transient transport failures are retried (with cancel-aware
    delays), then the round completes normally — no interruption raised."""
    monkeypatch.setattr(oc_mod, "TRANSPORT_RETRY_DELAYS", (0.0, 0.0))
    fake = FlakySession(
        [("idle", completed_round("after reconnect"))], fail_for=2
    )
    result = run(fake, new_dispatch(), timeout=0.0)
    assert result.final_text == "after reconnect"
    assert fake.call_log.count("transport_fail") == 2


def test_transport_failures_beyond_limit_escalate_to_interruption(monkeypatch):
    """More than MAX_TRANSPORT_RETRIES transport failures escalate to an
    interrupted continuation (the relay then auto-recovers once)."""
    monkeypatch.setattr(oc_mod, "TRANSPORT_RETRY_DELAYS", (0.0, 0.0))
    fake = FlakySession(
        [("idle", completed_round("late"))], fail_for=5
    )
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    assert excinfo.value.reason == "transport_failure"


# ---------------------------------------------------------------------- #
# auto-recovery: 10s re-check and self-recovery
# ---------------------------------------------------------------------- #


def _make_workflow(tmp_path, oc, settings):
    return RelayWorkflow(
        reasonix=None,
        registry=TaskRegistry(tmp_path / "tasks.json"),
        settings=settings,
        openchamber=oc,
        replies_dir=tmp_path / "replies",
    )


def _task(target=TARGET_OPENCHAMBER, body="do it"):
    headers = [
        "AI_RELAY/1",
        "MESSAGE_ID: task-001",
        "SOURCE: CHATGPT",
        f"TARGET: {target}",
        "TYPE: TASK",
        "ROUND: 1",
        "MAX_ROUNDS: 3",
    ]
    return "\n".join((*headers, "", body))


def _settings(completion_timeout=5.0):
    return RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="4090/qwen3.8-27b",
        openchamber_agent="build",
        completion_timeout=completion_timeout,
        poll_interval=0.01,
    )


def test_auto_recover_recheck_waits_then_continues_once(tmp_path, monkeypatch):
    """First interrupt: after the 10s re-check the round is STILL interrupted,
    so the relay sends exactly ONE automatic continuation in the original
    session (same TASK_ID, same Agent and Model)."""
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    from tests.test_relay_workflow import RecoveryOpenChamber  # reuse

    oc = RecoveryOpenChamber(recover_at=1)
    wf = _make_workflow(tmp_path, oc, _settings())
    statuses: list[str] = []
    response = wf.process(_task(), statuses.append)
    assert "recovered answer" in response
    assert any("正在原会话自动续接（1/1）" in s for s in statuses)
    # original task + ONE auto continuation only
    assert sum(c.startswith("send:") for c in oc.call_log) == 2
    assert wf.recovery_attempted is True


def test_auto_recover_self_recovery_does_not_send_continue_does_not_consume(
    tmp_path, monkeypatch
):
    """When the session recovers ON ITS OWN during the re-check window, the
    relay returns that reply WITHOUT sending a continuation and WITHOUT
    consuming the automatic recovery slot."""
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    # 3 empty-idle polls (confirmation), then the session resumes and completes.
    steps = (
        [("idle", empty_round())] * 3
        + [("busy", completed_round("self recovered")),
           ("idle", completed_round("self recovered"))]
    )
    oc = StepSession(steps)
    wf = _make_workflow(tmp_path, oc, _settings(completion_timeout=0.0))
    statuses: list[str] = []
    response = wf.process(_task(), statuses.append)
    assert "self recovered" in response
    # The original task send happened, but NO auto-continuation was sent:
    # the session recovered on its own.
    assert sum(c.startswith("send:") for c in oc.call_log) == 1
    assert wf.recovery_attempted is False


def test_cancel_interrupts_auto_recover_recheck(tmp_path, monkeypatch):
    """User cancellation aborts the 10s re-check window and marks the task
    stopped, keeping the OpenChamber session."""
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 30.0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = StepSession([("idle", empty_round())])
    cancel = threading.Event()
    wf = _make_workflow(tmp_path, oc, _settings())

    def delayed_cancel():
        import time

        time.sleep(0.05)
        cancel.set()

    th = threading.Thread(target=delayed_cancel, daemon=True)
    th.start()
    with pytest.raises(Exception) as excinfo:
        wf.process(_task(), cancel_event=cancel)
    th.join(timeout=2)

    # Should surface as a user-cancellation (the CANCELLED marker is added
    # when OpenChamberCancelledError is re-raised by the relay).
    value = excinfo.value
    assert isinstance(value, OpenChamberCancelledError) or CANCELLED_MARKER in str(value)
    record = wf.registry.record("task-001")
    assert record["state"] == "STOPPED_BY_USER"


# ---------------------------------------------------------------------- #
# status map vanishing: missing_from_status_map confirmation
# ---------------------------------------------------------------------- #


def in_progress_round() -> list:
    return HISTORY + [
        user_message("u2", "new task", 1000),
        assistant_message(
            "a2", 1100, completed=None, finish=None,
            parts=[text_part("working")], parent_id="u2",
        ),
    ]


def tool_calls_round() -> list:
    return HISTORY + [
        user_message("u2", "new task", 1000),
        assistant_message(
            "a2", 1100, completed=1200, finish="tool-calls",
            parts=[tool_part("shell")],
            parent_id="u2",
        ),
    ]


MISSING = "missing_from_status_map"


def test_missing_after_activity_confirms_interruption(caplog):
    """busy then the session vanishes from the status map with no valid
    final reply: three consecutive polls raise the interrupted error with
    the status-missing reason (never a plain idle reason)."""
    import logging

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    fake = StepSession([("busy", in_progress_round()), (MISSING, empty_round())])
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    assert excinfo.value.reason == "status_missing_after_activity"
    assert "reason=status_missing_after_activity" in caplog.text
    assert "idle_confirmations=1/3" in caplog.text
    assert "idle_confirmations=3/3" in caplog.text


def test_missing_from_start_confirms_after_three_polls():
    """The dispatch is a confirmed send (the prompt is recorded in the
    session), so even a status map that NEVER listed the session (service
    restart / degraded status endpoint) confirms the stall after three
    polls -- it cannot mask a stuck round forever."""
    fake = StepSession([(MISSING, empty_round())])
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    assert excinfo.value.reason == "status_missing_after_activity"


def test_missing_then_busy_resets_and_completion_wins():
    """A vanished session that comes back busy (and later completes) must
    never be declared interrupted: busy resets the missing confirmations."""
    fake = StepSession(
        [
            ("busy", in_progress_round()),
            (MISSING, empty_round()),          # confirmation 1
            ("busy", in_progress_round()),     # reset
            (MISSING, empty_round()),          # confirmation 1 again
            ("idle", completed_round("final answer")),
        ]
    )
    result = run(fake, new_dispatch(), timeout=0.0)
    assert result.final_text == "final answer"
    assert result.finish == "stop"


def test_missing_then_explicit_idle_completion_succeeds():
    """missing (confirmation) -> busy (reset) -> explicit idle with a
    completed stop reply is a normal success."""
    fake = StepSession(
        [
            ("busy", in_progress_round()),
            (MISSING, empty_round()),
            ("busy", in_progress_round()),
            ("idle", completed_round("late final")),
        ]
    )
    result = run(fake, new_dispatch(), timeout=0.0)
    assert result.final_text == "late final"


def test_missing_after_tool_calls_waits_grace_then_confirms():
    """The session vanished right after a tool-calls message with no
    continuation reply: after a short grace window the same 3-poll
    confirmation raises the interruption (status-missing reason)."""
    fake = StepSession([(MISSING, tool_calls_round())])
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    assert excinfo.value.reason == "status_missing_after_activity"
    assert excinfo.value.last_finish == "tool-calls"


def test_missing_and_idle_alternation_share_confirmation(caplog):
    """missing -> idle -> missing with an unchanged round keep counting
    (the abnormal signature is identical): the third poll interrupts."""
    import logging

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    fake = StepSession(
        [
            ("busy", in_progress_round()),
            (MISSING, empty_round()),   # 1
            ("idle", empty_round()),    # 2 (same signature)
            (MISSING, empty_round()),   # 3 -> interrupted
        ]
    )
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    assert excinfo.value.reason == "status_missing_after_activity"
    assert "idle_confirmations=3/3" in caplog.text


def test_missing_text_growth_resets_confirmations():
    """Growing assistant text while the session stays vanished from the
    status map resets the counter (the round is still producing)."""
    def round_with(body: str) -> list:
        return HISTORY + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, completed=None, finish=None,
                parts=[text_part(body)], parent_id="u2",
            ),
        ]

    fake = StepSession(
        [
            ("busy", in_progress_round()),
            (MISSING, round_with("g")),          # 1
            (MISSING, round_with("gr")),         # growth -> reset
            (MISSING, round_with("gre")),        # 1
            (MISSING, round_with("gre")),        # 2
            (MISSING, round_with("gre")),        # 3 -> interrupted
        ]
    )
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    assert excinfo.value.reason == "status_missing_after_activity"


def test_unknown_status_resets_missing_confirmations(monkeypatch):
    """A malformed status payload (unknown) between vanished polls is a
    transport/parse issue: it resets the missing counter, so with
    interleaved unknown polls the stall is never confirmed before the
    deadline -> timeout, not interruption."""
    import core.openchamber as oc

    from core.openchamber import OpenChamberTimeoutError

    clock = {"now": 0.0}

    def slow_monotonic() -> float:
        clock["now"] += 0.5
        return clock["now"]

    monkeypatch.setattr(oc.time, "monotonic", slow_monotonic)
    fake = StepSession(
        [
            ("busy", in_progress_round()),
            (MISSING, empty_round()),    # 1
            ("unknown", empty_round()),  # reset
            (MISSING, empty_round()),    # 1
            ("unknown", empty_round()),  # reset
        ]
    )
    with pytest.raises(OpenChamberTimeoutError):
        run(fake, new_dispatch(), timeout=10.0)


def test_missing_with_pending_user_action_waits_never_confirms():
    """A round waiting on the operator's question/permission must never be
    declared interrupted just because the status map lost the session."""
    from core.openchamber import OpenChamberTimeoutError

    pending_round = HISTORY + [
        user_message("u2", "new task", 1000),
        assistant_message(
            "a2", 1100, completed=None, finish=None,
            parts=[question_part("pending")], parent_id="u2",
        ),
    ]
    fake = StepSession([(MISSING, pending_round)])
    with pytest.raises(OpenChamberTimeoutError):
        run(fake, new_dispatch(), timeout=5.0)


def test_cancel_during_missing_wait():
    fake = StepSession([("busy", in_progress_round()), (MISSING, empty_round())])
    cancel = threading.Event()

    def delayed_cancel():
        import time

        time.sleep(0.05)
        cancel.set()

    th = threading.Thread(target=delayed_cancel, daemon=True)
    th.start()
    with pytest.raises(OpenChamberCancelledError):
        wait_for_completion(
            fake, new_dispatch(), 0.0,
            poll_interval=0.01, grace_seconds=0.05, cancel_event=cancel,
        )
    th.join(timeout=2)


def test_missing_interruption_carries_round_context():
    fake = StepSession([("busy", in_progress_round()), (MISSING, empty_round())])
    with pytest.raises(OpenChamberInterruptedError) as excinfo:
        run(fake, new_dispatch(), timeout=0.0)
    err = excinfo.value
    assert err.reason == "status_missing_after_activity"
    assert err.session_id == "ses_test123"
    assert err.last_message_id == "a2"


# ---------------------------------------------------------------------- #
# relay level: status-missing interruption feeds the automatic recovery
# ---------------------------------------------------------------------- #


class MissingRecoverySession(StepSession):
    """A fixed session whose status follows an explicit timeline (a step
    list would clamp forever); after the relay (or a monitor) re-sends a
    continuation, the session replies with a verified completed round
    (``replies_after_send=False`` keeps it vanished for exhaustion tests)."""

    def __init__(
        self,
        timeline,
        replies_after_send: bool = True,
        session_id="ses_test123",
        directory="D:/proj",
    ):
        super().__init__([("idle", [])], session_id=session_id, directory=directory)
        self._timeline = list(timeline)
        self._t = 0
        self._post_send = False
        self._replies_after_send = replies_after_send
        self._send_count = 0

    def session_status(self, session_id, directory):
        idx = min(self._t, len(self._timeline) - 1)
        self._t += 1
        return self._timeline[idx]

    def messages(self, session_id, directory):
        # The round only replies once a CONTINUATION prompt (send #2) was
        # delivered; after the original task send (send #1) the session
        # still shows the in-progress round.
        if self._send_count >= 2 and self._replies_after_send:
            return completed_round("recovered after missing")
        return empty_round()

    def send(self, session_id, prompt, directory, agent=None, model=None):
        self.call_log.append(f"send:{prompt!r}")
        self._send_count += 1
        if self._send_count >= 2:
            self._post_send = True
        if self.send_dispatch is None:
            self.send_dispatch = make_dispatch(
                session_id=session_id,
                directory=directory,
                pre_ids=set(HISTORY_IDS),
                user_message_id="u2",
            )
        return self.send_dispatch


def test_status_missing_triggers_auto_recover_then_completes(tmp_path, monkeypatch):
    """The reported failure shape: the status map lost the busy session.
    Three vanished polls raise the interruption; the relay's ONE automatic
    continuation (the status re-check sees the vanished session as idle)
    lands and the round completes under the ORIGINAL TASK_ID."""
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = MissingRecoverySession(
        [
            "busy",                       # original round working
            MISSING, MISSING, MISSING,    # vanished -> 3 confirmations
            MISSING,                      # re-check loop poll (== idle)
            MISSING,                      # final re-check (== idle)
            "idle",                       # post-continuation wait polls
        ]
    )
    wf = _make_workflow(tmp_path, oc, _settings(completion_timeout=0.0))
    statuses: list[str] = []
    response = wf.process(_task(), statuses.append)
    assert "recovered after missing" in response
    assert any("正在原会话自动续接（1/1）" in s for s in statuses)
    # original task + exactly ONE automatic continuation
    assert sum(c.startswith("send:") for c in oc.call_log) == 2
    assert wf.recovery_attempted is True
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"


def test_status_missing_recovery_exhausted_fails_and_keeps_pending(
    tmp_path, monkeypatch
):
    """When even the automatic continuation leaves the round vanished, the
    task ends FAILED (never PROCESSING) with the manual continue/stop offer
    kept for the operator."""
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    oc = MissingRecoverySession(
        ["busy"] + [MISSING] * 12, replies_after_send=False
    )
    wf = _make_workflow(tmp_path, oc, _settings(completion_timeout=0.0))
    with pytest.raises(Exception) as excinfo:
        wf.process(_task())
    # the relay re-raises the exhausted-recovery interrupted error
    assert "自动恢复失败" in str(excinfo.value)
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert wf.pending_continue is not None
    assert wf.pending_continue.session_id == "ses_test123"
    assert wf.pending_continue.directory == "D:/proj"
