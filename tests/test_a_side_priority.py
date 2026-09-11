"""A-side TASK priority invariants: highest priority + final-reply exclusivity.

These tests pin the CODE guarantees (not operator discipline) behind the
A-side priority contract:

1. While a legitimate A-side TASK is in flight on a session, the manual
   OpenChamber monitor may only OBSERVE that session: it must NOT wrap its
   replies, auto-continue it, or raise stall interruptions for it.  The
   monitor resumes wrapping only after the task reaches a terminal state.
2. The final reply of an in-flight A-side task is consumed ONLY by the auto
   relay and stays bound to the ORIGINAL TASK_ID (never a manual-* id); the
   monitor never re-wraps it afterwards.
3. A task that cannot finish (failure or user stop) still answers side A with
   a structured RESPONSE bound to the original TASK_ID and lands in a
   terminal registry state -- never left silently in PROCESSING.
4. A repeated TASK_ID is never re-executed: side A gets the existing state or
   the already-saved reply.
5. Ordinary (non-protocol) clipboard text is never an A-side task and never
   enters the task chain.
"""

from __future__ import annotations

import time

from PySide6.QtTest import QTest

from core.openchamber import OpenChamberSessionError, is_message_wrapped
from core.protocol import parse_message
from core.relay import directory_key
from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
from tests.fakes import (
    ScriptedOpenChamber,
    assistant_message,
    text_part,
    user_message,
)
from tests.test_oc_monitor import (
    MutableMessages,
    completed_reply,
    manual_completed_ids,
)
from tests.test_relay_workflow import (
    BusyOpenChamber,
    v1_task,
)
from tests.test_ui_startup import build_window, shutdown_window, wait_until

import ui as ui_mod


def _completing_oc(tmp_path, reply_id: str, reply_text: str) -> ScriptedOpenChamber:
    """An OpenChamber fake that completes the task round with a UNIQUE reply
    id (``reply_id``).  A unique id keeps the process-global
    ``mark_message_wrapped`` registry hermetic: these tests run before
    ``test_oc_monitor`` and must not pre-wrap the ids that module's
    ``monitor_scan`` unit tests rely on (``a_new`` / ``a_later`` / ...)."""
    oc = ScriptedOpenChamber(directory=str(tmp_path))
    oc.sessions = [("ses_test123", "Test session", str(tmp_path))]
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_pri", "task body", 1000),
            assistant_message(
                reply_id,
                1100,
                completed=1200,
                finish="stop",
                parts=[text_part(reply_text)],
                parent_id="u_pri",
            ),
        ]
    ]
    return oc


def build_priority_window(qapp, monkeypatch, tmp_path, monitor_fake, task_oc):
    """A window whose A-side relay runs against ``task_oc`` while the manual
    monitor polls ``monitor_fake``; both reference the SAME session id and
    project directory so the in-flight ownership guard can match them."""
    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory=str(tmp_path),
        openchamber_session_id="ses_test123",
        openchamber_sessions={directory_key(str(tmp_path)): "ses_test123"},
        poll_interval=0.05,
        completion_timeout=0,  # no timeout: a busy session waits until cancelled
    )
    monkeypatch.setattr(ui_mod, "OpenChamberClient", lambda url: monitor_fake)
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=task_oc
    )
    window._startup_check_pending = False
    return window


def _start_task(qapp, window, task_id: str, workdir: str):
    qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", task_id, workdir))
    assert wait_until(lambda: window._busy), "A-side task did not start"


def test_monitor_yields_to_in_flight_a_side_task_and_resumes_after_stop(
    qapp, monkeypatch, tmp_path
):
    """Invariant 1/4: an in-flight A-side task owns the session; the monitor
    observes but neither wraps nor auto-continues it.  After the task is
    stopped (terminal), the monitor resumes wrapping new replies."""
    qapp.clipboard().clear()
    # pre-existing history reply: keeps the monitor quiet (no false stall)
    monitor_fake = MutableMessages([completed_reply("a_pri_hist", created=1000, text="history")])
    task_oc = BusyOpenChamber(session_id="ses_test123", directory=str(tmp_path))
    window = build_priority_window(
        qapp, monkeypatch, tmp_path, monitor_fake, task_oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"

        _start_task(qapp, window, "task-pri-001", str(tmp_path))
        # in-flight ownership registered at claim time (the project directory)
        assert directory_key(str(tmp_path)) in window._a_side_inflight_dirs

        # start the manual monitor on the SAME session as the in-flight task
        window._toggle_monitor()
        assert window._oc_monitor_active

        # a final reply "arrives" while the task is still in flight: the
        # monitor must observe it but NOT wrap it (no manual-* id, and the
        # message is left un-wrapped for the auto relay to consume).
        monitor_fake.add(
            completed_reply("a_pri_task", created=4000, text="task final")
        )
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            QTest.qWait(40)
        assert manual_completed_ids(window) == set(), "monitor wrapped during a task"
        assert not is_message_wrapped("a_pri_task", "ses_test123"), "auto-owned reply was consumed"
        # no auto-continue was ever fired into the task's session
        assert window._oc_monitor_auto_count == 0
        assert monitor_fake.sent == []

        # stop the A-side task -> terminal STOPPED_BY_USER, and side A gets a
        # structured stop response bound to the ORIGINAL TASK_ID.
        window.stop_button.click()
        assert wait_until(lambda: not window._busy), "task did not stop"
        assert not window._a_side_inflight_dirs
        assert not window._a_side_inflight_sessions
        record = window._workflow.registry.record("task-pri-001")
        assert record is not None and record["state"] == "STOPPED_BY_USER"
        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-pri-001" in (clipboard.text() or "")
        ), "stop response was not returned to side A"
        stopped = parse_message(clipboard.text())
        assert stopped.in_reply_to == "task-pri-001"
        assert "手动停止" in stopped.body

        # after the task ended, the monitor resumes wrapping: a NEW reply
        # (different id) now gets a manual-* wrap, while the yielded reply is
        # never re-wrapped by the monitor.
        monitor_fake.add(completed_reply("a_pri_later", created=5000, text="later reply"))
        assert wait_until(
            lambda: "later reply" in (clipboard.text() or "")
        ), "monitor did not resume wrapping after the task ended"
        later = parse_message(clipboard.text())
        assert later.in_reply_to.startswith("manual-")
        assert not is_message_wrapped("a_pri_task", "ses_test123")
    finally:
        shutdown_window(window)


def test_a_side_final_reply_bound_to_original_task_id_and_exclusive(
    qapp, monkeypatch, tmp_path
):
    """Invariant 2: the completed task's final reply is written under the
    ORIGINAL TASK_ID with a fresh RESPONSE_ID; a monitor running on the same
    session never re-wraps that reply (the auto relay owns it), and a
    genuinely new reply wraps as manual-* without touching the completed
    task record."""
    qapp.clipboard().clear()
    monitor_fake = MutableMessages([completed_reply("a_pri_hist", created=1000, text="history")])
    task_oc = _completing_oc(tmp_path, "a_pri_final", "final answer")
    window = build_priority_window(
        qapp, monkeypatch, tmp_path, monitor_fake, task_oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"

        # monitor starts BEFORE the task: its baseline has no knowledge of
        # the reply the task will produce.
        window._toggle_monitor()
        assert window._oc_monitor_active

        _start_task(qapp, window, "task-pri-002", str(tmp_path))
        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-pri-002" in (clipboard.text() or "")
        ), "A-side final reply was not returned under the original TASK_ID"
        response = parse_message(clipboard.text())
        assert response.in_reply_to == "task-pri-002"
        assert response.message_id != "task-pri-002", "RESPONSE_ID must be fresh"
        record = window._workflow.registry.record("task-pri-002")
        assert record is not None and record["state"] == "COMPLETED"
        # the auto relay consumed the task's reply
        assert is_message_wrapped("a_pri_final", "ses_test123")

        # surface the SAME reply id into the monitor's view: the poller sees
        # it, but the auto relay already wrapped it -> no manual re-wrap.
        monitor_fake.add(
            completed_reply("a_pri_final", created=1200, text="final answer")
        )
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            QTest.qWait(40)
        assert manual_completed_ids(window) == set(), "monitor re-wrapped the task reply"

        # a genuinely new late reply wraps as manual-* and leaves the
        # completed task record untouched.
        monitor_fake.add(completed_reply("a_pri_late", created=6000, text="late reply"))
        assert wait_until(
            lambda: "late reply" in (clipboard.text() or "")
        ), "monitor did not wrap the late reply"
        late = parse_message(clipboard.text())
        assert late.in_reply_to.startswith("manual-")
        assert window._workflow.registry.record("task-pri-002")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


class _FailingOc(BusyOpenChamber):
    """A session whose task send aborts: a terminal, non-retryable failure
    that must still produce a structured failure response."""

    def send(self, session_id, prompt, directory, agent=None, model=None):
        raise OpenChamberSessionError("Aborted")


def test_a_side_failure_returns_structured_response_and_reaches_terminal(
    qapp, monkeypatch, tmp_path
):
    """Invariant 3/7: a task that truly fails still answers side A with a
    structured failure bound to the ORIGINAL TASK_ID and lands in a terminal
    (FAILED) state, never silently left in PROCESSING."""
    qapp.clipboard().clear()
    monitor_fake = MutableMessages([completed_reply("a_pri_hist", created=1000, text="history")])
    task_oc = _FailingOc(session_id="ses_test123", directory=str(tmp_path))
    window = build_priority_window(
        qapp, monkeypatch, tmp_path, monitor_fake, task_oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"

        _start_task(qapp, window, "task-pri-003", str(tmp_path))
        assert wait_until(lambda: not window._busy), "task did not settle"
        record = window._workflow.registry.record("task-pri-003")
        assert record is not None and record["state"] == "FAILED"
        assert record.get("error"), "terminal failure must record an explicit error"

        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-pri-003" in (clipboard.text() or "")
        ), "failure response was not returned to side A"
        failure = parse_message(clipboard.text())
        assert failure.in_reply_to == "task-pri-003"
        assert "任务执行失败" in failure.body
        # the in-flight ownership is released so the monitor may act again
        assert not window._a_side_inflight_dirs
        assert not window._a_side_inflight_sessions
    finally:
        shutdown_window(window)


def test_duplicate_task_id_not_reexecuted_returns_existing_state_or_reply(
    qapp, monkeypatch, tmp_path
):
    """Invariant 8: re-sending the same TASK_ID never re-executes the task;
    side A is answered with the existing state (in flight) or the saved
    reply (after completion), always bound to the original TASK_ID."""
    qapp.clipboard().clear()
    monitor_fake = MutableMessages([completed_reply("a_pri_hist", created=1000, text="history")])
    task_oc = _completing_oc(tmp_path, "a_pri_dup", "dup final answer")
    window = build_priority_window(
        qapp, monkeypatch, tmp_path, monitor_fake, task_oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"

        task_text = v1_task("OPENCHAMBER", "do it", "task-dup-001", str(tmp_path))
        clipboard = qapp.clipboard()
        clipboard.setText(task_text)
        # the FIRST copy is picked up and completed
        assert wait_until(
            lambda: "IN_REPLY_TO: task-dup-001" in (clipboard.text() or "")
        ), "first task copy was not relayed"
        first_reply = clipboard.text()
        assert len([c for c in task_oc.call_log if c.startswith("send:")]) == 1

        # re-send the SAME task id: it must NOT re-execute; side A gets the
        # saved reply again (bound to the original TASK_ID).  The saved reply
        # is byte-identical to the first one, so the duplicate handling is
        # observed via the window's "重复任务" status, not the clipboard.
        clipboard.setText(task_text)
        assert wait_until(
            lambda: "重复任务" in (window.detail_label.text() or "")
        ), "duplicate TASK_ID was not answered with the existing state/reply"
        duplicate = parse_message(clipboard.text())
        assert duplicate.in_reply_to == "task-dup-001"
        assert duplicate.body == parse_message(first_reply).body, (
            "duplicate must return the SAVED reply, not a new execution"
        )
        assert len([c for c in task_oc.call_log if c.startswith("send:")]) == 1
        assert window._workflow.registry.record("task-dup-001")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


def test_ordinary_clipboard_text_is_not_an_a_side_task(
    qapp, monkeypatch, tmp_path
):
    """Invariant 8: arbitrary (non-protocol) clipboard text is never treated
    as an A-side task, never registered and never executed."""
    qapp.clipboard().clear()
    monitor_fake = MutableMessages()
    task_oc = BusyOpenChamber(session_id="ses_test123", directory=str(tmp_path))
    window = build_priority_window(
        qapp, monkeypatch, tmp_path, monitor_fake, task_oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"

        qapp.clipboard().setText("just some random notes that are not a relay message")
        QTest.qWait(200)
        # nothing was registered or executed
        assert window._busy is False
        assert not window._a_side_active()
        assert task_oc.call_log == []
    finally:
        shutdown_window(window)