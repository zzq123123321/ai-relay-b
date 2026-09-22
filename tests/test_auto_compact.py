"""Post-reply auto-compaction of the OpenChamber session.

Pins the contract added on top of the reply pipeline:

A. After a reply is WRAPPED and COPIED to the clipboard, and only then, a
   first-class opencode compaction is triggered on the session that produced
   it.  The call goes to the dedicated ``/api/session/{id}/compact`` API, NOT
   a text prompt: the ``/compact`` command is absent from the resolvable
   command list, so sending it as a prompt just makes the model answer that
   it is not an opencode command.  The compaction never passes through the
   AI_RELAY formatter (it is never a wrapped protocol message).
B. While the compaction runs the task lane stays busy: a new A-side task is
   accepted into the persistent queue but is NOT started until the compaction
   completes (or fails).
C. After the compaction completes, the next queued task starts automatically.
D. A failed compaction never affects the already-delivered reply: the
   clipboard entry and the COMPLETED registry state are preserved, the queue
   still advances, and the status reports exactly the compact failure.
E. Disabling ``auto_compact_after_response`` restores the old behaviour (no
   compaction is ever requested).
"""

from __future__ import annotations

import threading

import core.relay as relay_mod
from core.openchamber import OpenChamberUnavailableError
from core.protocol import parse_message
from core.relay import directory_key
from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
from tests.test_fifo_queue import ControllableRoundOc
from tests.test_oc_monitor import MutableMessages, completed_reply
from tests.test_relay_workflow import v1_task
from tests.test_ui_startup import build_window, shutdown_window, wait_until

import ui as ui_mod


class CompactBlockedOc(ControllableRoundOc):
    """A controllable fake whose compaction call always fails."""

    def compact(
        self, session_id: str, directory: str | None = None, model=None
    ) -> None:
        raise OpenChamberUnavailableError("compact blocked in test")


def build_compact_window(
    qapp, monkeypatch, tmp_path, oc, auto: bool = True
):
    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory=str(tmp_path),
        openchamber_session_id="ses_test123",
        openchamber_sessions={directory_key(str(tmp_path)): "ses_test123"},
        poll_interval=0.05,
        completion_timeout=0,  # no timeout: rounds wait until completed
        auto_compact_after_response=auto,
    )
    monitor_fake = MutableMessages(
        [completed_reply("a_ac_hist", created=1000, text="history")]
    )
    monkeypatch.setattr(ui_mod, "OpenChamberClient", lambda url, **kwargs: monitor_fake)
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=oc
    )
    window._startup_check_pending = False
    return window


def _idle(qapp, window):
    assert wait_until(lambda: window._listener.enabled), "listener never started"
    assert wait_until(lambda: not window._busy), "startup tasks never settled"


def test_compact_after_reply_is_copied(qapp, monkeypatch, tmp_path):
    """A: the wrapped reply is written to the clipboard BEFORE the
    compaction is requested; the compaction is a direct API call (never a
    text prompt, never a protocol message and never a wrapped reply)."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply one"])
    oc.compact_gate = threading.Event()
    window = build_compact_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        qapp.clipboard().setText(
            v1_task("OPENCHAMBER", "one", "task-cmp-001", str(tmp_path))
        )
        assert wait_until(lambda: window._busy), "first task did not start"
        assert wait_until(
            lambda: [c for c in oc.call_log if c.startswith("send:")]
        ), "first task was not sent"

        oc.complete_last()  # the REPLY round finishes

        assert wait_until(lambda: oc.compact_calls), "compaction was never requested"
        assert window._compacting, "window is not holding the lane while compacting"
        # The compaction is a direct API call, not an extra text prompt.
        assert oc.sent_prompts == ["one\n\n[AI_RELAY_TASK_ID: task-cmp-001]"]
        # The reply was already wrapped AND copied before the compaction.
        clipboard_text = qapp.clipboard().text()
        assert "IN_REPLY_TO: task-cmp-001" in clipboard_text
        parse_message(clipboard_text)  # the clipboard entry is a valid protocol msg

        oc.compact_gate.set()  # the compaction finishes
        assert wait_until(lambda: not window._busy), "lane never released after compact"
        assert not window._compacting
    finally:
        shutdown_window(window)


def test_queued_task_waits_until_compact_finishes_and_then_continues(
    qapp, monkeypatch, tmp_path
):
    """B+C: a task arriving while the compaction runs is QUEUED and not
    started; once the compaction finishes the next task runs and completes."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc(
        "ses_test123", str(tmp_path), ["reply one", "reply two"]
    )
    oc.compact_gate = threading.Event()
    window = build_compact_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        qapp.clipboard().setText(
            v1_task("OPENCHAMBER", "one", "task-cmp-101", str(tmp_path))
        )
        assert wait_until(lambda: window._busy), "first task did not start"
        assert wait_until(
            lambda: [c for c in oc.call_log if c.startswith("send:")]
        ), "first task was not sent"
        oc.complete_last()  # reply round done -> compaction starts
        assert wait_until(lambda: oc.compact_calls), "compaction did not start"
        assert window._compacting

        # A task arriving DURING compaction is queued and never started.
        qapp.clipboard().setText(
            v1_task("OPENCHAMBER", "two", "task-cmp-102", str(tmp_path))
        )
        assert wait_until(
            lambda: (window._workflow.registry.record("task-cmp-102") or {}).get(
                "state"
            )
            == "QUEUED"
        ), "second task was not queued while compacting"
        assert len(oc.sent_prompts) == 1, "no extra send while compacting"
        assert not any("two" in p for p in oc.sent_prompts), (
            "queued task started before compact done"
        )

        oc.compact_gate.set()  # the compaction finishes
        assert wait_until(
            lambda: not window._compacting
        ), "compact lane never released"
        assert wait_until(
            lambda: any("two" in p for p in oc.sent_prompts)
        ), "next queued task did not start after compact"

        oc.complete_last()  # the second task's reply round finishes
        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-cmp-102" in clipboard.text()
        ), "second task reply was never copied"
        # Every reply is followed by its OWN compaction: the second reply must
        # trigger compact #2, and the lane only clears after that one too.
        assert wait_until(
            lambda: len(oc.compact_calls) == 2
        ), "second reply did not trigger its own compaction"
        assert wait_until(lambda: not window._busy), "lane never released"
        assert not window._compacting
    finally:
        shutdown_window(window)


def test_compact_failure_preserves_reply_and_queue_still_advances(
    qapp, monkeypatch, tmp_path
):
    """D: a failed compaction reports "回复完成，但上下文压缩失败" while the
    already-copied reply and the COMPLETED registry state stay intact, and the
    next queued task still runs."""
    qapp.clipboard().clear()
    oc = CompactBlockedOc("ses_test123", str(tmp_path), ["reply one", "reply two"])
    window = build_compact_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        qapp.clipboard().setText(
            v1_task("OPENCHAMBER", "one", "task-cmp-201", str(tmp_path))
        )
        assert wait_until(lambda: window._busy), "first task did not start"
        assert wait_until(
            lambda: [c for c in oc.call_log if c.startswith("send:")]
        ), "first task was not sent"
        oc.complete_last()  # reply round done -> compaction FAILS

        assert wait_until(
            lambda: (window._workflow.registry.record("task-cmp-201") or {}).get(
                "state"
            )
            == "COMPLETED"
        ), "reply task was not COMPLETED"
        assert wait_until(lambda: not window._busy), "lane never released"
        clipboard_text = qapp.clipboard().text()
        assert "IN_REPLY_TO: task-cmp-201" in clipboard_text
        parse_message(clipboard_text)
        assert "回复完成，但上下文压缩失败" in window.status_label.text()

        # The queue still advances after the failed compaction.
        qapp.clipboard().setText(
            v1_task("OPENCHAMBER", "two", "task-cmp-202", str(tmp_path))
        )
        assert wait_until(
            lambda: any("two" in p for p in oc.sent_prompts)
        ), "next task did not start after compact failure"
        oc.complete_last()
        assert wait_until(
            lambda: (window._workflow.registry.record("task-cmp-202") or {}).get(
                "state"
            )
            == "COMPLETED"
        )
    finally:
        shutdown_window(window)


def test_compact_disabled_never_requests_compact(qapp, monkeypatch, tmp_path):
    """E: with ``auto_compact_after_response=False`` nothing is requested and
    the lane clears directly after the reply."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply one"])
    window = build_compact_window(qapp, monkeypatch, tmp_path, oc, auto=False)
    try:
        _idle(qapp, window)
        qapp.clipboard().setText(
            v1_task("OPENCHAMBER", "one", "task-cmp-301", str(tmp_path))
        )
        assert wait_until(lambda: window._busy), "task did not start"
        assert wait_until(
            lambda: [c for c in oc.call_log if c.startswith("send:")]
        ), "task was not sent"
        oc.complete_last()
        assert wait_until(
            lambda: (window._workflow.registry.record("task-cmp-301") or {}).get(
                "state"
            )
            == "COMPLETED"
        )
        assert wait_until(lambda: not window._busy), "lane never released"
        assert not oc.compact_calls, "compaction must be disabled"
    finally:
        shutdown_window(window)


def test_compact_session_workflow_level():
    """Workflow-level: compact_session requests the first-class compaction of
    the session, returns True on success and False on a transport failure,
    without ever touching registry or the reply protocol."""
    oc = ControllableRoundOc("D:/proj", "D:/proj", [])
    workflow = _compact_workflow(oc, timeout=1.0)

    assert workflow.compact_session("D:/proj", "D:/proj") is True
    assert oc.compact_calls == [("D:/proj", "D:/proj")]
    assert not oc.sent_prompts, "compaction must not be sent as a text prompt"

    blocked = CompactBlockedOc("D:/proj", "D:/proj", [])
    assert (
        _compact_workflow(blocked, timeout=1.0).compact_session(
            "D:/proj", "D:/proj"
        )
        is False
    ), "a transport failure must report compact as failed"


def _compact_workflow(oc, timeout: float):
    return relay_mod.RelayWorkflow(
        reasonix=object(),  # unused by compact_session
        registry=None,
        settings=RelaySettings(
            openchamber_directory="D:/proj",
            poll_interval=0.01,
            completion_timeout=timeout,
            auto_compact_after_response=True,
        ),
        openchamber=oc,
    )
