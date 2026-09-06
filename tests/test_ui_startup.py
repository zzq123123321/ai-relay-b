"""PySide6 window startup, clipboard wiring and executor-independent monitoring."""

from __future__ import annotations

import time

import pytest

from core.protocol import ProtocolFormat, parse_message
from core.relay import TaskOutcome
from core.relay_settings import RelaySettings
from core.task_registry import TaskRegistry
from tests.test_relay_workflow import FakeReasonix, v1_task


class FailingReasonix:
    def self_check(self):
        raise RuntimeError("Reasonix window was not found")

    def execute(self, task: str) -> str:
        raise RuntimeError("Reasonix window was not found")


def build_window(qapp, monkeypatch, tmp_path, reasonix_cls=FakeReasonix):
    import ui as ui_mod
    from core.relay import RelayWorkflow as BaseRelayWorkflow

    class TestWorkflow(BaseRelayWorkflow):
        def __init__(self, reasonix, settings=None):
            super().__init__(
                reasonix,
                registry=TaskRegistry(tmp_path / "tasks.json"),
                settings=settings
                or RelaySettings(openchamber_directory="D:/proj"),
                openchamber=None,
                replies_dir=tmp_path / "replies",
            )

    class FakeSettingsType:
        @staticmethod
        def load(path=None):
            return RelaySettings(openchamber_directory="D:/proj")

    monkeypatch.setattr(ui_mod, "ReasonixAutomation", reasonix_cls)
    monkeypatch.setattr(ui_mod, "RelayWorkflow", TestWorkflow)
    monkeypatch.setattr(ui_mod, "RelaySettings", FakeSettingsType)

    window = ui_mod.RelayWindow(qapp)
    window.show()
    return window


def wait_until(predicate, timeout: float = 6.0):
    from PySide6.QtTest import QTest

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QTest.qWait(40)
        if predicate():
            return True
    return False


def test_window_starts_and_processes_clipboard_task(qapp, monkeypatch, tmp_path):
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert window.start_button.isEnabled() is False

        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("REASONIX", "say hi", "task-ui-001"))

        def done():
            text = clipboard.text()
            return "IN_REPLY_TO: task-ui-001" in text

        assert wait_until(done), "clipboard task was not relayed in time"

        response_text = clipboard.text()
        message = parse_message(response_text)
        assert message.protocol_format is ProtocolFormat.V1
        assert message.in_reply_to == "task-ui-001"
        assert "reasonix-reply: say hi" in message.body
        assert window._last_response == response_text
        # reply persisted for re-copy
        assert window._workflow.load_reply("task-ui-001") == response_text
        assert window.recopy_button.isEnabled()
    finally:
        window._listener.pause()
        window.close()


def test_window_starts_even_when_reasonix_self_check_fails(qapp, monkeypatch, tmp_path):
    window = build_window(qapp, monkeypatch, tmp_path, reasonix_cls=FailingReasonix)
    try:
        assert wait_until(lambda: window._listener.enabled)
        # monitoring is up; the self-check failure is only reported
        assert "Reasonix 自检失败" in window.detail_label.text()
        assert window.start_button.isEnabled() is False
    finally:
        window._listener.pause()
        window.close()


def test_non_protocol_clipboard_text_is_ignored(qapp, monkeypatch, tmp_path):
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        clipboard = qapp.clipboard()
        clipboard.setText("plain shopping list, no protocol")
        from PySide6.QtTest import QTest

        QTest.qWait(300)
        assert clipboard.text() == "plain shopping list, no protocol"
        assert window._workflow.outcome is None
    finally:
        window._listener.pause()
        window.close()


def test_response_message_on_clipboard_is_not_processed(qapp, monkeypatch, tmp_path):
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        clipboard = qapp.clipboard()
        clipboard.setText(
            "AI_RELAY/1\nMESSAGE_ID: r1\nSOURCE: CHATGPT\nTARGET: REASONIX\n"
            "TYPE: RESPONSE\n\nnot a task"
        )
        from PySide6.QtTest import QTest

        QTest.qWait(300)
        assert window._workflow.outcome is None
        assert clipboard.text().startswith("AI_RELAY/1")
    finally:
        window._listener.pause()
        window.close()


def test_open_session_tracks_current_task_not_last_success(qapp, monkeypatch, tmp_path):
    """The 'open current session' action uses the task which owns this
    moment's session, never the last successful reply; it stays usable
    after the task fails."""
    import ui as ui_mod

    window = build_window(qapp, monkeypatch, tmp_path)
    window._startup_check_pending = False
    opened: list[str] = []
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "open_session",
        lambda self, session_id: opened.append(session_id),
    )
    try:
        # a previous task succeeded (session A) but the current live one is B
        window._last_outcome = TaskOutcome(
            "task-A", "OPENCHAMBER", "ses_A", "D:/proj"
        )
        window._current_outcome = TaskOutcome(
            "task-B", "OPENCHAMBER", "ses_B", "D:/proj"
        )
        window._open_current_session()
        assert opened == ["ses_B"]  # never task A while B is current

        # after a failure the current session must still be openable
        window._task_failed("timeout")
        assert window.open_session_button.isEnabled()
        window._open_current_session()
        assert opened == ["ses_B", "ses_B"]
    finally:
        window._listener.pause()
        window.close()


def test_starting_new_task_clears_current_session(qapp, monkeypatch, tmp_path):
    """Accepting task B must clear task A's session so the button cannot
    open task A while B is in progress."""
    window = build_window(qapp, monkeypatch, tmp_path)
    window._startup_check_pending = False
    try:
        window._current_outcome = TaskOutcome(
            "task-A", "OPENCHAMBER", "ses_A", "D:/proj"
        )
        window.open_session_button.setEnabled(True)

        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("REASONIX", "say hi", "task-ui-clear-001"))
        window._on_clipboard_text(clipboard.text())
        assert window._current_outcome is None
        assert not window.open_session_button.isEnabled()

        def finished():
            return "IN_REPLY_TO: task-ui-clear-001" in clipboard.text()

        assert wait_until(finished)
        # a REASONIX task has no OpenChamber session to open
        assert not window.open_session_button.isEnabled()
    finally:
        window._listener.pause()
        window.close()


def test_recopy_reply_after_restart_uses_saved_registry(qapp, monkeypatch, tmp_path):
    """After a restart (no in-memory response) the saved completed task's
    reply can still be re-copied from the persisted registry."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("REASONIX", "hello persisted", "task-persist-001"))
        window._on_clipboard_text(clipboard.text())

        def done():
            return "IN_REPLY_TO: task-persist-001" in clipboard.text()

        assert wait_until(done)
    finally:
        window._listener.pause()
        window.close()

    window2 = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert window2._saved_task_combo.count() >= 1
        clipboard = qapp.clipboard()
        clipboard.setText("")
        window2._saved_task_combo.setCurrentIndex(0)
        window2._recopy_reply()
        assert "hello persisted" in clipboard.text()
    finally:
        window2._listener.pause()
        window2.close()