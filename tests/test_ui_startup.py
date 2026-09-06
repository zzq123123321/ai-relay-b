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


def build_window(
    qapp,
    monkeypatch,
    tmp_path,
    reasonix_cls=FakeReasonix,
    settings=None,
    openchamber=None,
):
    import ui as ui_mod
    from core.relay import RelayWorkflow as BaseRelayWorkflow

    def effective_settings():
        return settings or RelaySettings(openchamber_directory="D:/proj")

    class TestWorkflow(BaseRelayWorkflow):
        def __init__(self, reasonix, settings=None):
            super().__init__(
                reasonix,
                registry=TaskRegistry(tmp_path / "tasks.json"),
                settings=effective_settings(),
                openchamber=openchamber,
                replies_dir=tmp_path / "replies",
            )

    class FakeSettingsType:
        @staticmethod
        def load(path=None):
            return effective_settings()

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
    """After a restart (no in-memory response) a saved completed task's
    reply can still be re-copied from the persisted registry by selecting
    it in the dropdown."""
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
        combo = window2._saved_task_combo
        assert combo.findData("task-persist-001") >= 0
        clipboard = qapp.clipboard()
        clipboard.setText("")
        combo.setCurrentIndex(combo.findData("task-persist-001"))
        window2._recopy_reply()
        assert "hello persisted" in clipboard.text()
    finally:
        window2._listener.pause()
        window2.close()


def _complete_task(window, qapp, body: str, task_id: str):
    clipboard = qapp.clipboard()
    clipboard.setText(v1_task("REASONIX", body, task_id))
    window._on_clipboard_text(clipboard.text())

    def done():
        return f"reasonix-reply: {body}" in clipboard.text()

    assert wait_until(done)


def test_recopy_uses_selected_task_not_last_response(qapp, monkeypatch, tmp_path):
    """Recopy must use the CURRENT drop-down selection.  After A and B both
    completed, selecting A copies A even though B is the last response;
    selecting B copies B."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        _complete_task(window, qapp, "A", "task-a")
        _complete_task(window, qapp, "B", "task-b")

        combo = window._saved_task_combo
        # the newest completed task is selected automatically; override to A
        combo.setCurrentIndex(combo.findData("task-a"))
        clipboard = qapp.clipboard()
        clipboard.setText("")
        window._recopy_reply()
        assert "reasonix-reply: A" in clipboard.text()

        combo.setCurrentIndex(combo.findData("task-b"))
        clipboard.setText("")
        window._recopy_reply()
        assert "reasonix-reply: B" in clipboard.text()
    finally:
        window._listener.pause()
        window.close()


def test_recopy_after_restart_selects_saved_a(qapp, monkeypatch, tmp_path):
    """After a restart, selecting saved task A still recopies A's reply."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        _complete_task(window, qapp, "A", "task-a")
        _complete_task(window, qapp, "B", "task-b")
    finally:
        window._listener.pause()
        window.close()

    window2 = build_window(qapp, monkeypatch, tmp_path)
    try:
        combo = window2._saved_task_combo
        clipboard = qapp.clipboard()
        clipboard.setText("")
        combo.setCurrentIndex(combo.findData("task-a"))
        window2._recopy_reply()
        assert "reasonix-reply: A" in clipboard.text()
    finally:
        window2._listener.pause()
        window2.close()


def test_recopy_selected_reply_missing_does_not_fallback(qapp, monkeypatch, tmp_path):
    """When the selected task's reply file is missing, recopy must FAIL with
    a clear error and must NOT fall back to copying another task's reply."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        _complete_task(window, qapp, "A", "task-a")
        _complete_task(window, qapp, "B", "task-b")

        # remove A's reply file only
        reply_a = window._workflow.reply_file_for("task-a")
        assert reply_a.exists()
        reply_a.unlink()

        combo = window._saved_task_combo
        combo.setCurrentIndex(combo.findData("task-a"))
        clipboard = qapp.clipboard()
        clipboard.setText("")
        window._recopy_reply()
        assert "回复文件缺失" in window.detail_label.text()
        # B's reply must NOT have been copied (no fallback)
        assert "reasonix-reply: B" not in clipboard.text()
    finally:
        window._listener.pause()
        window.close()


def test_model_details_three_layers_shown_in_ui(qapp, monkeypatch, tmp_path):
    """Requested A / resolved B / actual B must keep a '模型不一致' note and
    show the three layers in the completion detail; selecting the saved
    task restores the details from the registry record."""
    import ui as ui_mod

    from core.openchamber import ModelRef
    from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
    from tests.fakes import (
        ScriptedOpenChamber,
        assistant_message,
        make_dispatch,
        text_part,
        user_message,
    )

    requested = ModelRef("provA", "modelA")
    resolved = ModelRef("provB", "modelB")
    oc = ScriptedOpenChamber()
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
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="provA/modelA",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=oc
    )
    window._startup_check_pending = False
    try:
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "do it", "task-model-ui"))
        window._on_clipboard_text(clipboard.text())

        def done():
            return "IN_REPLY_TO: task-model-ui" in clipboard.text()

        assert wait_until(done)
        # completion detail carries the mismatch note with all three layers
        text = window.detail_label.text()
        assert "模型不一致" in text
        assert "provA/modelA" in text
        assert "provB/modelB" in text

        # selecting the saved task restores the details from the registry
        combo = window._saved_task_combo
        combo.setCurrentIndex(0)  # placeholder
        combo.setCurrentIndex(combo.findData("task-model-ui"))
        restored = window.detail_label.text()
        assert "已保存任务" in restored
        assert "模型不一致" in restored
        assert "provA/modelA" in restored
        assert "provB/modelB" in restored
    finally:
        window._listener.pause()
        window.close()


def test_session_refresh_lists_candidates_and_saves_id(qapp, monkeypatch, tmp_path):
    """The session combo lists existing sessions from the OpenChamber API;
    selecting one and saving settings stores its id (no auto-create, a
    free-typed id is kept verbatim, and a re-refresh restores by id — never
    by the '标题（ses_xxx）' label)."""
    import ui as ui_mod

    from core.relay_settings import RelaySettings

    window = build_window(qapp, monkeypatch, tmp_path)
    candidates = [
        ("ses_A", "候选会话A"),
        ("ses_B", "候选会话B"),
        ("ses_gone", "已消失会话"),
    ]
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "list_sessions",
        lambda self, directory: list(candidates),
    )
    try:
        window._directory_edit.setText("D:/proj")
        window._refresh_sessions()

        def populated():
            return window._session_combo.count() == 4  # placeholder + 3

        assert wait_until(populated)
        combo = window._session_combo
        assert combo.findData("ses_A") >= 0
        assert combo.findData("ses_B") >= 0

        # pick a candidate and save: the stored session id is the data value
        combo.setCurrentIndex(combo.findData("ses_B"))
        monkeypatch.setattr(RelaySettings, "save", lambda self, path=None: None)
        window._save_settings()
        assert window._settings.openchamber_session_id == "ses_B"

        # regression: selecting a titled candidate, then refreshing again,
        # then saving must still store the REAL id (never the display label)
        candidates.append(("ses_C", "候选会话C"))
        window._refresh_sessions()

        def refreshed(count):
            return (
                window._session_combo.count() == count
                and window._session_combo.currentData() == "ses_B"
            )

        assert wait_until(lambda: refreshed(5))
        assert combo.currentText() == "候选会话B（ses_B）"
        window._save_settings()
        assert window._settings.openchamber_session_id == "ses_B"

        # a free-typed id is preserved across a refresh, not treated as the
        # selected candidate's label
        combo.setCurrentText("ses_free_typed")
        window._refresh_sessions()
        assert wait_until(lambda: combo.currentText() == "ses_free_typed")
        window._save_settings()
        assert window._settings.openchamber_session_id == "ses_free_typed"

        # re-selecting the placeholder clears the id again
        combo.setCurrentIndex(0)
        window._save_settings()
        assert window._settings.openchamber_session_id == ""
    finally:
        window._listener.pause()
        window.close()


def test_refresh_completion_does_not_unlock_controls_during_task(
    qapp, monkeypatch, tmp_path
):
    """While a refresh is pending a clipboard task may arrive; the refresh
    success/failure callback must not re-enable execution controls until the
    task is done (busy-aware restore)."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        # a refresh was started: controls are disabled while the async list
        # call is in flight, and a task starts executing meanwhile
        window._set_controls_enabled(False)
        window._busy = True

        window._sessions_loaded([("ses_A", "候选会话A")])

        def execution_controls_enabled():
            return any(
                b.isEnabled()
                for b in (
                    window.start_button,
                    window.pause_button,
                    window.check_button,
                    window._save_settings_button,
                    window._refresh_sessions_button,
                )
            )

        assert not execution_controls_enabled()

        # the failure callback must respect the running task as well
        window._sessions_failed("boom")
        assert not execution_controls_enabled()

        # once the task has finished, both completion paths unlock again
        window._busy = False
        window._sessions_loaded([("ses_A", "候选会话A"), ("ses_B", "候选会话B")])
        assert execution_controls_enabled()

        window._busy = False
        window._sessions_failed("boom2")
        assert execution_controls_enabled()
    finally:
        window._listener.pause()
        window.close()