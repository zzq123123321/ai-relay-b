"""PySide6 window startup, clipboard wiring and executor-independent monitoring."""

from __future__ import annotations

import os
import threading
import time

import pytest

import core.relay as relay_mod
from core.protocol import ProtocolFormat, parse_message
from core.relay import OpenChamberContinue, TaskOutcome, directory_key
from core.relay_settings import RelaySettings
from core.task_registry import TaskRegistry
from PySide6.QtCore import QRunnable
from tests.test_relay_workflow import (
    AlwaysRejectedOpenChamber,
    BadRequestOpenChamber,
    BusyOpenChamber,
    FakeReasonix,
    RejectedThenWorkingOpenChamber,
    RecoveryOpenChamber,
    v1_task,
)
from tests.fakes import (
    assistant_message,
    text_part,
    user_message,
)


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
        return settings or RelaySettings(
            openchamber_directory="D:/proj",
            # post-reply compaction is a separate feature: existing UI tests
            # keep exercising their individual flows without an extra /compact
            auto_compact_after_response=False,
        )

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

    from PySide6.QtCore import QThreadPool

    monkeypatch.setattr(ui_mod, "ReasonixAutomation", reasonix_cls)
    monkeypatch.setattr(ui_mod, "RelayWorkflow", TestWorkflow)
    monkeypatch.setattr(ui_mod, "RelaySettings", FakeSettingsType)

    # Isolate this window's workers on a private pool so tests never compete
    # for the process-global QThreadPool.globalInstance().
    window = ui_mod.RelayWindow(qapp, pool=QThreadPool())
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


def shutdown_window(window) -> None:
    """Strict teardown for one window (mirrors the conftest fixture).

    Stops the clipboard listener, sets every stop/cancel event, waits until
    all monitor/general workers emitted ``finished`` (bounded), closes the
    window and confirms the worker sets are empty again - so a worker can
    never outlive its window or leak across tests (the access-violation race).
    Every Qt-worker test must call this in its ``finally``; the autouse
    conftest fixture enforces the same sequence regardless."""
    from PySide6.QtCore import QCoreApplication

    window._listener.pause()
    stop = getattr(window, "_oc_monitor_stop", None)
    if stop is not None:
        stop.set()
    cancel = getattr(window, "_task_cancel_event", None)
    if cancel is not None:
        cancel.set()
    active = getattr(window, "_oc_monitor_active", False)
    stopping = getattr(window, "_oc_monitor_stopping", False)
    if active and not stopping:
        window._stop_monitor()
    assert wait_until(
        lambda: (
            not window._monitor_workers
            and not window._general_workers
            and not getattr(window, "_oc_monitor_stopping", False)
        ),
        timeout=5.0,
    ), (
        f"worker leak in {window.__class__.__name__}: "
        f"monitor={sorted(window._monitor_workers)} "
        f"general={sorted(window._general_workers)}"
    )
    QCoreApplication.processEvents()
    window.close()
    QCoreApplication.processEvents()
    assert not window._monitor_workers and not window._general_workers


def test_window_starts_and_processes_clipboard_task(qapp, monkeypatch, tmp_path):
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy)
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
        shutdown_window(window)


def test_window_starts_even_when_reasonix_self_check_fails(qapp, monkeypatch, tmp_path):
    window = build_window(qapp, monkeypatch, tmp_path, reasonix_cls=FailingReasonix)
    try:
        assert wait_until(lambda: window._listener.enabled)
        # monitoring is up; the self-check failure is only reported
        assert wait_until(lambda: "Reasonix 自检失败" in window.detail_label.text())
        assert window.start_button.isEnabled() is False
    finally:
        shutdown_window(window)


def test_openchamber_default_skips_reasonix_startup_self_check(
    qapp, monkeypatch, tmp_path
):
    from core.relay_settings import TARGET_OPENCHAMBER

    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory="D:/proj",
        auto_compact_after_response=False,
    )
    window = build_window(
        qapp,
        monkeypatch,
        tmp_path,
        reasonix_cls=FailingReasonix,
        settings=settings,
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert window._startup_check_pending is False
        assert not window._busy
        assert "Reasonix 自检失败" not in window.detail_label.text()
    finally:
        shutdown_window(window)

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
        shutdown_window(window)


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
        shutdown_window(window)


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
        shutdown_window(window)


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
        shutdown_window(window)


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
        shutdown_window(window)

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
        shutdown_window(window2)


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
        shutdown_window(window)


def test_recopy_after_restart_selects_saved_a(qapp, monkeypatch, tmp_path):
    """After a restart, selecting saved task A still recopies A's reply."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        _complete_task(window, qapp, "A", "task-a")
        _complete_task(window, qapp, "B", "task-b")
    finally:
        shutdown_window(window)

    window2 = build_window(qapp, monkeypatch, tmp_path)
    try:
        combo = window2._saved_task_combo
        clipboard = qapp.clipboard()
        clipboard.setText("")
        combo.setCurrentIndex(combo.findData("task-a"))
        window2._recopy_reply()
        assert "reasonix-reply: A" in clipboard.text()
    finally:
        shutdown_window(window2)


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
        shutdown_window(window)


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
        auto_compact_after_response=False,  # keeps this test about model details
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
        shutdown_window(window)


def test_save_keeps_preconfigured_fixed_session(qapp, monkeypatch, tmp_path):
    """The fixed-session UI row is gone; an id persisted in the settings
    file is kept verbatim by saving (edit the file to change/clear it;
    empty id = task goes to the currently active OpenChamber session)."""
    from core.relay_settings import RelaySettings

    settings = RelaySettings(openchamber_session_id="ses_fixed_in_file")
    window = build_window(qapp, monkeypatch, tmp_path, settings=settings)
    try:
        monkeypatch.setattr(RelaySettings, "save", lambda self, path=None: None)
        assert window._save_settings() is True
        assert window._settings.openchamber_session_id == "ses_fixed_in_file"
    finally:
        shutdown_window(window)


def test_browse_directory_fills_project_path(qapp, monkeypatch, tmp_path):
    import ui as ui_mod

    from core.relay import directory_key
    from core.relay_settings import RelaySettings

    selected = str(tmp_path / "kart-game")
    (tmp_path / "kart-game").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        ui_mod.QFileDialog,
        "getExistingDirectory",
        lambda *args: selected,
    )
    # pre-save a fixed session so we can assert it is cleared when the
    # directory changes
    settings = RelaySettings(
        openchamber_session_id="old_ses",
        openchamber_sessions={directory_key(selected): "old_ses"},
    )
    window = build_window(qapp, monkeypatch, tmp_path, settings=settings)
    try:
        # avoid writing the real config on disk during the auto-save and avoid
        # a real network call during the automatic Agent/Model refresh
        monkeypatch.setattr(RelaySettings, "save", lambda self, path=None: None)
        monkeypatch.setattr(
            ui_mod.OpenChamberClient, "list_sessions", lambda self, directory: []
        )

        window._browse_directory()
        assert os.path.normcase(window._directory_edit.text()) == os.path.normcase(
            selected
        )
        # auto-saved (canonicalized/normalized form)
        assert os.path.normcase(
            window._settings.openchamber_directory
        ) == os.path.normcase(selected)
        # stale fixed session cleared when the directory changes
        assert window._settings.openchamber_session_id == ""
        assert (
            window._settings.openchamber_sessions.get(directory_key(selected))
            is None
        )
    finally:
        shutdown_window(window)


class StubSelfCheckTask(QRunnable):
    """Deterministic stand-in for SelfCheckTask: the worker finishes at once
    but the ``succeeded`` signal is only fired when the TEST calls
    ``succeed()`` -- the startup self-check completion can therefore be
    ordered before/after any higher-priority UI operation."""

    instances: list["StubSelfCheckTask"] = []

    def __init__(self, reasonix=None):
        import ui as ui_mod

        super().__init__()
        self.signals = ui_mod.WorkerSignals()
        StubSelfCheckTask.instances.append(self)

    def run(self):
        self.signals.finished.emit()

    def succeed(self):
        self.signals.succeeded.emit("unused")


def _build_with_stub_selfcheck(qapp, monkeypatch, tmp_path, wait=True):
    import ui as ui_mod

    StubSelfCheckTask.instances.clear()
    monkeypatch.setattr(ui_mod, "SelfCheckTask", StubSelfCheckTask)
    window = build_window(qapp, monkeypatch, tmp_path)
    stub = None
    if wait:
        assert wait_until(
            lambda: len(StubSelfCheckTask.instances) == 1
        ), "startup self-check did not run"
        stub = StubSelfCheckTask.instances[0]
    return window, stub


def test_startup_selfcheck_hint_shows_when_idle(
    qapp, monkeypatch, tmp_path
):
    """Requirement: with no higher-priority UI activity the completed startup
    self-check MAY write its UIA hint into the detail label."""
    window, stub = _build_with_stub_selfcheck(qapp, monkeypatch, tmp_path)
    try:
        window._listener.pause()
        stub.succeed()
        assert window.detail_label.text() == (
            "Reasonix 窗口、输入框和发送按钮均可通过 UIA 识别。"
        )
        assert wait_until(lambda: not window._general_workers)
    finally:
        shutdown_window(window)


def test_startup_selfcheck_does_not_clobber_monitor_text(
    qapp, monkeypatch, tmp_path
):
    """Requirement: an OpenChamber monitor started BEFORE the startup
    self-check may never be overwritten by the self-check hint."""
    import ui as ui_mod

    window, stub = _build_with_stub_selfcheck(qapp, monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "session_exists",
        lambda self, session_id, directory: True,
    )
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "messages",
        lambda self, session_id, directory: [],
    )
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "session_status",
        lambda self, session_id, directory: "idle",
    )
    try:
        window._listener.pause()
        window._monitor_busy_without_progress_threshold = 0.2
        window._monitor_busy_stale_log_interval = 0.1
        window._settings.openchamber_url = "http://test:8888"
        window._settings.openchamber_directory = str(tmp_path)
        window._settings.openchamber_session_id = "ses_mon"
        window._toggle_monitor()
        assert wait_until(lambda: window._oc_monitor_active)
        stub.succeed()
        # monitor start writes the STATUS label; the detail area must simply
        # not receive the self-check hint while a monitor owns the window
        assert "可通过 UIA 识别" not in window.detail_label.text()
        assert "Reasonix 窗口" not in window.detail_label.text()
    finally:
        shutdown_window(window)


def test_startup_selfcheck_late_after_close_is_ignored(
    qapp, monkeypatch, tmp_path
):
    """Requirement: the self-check result arriving AFTER the window started
    closing is ignored -- it must neither crash nor overwrite any label."""
    window, stub = _build_with_stub_selfcheck(qapp, monkeypatch, tmp_path)
    window._listener.pause()
    window.detail_label.setText("marker-before-close")
    window.close()
    assert window._close_requested
    stub.succeed()
    assert window.detail_label.text() == "marker-before-close"
    shutdown_window(window)


def test_browse_directory_rejects_nonexistent_dir(qapp, monkeypatch, tmp_path):
    import ui as ui_mod

    selected = str(tmp_path / "does-not-exist")
    monkeypatch.setattr(
        ui_mod.QFileDialog,
        "getExistingDirectory",
        lambda *args: selected,
    )
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        window._directory_edit.setText("D:/proj")
        window._browse_directory()
        # invalid directory is not adopted and OpenChamber is not called
        assert window._directory_edit.text() == "D:/proj"
        assert window._settings.openchamber_directory == "D:/proj"
        assert "不存在或不是文件夹" in window.detail_label.text()
    finally:
        shutdown_window(window)


def test_agent_model_combos_have_defaults_and_refresh_preserves_selection(
    qapp, monkeypatch, tmp_path
):
    """Agent/Model are combo boxes with default candidates; the '刷新
    Agent/Model' task populates both from the project's session
    messages while preserving the current selection, and a failure keeps the
    original values.  The model combo is NOT editable so an invalid or
    truncated name can never be typed."""
    import ui as ui_mod

    from core.openchamber import ModelRef
    from core.relay_settings import RelaySettings

    model = ModelRef("4090", "qwen3.8-27b")
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "list_sessions",
        lambda self, directory: [("ses_1", "项目A")],
    )
    monkeypatch.setattr(
        ui_mod.OpenChamberClient,
        "messages",
        lambda self, session_id, directory: [
            user_message("u1", "task", 1000),
            assistant_message(
                "a1", 1100, completed=1200, finish="stop",
                parts=[text_part("ok")], model=model, agent="build",
                parent_id="u1",
            ),
        ],
    )
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        # defaults seeded even before any refresh
        defaults = [
            window._agent_combo.itemText(i) for i in range(window._agent_combo.count())
        ]
        assert "build" in defaults and "plan" in defaults
        default_models = [
            window._model_combo.itemText(i) for i in range(window._model_combo.count())
        ]
        assert "4090/qwen3.8-27b" in default_models
        assert "opencode/big-pickle" in default_models

        # the model combo is a pick-only dropdown: no free typing
        assert window._model_combo.isEditable() is False

        # a valid selection survives a successful refresh
        window._agent_combo.setCurrentText("plan")
        window._model_combo.setCurrentText("opencode/big-pickle")
        proj = tmp_path / "proj"
        proj.mkdir(parents=True, exist_ok=True)
        window._directory_edit.setText(str(proj))
        # wait out the startup self-check (it sets _busy and would overwrite
        # the status the meta-refresh completion reports)
        assert wait_until(lambda: not window._busy)
        # listener becomes enabled only after the self-check fully finishes;
        # use it as the barrier so the refresh below is not racy
        assert wait_until(lambda: window._listener.enabled)
        window._refresh_meta()

        def updated():
            return "Agent/Model 已刷新" in window.status_label.text()

        was_updated = wait_until(updated)
        assert was_updated
        assert window._agent_combo.currentText() == "plan"
        assert window._model_combo.currentText() == "opencode/big-pickle"
        agents = {
            window._agent_combo.itemText(i)
            for i in range(window._agent_combo.count())
        }
        models = {
            window._model_combo.itemText(i)
            for i in range(window._model_combo.count())
        }
        assert "build" in agents  # extracted from session messages
        assert "4090/qwen3.8-27b" in models

        # typing is impossible: a non-item text is simply never selected
        window._model_combo.setCurrentText("custom/model")
        assert window._model_combo.currentText() != "custom/model"

        # a failed refresh keeps the previously selected values untouched
        window._agent_combo.setCurrentText("plan")
        window._model_combo.setCurrentText("opencode/big-pickle")
        monkeypatch.setattr(
            ui_mod.OpenChamberClient,
            "list_sessions",
            lambda self, directory: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        window._refresh_meta()

        def failed():
            return "刷新 Agent/Model 失败" in window.detail_label.text()

        assert wait_until(failed)
        assert window._agent_combo.currentText() == "plan"
        assert window._model_combo.currentText() == "opencode/big-pickle"
    finally:
        shutdown_window(window)


def test_save_settings_reads_agent_and_model_from_combos(
    qapp, monkeypatch, tmp_path
):
    """Saving settings stores the combo text values for agent/model.  The
    model must be a full value from the dropdown list: typing an unknown
    name is impossible (no-op on a non-editable combo) and an empty model
    selection is rejected with a Chinese prompt instead of silently saving."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        window._agent_combo.setCurrentText("plan")
        window._model_combo.setCurrentText("opencode/big-pickle")
        assert window._save_settings() is True
        assert window._settings.openchamber_agent == "plan"
        assert window._settings.openchamber_model == "opencode/big-pickle"

        # a non-list text can never be typed in: setCurrentText is a no-op
        # and the previously selected model stays untouched
        window._model_combo.setCurrentText("some/vendor:model")
        assert window._model_combo.currentText() == "opencode/big-pickle"
        assert window._settings.openchamber_model == "opencode/big-pickle"

        # an empty selection is rejected: nothing is written
        window._model_combo.setCurrentIndex(-1)
        assert window._save_settings() is False
        assert window._settings.openchamber_model == "opencode/big-pickle"
        assert "模型" in window.detail_label.text()
    finally:
        shutdown_window(window)


def test_wrap_button_always_enabled_and_rejects_initial_copy(
    qapp, monkeypatch, tmp_path
):
    """The 包装内容 button is independent of listener/busy state and always
    enabled.  An empty clipboard only produces a prompt, never a wrapped copy."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert window.wrap_button.isEnabled()
        assert window.wrap_button.text() == "包装内容"

        # even while the executor is busy, the button stays usable
        window._set_controls_enabled(False)
        assert window.wrap_button.isEnabled()

        clipboard = qapp.clipboard()
        clipboard.clear()
        from PySide6.QtTest import QTest

        QTest.qWait(200)
        window.wrap_button.click()
        assert "剪切板没有可包装内容" in window.detail_label.text()
        assert clipboard.text() == ""
    finally:
        shutdown_window(window)


def test_wrap_button_wraps_clipboard_text_into_manual_response(
    qapp, monkeypatch, tmp_path
):
    """Clicking 包装内容 wraps arbitrary clipboard text into a standalone
    manual RESPONSE (manual-{uuid}, ROUND 0, MAX_ROUNDS 1) and overwrites the
    clipboard.  Each click generates unique TASK_ID and RESPONSE_ID; the auto
    relay never sees the wrapped text."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        clipboard = qapp.clipboard()
        clipboard.setText("我的任意回答文本")

        window.wrap_button.click()
        wrapped = clipboard.text()
        message = parse_message(wrapped)
        assert message.message_type.value == "RESPONSE"
        assert message.in_reply_to.startswith("manual-")
        assert message.round_number == 0
        assert message.max_rounds == 1
        assert "我的任意回答文本" in message.body

        # listener self-write digest prevents reprocessing the wrapped text
        from PySide6.QtTest import QTest

        QTest.qWait(300)
        assert clipboard.text() == wrapped

        # second click → brand new TASK_ID and RESPONSE_ID
        clipboard.setText("另一段内容")
        window.wrap_button.click()
        second = clipboard.text()
        second_message = parse_message(second)
        assert second_message.in_reply_to.startswith("manual-")
        assert second_message.message_id != message.message_id
        assert second_message.in_reply_to != message.in_reply_to
    finally:
        shutdown_window(window)


def test_wrap_button_rejects_existing_relay_content(qapp, monkeypatch, tmp_path):
    """Already-complete AI Relay content on the clipboard is never re-wrapped:
    only a prompt is shown and the clipboard stays untouched."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        clipboard = qapp.clipboard()

        from core.protocol import wrap_response

        existing = wrap_response(
            "already a response",
            "manual-existing",
            ProtocolFormat.V1,
            round_number=0,
            max_rounds=1,
        )
        clipboard.setText(existing)
        window.wrap_button.click()
        assert "拒绝重复包装" in window.detail_label.text()
        assert clipboard.text() == existing
    finally:
        shutdown_window(window)


# ------------------------------------------------------------------ #
# stop / continue controls for abnormally interrupted OpenChamber tasks
# ------------------------------------------------------------------ #


def test_continue_button_state_follows_pending_interruption(
    qapp, monkeypatch, tmp_path
):
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert window.continue_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False

        # an interrupted task leaves a pending continue offer → button on
        message = parse_message(v1_task("OPENCHAMBER", "do it"))
        window._workflow.pending_continue = OpenChamberContinue(
            message=message, session_id="ses_test123", directory="D:/proj",
        )
        window._set_controls_enabled(True)
        assert window.continue_button.isEnabled() is True

        # while a task is RUNNING with a pending manual continue (recovery
        # exhausted, "请人工继续或停止") BOTH continue and stop stay
        # clickable: the operator is the only way out of that state.
        window._busy = True
        window._task_cancel_event = threading.Event()
        window._set_controls_enabled(False)
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True

        # but while a plain task is RUNNING and nothing is interrupted,
        # the continue button stays disabled (stop remains clickable).
        window._workflow.pending_continue = None
        window._set_controls_enabled(False)
        assert window.continue_button.isEnabled() is False
        assert window.stop_button.isEnabled() is True

        # after the auto-recovery fails the task awaits a manual continue:
        # BOTH continue and stop become clickable (the operator chooses
        # whether to keep pursuing it or stop).
        window._busy = False
        window._workflow.pending_continue = OpenChamberContinue(
            message=message, session_id="ses_test123", directory="D:/proj",
        )
        window._set_controls_enabled(True)
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True
    finally:
        shutdown_window(window)


def test_update_controls_without_cancel_event_keeps_stop_disabled(
    qapp, monkeypatch, tmp_path
):
    """The stop button only appears for RUNNING clipboard OpenChamber waits,
    not for setting/session tasks that have no cancel event."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        window._busy = True
        window._task_cancel_event = None
        window._set_controls_enabled(False)
        assert window.stop_button.isEnabled() is False
    finally:
        shutdown_window(window)


def test_stop_interrupted_task_marks_stopped_and_disables(
    qapp, monkeypatch, tmp_path
):
    """After auto-recovery fails the task awaits a manual continue; the stop
    button is enabled and clicking it marks the task STOPPED_BY_USER, keeps
    the OpenChamber session and disables both recovery buttons."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        message = parse_message(v1_task("OPENCHAMBER", "do it", "task-interrupted"))
        window._workflow.pending_continue = OpenChamberContinue(
            message=message, session_id="ses_test123", directory="D:/proj",
        )
        window._busy = False
        window._task_cancel_event = threading.Event()
        window._set_controls_enabled(True)

        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True
        assert window.new_session_button.isEnabled() is False

        window.stop_button.click()
        record = window._workflow.registry.record("task-interrupted")
        assert record is not None and record["state"] == "STOPPED_BY_USER"
        assert window.continue_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
    finally:
        shutdown_window(window)


def test_stop_button_interrupts_wait_and_preserves_session(
    qapp, monkeypatch, tmp_path
):
    oc = BusyOpenChamber()
    window = build_window(qapp, monkeypatch, tmp_path, openchamber=oc)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-stop-001"))

        assert wait_until(lambda: window._busy), "task did not start"
        assert window.stop_button.isEnabled()
        window.stop_button.click()

        assert wait_until(lambda: not window._busy), "task did not stop"
        assert "已停止等待" in window.status_label.text()
        # the OpenChamber session stays available for a later manual listen
        assert oc.session_exists("ses_test123", "D:/proj")
        record = window._workflow.registry.record("task-stop-001")
        assert record["state"] == "STOPPED_BY_USER"
        assert window.continue_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
    finally:
        shutdown_window(window)


def test_continue_button_relays_interrupted_task(qapp, monkeypatch, tmp_path):
    """Auto-recovery fails once, then the operator clicks "继续当前任务" and
    the continuation wraps under the ORIGINAL task id."""
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.02)
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0)
    oc = RecoveryOpenChamber(recover_at=2)
    window = build_window(qapp, monkeypatch, tmp_path, openchamber=oc)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-cont-001"))

        assert wait_until(lambda: window._busy), "task did not start"
        # original interruption → one failed auto-recovery → manual offer
        assert wait_until(lambda: not window._busy), "task did not settle"
        assert window.continue_button.isEnabled(), "continue offer expected"
        assert window._workflow.pending_continue is not None
        assert (
            window._workflow.registry.record("task-cont-001")["state"] == "FAILED"
        )

        window.continue_button.click()
        assert wait_until(lambda: not window._busy), "continue did not settle"

        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-cont-001" in clipboard.text()
        ), "manual continuation result was not written"
        message = parse_message(clipboard.text())
        assert message.in_reply_to == "task-cont-001"
        assert "recovered answer" in message.body
        record = window._workflow.registry.record("task-cont-001")
        assert record["state"] == "COMPLETED"
        assert window._workflow.pending_continue is None
    finally:
        shutdown_window(window)


def _rejection_settings() -> RelaySettings:
    return RelaySettings(
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_rejected",
        openchamber_sessions={directory_key("D:/proj"): "ses_rejected"},
        openchamber_model="4090/qwen3.8-27b",
        completion_timeout=0,
        poll_interval=0.01,
    )


def test_model_rejection_offers_new_session_retry_and_adopts_fresh_session(
    qapp, monkeypatch, tmp_path
):
    """A non-retryable 400 stops the wait without auto-recovery, enables the
    one-shot "新会话重试", and a working retry switches the session selector
    to the FRESH session and persists it as the project preference."""
    oc = RejectedThenWorkingOpenChamber()
    settings = _rejection_settings()
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-rej-001"))

        assert wait_until(lambda: window._busy), "task did not start"
        assert wait_until(lambda: not window._busy), "task did not settle"
        assert window._workflow.pending_rejection is not None
        assert "新会话重试" in window.status_label.text()
        assert window.new_session_button.isEnabled() is True
        assert window.continue_button.isEnabled() is False
        assert window._workflow.registry.record("task-rej-001")["state"] == "FAILED"

        window.new_session_button.click()
        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "fresh session answer" in clipboard.text()
        ), "fresh-session retry result was not written"
        assert parse_message(clipboard.text()).in_reply_to == "task-rej-001"
        assert oc.history["ses_new"], "fresh session must have received the task"

        assert settings.openchamber_session_id == "ses_new"
        assert settings.openchamber_sessions[directory_key("D:/proj")] == "ses_new"
        assert window._workflow.registry.record("task-rej-001")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


def test_second_rejection_stops_after_one_retry(qapp, monkeypatch, tmp_path):
    """A rejection on the FRESH session consumes the one-shot offer: no third
    session is created and the model must be checked, never auto-switched."""
    oc = AlwaysRejectedOpenChamber()
    settings = _rejection_settings()
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=oc
    )
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-rej-002"))

        assert wait_until(lambda: window._busy), "task did not start"
        assert wait_until(lambda: not window._busy), "task did not settle"
        assert window.new_session_button.isEnabled() is True

        window.new_session_button.click()
        assert wait_until(lambda: not window._busy), "retry did not settle"
        assert oc.created == ["ses_new_too"], "exactly ONE fresh session"
        assert window._workflow.pending_rejection is None
        assert window.new_session_button.isEnabled() is False
        assert "新会话重试失败" in window.status_label.text()
        assert "检查4090模型服务日志" in window.detail_label.text()
        assert (
            window._workflow.registry.record("task-rej-002")["state"] == "FAILED"
        )
    finally:
        shutdown_window(window)


def test_http_400_failure_offers_no_continue(qapp, monkeypatch, tmp_path):
    """An ordinary HTTP 400 is reported and offers neither a same-session
    continue nor a fresh-session retry."""
    oc = BadRequestOpenChamber()
    window = build_window(qapp, monkeypatch, tmp_path, openchamber=oc)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-400-001"))

        assert wait_until(lambda: window._busy), "task did not start"
        assert wait_until(lambda: not window._busy), "task did not settle"
        assert window._workflow.pending_continue is None
        assert window._workflow.pending_rejection is None
        assert window.continue_button.isEnabled() is False
        assert window.new_session_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
        assert "已跳过原会话自动恢复" in window.detail_label.text()
        assert (
            window._workflow.registry.record("task-400-001")["state"] == "FAILED"
        )
    finally:
        shutdown_window(window)


def test_auto_rotation_ignores_reasonix_tasks(qapp, monkeypatch, tmp_path):
    from core.relay_settings import RelaySettings

    settings = RelaySettings(auto_rotate_enabled=True, auto_rotate_threshold=1)
    window = build_window(qapp, monkeypatch, tmp_path, settings=settings)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy), "startup self-check never settled"
        qapp.clipboard().setText(v1_task("REASONIX", "say hi", "task-rot-rx"))
        assert wait_until(
            lambda: "IN_REPLY_TO: task-rot-rx" in qapp.clipboard().text()
        )
        # Reasonix success is not an OpenChamber auto task: never counted,
        # never rotated, no progress hint
        assert window._rotation.count("D:/proj") == 0
        assert window._rotation_pending is False
        assert "自动轮换进度" not in window.detail_label.text()
    finally:
        shutdown_window(window)


def test_auto_rotation_creates_and_adopts_session_at_threshold(qapp, monkeypatch, tmp_path):
    """A completed automatic OpenChamber task at the threshold creates a
    fresh session, persists it as the fixed session and adopts it."""
    import ui as ui_mod

    from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
    from tests.fakes import (
        ScriptedOpenChamber,
        assistant_message,
        make_dispatch,
        text_part,
        user_message,
    )

    oc = ScriptedOpenChamber()
    oc.next_session_id = "ses_rotated1"
    oc.send_dispatch = make_dispatch(session_id="ses_test123", directory="D:/proj")
    oc.message_timelines = [
        [
            user_message("u_rot", "task body", 1000),
            assistant_message(
                "a_rot", 1100, completed=1200, finish="stop",
                parts=[text_part("final now")], parent_id="u_rot",
            ),
        ]
    ]

    class RotatingOpenChamber(ScriptedOpenChamber):
        """Every UI-constructed client (rotation worker / session refresh)
        returns the same fresh session id."""

        def __init__(self, url=None, directory="D:/proj", **kwargs):
            super().__init__(directory=directory)
            self.url = url
            self.next_session_id = "ses_rotated1"

    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory="D:/proj",
        openchamber_session_id="ses_test123",
        openchamber_model="4090/qwen3.8-27b",
        completion_timeout=5.0,
        poll_interval=0.01,
        auto_rotate_enabled=True,
        auto_rotate_threshold=1,
    )
    monkeypatch.setattr(ui_mod, "OpenChamberClient", RotatingOpenChamber)
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=oc
    )
    window._startup_check_pending = False
    try:
        key = directory_key("D:/proj")
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-rot-001"))
        window._on_clipboard_text(qapp.clipboard().text())

        assert wait_until(
            lambda: window._settings.openchamber_session_id == "ses_rotated1"
        )
        # rotation success resets the project counter (settings persist
        # before the UI signal handler runs, so wait for the reset itself)
        assert wait_until(lambda: window._rotation.count("D:/proj") == 0)
        assert window._settings.openchamber_sessions[key] == "ses_rotated1"
        assert window.recopy_button.isEnabled()
    finally:
        shutdown_window(window)


# -------------------------------------------------------------------- #
# 任务接收: automatic clipboard start, distinct labels, task dedup
# -------------------------------------------------------------------- #


def test_auto_start_enables_a_side_receiving(qapp, monkeypatch, tmp_path):
    """Scenario 1: launching AI Relay B automatically enables the clipboard
    listener (任务接收) without any click, even before the Reasonix
    self-check finishes."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy)
        assert window.start_button.isEnabled() is False
        assert window.pause_button.isEnabled()
    finally:
        shutdown_window(window)


def test_action_buttons_label_a_side_receiving_and_monitor(qapp, monkeypatch, tmp_path):
    """Scenario 2: the two listening features have distinct, non-confusing
    button labels."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert window.start_button.text() == "启动任务接收"
        assert window.pause_button.text() == "暂停任务接收"
        assert window.monitor_button.text() == "监控 OpenChamber"
    finally:
        shutdown_window(window)


def test_startup_delivers_existing_clipboard_task_once(qapp, monkeypatch, tmp_path):
    """Scenario 3: a valid NEW task already sitting on the clipboard when the
    app launches is delivered exactly once, and a second copy of the same
    text is ignored (never re-executed)."""
    from PySide6.QtTest import QTest

    qapp.clipboard().setText(
        v1_task("REASONIX", "leftover work", "task-pickup-001")
    )
    window = build_window(qapp, monkeypatch, tmp_path)
    reasonix = window._workflow.reasonix
    try:
        assert wait_until(
            lambda: "IN_REPLY_TO: task-pickup-001" in qapp.clipboard().text()
        ), "leftover clipboard task was never delivered once"
        assert len(reasonix.executed) == 1
        record = window._workflow.registry.record("task-pickup-001")
        assert record is not None and record["state"] == "COMPLETED"

        qapp.clipboard().setText(
            v1_task("REASONIX", "same task resent", "task-pickup-001")
        )
        QTest.qWait(300)
        assert len(reasonix.executed) == 1, "duplicate clipboard task re-executed"
        assert window._workflow.registry.record("task-pickup-001")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


def test_historical_registered_task_never_reexecuted(qapp, monkeypatch, tmp_path):
    """Scenario 4: a task already persisted in tasks.json (any state) at
    launch is never executed again; the startup pickup only logs the ignore."""
    from PySide6.QtTest import QTest

    TaskRegistry(tmp_path / "tasks.json").mark("task-hist-001", "COMPLETED")
    qapp.clipboard().setText(
        v1_task("REASONIX", "already done", "task-hist-001")
    )
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        QTest.qWait(300)
        assert len(window._workflow.reasonix.executed) == 0
        record = window._workflow.registry.record("task-hist-001")
        assert record is not None and record["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


def test_pause_is_explicit_and_incoming_tasks_ignored(qapp, monkeypatch, tmp_path):
    """Scenario 7: Pause stops ONLY the clipboard listener, stays paused (no
    auto-resume by itself), and incoming clipboard tasks are ignored while
    paused."""
    from PySide6.QtTest import QTest

    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy)
        window._pause()
        assert not window._listener.enabled
        assert window._clipboard_status_label.text() == "任务接收：已暂停"
        assert window.start_button.isEnabled() is True
        assert window.pause_button.isEnabled() is False

        qapp.clipboard().setText(
            v1_task("REASONIX", "while paused", "task-pause-001")
        )
        QTest.qWait(300)
        assert len(window._workflow.reasonix.executed) == 0
        assert window._workflow.registry.record("task-pause-001") is None
    finally:
        shutdown_window(window)


def test_runtime_status_labels_reflect_a_side_state(qapp, monkeypatch, tmp_path):
    """Scenario 11: the status card shows BOTH listener states independently;
    the OpenChamber monitor label stays untouched by clipboard pause."""
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert window._clipboard_status_label.text() == "任务接收：运行中"
        assert window._monitor_status_label.text() == "OpenChamber监控：未启动"

        window._pause()
        assert window._clipboard_status_label.text() == "任务接收：已暂停"
        assert window._monitor_status_label.text() == "OpenChamber监控：未启动"

        window._start()
        assert window._clipboard_status_label.text() == "任务接收：运行中"
        assert window._monitor_status_label.text() == "OpenChamber监控：未启动"
    finally:
        shutdown_window(window)


def test_a_side_receiving_logs_not_full_body(qapp, monkeypatch, tmp_path, caplog):
    """A received task logs its task_id and round but never the whole body;
    a duplicate logs that it returns the existing state/reply (never
    re-executes) with its persisted state."""
    import logging

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().setText(v1_task("REASONIX", "secret body xyz", "task-log-001"))
    window = build_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(
            lambda: (
                window._workflow.registry.record("task-log-001") or {}
            ).get("state")
            == "COMPLETED"
        )
        messages = [r.message for r in caplog.records]
        assert any("收到A端TASK" in m and "task_id=task-log-001" in m for m in messages)
        assert any("task-log-001" in m and "round=1/3" in m for m in messages)
        assert not any("secret body xyz" in m for m in messages)

        qapp.clipboard().setText(
            v1_task("REASONIX", "second copy of the same task", "task-log-001")
        )
        assert wait_until(
            lambda: any(
                "重复TASK" in m
                and "task_id=task-log-001" in m
                and "existing_state=COMPLETED" in m
                for m in (r.message for r in caplog.records)
            )
        )
        assert not any("secret body xyz" in m for m in messages)
    finally:
        shutdown_window(window)


def test_listener_still_delivers_tasks_when_self_check_fails(
    qapp, monkeypatch, tmp_path
):
    """Scenario 10: the automatic clipboard start is independent of the
    Reasonix self-check -- a task shipped right after a failed self-check is
    still received (and its failure recorded)."""
    window = build_window(qapp, monkeypatch, tmp_path, reasonix_cls=FailingReasonix)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: "Reasonix 自检失败" in window.detail_label.text())
        qapp.clipboard().setText(
            v1_task("REASONIX", "boom", "task-failpick-001")
        )
        assert wait_until(
            lambda: (
                window._workflow.registry.record("task-failpick-001") or {}
            ).get("state")
            == "FAILED"
        ), "clipboard task was never received after a failed self-check"
    finally:
        shutdown_window(window)
