"""Minimal PySide6 control window for AI Relay."""

from __future__ import annotations

import logging
from datetime import datetime

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.clipboard import ClipboardListener
from core.openchamber import OpenChamberClient
from core.protocol import MessageType, RelayProtocolError, parse_message
from core.reasonix_uia import ReasonixAutomation
from core.relay import RelayWorkflow, TaskOutcome
from core.relay_settings import (
    DEFAULT_OPENCHAMBER_URL,
    TARGET_OPENCHAMBER,
    TARGET_REASONIX,
    RelaySettings,
)
from core.runtime_paths import data_dir

LOG_PATH = data_dir() / "relay.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
LOGGER = logging.getLogger("ai_relay_b")


class WorkerSignals(QObject):
    status = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)


class RelayTask(QRunnable):
    def __init__(self, workflow: RelayWorkflow, text: str):
        super().__init__()
        self.workflow = workflow
        self.text = text
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.workflow.process(self.text, self.signals.status.emit)
            self.signals.succeeded.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class SelfCheckTask(QRunnable):
    def __init__(self, reasonix: ReasonixAutomation):
        super().__init__()
        self.reasonix = reasonix
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            checks = self.reasonix.self_check()
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(f"自检未通过: {', '.join(failed)}")
            self.signals.succeeded.emit("")
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class RelayWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.setWindowTitle("AI Relay")
        self.setMinimumWidth(520)

        self._busy = False
        self._startup_check_pending = True
        self._pool = QThreadPool.globalInstance()
        self._reasonix = ReasonixAutomation()
        self._settings = RelaySettings.load()
        self._workflow = RelayWorkflow(
            self._reasonix, settings=self._settings
        )
        self._last_response: str | None = None
        self._last_outcome: TaskOutcome | None = None
        self._listener = ClipboardListener(app.clipboard())
        self._listener.task_received.connect(self._on_clipboard_text)

        self.status_label = QLabel("未监听")
        self.detail_label = QLabel(
            "启动监听后，处理 AI_RELAY/1 与旧版剪贴板任务；"
            "TARGET 为 REASONIX / OPENCHAMBER / EXECUTOR（默认执行端）。"
        )
        self.detail_label.setWordWrap(True)
        self.start_button = QPushButton("启动监听")
        self.pause_button = QPushButton("暂停监听")
        self.check_button = QPushButton("测试 Reasonix 连接")
        self.open_session_button = QPushButton("打开当前会话")
        self.recopy_button = QPushButton("重新复制回复")
        self.pause_button.setEnabled(False)
        self.open_session_button.setEnabled(False)
        self.recopy_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.check_button)
        layout.addWidget(self._build_settings_group())
        row1 = self._button_row(self.open_session_button)
        row2 = self._button_row(self.recopy_button)
        layout.addLayout(row1)
        layout.addLayout(row2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.check_button.clicked.connect(self._self_check)
        self.open_session_button.clicked.connect(self._open_current_session)
        self.recopy_button.clicked.connect(self._recopy_reply)
        self._save_settings_button.clicked.connect(self._save_settings)

        QTimer.singleShot(0, self._self_check)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _button_row(self, button: QPushButton):
        from PySide6.QtWidgets import QHBoxLayout

        row = QHBoxLayout()
        row.addWidget(button)
        return row

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("执行端设置")
        form = QFormLayout()

        self._executor_combo = QComboBox()
        self._executor_combo.addItem("Reasonix", TARGET_REASONIX)
        self._executor_combo.addItem("OpenChamber", TARGET_OPENCHAMBER)
        index = self._executor_combo.findData(self._settings.default_target)
        self._executor_combo.setCurrentIndex(max(index, 0))

        self._url_edit = QLineEdit(self._settings.openchamber_url or DEFAULT_OPENCHAMBER_URL)
        self._directory_edit = QLineEdit(self._settings.openchamber_directory)
        self._agent_edit = QLineEdit(self._settings.openchamber_agent)
        self._agent_edit.setPlaceholderText("可选，如 build")
        self._model_edit = QLineEdit(self._settings.openchamber_model)
        self._model_edit.setPlaceholderText("可选，格式 providerID/modelID")

        form.addRow("默认执行端（TARGET: EXECUTOR）", self._executor_combo)
        form.addRow("OpenChamber 地址", self._url_edit)
        form.addRow("项目目录", self._directory_edit)
        form.addRow("Agent", self._agent_edit)
        form.addRow("Model", self._model_edit)

        self._save_settings_button = QPushButton("保存设置")
        form.addRow(self._save_settings_button)

        group.setLayout(form)
        return group

    # ------------------------------------------------------------------ #
    # monitoring control
    # ------------------------------------------------------------------ #

    @Slot()
    def _start(self):
        self._listener.start()
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self._set_status("空闲：正在监听剪贴板")

    @Slot()
    def _pause(self):
        self._listener.pause()
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self._set_status("已暂停")

    @Slot(str)
    def _on_clipboard_text(self, text: str):
        if not self._listener.looks_like_relay_message(text):
            return

        try:
            message = parse_message(text)
        except RelayProtocolError as exc:
            self._show_error(f"剪贴板 AI_RELAY 消息格式错误：{exc}")
            return

        if message.message_type is not MessageType.TASK:
            return

        if self._busy:
            self._show_error("当前任务尚未完成，已拒绝新的剪贴板任务。")
            return

        self._busy = True
        self._set_controls_enabled(False)

        task = RelayTask(self._workflow, text)
        task.signals.status.connect(self._set_status)
        task.signals.succeeded.connect(self._task_succeeded)
        task.signals.failed.connect(self._task_failed)
        self._pool.start(task)

    # ------------------------------------------------------------------ #
    # task results
    # ------------------------------------------------------------------ #

    @Slot(str)
    def _task_succeeded(self, response: str):
        LOGGER.info("task completed response_length=%d", len(response))
        self._last_response = response
        self._last_outcome = self._workflow.outcome
        self._listener.write_response(response)

        detail = "已将包装后的回复写入剪贴板。"
        outcome = self._workflow.outcome
        if outcome is not None:
            if outcome.executor == TARGET_OPENCHAMBER and outcome.session_id:
                detail += f"（执行端 OpenChamber，会话 {outcome.session_id}）"
            if outcome.note:
                detail += f" {outcome.note}"
        self.detail_label.setText(detail)
        self.open_session_button.setEnabled(
            outcome is not None and outcome.session_id is not None
        )
        self.recopy_button.setEnabled(True)
        self._finish_task("完成：等待下一个任务")

    @Slot(str)
    def _task_failed(self, error: str):
        LOGGER.error("task failed: %s", error)
        if self._startup_check_pending:
            self._startup_check_pending = False
            self._finish_task("Reasonix 自检失败（不影响 OpenChamber）")
            self.detail_label.setText(
                f"Reasonix 自检失败：{error}。"
                "监听仍会启动；REASONIX 任务将失败，OPENCHAMBER 任务不受影响。"
            )
            self._auto_start_after_self_check()
            return
        self._finish_task("错误")
        self._show_error(error)

    def _finish_task(self, status: str):
        self._busy = False
        self._set_controls_enabled(True)
        self._set_status(status)

    # ------------------------------------------------------------------ #
    # self check (informational only: must never block monitoring)
    # ------------------------------------------------------------------ #

    def _auto_start_after_self_check(self):
        if not self._listener.enabled:
            self._start()

    @Slot()
    def _self_check(self):
        if self._busy:
            return
        self._busy = True
        self._set_controls_enabled(False)
        self._set_status("正在检查 Reasonix")

        task = SelfCheckTask(self._reasonix)
        task.signals.succeeded.connect(self._self_check_succeeded)
        task.signals.failed.connect(self._task_failed)
        self._pool.start(task)

    @Slot(str)
    def _self_check_succeeded(self, _unused: str):
        self.detail_label.setText("Reasonix 窗口、输入框和发送按钮均可通过 UIA 识别。")
        self._finish_task("Reasonix 连接自检通过")
        if self._startup_check_pending:
            self._startup_check_pending = False
            self._auto_start_after_self_check()

    # ------------------------------------------------------------------ #
    # session / reply actions
    # ------------------------------------------------------------------ #

    @Slot()
    def _open_current_session(self):
        outcome = self._last_outcome or self._workflow.outcome
        if outcome is None or not outcome.session_id:
            self._show_error("当前没有可打开的 OpenChamber 会话")
            return
        try:
            OpenChamberClient(self._settings.openchamber_url).open_session(
                outcome.session_id
            )
        except Exception as exc:
            self._show_error(f"打开会话失败：{exc}")
            return
        self.detail_label.setText(
            "已请求 OpenChamber 打开会话"
            f"（{outcome.session_id}）；请在桌面窗口确认显示。"
        )
        self._set_status("已请求打开 OpenChamber 会话")

    @Slot()
    def _recopy_reply(self):
        response = self._last_response
        if response is None and self._last_outcome is not None:
            response = self._workflow.load_reply(self._last_outcome.task_id)
        if response is None:
            self._show_error("没有可重新复制的回复")
            return
        self._listener.write_response(response)
        self.detail_label.setText("已重新复制上次回复到剪贴板（未重复执行任务）。")
        self._set_status("已重新复制回复")

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #

    @Slot()
    def _save_settings(self):
        try:
            self._settings.default_target = (
                self._executor_combo.currentData() or TARGET_REASONIX
            )
            self._settings.openchamber_url = self._url_edit.text().strip()
            self._settings.openchamber_directory = self._directory_edit.text().strip()
            self._settings.openchamber_agent = self._agent_edit.text().strip()
            self._settings.openchamber_model = self._model_edit.text().strip()
            self._settings.validate()
            self._settings.save()
        except Exception as exc:
            self._show_error(f"保存设置失败：{exc}")
            return
        self.detail_label.setText("设置已保存。")
        self._set_status("设置已保存")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _set_controls_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled and not self._listener.enabled)
        self.pause_button.setEnabled(enabled and self._listener.enabled)
        self.check_button.setEnabled(enabled)
        self._save_settings_button.setEnabled(enabled)
        if enabled:
            self.open_session_button.setEnabled(
                (self._last_outcome or self._workflow.outcome) is not None
                and (self._last_outcome or self._workflow.outcome).session_id
                is not None
            )
            self.recopy_button.setEnabled(self._last_response is not None)

    def _set_status(self, status: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"[{now}] {status}")

    def _show_error(self, error: str):
        self.detail_label.setText(error)
        self._set_status("错误")