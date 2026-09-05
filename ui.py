"""Minimal PySide6 control window for AI Relay."""

from __future__ import annotations

import logging
from datetime import datetime

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.clipboard import ClipboardListener
from core.protocol import MessageType, RelayProtocolError, parse_message
from core.reasonix_uia import ReasonixAutomation
from core.relay import RelayWorkflow
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
        self.setMinimumWidth(460)

        self._busy = False
        self._startup_check_pending = True
        self._pool = QThreadPool.globalInstance()
        self._reasonix = ReasonixAutomation()
        self._workflow = RelayWorkflow(self._reasonix)
        self._listener = ClipboardListener(app.clipboard)
        self._listener.task_received.connect(self._on_clipboard_text)

        self.status_label = QLabel("未监听")
        self.detail_label = QLabel("启动监听后，只处理 AI_RELAY/1 格式的 Reasonix 任务。")
        self.detail_label.setWordWrap(True)
        self.start_button = QPushButton("启动监听")
        self.pause_button = QPushButton("暂停监听")
        self.check_button = QPushButton("测试 Reasonix 连接")
        self.pause_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.check_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.check_button.clicked.connect(self._self_check)

        QTimer.singleShot(0, self._self_check)

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

        if message.target not in frozenset({"REASONIX", "EXECUTOR"}):
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

    @Slot(str)
    def _task_succeeded(self, response: str):
        LOGGER.info("task completed response_length=%d", len(response))
        self._listener.write_response(response)
        self.detail_label.setText("已将包装后的 Reasonix 回复写入剪贴板。")
        self._finish_task("完成：等待下一个任务")

    @Slot(str)
    def _task_failed(self, error: str):
        LOGGER.error("task failed: %s", error)
        if self._startup_check_pending:
            self._startup_check_pending = False
            self._finish_task("启动自检失败")
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.check_button.setEnabled(True)
            self.detail_label.setText(f"监听未启动：{error}")
            return
        self._finish_task("错误")
        self._show_error(error)

    def _finish_task(self, status: str):
        self._busy = False
        self._set_controls_enabled(True)
        self._set_status(status)

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
            self._start()

    def _set_controls_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled and not self._listener.enabled)
        self.pause_button.setEnabled(enabled and self._listener.enabled)
        self.check_button.setEnabled(enabled)

    def _set_status(self, status: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"[{now}] {status}")

    def _show_error(self, error: str):
        self.detail_label.setText(error)
