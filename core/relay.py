"""Relay workflow orchestration independent from the user interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.openchamber import (
    CompletionResult,
    ModelRef,
    OpenChamberClient,
    wait_for_completion,
)
from core.protocol import MessageType, RelayProtocolError, RelayMessage, parse_message, wrap_response
from core.reasonix_uia import ReasonixAutomation
from core.relay_settings import (
    TARGET_EXECUTOR,
    TARGET_OPENCHAMBER,
    TARGET_REASONIX,
    KNOWN_TARGETS,
    RelaySettings,
)
from core.runtime_paths import data_dir
from core.task_registry import TaskRegistry


class RelayWorkflowError(RuntimeError):
    """Raised when an incoming task cannot be relayed safely."""


def resolve_executor_kind(target: str, settings: RelaySettings) -> str:
    """Map a protocol TARGET to an executor.

    An explicit TARGET (REASONIX / OPENCHAMBER) is never overridden by the
    configured default; only TARGET: EXECUTOR falls back to it.
    """
    if target == TARGET_EXECUTOR:
        return settings.default_target
    return target


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """What the UI needs after a finished task (success path only)."""

    task_id: str
    executor: str
    session_id: str | None = None
    directory: str | None = None
    note: str | None = None


class RelayWorkflow:
    def __init__(
        self,
        reasonix: ReasonixAutomation | None = None,
        registry: TaskRegistry | None = None,
        settings: RelaySettings | None = None,
        openchamber: OpenChamberClient | None = None,
        replies_dir: Path | None = None,
    ):
        self.reasonix = reasonix or ReasonixAutomation()
        self.registry = registry or TaskRegistry()
        self.settings = settings or RelaySettings()
        self.openchamber = openchamber
        self.replies_dir = replies_dir or (data_dir() / "replies")
        self.outcome: TaskOutcome | None = None

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #

    def process(
        self,
        text: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> str:
        update = status_callback or (lambda _status: None)
        try:
            message = parse_message(text)

            if message.source != "CHATGPT":
                raise RelayWorkflowError(f"unsupported task source: {message.source}")
            if message.target not in KNOWN_TARGETS:
                raise RelayWorkflowError(
                    f"task target is not executable: {message.target}"
                )
            if message.message_type is not MessageType.TASK:
                raise RelayWorkflowError(
                    f"clipboard message is not a TASK: {message.message_type.value}"
                )
            if self.registry.contains(message.message_id):
                raise RelayWorkflowError(
                    f"task was already processed: {message.message_id}"
                )

            executor_kind = resolve_executor_kind(message.target, self.settings)
            if executor_kind == TARGET_OPENCHAMBER:
                return self._run_openchamber(message, update)
            return self._run_reasonix(message, update)
        except RelayProtocolError as exc:
            raise RelayWorkflowError(f"invalid clipboard task: {exc}") from exc

    # ------------------------------------------------------------------ #
    # executors
    # ------------------------------------------------------------------ #

    def _run_reasonix(self, message: RelayMessage, update) -> str:
        self.registry.mark(message.message_id, "PROCESSING", executor=TARGET_REASONIX)
        update("正在发送到 Reasonix")
        try:
            reply = self.reasonix.execute(message.body)
            update("正在包装 Reasonix 回复")
            response = wrap_response(
                reply,
                message.message_id,
                message.protocol_format,
                message.round_number,
                message.max_rounds,
            )
            reply_file = self.save_reply(message.message_id, response)
            self.registry.mark(
                message.message_id,
                "COMPLETED",
                executor=TARGET_REASONIX,
                reply_file=str(reply_file),
            )
            self.outcome = TaskOutcome(
                task_id=message.message_id, executor=TARGET_REASONIX
            )
            return response
        except Exception as exc:
            error = f"reasonix_execute:{type(exc).__name__}: {exc}"
            self.registry.mark(
                message.message_id, "FAILED", error, executor=TARGET_REASONIX
            )
            raise

    def _openchamber_client(self) -> OpenChamberClient:
        if self.openchamber is not None:
            return self.openchamber
        return OpenChamberClient(self.settings.openchamber_url)

    def _run_openchamber(self, message: RelayMessage, update) -> str:
        directory = self.settings.openchamber_directory.strip()
        if not directory:
            self.registry.mark(
                message.message_id, "FAILED", "openchamber_config:missing directory",
                executor=TARGET_OPENCHAMBER,
            )
            raise RelayWorkflowError("OpenChamber 项目目录未配置，请在设置中填写")

        client = self._openchamber_client()
        session_id: str | None = None
        try:
            update("正在验证 OpenChamber 连接")
            client.verify()

            update("正在创建 OpenChamber 会话")
            title = f"AI Relay {message.message_id[:8]}"
            session_id = client.create_session(title, directory)
            self.registry.mark(
                message.message_id,
                "PROCESSING",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
            )

            update("正在请求 OpenChamber 打开会话")
            client.open_session(session_id)  # failure: do NOT send the task

            update("正在发送任务到 OpenChamber")
            model = ModelRef.parse(self.settings.openchamber_model or None)
            dispatch = client.send(
                session_id,
                message.body,
                directory,
                agent=self.settings.openchamber_agent.strip() or None,
                model=model,
            )
            if dispatch.resolved_model is not None and model is not None \
                    and dispatch.resolved_model != model:
                update(
                    "注意：OpenChamber 解析的模型为 "
                    f"{dispatch.resolved_model.label()}，与请求的 "
                    f"{model.label()} 不一致"
                )

            update(f"正在等待 OpenChamber 完成（会话 {session_id}）")
            result = wait_for_completion(
                client,
                dispatch,
                self.settings.completion_timeout,
                self.settings.poll_interval,
                status_callback=update,
            )
            if result.model_mismatch:
                actual = result.actual_model.label() if result.actual_model else "未知"
                expected = (
                    (result.resolved_model or result.requested_model).label()
                    if (result.resolved_model or result.requested_model)
                    else "未指定"
                )
                update(f"警告：实际执行模型 {actual} 与解析模型 {expected} 不一致")

            update("正在包装 OpenChamber 回复")
            response = wrap_response(
                result.final_text,
                message.message_id,
                message.protocol_format,
                message.round_number,
                message.max_rounds,
            )
            reply_file = self.save_reply(message.message_id, response)
            self.registry.mark(
                message.message_id,
                "COMPLETED",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
                reply_file=str(reply_file),
            )
            note = None
            if result.model_mismatch:
                actual = result.actual_model.label() if result.actual_model else "未知"
                expected = (
                    (result.resolved_model or result.requested_model).label()
                    if (result.resolved_model or result.requested_model)
                    else "未指定"
                )
                note = f"模型不一致：实际 {actual}，请求/解析 {expected}"
            self.outcome = TaskOutcome(
                task_id=message.message_id,
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
                note=note,
            )
            return response
        except Exception as exc:
            error = f"openchamber_execute:{type(exc).__name__}: {exc}"
            self.registry.mark(
                message.message_id,
                "FAILED",
                error,
                executor=TARGET_OPENCHAMBER,
                **(
                    {"session_id": session_id, "directory": directory}
                    if session_id
                    else {"directory": directory}
                ),
            )
            raise

    # ------------------------------------------------------------------ #
    # reply persistence
    # ------------------------------------------------------------------ #

    def save_reply(self, task_id: str, response: str) -> Path:
        if not response.strip():
            raise RelayWorkflowError("reply must not be empty")
        self.replies_dir.mkdir(parents=True, exist_ok=True)
        path = self.replies_dir / f"{task_id}.response.txt"
        path.write_text(response, encoding="utf-8")
        return path

    def load_reply(self, task_id: str) -> str | None:
        path = self.replies_dir / f"{task_id}.response.txt"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None