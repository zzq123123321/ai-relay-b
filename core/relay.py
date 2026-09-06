"""Relay workflow orchestration independent from the user interface."""

from __future__ import annotations

import hashlib
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
    """Task/session identity after a task started or finished.

    ``current_session`` is set as soon as the OpenChamber session exists and
    is persisted, and is kept after failure so the UI can open the session
    of the task that is currently running or just failed.  ``outcome`` is
    the LAST SUCCESS reply reference, kept separately for repeat copy.
    ``model_info`` carries the three-layer model detail (requested /
    resolved / actual) when an OpenChamber task completed.
    """

    task_id: str
    executor: str
    session_id: str | None = None
    directory: str | None = None
    note: str | None = None
    model_info: str | None = None


def model_details(
    requested: ModelRef | None,
    resolved: ModelRef | None,
    actual: ModelRef | None,
) -> tuple[str | None, str]:
    """Compute model detail display and the mismatch note.

    Two INDEPENDENT mismatch conditions are distinguished:
    * requested != resolved (the provider resolved to a different model
      than the one asked for);
    * resolved != actual (the message the model actually ran on differs
      from the resolved model).

    Missing values are shown as 未指定 (requested/resolved) or 未知
    (actual), never guessed.  Returns ``(note, info)`` where ``info``
    always shows the three layers and ``note`` is the "模型不一致：…"
    prefix when at least one condition holds.
    """
    requested_label = requested.label() if requested is not None else "未指定"
    resolved_label = resolved.label() if resolved is not None else "未指定"
    actual_label = actual.label() if actual is not None else "未知"
    info = f"请求 {requested_label}，解析 {resolved_label}，实际 {actual_label}"
    mismatch = (
        requested is not None
        and resolved is not None
        and requested != resolved
    ) or (
        resolved is not None
        and actual is not None
        and resolved != actual
    )
    note = f"模型不一致：{info}" if mismatch else None
    return note, info


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
        self.current_session: TaskOutcome | None = None

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #

    def process(
        self,
        text: str,
        status_callback: Callable[[str], None] | None = None,
        session_callback: Callable[[TaskOutcome], None] | None = None,
    ) -> str:
        update = status_callback or (lambda _status: None)
        # A new task invalidates the previous task's current session so the
        # UI's open-session button never points at an older task while this
        # one is being routed.
        self.current_session = None
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
                return self._run_openchamber(message, update, session_callback)
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

    def _run_openchamber(
        self,
        message: RelayMessage,
        update,
        session_callback: Callable[[TaskOutcome], None] | None = None,
    ) -> str:
        directory = self.settings.openchamber_directory.strip()
        if not directory:
            self.registry.mark(
                message.message_id, "FAILED", "openchamber_config:missing directory",
                executor=TARGET_OPENCHAMBER,
            )
            raise RelayWorkflowError("OpenChamber 项目目录未配置，请在设置中填写")

        session_id = self.settings.openchamber_session_id.strip()
        if not session_id:
            self.registry.mark(
                message.message_id,
                "FAILED",
                "openchamber_config:missing session id",
                executor=TARGET_OPENCHAMBER,
                directory=directory,
            )
            raise RelayWorkflowError(
                "OpenChamber 会话 ID 未配置：请在设置中选择或填写已有会话 ID；"
                "中继不会自动创建或更换会话"
            )

        client = self._openchamber_client()
        try:
            update("正在验证 OpenChamber 连接")
            client.verify()

            update("正在确认 OpenChamber 会话")
            if not client.session_exists(session_id, directory):
                raise RelayWorkflowError(
                    f"OpenChamber 会话 {session_id} 不存在或不属于项目目录"
                    f"{directory}，任务未发送；中继不会自动创建或更换会话，"
                    "请在设置中重新选择已有会话"
                )
            self.registry.mark(
                message.message_id,
                "PROCESSING",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
            )
            # Notify the UI as soon as the session exists AND is persisted:
            # from this moment the open-session button must point at THIS
            # task, during execution as well as after failure.
            self.current_session = TaskOutcome(
                task_id=message.message_id,
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
            )
            if session_callback is not None:
                session_callback(self.current_session)

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
            # Persist requested/resolved model NOW: it must survive any
            # later state update (a mismatch warning must not vanish).
            self.registry.mark(
                message.message_id,
                "PROCESSING",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
                requested_model=(
                    dispatch.requested_model.label()
                    if dispatch.requested_model is not None
                    else None
                ),
                resolved_model=(
                    dispatch.resolved_model.label()
                    if dispatch.resolved_model is not None
                    else None
                ),
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
            note, _info = model_details(
                result.requested_model, result.resolved_model, result.actual_model
            )
            if note:
                update(f"警告：{note}")

            update("正在包装 OpenChamber 回复")
            response = wrap_response(
                result.final_text,
                message.message_id,
                message.protocol_format,
                message.round_number,
                message.max_rounds,
            )
            reply_file = self.save_reply(message.message_id, response)
            note, model_info = model_details(
                result.requested_model, result.resolved_model, result.actual_model
            )
            self.registry.mark(
                message.message_id,
                "COMPLETED",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
                reply_file=str(reply_file),
                requested_model=(
                    result.requested_model.label()
                    if result.requested_model is not None
                    else None
                ),
                resolved_model=(
                    result.resolved_model.label()
                    if result.resolved_model is not None
                    else None
                ),
                actual_model=(
                    result.actual_model.label()
                    if result.actual_model is not None
                    else None
                ),
                model_note=note,
            )
            self.outcome = TaskOutcome(
                task_id=message.message_id,
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
                note=note,
                model_info=model_info,
            )
            self.current_session = self.outcome
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
    # reply persistence (untrusted task ids never touch the file name)
    # ------------------------------------------------------------------ #

    def reply_file_for(self, task_id: str) -> Path:
        """Hash-derived reply file path: safe for any task id (traversal,
        absolute paths, drive letters, reserved names, illegal characters,
        CJK ids all collapse to the same hex name pattern)."""
        digest = hashlib.sha256(task_id.encode("utf-8", "surrogatepass")).hexdigest()
        return self.replies_dir / f"rel_{digest}.response.txt"

    def save_reply(self, task_id: str, response: str) -> Path:
        if not response.strip():
            raise RelayWorkflowError("reply must not be empty")
        self.replies_dir.mkdir(parents=True, exist_ok=True)
        path = self.reply_file_for(task_id)
        path.write_text(response, encoding="utf-8")
        return path

    def _contained_reply_path(self, path: Path) -> Path | None:
        """Return ``path`` only if it resolves inside the replies dir."""
        try:
            resolved = path.resolve()
            root = self.replies_dir.resolve()
        except OSError:
            return None
        if not resolved.is_relative_to(root):
            return None
        return resolved

    def load_reply(self, task_id: str) -> str | None:
        # 1) the registry mapping recorded when the reply was saved
        record = self.registry.record(task_id)
        if record and record.get("reply_file"):
            path = self._contained_reply_path(Path(record["reply_file"]))
            if path is not None and path.is_file():
                try:
                    return path.read_text(encoding="utf-8")
                except OSError:
                    return None
        # 2) the current hash-derived name
        path = self._contained_reply_path(self.reply_file_for(task_id))
        if path is not None and path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
        # 3) legacy name written before the hash-naming change
        legacy = self._contained_reply_path(self.replies_dir / f"{task_id}.response.txt")
        if legacy is not None and legacy.is_file():
            try:
                return legacy.read_text(encoding="utf-8")
            except OSError:
                return None
        return None