"""Relay workflow orchestration independent from the user interface."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core.openchamber import (
    CompletionResult,
    MISSING_FROM_STATUS_MAP,
    ModelRef,
    OpenChamberBadRequestError,
    OpenChamberCancelledError,
    OpenChamberError,
    OpenChamberInterruptedError,
    OpenChamberModelRequestRejectedError,
    OpenChamberClient,
    _message_id,
    _message_text,
    _role,
    _round_assistant_messages,
    _sleep_interruptible,
    error_detail,
    extract_final_text,
    has_pending_user_action,
    mark_message_wrapped,
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


_LOG = logging.getLogger("ai_relay_b")


RECOVERY_DELAY_SECONDS = 10.0
COMPLETION_GRACE_SECONDS = 5.0
CANCELLED_MARKER = "已停止等待，OpenChamber 会话仍保留"
OPENCHAMBER_CONTINUE_PROMPT = (
    "检测到上一次工具执行后的续接中断。请从最后一个未完成步骤继续，"
    "不要重复已经完成的调查和修改；完成剩余验证并返回一份完整最终报告。"
    "不要只回复“继续”或过程说明。"
)
MODEL_REJECTION_PROMPT = "Qwen 拒绝当前会话请求，可能是上下文过长。请点击“新会话重试”。"
AR_RECOVERY_FAILED = (
    "OpenCode 工具续接再次中断，已停止自动恢复。自动恢复失败，"
    "请点击“继续当前任务”重试，或点击“停止当前任务”结束。"
)
MODEL_REJECTION_AGAIN = "新会话仍被模型服务拒绝，请检查4090模型服务日志。"
MODEL_REJECTION_RETRY_PREFIX = (
    "这是从旧会话恢复的新会话。请先检查当前项目文件和已有实现，"
    "确认当前实际状态，不要重复已经完成的修改。\n\n"
    "请执行下面的原始任务，完成必要的修改、测试和验证后返回完整最终报告。\n\n"
    "原始任务："
)


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


def directory_key(directory: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(directory)))


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


@dataclass(frozen=True, slots=True)
class OpenChamberContinue:
    """Everything needed to continue an interrupted OpenChamber task.

    Filled when an abnormal interruption is detected (so the operator can
    click "继续当前任务" again even after the automatic recovery attempt),
    and cleared as soon as the round completes successfully.
    """

    message: RelayMessage
    session_id: str
    directory: str
    agent: str | None = None
    model: ModelRef | None = None


@dataclass(frozen=True, slots=True)
class OpenChamberModelRejection:
    """Everything needed to retry a model-rejected task in a FRESH session.

    Filled when the upstream model service returns a non-retryable HTTP 400
    (never auto-recovered in the rejected session), so the operator can click
    "新会话重试".  One-shot per task: the offer is cleared as soon as the
    retry is attempted, regardless of its outcome.
    """

    message: RelayMessage
    session_id: str
    directory: str
    agent: str | None = None
    model: ModelRef | None = None


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
        self.recovery_attempted: bool = False
        self.pending_continue: OpenChamberContinue | None = None
        self.pending_rejection: OpenChamberModelRejection | None = None
        # Structured failure/stopped RESPONSE (IN_REPLY_TO = original TASK_ID)
        # produced by the LAST terminal failure of the current task.  The UI
        # hands it to the clipboard so side A never waits on a silent stop.
        self.failure_response: str | None = None

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #

    def _failure_body(self, state: str, error: str | None) -> str:
        if state == "STOPPED_BY_USER":
            return (
                "任务已由用户手动停止，未产生最终结果。OpenChamber 会话已保留，"
                f"未删除任何工作成果。停止原因：{error or '用户停止等待'}"
            )
        return (
            "任务执行失败，未产生最终结果。"
            f"错误：{error or '未知错误'}。请检查错误信息后重新发送任务。"
        )

    def _finalize_failure(
        self,
        message: RelayMessage,
        state: str,
        error: str,
        executor: str = "",
        session_id: str | None = None,
        directory: str | None = None,
        force: bool = False,
    ) -> None:
        """Move a task to a terminal state and still answer side A.

        Invariant: a task can never be left in RECEIVED/PROCESSING without
        a reason.  When it truly fails (or is stopped by the user) side A
        gets a structured RESPONSE bound to the ORIGINAL TASK_ID and the
        registry record carries both the explicit error and the saved
        failure response.  ``force=True`` is the explicit operator re-entry
        form (e.g. stopping an already FAILED interrupted task): it writes
        the terminal state even though the task is already terminal;
        ordinary late workers always use the non-forced form, which is a
        no-op once any terminal state has been reached.
        """
        response = wrap_response(
            self._failure_body(state, error),
            message.message_id,
            message.protocol_format,
            message.round_number,
            message.max_rounds,
        )
        extra: dict[str, str] = {}
        if executor:
            extra["executor"] = executor
        if session_id:
            extra["session_id"] = session_id
        if directory:
            extra["directory"] = directory
        try:
            extra["reply_file"] = str(self.save_reply(message.message_id, response))
        except Exception as exc:
            _LOG.warning(
                "failure response save failed task=%s error=%s",
                message.message_id, exc,
            )
        # A late worker must never clobber a round the operator already
        # resolved: only apply this terminal write if the task is not yet
        # terminal (see TaskRegistry.mark_if_active).  Explicit operator
        # re-entries (force=True) use mark_reentry instead.
        if force:
            self.registry.mark_reentry(message.message_id, state, error, **extra)
        else:
            self.registry.mark_if_active(message.message_id, state, error, **extra)
        self.failure_response = response
        _LOG.info(
            "task terminal %s task=%s session=%s directory=%s error=%s",
            state, message.message_id, session_id or "-", directory or "-", error,
        )

    # ------------------------------------------------------------------ #
    # startup recovery (stale non-terminal records from a previous run)
    # ------------------------------------------------------------------ #

    def reconcile_stale_task(
        self,
        task_id: str,
        record: Mapping[str, str],
        status_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """One-shot, bounded reconciliation of a stale RECEIVED/PROCESSING
        record left behind when the previous run died mid-task.

        NEVER re-sends anything and NEVER writes to the registry: it only
        inspects the recorded OpenChamber session and reports the most
        defensible terminal state so the UI (main thread) can persist it:

        * COMPLETED  -- the round's final assistant message is verifiably
          completed with ``finish=stop`` and a non-empty final text; the
          text is returned so the UI can wrap it under the ORIGINAL task id;
        * FAILED     -- a clear, recorded failure (message error, or a reply
          truncated by the model output limit);
        * RECOVERY_REQUIRED -- everything else (no session record, service
          unreachable, round not uniquely locatable, still busy, pending
          user action, unknown finish): the result is genuinely unknown and
          must be inspected by a human.
        """
        update = status_callback or (lambda _status: None)
        executor = str(record.get("executor") or "").strip()
        session_id = str(record.get("session_id") or "").strip()
        directory = str(record.get("directory") or "").strip()

        def _result(state: str, error: str | None, **extra) -> dict:
            out = {
                "task_id": task_id,
                "state": state,
                "error": error or "",
                "final_text": None,
                "reply_message_ids": (),
                "session_id": session_id,
            }
            out.update(extra)
            return out

        if executor != TARGET_OPENCHAMBER or not session_id or not directory:
            # REASONIX tasks (and tasks that never reached PROCESSING) have
            # no after-the-fact source of truth.
            return _result(
                "RECOVERY_REQUIRED",
                "无法事后核实该任务的执行结果（无 OpenChamber 会话记录），需人工检查",
            )
        client = OpenChamberClient(self.settings.openchamber_url)
        try:
            update(f"正在核对遗留任务 {task_id[:12]}…")
            client.verify()
            messages = client.messages(session_id, directory)
        except Exception as exc:
            return _result(
                "RECOVERY_REQUIRED",
                f"无法查询 OpenChamber 会话核实任务结果：{exc}",
            )
        finally:
            client.close()

        marker = f"[AI_RELAY_TASK_ID: {task_id}]"
        hits = [
            index
            for index, message in enumerate(messages)
            if _role(message) == "user" and marker in _message_text(message)
        ]
        if not hits:
            return _result(
                "RECOVERY_REQUIRED",
                "OpenChamber 会话中未找到该任务的发送记录，无法核实结果",
            )
        if len(hits) > 1:
            return _result(
                "RECOVERY_REQUIRED",
                "OpenChamber 会话中存在多条该任务的发送记录（歧义），不做猜测",
            )
        user_index = hits[0]
        anchor_id = _message_id(messages[user_index])
        try:
            round_messages = _round_assistant_messages(
                messages, user_index, anchor_id
            )
        except OpenChamberError as exc:
            return _result("RECOVERY_REQUIRED", f"该任务轮次归属无法唯一确认：{exc}")
        if not round_messages:
            return _result(
                "RECOVERY_REQUIRED", "OpenChamber 会话中未找到该任务的回复，无法核实结果"
            )
        if has_pending_user_action(round_messages):
            return _result(
                "RECOVERY_REQUIRED",
                "该任务仍在等待人工确认（问题/权限），无法自动核实结果",
            )

        last = round_messages[-1]
        info = last.get("info") if isinstance(last.get("info"), Mapping) else {}
        error = info.get("error") if isinstance(info, Mapping) else None
        completed_ts = (
            (info.get("time") or {}).get("completed")
            if isinstance(info, Mapping)
            else None
        )
        completed = (
            isinstance(completed_ts, int)
            and not isinstance(completed_ts, bool)
            and completed_ts > 0
        )
        reply_message_ids = tuple(
            mid for mid in (_message_id(m) for m in round_messages) if mid
        )
        if error:
            return _result(
                "FAILED",
                f"任务在 OpenChamber 中已记录失败：{error_detail(error)}",
                reply_message_ids=reply_message_ids,
            )
        if not completed:
            return _result(
                "RECOVERY_REQUIRED",
                "无法确认该任务已在 OpenChamber 中完成（可能仍在执行），需人工检查",
            )
        finish = info.get("finish") if isinstance(info, Mapping) else None
        if finish == "length":
            return _result(
                "FAILED",
                "任务回复被模型输出上限截断，不是成功结果，需人工检查",
                reply_message_ids=reply_message_ids,
            )
        if finish != "stop":
            return _result(
                "RECOVERY_REQUIRED",
                f"该任务轮次结束状态无法确认（finish={finish!r}），需人工检查",
            )
        final_text = extract_final_text(round_messages)
        if not final_text:
            return _result(
                "RECOVERY_REQUIRED",
                "该任务轮次已完成但没有可用的最终文本，需人工检查",
            )
        return _result(
            "COMPLETED",
            None,
            final_text=final_text,
            reply_message_ids=reply_message_ids,
        )

    def process(
        self,
        text: str,
        status_callback: Callable[[str], None] | None = None,
        session_callback: Callable[[TaskOutcome], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        update = status_callback or (lambda _status: None)
        # A new task invalidates the previous task's current session so the
        # UI's open-session button never points at an older task while this
        # one is being routed.
        self.current_session = None
        # A new task also resets the one-shot auto-recovery slot and any
        # earlier "继续当前任务" offer: only the newly started round may be
        # recovered once, and only from the state that task left behind.
        self.recovery_attempted = False
        self.pending_continue = None
        self.pending_rejection = None
        self.failure_response = None
        try:
            message = parse_message(text)

            if message.source != "CHATGPT":
                self._finalize_failure(
                    message,
                    "FAILED",
                    f"unsupported task source: {message.source}",
                    executor="",
                )
                raise RelayWorkflowError(f"unsupported task source: {message.source}")
            if message.target not in KNOWN_TARGETS:
                self._finalize_failure(
                    message,
                    "FAILED",
                    f"task target is not executable: {message.target}",
                    executor="",
                )
                raise RelayWorkflowError(
                    f"task target is not executable: {message.target}"
                )
            if message.message_type is not MessageType.TASK:
                self._finalize_failure(
                    message,
                    "FAILED",
                    f"clipboard message is not a TASK: {message.message_type.value}",
                    executor=resolve_executor_kind(message.target, self.settings),
                )
                raise RelayWorkflowError(
                    f"clipboard message is not a TASK: {message.message_type.value}"
                )
            existing = self.registry.record(message.message_id)
            if existing is not None and existing.get("state") != "RECEIVED":
                raise RelayWorkflowError(
                    f"task was already processed: {message.message_id}"
                )

            executor_kind = resolve_executor_kind(message.target, self.settings)
            if executor_kind == TARGET_OPENCHAMBER:
                return self._run_openchamber(
                    message, update, session_callback, cancel_event
                )
            return self._run_reasonix(message, update)
        except RelayProtocolError as exc:
            raise RelayWorkflowError(f"invalid clipboard task: {exc}") from exc

    # ------------------------------------------------------------------ #
    # executors
    # ------------------------------------------------------------------ #

    def _run_reasonix(self, message: RelayMessage, update) -> str:
        self.registry.mark_if_active(
            message.message_id,
            "PROCESSING",
            executor=TARGET_REASONIX,
        )
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
            self.registry.mark_if_active(
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
            self._finalize_failure(
                message, "FAILED", error, executor=TARGET_REASONIX
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
        cancel_event: threading.Event | None = None,
    ) -> str:
        directory = (message.workdir or self.settings.openchamber_directory).strip()
        if not directory:
            self._finalize_failure(
                message,
                "FAILED",
                "openchamber_config:missing directory",
                executor=TARGET_OPENCHAMBER,
            )
            raise RelayWorkflowError("OpenChamber 项目目录未配置，请在设置中填写")
        if self.openchamber is None and not Path(directory).is_dir():
            self._finalize_failure(
                message,
                "FAILED",
                "openchamber_config:directory does not exist",
                executor=TARGET_OPENCHAMBER,
                directory=directory,
            )
            raise RelayWorkflowError(f"OpenChamber 项目目录不存在：{directory}")

        client = self._openchamber_client()
        key = directory_key(directory)
        session_id = self.settings.openchamber_sessions.get(key, "").strip()
        default_directory = self.settings.openchamber_directory.strip()
        if not session_id and default_directory and directory_key(default_directory) == key:
            session_id = self.settings.openchamber_session_id.strip()
        try:
            update("正在验证 OpenChamber 连接")
            client.verify()

            if session_id:
                update("正在确认 OpenChamber 会话")
                if not client.session_exists(session_id, directory):
                    session_id = ""
            if not session_id:
                update("正在为项目创建 OpenChamber 会话")
                title = f"AI Relay - {Path(directory).name or '项目'}"
                session_id = client.create_session(title, directory)
                self.settings.openchamber_sessions[key] = session_id
                if default_directory and directory_key(default_directory) == key:
                    self.settings.openchamber_session_id = session_id
                if self.settings._path is not None:
                    self.settings.save()
            self.registry.mark_if_active(
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
            agent = self.settings.openchamber_agent.strip() or None
            tagged_prompt = (
                f"{message.body.rstrip()}\n\n"
                f"[AI_RELAY_TASK_ID: {message.message_id}]"
            )
            try:
                dispatch = client.send(
                    session_id,
                    tagged_prompt,
                    directory,
                    agent=agent,
                    model=model,
                )
            except OpenChamberModelRequestRejectedError as exc:
                # A non-retryable model rejection is NEVER retried in the
                # same session and is NOT treated as a plain interruption:
                # the round it produced would just be rejected again.  The
                # task info is kept so the operator can retry in a fresh
                # session via "新会话重试".  Structured upstream fields
                # (status_code / is_retryable / request_url) are preserved.
                self._note_model_rejection(
                    message, session_id, directory, agent, model
                )
                raise OpenChamberModelRequestRejectedError(
                    f"{MODEL_REJECTION_PROMPT} {exc}",
                    status_code=exc.status_code,
                    is_retryable=exc.is_retryable,
                    request_url=exc.request_url,
                ) from exc
            except OpenChamberBadRequestError as exc:
                # An ordinary HTTP 400 (no upstream isRetryable=false
                # marker) is also never auto-recovered in the same session,
                # but it offers no fresh-session retry either.
                raise OpenChamberBadRequestError(
                    f"OpenChamber 返回 HTTP 400，已跳过原会话自动恢复：{exc}"
                ) from exc
            # Persist requested/resolved model NOW: it must survive any
            # later state update (a mismatch warning must not vanish).
            self.registry.mark_if_active(
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
            try:
                result = wait_for_completion(
                    client,
                    dispatch,
                    self.settings.completion_timeout,
                    self.settings.poll_interval,
                    grace_seconds=COMPLETION_GRACE_SECONDS,
                    status_callback=update,
                    cancel_event=cancel_event,
                )
            except OpenChamberModelRequestRejectedError as exc:
                self._note_model_rejection(
                    message, session_id, directory, agent, model
                )
                raise OpenChamberModelRequestRejectedError(
                    f"{MODEL_REJECTION_PROMPT} {exc}",
                    status_code=exc.status_code,
                    is_retryable=exc.is_retryable,
                    request_url=exc.request_url,
                ) from exc
            except OpenChamberInterruptedError as exc:
                try:
                    recovered = self._auto_recover(
                        message, update, directory, session_id, client, agent,
                        model, cancel_event, dispatch,
                    )
                except OpenChamberModelRequestRejectedError as rejected:
                    self._note_model_rejection(
                        message, session_id, directory, agent, model
                    )
                    raise OpenChamberModelRequestRejectedError(
                        f"{MODEL_REJECTION_PROMPT} {rejected}",
                        status_code=rejected.status_code,
                        is_retryable=rejected.is_retryable,
                        request_url=rejected.request_url,
                    ) from rejected
                if recovered is None:
                    raise OpenChamberInterruptedError(
                        f"{AR_RECOVERY_FAILED}：{exc}"
                    ) from exc
                result = recovered
            return self._complete_openchamber_result(
                result, message, session_id, directory, update
            )
        except OpenChamberCancelledError as exc:
            error = f"openchamber_cancel:{type(exc).__name__}: {exc}"
            self._finalize_failure(
                message,
                "STOPPED_BY_USER",
                error,
                executor=TARGET_OPENCHAMBER,
                session_id=session_id or None,
                directory=directory,
            )
            self.pending_continue = None
            raise OpenChamberCancelledError(f"{CANCELLED_MARKER}: {exc}") from exc
        except Exception as exc:
            error = f"openchamber_execute:{type(exc).__name__}: {exc}"
            if isinstance(exc, OpenChamberModelRequestRejectedError):
                # A fresh "new-session retry" is still possible, so the task
                # is NOT terminal yet: record FAILED but do not answer side A
                # with a failure response; the later retry (or stop) will.
                self.registry.mark_if_active(
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
            else:
                self._finalize_failure(
                    message,
                    "FAILED",
                    error,
                    executor=TARGET_OPENCHAMBER,
                    session_id=session_id or None,
                    directory=directory,
                )
            raise

    # ------------------------------------------------------------------ #
    # abnormal interruption recovery (auto once, then manual continue)
    # ------------------------------------------------------------------ #

    def _auto_recover(
        self,
        message: RelayMessage,
        update,
        directory: str,
        session_id: str,
        client: OpenChamberClient,
        agent: str | None,
        model: ModelRef | None,
        cancel_event: threading.Event | None,
        original_dispatch,
    ) -> CompletionResult | None:
        """Offer ONE automatic recovery of an interrupted round.

        First, wait about 10 seconds and re-check whether the session has
        recovered ON ITS OWN (busy/resumed and completed): if so, return that
        result WITHOUT sending a continuation and WITHOUT consuming the
        one-shot automatic recovery slot.  Only if the round is still idle
        and still interrupted after the re-check does the relay send ONE
        automatic continuation in the ORIGINAL session (same TASK_ID, Agent
        and Model).  Returns ``None`` when recovery fails, keeping
        ``pending_continue`` so the operator can click "继续当前任务".
        """
        if self.recovery_attempted:
            update(AR_RECOVERY_FAILED)
            _LOG.warning(
                "auto_recover: already attempted, not retrying task=%s "
                "session=%s recovery_attempted=true",
                message.message_id, session_id,
            )
            return None
        self.pending_continue = OpenChamberContinue(
            message=message,
            session_id=session_id,
            directory=directory,
            agent=agent,
            model=model,
        )
        update("检测到 OpenCode 工具续接中断，10 秒后复查……")
        _LOG.info(
            "auto_recover: start re-check task=%s session=%s "
            "recovery_attempted=false",
            message.message_id, session_id,
        )
        deadline = time.monotonic() + RECOVERY_DELAY_SECONDS
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise OpenChamberCancelledError(
                    "OpenChamber wait cancelled during auto-recovery re-check"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                status = client.session_status(session_id, directory)
            except OpenChamberError:
                status = "unknown"
            if status in ("busy", "retry"):
                # The session resumed working on its own: await its final
                # reply in the ORIGINAL round without sending a continuation,
                # and do not consume the automatic recovery slot.
                update("会话自行恢复，不发送继续……")
                _LOG.info(
                    "auto_recover: session self-recovered, skip continue "
                    "task=%s session=%s",
                    message.message_id, session_id,
                )
                try:
                    return wait_for_completion(
                        client,
                        original_dispatch,
                        self.settings.completion_timeout,
                        self.settings.poll_interval,
                        grace_seconds=COMPLETION_GRACE_SECONDS,
                        status_callback=update,
                        cancel_event=cancel_event,
                    )
                except (OpenChamberInterruptedError, OpenChamberBadRequestError):
                    # Not actually recovered; fall back to continued polling.
                    pass
            _sleep_interruptible(min(0.5, remaining), cancel_event)
        # 10 秒复查后仍为中断状态：检查会话是否已被 Monitor 续接。
        # 如果 Monitor 已经续接了会话（检测到新用户消息），则跳过自己的
        # 续接，直接用原始 dispatch 继续等待（_follow_continuation_chain
        # 会跟踪 Monitor 的续接链）。只有会话真正空闲（无新消息）时才
        # 发送自己的续接。
        try:
            status = client.session_status(session_id, directory)
        except OpenChamberError:
            status = "unknown"
        # A session that vanished from the status map is server-side idle:
        # it still counts as "idle" here (a fake client may report the
        # detailed status verbatim).
        if status not in ("idle", MISSING_FROM_STATUS_MAP):
            update(AR_RECOVERY_FAILED)
            _LOG.warning(
                "auto_recover: session not idle after re-check task=%s session=%s",
                message.message_id, session_id,
            )
            return None
        # Session is idle: check if the monitor already continued it.
        try:
            messages = client.messages(session_id, directory)
        except OpenChamberError:
            messages = []
        if messages and original_dispatch.pre_send_snapshot_ok:
            # Count user messages NOT in the original dispatch snapshot.
            new_user_msgs = [
                m for m in messages
                if _role(m) == "user"
                and _message_id(m) not in original_dispatch.pre_send_message_ids
            ]
            if new_user_msgs:
                # Monitor already continued: skip own continuation, wait for
                # the chain to resolve via the original dispatch.
                _LOG.info(
                    "auto_recover: monitor already continued session "
                    "task=%s session=%s new_user_msgs=%d, skip own continue",
                    message.message_id, session_id, len(new_user_msgs),
                )
                update("Monitor 已续接会话，等待结果……")
                try:
                    return wait_for_completion(
                        client,
                        original_dispatch,
                        self.settings.completion_timeout,
                        self.settings.poll_interval,
                        grace_seconds=COMPLETION_GRACE_SECONDS,
                        status_callback=update,
                        cancel_event=cancel_event,
                    )
                except (OpenChamberInterruptedError, OpenChamberBadRequestError):
                    # Not actually recovered; fall back.
                    pass
        self.recovery_attempted = True
        update("正在原会话自动续接（1/1）……")
        _LOG.info(
            "auto_recover: auto continue once task=%s session=%s "
            "recovery_attempted=true",
            message.message_id, session_id,
        )
        try:
            dispatch = client.send(
                session_id,
                OPENCHAMBER_CONTINUE_PROMPT,
                directory,
                agent=agent,
                model=model,
            )
            return wait_for_completion(
                client,
                dispatch,
                self.settings.completion_timeout,
                self.settings.poll_interval,
                grace_seconds=COMPLETION_GRACE_SECONDS,
                status_callback=update,
                cancel_event=cancel_event,
            )
        except (OpenChamberInterruptedError, OpenChamberBadRequestError):
            update(AR_RECOVERY_FAILED)
            _LOG.warning(
                "auto_recover: continue failed again task=%s session=%s "
                "pending_continue_kept=true",
                message.message_id, session_id,
            )
            return None

    def _note_model_rejection(
        self,
        message: RelayMessage,
        session_id: str,
        directory: str,
        agent: str | None,
        model: ModelRef | None,
    ) -> None:
        """Record everything needed for a fresh-session retry.

        Filled when the upstream model service rejects the request with a
        non-retryable 400, so the operator can click "新会话重试".  The
        rejected session id is kept for the record but never reused by the
        retry (a fresh session is created instead).
        """
        self.pending_rejection = OpenChamberModelRejection(
            message=message,
            session_id=session_id,
            directory=directory,
            agent=agent,
            model=model,
        )

    def _complete_openchamber_result(
        self,
        result: CompletionResult,
        message: RelayMessage,
        session_id: str,
        directory: str,
        update,
    ) -> str:
        """Wrap a verified OpenChamber result into the task's response.

        Shared by the normal path, the automatic recovery and the manual
        continue path, so every success gets identical packaging,
        persistence, registry updates and wrapped-message bookkeeping.  The
        TASK_ID is always the original one.
        """
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
        self.registry.mark_if_active(
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
        # The reply has now been packaged.  Record every assistant message of
        # this verified round as wrapped so the manual OpenChamber monitor
        # never re-wraps the same reply.
        for msg_id in result.round_message_ids:
            mark_message_wrapped(msg_id, session_id)
        self.pending_continue = None
        update("恢复成功，回复已包装并复制")
        return response

    def continue_openchamber_task(
        self,
        status_callback: Callable[[str], None] | None = None,
        session_callback: Callable[[TaskOutcome], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Manually continue the interrupted task once, in its own session.

        Called by the "继续当前任务" button.  Each click performs exactly ONE
        continuation; a failed continuation keeps ``pending_continue`` so the
        operator can click again.  Success wraps under the ORIGINAL TASK_ID
        and clears the pending offer.
        """
        update = status_callback or (lambda _status: None)
        pending = self.pending_continue
        if pending is None:
            raise RelayWorkflowError("没有可继续的 OpenChamber 任务")
        session_id = pending.session_id
        directory = pending.directory
        client = self._openchamber_client()
        # Explicit active re-entry: the task may currently sit in a terminal
        # state (FAILED after the automatic recovery was used up), so the
        # move back to PROCESSING is an operator-driven transition, not an
        # ordinary late-worker write.
        self.registry.mark_reentry(
            pending.message.message_id,
            "PROCESSING",
            executor=TARGET_OPENCHAMBER,
            session_id=session_id,
            directory=directory,
        )
        update("正在人工继续当前任务")
        if not client.session_exists(session_id, directory):
            raise RelayWorkflowError(
                "OpenChamber 会话已不存在，无法继续；请检查会话或重新发送任务"
            )
        update("正在请求 OpenChamber 打开会话")
        client.open_session(session_id)
        try:
            update(f"正在向会话 {session_id} 发送继续指令")
            dispatch = client.send(
                session_id,
                OPENCHAMBER_CONTINUE_PROMPT,
                directory,
                agent=pending.agent,
                model=pending.model,
            )
            update(f"正在等待 OpenChamber 完成（会话 {session_id}）")
            result = wait_for_completion(
                client,
                dispatch,
                self.settings.completion_timeout,
                self.settings.poll_interval,
                grace_seconds=COMPLETION_GRACE_SECONDS,
                status_callback=update,
                cancel_event=cancel_event,
            )
            return self._complete_openchamber_result(
                result, pending.message, session_id, directory, update
            )
        except OpenChamberCancelledError as exc:
            self.pending_continue = None
            self._finalize_failure(
                pending.message,
                "STOPPED_BY_USER",
                f"openchamber_cancel:OpenChamberCancelledError: {exc}",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
            )
            raise OpenChamberCancelledError(f"{CANCELLED_MARKER}: {exc}") from exc
        # Any other failure intentionally KEEPS ``pending_continue`` so the
        # operator can retry the continuation by clicking the button again.

    # ------------------------------------------------------------------ #
    # model request rejection -> fresh-session retry (once per task)
    # ------------------------------------------------------------------ #

    def retry_model_rejected_task(
        self,
        status_callback: Callable[[str], None] | None = None,
        session_callback: Callable[[TaskOutcome], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Retry a model-rejected task in a brand-new OpenChamber session.

        Called by the "新会话重试" button.  Creates a FRESH session in the
        same project directory (never forks the rejected session, so the
        problematic context is not copied), sends the original task body with
        the same Agent and Model, waits for the real completion and wraps it
        under the ORIGINAL TASK_ID with a fresh RESPONSE_ID.  The new session
        id is persisted as the project's preference only on success, and the
        retry is one-shot per task.
        """
        update = status_callback or (lambda _status: None)
        pending = self.pending_rejection
        if pending is None:
            raise RelayWorkflowError("没有可重试的模型拒绝任务")
        client = self._openchamber_client()
        directory = pending.directory
        key = directory_key(directory)
        # One-shot per task: once a retry starts there is no further offer,
        # and a rejection on the NEW session never creates a third one.
        self.pending_rejection = None
        update("正在创建全新会话重试原任务")
        try:
            session_id = client.create_session(
                f"AI Relay 重试 - {Path(directory).name or '项目'}", directory
            )
            # Explicit active re-entry: the task is FAILED (the rejection is
            # not terminal until the retry is attempted), so moving it back
            # to PROCESSING is an operator-driven transition.
            self.registry.mark_reentry(
                pending.message.message_id,
                "PROCESSING",
                executor=TARGET_OPENCHAMBER,
                session_id=session_id,
                directory=directory,
            )
            update(f"正在向新会话 {session_id} 发送原始任务")
            prompt = (
                f"{MODEL_REJECTION_RETRY_PREFIX}\n"
                f"{pending.message.body.rstrip()}\n\n"
                f"[AI_RELAY_TASK_ID: {pending.message.message_id}]"
            )
            dispatch = client.send(
                session_id,
                prompt,
                directory,
                agent=pending.agent,
                model=pending.model,
            )
            update(f"正在等待 OpenChamber 完成（会话 {session_id}）")
            result = wait_for_completion(
                client,
                dispatch,
                self.settings.completion_timeout,
                self.settings.poll_interval,
                grace_seconds=COMPLETION_GRACE_SECONDS,
                status_callback=update,
                cancel_event=cancel_event,
            )
            # Persist the fresh session ONLY on success: it becomes the new
            # preference for the project, the fixed session and the next task.
            self.settings.openchamber_sessions[key] = session_id
            if self.settings.openchamber_directory.strip():
                default_key = directory_key(
                    self.settings.openchamber_directory.strip()
                )
                if default_key == key:
                    self.settings.openchamber_session_id = session_id
            if self.settings._path is not None:
                self.settings.save()
            if session_callback is not None:
                session_callback(
                    TaskOutcome(
                        task_id=pending.message.message_id,
                        executor=TARGET_OPENCHAMBER,
                        session_id=session_id,
                        directory=directory,
                    )
                )
            return self._complete_openchamber_result(
                result, pending.message, session_id, directory, update
            )
        except OpenChamberModelRequestRejectedError as exc:
            error = f"openchamber_execute:{type(exc).__name__}: {exc}"
            self._finalize_failure(
                pending.message,
                "FAILED",
                error,
                executor=TARGET_OPENCHAMBER,
                session_id=pending.session_id,
                directory=directory,
            )
            raise OpenChamberModelRequestRejectedError(
                f"{MODEL_REJECTION_AGAIN} {exc}",
                status_code=exc.status_code,
                is_retryable=exc.is_retryable,
                request_url=exc.request_url,
            ) from exc
        except OpenChamberCancelledError as exc:
            self._finalize_failure(
                pending.message,
                "STOPPED_BY_USER",
                f"openchamber_cancel:OpenChamberCancelledError: {exc}",
                executor=TARGET_OPENCHAMBER,
                session_id=pending.session_id,
                directory=directory,
            )
            raise OpenChamberCancelledError(f"{CANCELLED_MARKER}: {exc}") from exc
        except Exception as exc:
            error = f"openchamber_execute:{type(exc).__name__}: {exc}"
            self._finalize_failure(
                pending.message,
                "FAILED",
                error,
                executor=TARGET_OPENCHAMBER,
                session_id=pending.session_id,
                directory=directory,
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
