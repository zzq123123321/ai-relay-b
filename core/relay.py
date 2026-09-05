"""Relay workflow orchestration independent from the user interface."""

from __future__ import annotations

from collections.abc import Callable

from core.protocol import MessageType, RelayProtocolError, parse_message, wrap_response
from core.reasonix_uia import ReasonixAutomation
from core.task_registry import TaskRegistry


class RelayWorkflowError(RuntimeError):
    """Raised when an incoming task cannot be relayed safely."""


class RelayWorkflow:
    def __init__(
        self,
        reasonix: ReasonixAutomation | None = None,
        registry: TaskRegistry | None = None,
    ):
        self.reasonix = reasonix or ReasonixAutomation()
        self.registry = registry or TaskRegistry()

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
            if message.target not in frozenset({"REASONIX", "EXECUTOR"}):
                raise RelayWorkflowError(f"task target is not EXECUTOR: {message.target}")
            if message.message_type is not MessageType.TASK:
                raise RelayWorkflowError(
                    f"clipboard message is not a TASK: {message.message_type.value}"
                )
            if self.registry.contains(message.message_id):
                raise RelayWorkflowError(f"task was already processed: {message.message_id}")

            self.registry.mark(message.message_id, "PROCESSING")
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
                self.registry.mark(message.message_id, "COMPLETED")
                return response
            except Exception as exc:
                error = f"reasonix_execute:{type(exc).__name__}: {exc}"
                self.registry.mark(message.message_id, "FAILED", error)
                raise
        except RelayProtocolError as exc:
            raise RelayWorkflowError(f"invalid clipboard task: {exc}") from exc
