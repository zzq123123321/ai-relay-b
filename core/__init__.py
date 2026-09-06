"""AI Relay core domain models."""

from core.agent import Agent, AgentConfigurationError
from core.openchamber import (
    CompletionResult,
    ModelRef,
    OpenChamberAuthError,
    OpenChamberClient,
    OpenChamberDispatch,
    OpenChamberError,
    OpenChamberSessionError,
    OpenChamberTimeoutError,
    OpenChamberUnavailableError,
    OpenChamberUserActionRequired,
)
from core.protocol import (
    MessageType,
    ProtocolFormat,
    RelayMessage,
    RelayProtocolError,
    parse_message,
    wrap_response,
)
from core.relay import RelayWorkflow, TaskOutcome, resolve_executor_kind
from core.relay_settings import RelaySettings
from core.state_machine import (
    DEFAULT_TRANSITIONS,
    InvalidStateError,
    InvalidTransitionError,
    RelayState,
    StateMachine,
)

__all__ = [
    "Agent",
    "AgentConfigurationError",
    "CompletionResult",
    "DEFAULT_TRANSITIONS",
    "InvalidStateError",
    "InvalidTransitionError",
    "ModelRef",
    "MessageType",
    "OpenChamberAuthError",
    "OpenChamberClient",
    "OpenChamberDispatch",
    "OpenChamberError",
    "OpenChamberSessionError",
    "OpenChamberTimeoutError",
    "OpenChamberUnavailableError",
    "OpenChamberUserActionRequired",
    "ProtocolFormat",
    "RelayMessage",
    "RelayProtocolError",
    "RelaySettings",
    "RelayState",
    "RelayWorkflow",
    "StateMachine",
    "TaskOutcome",
    "parse_message",
    "resolve_executor_kind",
    "wrap_response",
]
