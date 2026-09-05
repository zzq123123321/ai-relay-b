"""Parameterized state machine for the AI Relay workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum


class RelayState(Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    SENDING = "sending"
    WAITING_RESPONSE = "waiting_response"
    COPYING_RESPONSE = "copying_response"
    RETURNING = "returning"
    ERROR = "error"


class InvalidStateError(ValueError):
    """Raised when a state is not part of the configured state machine."""


class InvalidTransitionError(RuntimeError):
    """Raised when the requested transition is not allowed."""


DEFAULT_TRANSITIONS: Mapping[RelayState, frozenset[RelayState]] = {
    RelayState.IDLE: frozenset({RelayState.CAPTURING}),
    RelayState.CAPTURING: frozenset({RelayState.SENDING, RelayState.ERROR}),
    RelayState.SENDING: frozenset({RelayState.WAITING_RESPONSE, RelayState.ERROR}),
    RelayState.WAITING_RESPONSE: frozenset({RelayState.COPYING_RESPONSE, RelayState.ERROR}),
    RelayState.COPYING_RESPONSE: frozenset({RelayState.RETURNING, RelayState.ERROR}),
    RelayState.RETURNING: frozenset({RelayState.IDLE, RelayState.ERROR}),
    RelayState.ERROR: frozenset({RelayState.IDLE}),
}


class StateMachine:
    """Small state machine whose states and transitions are configurable."""

    def __init__(
        self,
        initial_state: RelayState,
        transitions: Mapping[RelayState, Iterable[RelayState]],
    ) -> None:
        normalized = {state: frozenset(targets) for state, targets in transitions.items()}
        known_states = set(normalized)
        for targets in normalized.values():
            known_states.update(targets)
        if initial_state not in known_states:
            raise InvalidStateError(f"initial state is not configured: {initial_state.value}")
        self._state = initial_state
        self._transitions = normalized

    @property
    def state(self) -> RelayState:
        return self._state

    def can_transition_to(self, target: RelayState) -> bool:
        return target in self._transitions.get(self._state, frozenset())

    def transition_to(self, target: RelayState) -> RelayState:
        if target not in self._transitions and not all(
            target not in targets for targets in self._transitions.values()
        ):
            raise InvalidStateError(f"state is not configured: {target.value}")
        if not self.can_transition_to(target):
            raise InvalidTransitionError(
                f"transition is not allowed: {self._state.value} -> {target.value}"
            )
        self._state = target
        return self._state

    def reset(self):
        """Recover the workflow to IDLE through an allowed transition."""
        if self._state is RelayState.IDLE:
            return self._state
        return self.transition_to(RelayState.IDLE)


def create_default_state_machine() -> StateMachine:
    return StateMachine(RelayState.IDLE, DEFAULT_TRANSITIONS)
