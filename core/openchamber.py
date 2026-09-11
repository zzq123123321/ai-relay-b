"""OpenChamber executor client.

All traffic is sent to the configured OpenChamber address (default
``http://127.0.0.1:57123``).  The ``/api`` prefix is proxied by OpenChamber
to its managed OpenCode instance; the proxy injects OpenCode's own
credentials, so this client never connects to the dynamic OpenCode port and
never handles OpenCode credentials itself.

Verified against the local OpenChamber 1.22.2 build with its managed
OpenCode 1.18.29 (live probes, 2026-09-06):

* ``POST /api/openchamber/sessions`` -> ``{"sessionId": ...}`` (created
  without any prompt).
* ``POST /api/openchamber/sessions/:id/send`` ->
  ``{"model", "agent", "promptDispatched", "dispatchedAsCommand"}``.
  The response carries NO baseline message id: OpenChamber computes the
  pre-prompt baseline user message id server-side only to verify that the
  prompt actually landed (``promptDispatched`` is false otherwise).  Round
  association must therefore use the message-id snapshot taken before the
  send (see :class:`OpenChamberDispatch`), never a time window.  If that
  pre-send snapshot CANNOT be read, the task is NOT sent: with no
  trustworthy snapshot the round cannot be attributed, so the prompt is
  aborted instead of being degraded to a guess.
* ``GET /api/session/status?directory=...`` ->
  ``{sessionId: {"type": "busy" | "retry" | ...}}``.  OpenChamber proxies
  this to OpenCode, whose ``SessionStatus.set()`` removes a session from the
  map the moment its status becomes ``idle`` (and ``get()`` falls back to
  ``{type: "idle"}`` for unknown ids).  A session id MISSING from the map
  therefore means idle; the map only ever contains busy/retry entries.  A
  session id PRESENT with a null value is malformed data -> ``unknown``,
  never idle.
* ``GET /api/session`` -> flat newest-first list of existing sessions (the
  server caps the unfiltered list at the 100 most recent); each item carries
  ``id``, ``title``, ``directory``/``path`` (the project directory), etc.
  ``GET /api/session?directory=...`` filters server-side to ONE project
  directory (both the backslash ``D:\\...`` and forward-slash ``D:/...``
  forms are accepted and match).  ``GET /api/session/:id/message?directory=...``
  -> message array; a MISSING session returns HTTP 404
  ``{"name": "NotFoundError", ...}``.
* ``GET /api/session/:id/message?directory=...`` (existing session) ->
  message array; assistant ``info`` carries ``parentID`` (the user message
  that triggered the turn), ``finish`` / ``error`` / ``time.completed`` /
  ``time.created``.  Round membership is verified by walking each assistant
  message's ``parentID`` chain up to the round's user message; the round is
  ordered by the strictly-increasing ``time.created`` timestamps, never by
  array position.  Text parts flagged ``synthetic: true`` (or whose state
  is ``ignored``) are real-stack markers for injected/internal text and are
  not answers.
* A model question is a tool part ``{"type": "tool", "tool": "question",
  "state": {"status": "pending"}}``; it completes (status ``completed``)
  only after the user answers in the OpenChamber UI.  OTHER pending
  tool/permission parts are treated as the same CANDIDATE waiting signal —
  the relay keeps waiting and prompts the operator instead of failing.  The
  real permission-popup flow is NOT verified on this stack, so no claim is
  made that the permission path is validated.  This pending detection runs
  both while the session reports ``busy``/``retry`` and while it is
  ``idle``.

The desktop deep link ``openchamber://session/<id>`` only asks the OS to
open the session; it does not prove the window displayed it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import requests


_LOG = logging.getLogger("ai_relay_b")


# Shared registry of OpenChamber messages that have already been wrapped into
# AI_RELAY responses.  Used by both the auto-relay path (via
# :func:`mark_message_wrapped` after :func:`wait_for_completion`) and the
# manual OpenChamber monitor to prevent double-wrapping the same reply.
#
# The consumption key is ``(session_id, message_id)``, NOT the message id
# alone: OpenChamber message ids come from the server's per-session message
# list and are not guaranteed globally unique across sessions.  Keying on the
# message id by itself would let one session's wrapped reply suppress an
# unrelated session's same-named message, so a reply is consumed under the
# session it belongs to.
_wrapped_message_ids: set[tuple[str, str]] = set()


def mark_message_wrapped(msg_id: str, session_id: str = "") -> None:
    """Record that an OpenChamber message has been wrapped into a response.

    ``msg_id`` is consumed under ``session_id`` (the session the message
    belongs to); an empty ``session_id`` keys the mark under no session and
    is only used where no session context exists (never in the auto-relay /
    monitor paths, which always know the session).
    """
    _wrapped_message_ids.add((session_id or "", msg_id))


def is_message_wrapped(msg_id: str, session_id: str = "") -> bool:
    """Check whether an OpenChamber message has already been wrapped."""
    return (session_id or "", msg_id) in _wrapped_message_ids


def reset_wrapped_message_ids() -> None:
    """TEST-ONLY: drop the process-global wrapped-message registry.

    Only tests call this (via the conftest autouse fixture) so that
    ``mark_message_wrapped`` state can never leak from one test into the
    next and test execution order becomes irrelevant.  Production flows must
    NEVER clear the registry mid-run: the marks stay for the process lifetime
    so no reply is ever double-wrapped.
    """
    _wrapped_message_ids.clear()


# Number of consecutive polls that must show the SAME abnormal-idle state
# before an interrupted continuation is confirmed (anti-flap protection for
# brief idle blips during streaming).
REQUIRED_IDLE_CONFIRMATIONS = 3
# Structured status: the status request SUCCEEDED but the session is absent
# from the returned map (the service only reports busy/retry entries).  On
# the server side that means idle, but the client must never conflate it
# with ``unknown`` (request failed / unparseable) nor with an explicit
# ``idle`` entry: only a session that was ACTIVELY observed (busy/retry or
# message growth) may be treated as idle once it disappears from the map.
MISSING_FROM_STATUS_MAP = "missing_from_status_map"
# Limited transport-layer reconnect for transient failures (SSE read timeout
# / connection reset, surfaced as OpenChamberUnavailableError).  After this
# many failures the wait escalates to an interrupted-continuation recovery.
MAX_TRANSPORT_RETRIES = 2
TRANSPORT_RETRY_DELAYS = (3.0, 8.0)

# --- Manual "监听 OpenChamber" monitor ----------------------------------- #
# The manual monitor never auto-resumes: it only DETECTS the abnormal idle
# after ``MONITOR_ONLY_IDLE_CONFIRMATIONS`` identical polls and lets the
# operator click "继续当前任务" once.  Transport failures are probed up to
# ``MONITOR_MAX_TRANSPORT_RETRIES`` times (awaits the same delays as the
# auto-relay path) before the state is reported as unknowable.
MONITOR_ONLY_IDLE_CONFIRMATIONS = 3
MONITOR_MAX_TRANSPORT_RETRIES = 2
MONITOR_TRANSPORT_RETRY_DELAYS = (3.0, 8.0)
# Sent to the ORIGINAL session when the operator continues a manually
# monitored round.  The relay does not know the original task text here, so
# the prompt only asks to resume from the interruption point.
MONITOR_CONTINUE_PROMPT = (
    "上一次执行可能在工具调用后的续接阶段中断。请从最后一个未完成步骤继续，"
    "先检查已有文件和执行结果，不要重复已经完成的修改；"
    "完成剩余工作并返回完整中文最终报告。"
)


class OpenChamberError(RuntimeError):
    """Base error for OpenChamber executor failures."""


class OpenChamberUnavailableError(OpenChamberError):
    """The OpenChamber service cannot be reached."""


class OpenChamberAuthError(OpenChamberError):
    """OpenChamber rejected the request (HTTP 401/403).

    The operator must configure normal OpenChamber UI authentication; this
    client never disables authentication or reads in-memory credentials.
    """


class OpenChamberSessionError(OpenChamberError):
    """A session create/open/send/state operation failed."""


class OpenChamberTimeoutError(OpenChamberError):
    """The task did not finish within the deadline.

    The POST may already have started execution upstream (or may still be
    waiting for the user in the OpenChamber UI); the caller must keep the
    session id and ask the operator to check the session instead of
    resending.
    """


class OpenChamberBadRequestError(OpenChamberSessionError):
    """A send request was rejected with HTTP 400.

    The session may still be idle; the caller can safely continue or retry
    once instead of discarding the task.
    """


class OpenChamberModelRequestRejectedError(OpenChamberSessionError):
    """The upstream model service rejected the request.

    Identified from STRUCTURED upstream fields relayed by OpenChamber (or a
    conservative string fallback): ``statusCode == 400`` combined with
    ``isRetryable == false``.  Retrying the same prompt in the SAME session
    would very likely be rejected again (the request context is the
    problem), so the relay never auto-recontinues these rounds; the operator
    is offered a fresh-session retry instead.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        is_retryable: bool | None = None,
        request_url: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.is_retryable = is_retryable
        self.request_url = request_url


class OpenChamberInterruptedError(OpenChamberSessionError):
    """The round stopped without producing a complete final answer.

    The session is idle but carries no usable completion (no reply, not
    completed, absent finish, or no final text).  The operator may safely
    ask the session to continue from where it stopped.
    """

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        last_message_id: str | None = None,
        last_finish: str | None = None,
        reason: str | None = None,
    ):
        super().__init__(message)
        self.session_id = session_id
        self.last_message_id = last_message_id
        self.last_finish = last_finish
        self.reason = reason


class OpenChamberCancelledError(OpenChamberError):
    """The completion wait was cancelled by the operator.

    The OpenChamber session itself is left untouched; the caller keeps it
    running and switches to manual monitoring instead of deleting it.
    """


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A ``providerID/modelID`` model reference."""

    provider_id: str
    model_id: str

    def as_payload(self) -> dict[str, str]:
        return {"providerID": self.provider_id, "modelID": self.model_id}

    @classmethod
    def parse(cls, value: str | None) -> "ModelRef | None":
        if not value:
            return None
        provider, sep, model = value.strip().partition("/")
        if not sep or not provider.strip() or not model.strip():
            raise OpenChamberSessionError(
                "model must be formatted as providerID/modelID: "
                f"{value!r}"
            )
        return cls(provider_id=provider.strip(), model_id=model.strip())

    def label(self) -> str:
        return f"{self.provider_id}/{self.model_id}"


@dataclass(frozen=True, slots=True)
class OpenChamberDispatch:
    """What the send call confirmed on the server side.

    ``pre_send_message_ids`` is the set of message ids that already existed
    in the session before the send POST (snapshot taken right before it).
    A user message whose id is NOT in that set is new; exactly one new user
    message is this round, more than one is ambiguous (for example the
    operator typed into the same session manually) and is reported as an
    error instead of being guessed.  ``pre_send_snapshot_ok`` is always
    True for a dispatch returned by :meth:`OpenChamberClient.send`: a failed
    snapshot aborts the send, so a False value here marks an untrustworthy
    dispatch that location must treat as ambiguous rather than guess.
    ``user_message_id`` is parsed from the send response when a build
    returns one (the local 1.22.2 build does not).
    """

    session_id: str
    directory: str
    requested_model: ModelRef | None
    resolved_model: ModelRef | None
    agent: str | None
    prompt_dispatched: bool
    dispatched_as_command: bool
    prompt_error: str | None
    user_message_id: str | None = None
    pre_send_message_ids: frozenset[str] = field(default=frozenset())
    pre_send_snapshot_ok: bool = True
    prompt_text: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Verified completion of one dispatched task round."""

    session_id: str
    final_text: str
    finish: str
    actual_model: ModelRef | None
    tool_calls: tuple[str, ...]
    requested_model: ModelRef | None
    resolved_model: ModelRef | None
    model_mismatch: bool
    # The ids of ALL assistant messages in the verified round (including the
    # final one).  Marked wrapped after the reply is packaged so the manual
    # OpenChamber monitor never duplicates a relay-produced reply.
    round_message_ids: tuple[str, ...] = ()


def _model_from_dict(value: Mapping[str, Any] | None) -> ModelRef | None:
    if not isinstance(value, Mapping):
        return None
    provider = value.get("providerID")
    model = value.get("modelID")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return ModelRef(provider_id=provider, model_id=model)
    return None


def _model_from_message_info(info: Mapping[str, Any] | None) -> ModelRef | None:
    if not isinstance(info, Mapping):
        return None
    provider = info.get("providerID")
    model = info.get("modelID")
    if isinstance(provider, str) and provider and isinstance(model, str) and model:
        return ModelRef(provider_id=provider, model_id=model)
    return None


def error_detail(error: Any) -> str:
    """Return the useful message from OpenCode/OpenChamber error payloads."""
    if not isinstance(error, Mapping):
        return str(error)
    direct = error.get("message")
    if direct:
        return str(direct)
    data = error.get("data")
    if isinstance(data, Mapping):
        nested = data.get("message") or data.get("error")
        if nested:
            return str(nested)
    return str(error.get("name") or "unknown error")


def normalize_directory(path: str) -> str:
    """Canonical, comparable form of a project directory path.

    Case-insensitively normalizes separators, resolves ``.``/``..`` and
    strips any trailing separator so ``D:\\proj/``, ``D:/proj`` and
    ``D:\\proj`` all compare equal (Windows paths are compared without
    case).
    """
    if not path or not path.strip():
        return ""
    return os.path.normcase(os.path.abspath(os.path.normpath(path.strip())))


def _extract_message_agent(message: Mapping[str, Any]) -> str | None:
    info = message.get("info")
    if isinstance(info, Mapping):
        agent = info.get("agent")
        if isinstance(agent, str) and agent.strip():
            return agent.strip()
    return None


def extract_agent_model_sets(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    """Collect distinct agent ids and ``providerID/modelID`` model refs from
    session messages (used by the UI's "刷新 Agent/Model").
    """
    agents: set[str] = set()
    models: set[str] = set()
    for message in messages:
        agent = _extract_message_agent(message)
        if agent:
            agents.add(agent)
        info = message.get("info")
        model = (
            _model_from_message_info(info) if isinstance(info, Mapping) else None
        ) or _model_from_dict(info.get("model") if isinstance(info, Mapping) else None)
        if model is not None:
            models.add(model.label())
    return agents, models


class OpenChamberClient:
    """Minimal OpenChamber desktop API client used by the relay."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:57123",
        timeout: float = 10.0,
        transport: requests.Session | None = None,
    ):
        if not base_url or not base_url.strip():
            raise OpenChamberSessionError("OpenChamber address must not be empty")
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = timeout
        self._http = transport if transport is not None else requests.Session()

    # ------------------------------------------------------------------ #
    # connection
    # ------------------------------------------------------------------ #

    def health(self) -> dict[str, Any]:
        try:
            response = self._http.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise OpenChamberUnavailableError(
                f"cannot reach OpenChamber at {self.base_url}: {exc}"
            ) from exc
        self._raise_for_api_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenChamberSessionError(
                "OpenChamber /health returned a non-JSON body"
            ) from exc
        if not isinstance(payload, dict):
            raise OpenChamberSessionError("OpenChamber /health has an invalid payload")
        return payload

    def verify(self) -> dict[str, Any]:
        """Step 1 of the task flow: prove the service and configuration work."""
        payload = self.health()
        if payload.get("status") != "ok":
            raise OpenChamberSessionError(
                f"OpenChamber reports unhealthy status: {payload.get('status')!r}"
            )
        return payload

    # ------------------------------------------------------------------ #
    # session lifecycle
    # ------------------------------------------------------------------ #

    def create_session(self, title: str, directory: str) -> str:
        payload = self._post_json(
            "/api/openchamber/sessions",
            {"title": title, "directory": directory},
        )
        session_id = payload.get("sessionId") if isinstance(payload, Mapping) else None
        if not isinstance(session_id, str) or not session_id:
            raise OpenChamberSessionError(
                f"session create response has no sessionId: {payload!r}"
            )
        return session_id

    def open_session(self, session_id: str) -> None:
        """Ask the desktop app to open the session via the native deep link.

        Success only proves the OS dispatched the protocol handler; it does
        not prove the window displayed the session.
        """
        uri = f"openchamber://session/{session_id}"
        try:
            if os.name == "nt":
                os.startfile(uri)  # type: ignore[attr-defined]
            else:
                import subprocess

                subprocess.Popen(
                    ["xdg-open", uri],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as exc:
            raise OpenChamberSessionError(
                f"failed to open session deep link {uri}: {exc}"
            ) from exc

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        """Existing sessions as ``(session_id, title)`` pairs, newest first.

        ``GET /api/session`` returns every session; appending
        ``?directory=...`` filters server-side to one project directory
        (both backslash and forward-slash directory forms are accepted by
        OpenChamber 1.22.2).  The unfiltered list is capped by the server at
        the 100 most recent sessions; the directory-filtered form is exact
        for one directory.  This is a read-only listing used to present
        existing-session candidates and to verify a configured session:
        the relay NEVER creates or auto-switches sessions.
        """
        path = "/api/session"
        if directory and directory.strip():
            encoded = requests.utils.quote(directory, safe="")
            path += f"?directory={encoded}"
        payload = self._get_json(path)
        if not isinstance(payload, list):
            raise OpenChamberSessionError(
                f"session list response is not a list: {type(payload).__name__}"
            )
        sessions: list[tuple[str, str]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            session_id = item.get("id")
            title = item.get("title")
            if not isinstance(session_id, str) or not session_id:
                continue
            sessions.append((session_id, title if isinstance(title, str) else ""))
        return sessions

    def list_sessions_with_projects(
        self, directory: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Sessions as ``(session_id, title, project_directory)`` triples.

        Unlike :meth:`list_sessions`, each item also carries the project
        directory it was created under, so callers can match a project by
        path client-side (for example when a server-side ``?directory=``
        filter disagrees with the locally canonicalized path).  The
        unfiltered form (``directory=None``) returns every session.
        """
        path = "/api/session"
        if directory and directory.strip():
            encoded = requests.utils.quote(directory, safe="")
            path += f"?directory={encoded}"
        payload = self._get_json(path)
        if not isinstance(payload, list):
            raise OpenChamberSessionError(
                f"session list response is not a list: {type(payload).__name__}"
            )
        sessions: list[tuple[str, str, str]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            session_id = item.get("id")
            title = item.get("title")
            project = item.get("directory")
            if not isinstance(session_id, str) or not session_id:
                continue
            sessions.append(
                (
                    session_id,
                    title if isinstance(title, str) else "",
                    project if isinstance(project, str) else "",
                )
            )
        return sessions

    @staticmethod
    def match_project_sessions(
        project_directory: str,
        all_sessions: Sequence[tuple[str, str, str]],
    ) -> list[tuple[str, str]]:
        """Client-side path-compatible filter of ``all_sessions``.

        Compares each session's project directory (case-insensitively, with
        normalized separators and trailing slashes stripped) against
        ``project_directory``.  Used when a server-side filter returns
        nothing even though the project really has sessions.
        """
        if not project_directory:
            return []
        key = normalize_directory(project_directory)
        matched: list[tuple[str, str]] = []
        for session_id, title, project in all_sessions:
            if project and normalize_directory(project) == key:
                matched.append((session_id, title))
        return matched

    def session_exists(self, session_id: str, directory: str) -> bool:
        """Whether ``session_id`` exists under ``directory``.

        Membership is checked against the server-filtered session list, so
        a configured session that does NOT exist under the project directory
        is reliably distinguished and reported (the relay must never fall
        back to another session).  A failed listing raises instead of
        guessing a False.
        """
        return any(
            existing == session_id for existing, _title in self.list_sessions(directory)
        )

    def auto_accept_snapshot(self) -> dict[str, bool]:
        """Current per-session permission auto-accept flags.

        ``GET /api/permission-auto-accept`` returns a snapshot whose
        ``sessions`` member maps session ids to booleans (the ``revision``
        field is ignored).  Used to inherit a policy onto a newly rotated
        session.
        """
        payload = self._get_json("/api/permission-auto-accept")
        sessions = payload.get("sessions") if isinstance(payload, Mapping) else None
        if not isinstance(sessions, Mapping):
            raise OpenChamberSessionError(
                f"permission auto-accept snapshot has no sessions: {payload!r}"
            )
        return {
            str(session_id): bool(enabled)
            for session_id, enabled in sessions.items()
            if isinstance(session_id, str)
        }

    def set_session_auto_accept(
        self,
        session_id: str,
        enabled: bool,
        directory: str | None = None,
    ) -> dict[str, bool]:
        """Set one session's permission auto-accept policy via PUT.

        The response echoes a snapshot with the same shape as
        :meth:`auto_accept_snapshot`.
        """
        if not session_id or not session_id.strip():
            raise OpenChamberSessionError("session id must not be empty")
        encoded = requests.utils.quote(session_id, safe="")
        body: dict[str, Any] = {"enabled": bool(enabled)}
        if directory and directory.strip():
            body["directory"] = directory
        payload = self._put_json(
            f"/api/permission-auto-accept/sessions/{encoded}", body
        )
        sessions = payload.get("sessions") if isinstance(payload, Mapping) else None
        if not isinstance(sessions, Mapping):
            raise OpenChamberSessionError(
                f"set auto-accept response has no sessions: {payload!r}"
            )
        return {
            str(session_id): bool(flag)
            for session_id, flag in sessions.items()
            if isinstance(session_id, str)
        }

    def _pre_send_snapshot(self, session_id: str, directory: str) -> tuple[frozenset[str], bool]:
        """Message ids already present in the session, taken before the send.

        A failed snapshot abort the send: with no trustworthy snapshot the
        round cannot be attributed, so the prompt is never dispatched.
        """
        try:
            messages = self.messages(session_id, directory)
        except OpenChamberError:
            return frozenset(), False
        ids = set()
        for message in messages:
            info = message.get("info")
            if isinstance(info, Mapping):
                message_id = info.get("id")
                if isinstance(message_id, str) and message_id:
                    ids.add(message_id)
        return frozenset(ids), True

    def send(
        self,
        session_id: str,
        prompt: str,
        directory: str,
        agent: str | None = None,
        model: ModelRef | None = None,
    ) -> OpenChamberDispatch:
        if not prompt or not prompt.strip():
            raise OpenChamberSessionError("OpenChamber task prompt must not be empty")
        pre_ids, snapshot_ok = self._pre_send_snapshot(session_id, directory)
        if not snapshot_ok:
            raise OpenChamberSessionError(
                "cannot attribute this round safely: the pre-send message "
                f"snapshot for session {session_id} could not be read; the "
                "task was NOT sent and the session is kept for manual check"
            )
        body: dict[str, Any] = {"prompt": prompt, "directory": directory}
        if agent:
            body["agent"] = agent
        if model is not None:
            body["model"] = model.label()
        payload = self._post_json(
            f"/api/openchamber/sessions/{session_id}/send", body
        )

        resolved = _model_from_dict(
            payload.get("model") if isinstance(payload, Mapping) else None
        )
        dispatched = (
            isinstance(payload, Mapping) and payload.get("promptDispatched") is True
        )
        prompt_error = payload.get("promptError") if isinstance(payload, Mapping) else None
        if not dispatched:
            detail = f": {prompt_error}" if prompt_error else ""
            raise OpenChamberSessionError(
                f"OpenChamber did not dispatch the prompt{detail} "
                f"(response: {payload!r}); session {session_id} kept for manual check"
            )
        user_message_id = None
        for key in ("userMessageId", "promptMessageId"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                user_message_id = value
                break
        agent_value = payload.get("agent") if isinstance(payload, Mapping) else None
        return OpenChamberDispatch(
            session_id=session_id,
            directory=directory,
            requested_model=model,
            resolved_model=resolved,
            agent=agent_value if isinstance(agent_value, str) else None,
            prompt_dispatched=dispatched,
            dispatched_as_command=(
                isinstance(payload, Mapping)
                and payload.get("dispatchedAsCommand") is True
            ),
            prompt_error=str(prompt_error) if prompt_error else None,
            user_message_id=user_message_id,
            pre_send_message_ids=pre_ids,
            pre_send_snapshot_ok=snapshot_ok,
            prompt_text=prompt,
        )

    # ------------------------------------------------------------------ #
    # status and messages
    # ------------------------------------------------------------------ #

    def session_status_detail(self, session_id: str, directory: str) -> str:
        """Return the session's status in the FULL client vocabulary.

        ``idle`` -- an explicit idle entry (or, per OpenCode's status
        service, a session id MISSING from the map, which the service
        defines as idle); ``busy`` / ``retry`` -- reported verbatim;
        ``missing_from_status_map`` -- the status request SUCCEEDED but the
        session is absent from the returned map (the map only ever holds
        busy/retry entries); ``unknown`` -- request parsed but the payload
        is malformed (non-dict map, null entry, unrecognized type).
        Transport-level failures (SSE read timeout / connection reset /
        service down) and HTTP errors raise instead of returning a status:
        they must be retried at the transport layer, never mistaken for
        ``missing_from_status_map``.
        """
        encoded = requests.utils.quote(directory, safe="")
        response = self._get_json(f"/api/session/status?directory={encoded}")
        if not isinstance(response, dict):
            return "unknown"
        if session_id not in response:
            return MISSING_FROM_STATUS_MAP
        entry = response[session_id]
        if entry is None:
            return "unknown"
        if isinstance(entry, str) and entry:
            return entry if entry in ("idle", "busy", "retry") else "unknown"
        if isinstance(entry, Mapping):
            # The desktop fork wraps the value as ``{"type": "busy"}``; be
            # tolerant of ``status`` / ``state`` keys from other versions so
            # an object-shaped idle is never silently compared to "idle" as
            # ``unknown``.  Non-whitelisted types (sleep/starting/... ) stay
            # ``unknown`` and must NOT be treated as idle.
            for key in ("type", "status", "state"):
                status_type = entry.get(key)
                if isinstance(status_type, str) and status_type:
                    return (
                        status_type if status_type in ("idle", "busy", "retry") else "unknown"
                    )
        return "unknown"

    def session_status(self, session_id: str, directory: str) -> str:
        """Return the session's status type.

        ``idle`` covers a MISSING session id: OpenCode's SessionStatus
        service deletes a session from the map exactly when it becomes idle
        (and falls back to idle for unknown ids), so the map only holds
        busy/retry entries.  A session id PRESENT with a null value is
        malformed data and is ``unknown``.  Anything else (unrecognized
        type, malformed payload) is ``unknown`` and must never be treated as
        idle or as success.  Callers that must tell a vanished session apart
        from an explicit idle entry use :meth:`session_status_detail`
        (``missing_from_status_map`` vs ``idle`` vs ``unknown``).
        """
        status = self.session_status_detail(session_id, directory)
        if status == MISSING_FROM_STATUS_MAP:
            return "idle"
        return status

    def messages(self, session_id: str, directory: str) -> list[dict[str, Any]]:
        encoded = requests.utils.quote(directory, safe="")
        payload = self._get_json(f"/api/session/{session_id}/message?directory={encoded}")
        if not isinstance(payload, list):
            raise OpenChamberSessionError(
                f"session message response is not a list: {type(payload).__name__}"
            )
        return [item for item in payload if isinstance(item, dict)]

    def round_has_pending_user_action(
        self,
        session_id: str,
        directory: str,
        dispatch: OpenChamberDispatch,
    ) -> bool:
        """True when this round currently shows a candidate pending
        question/permission tool part.

        Used while the session reports ``busy``/``retry`` so the wait
        detects pending interactions instead of only reporting busy.  Any
        interface or attribution failure returns False (the busy branch
        just keeps waiting; it never guesses).
        """
        try:
            messages = self.messages(session_id, directory)
        except OpenChamberError:
            return False
        user_index, location_error = locate_round(messages, dispatch)
        if location_error != "ok":
            return False
        try:
            round_messages = _round_assistant_messages(
                messages, user_index, _message_id(messages[user_index])
            )
        except OpenChamberError:
            return False
        return has_pending_user_action(round_messages)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _get_json(self, path: str) -> Any:
        try:
            response = self._http.get(f"{self.base_url}{path}", timeout=self.timeout)
        except requests.RequestException as exc:
            raise OpenChamberUnavailableError(
                f"cannot reach OpenChamber at {self.base_url}: {exc}"
            ) from exc
        self._raise_for_api_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise OpenChamberSessionError(
                f"OpenChamber {path} returned a non-JSON body"
            ) from exc

    def _post_json(self, path: str, body: Mapping[str, Any]) -> Any:
        try:
            response = self._http.post(
                f"{self.base_url}{path}", json=dict(body), timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise OpenChamberUnavailableError(
                f"cannot reach OpenChamber at {self.base_url}: {exc}"
            ) from exc
        self._raise_for_api_status(response)
        try:
            return response.json()
        except ValueError:
            try:
                return {"raw": response.text}
            except Exception:
                return {"raw": ""}

    def _put_json(self, path: str, body: Mapping[str, Any]) -> Any:
        try:
            response = self._http.put(
                f"{self.base_url}{path}", json=dict(body), timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise OpenChamberUnavailableError(
                f"cannot reach OpenChamber at {self.base_url}: {exc}"
            ) from exc
        self._raise_for_api_status(response)
        try:
            return response.json()
        except ValueError:
            try:
                return {"raw": response.text}
            except Exception:
                return {"raw": ""}

    @staticmethod
    def _extract_upstream_error(detail: str) -> dict[str, Any]:
        """Best-effort extraction of the structured upstream fields that
        OpenChamber relays for a rejected model-service request.

        Recognizes both the plain ``APIError`` block (``statusCode`` /
        ``isRetryable`` / ``url``) and the JSON form.  Returns a dict with
        an entry only for fields that were actually present, so classification
        never happens on guessed data.
        """
        result: dict[str, Any] = {}
        if not detail:
            return result
        patterns = {
            "status_code": r"(?:statusCode|[\"']statusCode[\"'])\s*[:=]\s*(\d+)",
            "is_retryable": (
                r"(?:isRetryable|[\"']isRetryable[\"'])\s*[:=]\s*(true|false)"
            ),
            "request_url": r"(?:url|[\"']url[\"'])\s*[:=]\s*(\S+)",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, detail, flags=re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1)
            if field == "is_retryable":
                result[field] = raw.lower() == "true"
            elif field == "status_code":
                result[field] = int(raw)
            else:
                result[field] = raw
        return result

    @staticmethod
    def _is_400_model_rejection(detail: str) -> dict[str, Any] | None:
        """Classify an HTTP 400 as a model-request rejection or ``None``.

        Structured detection is authoritative: the upstream relayed
        ``isRetryable`` field decides.  When OpenChamber relays no structured
        field at all, a CONSERVATIVE string fallback requires the full
        upstream ``APIError`` signature (statusCode + url + the APIError
        marker) before a 400 is treated as a model rejection — an ordinary
        400 is never blindly downgraded to a rejection.
        """
        structured = OpenChamberClient._extract_upstream_error(detail)
        if "is_retryable" in structured:
            if structured["is_retryable"] is False:
                return structured
            return None
        has_upstream_marker = (
            "APIError" in detail
            and structured.get("request_url")
            and structured.get("status_code") == 400
        )
        return structured if has_upstream_marker else None

    @staticmethod
    def _raise_for_api_status(response: requests.Response) -> None:
        if response.status_code in (401, 403):
            raise OpenChamberAuthError(
                "OpenChamber rejected the request (HTTP "
                f"{response.status_code}); configure normal OpenChamber UI "
                "authentication instead of disabling it"
            )
        if response.status_code >= 400:
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                    detail = payload["error"]
            except ValueError:
                detail = response.text[:300]
            if response.status_code == 400:
                rejected = OpenChamberClient._is_400_model_rejection(detail)
                if rejected is not None:
                    raise OpenChamberModelRequestRejectedError(
                        "OpenChamber rejected the request (HTTP 400) because "
                        f"the upstream model service rejected it: {detail}",
                        status_code=rejected.get("status_code", 400),
                        is_retryable=rejected.get("is_retryable"),
                        request_url=rejected.get("request_url"),
                    )
                raise OpenChamberBadRequestError(
                    f"OpenChamber rejected the request (HTTP 400): {detail}"
                )
            raise OpenChamberSessionError(
                f"OpenChamber returned HTTP {response.status_code}: {detail}"
            )


# ---------------------------------------------------------------------- #
# round location and completion verification
# ---------------------------------------------------------------------- #


def _role(message: Mapping[str, Any]) -> str | None:
    info = message.get("info")
    if isinstance(info, Mapping):
        role = info.get("role")
        if isinstance(role, str):
            return role
    return None


def _message_id(message: Mapping[str, Any]) -> str | None:
    info = message.get("info")
    if isinstance(info, Mapping):
        message_id = info.get("id")
        if isinstance(message_id, str) and message_id:
            return message_id
    return None


def _parent_id(message: Mapping[str, Any]) -> str | None:
    info = message.get("info")
    if isinstance(info, Mapping):
        parent = info.get("parentID")
        if isinstance(parent, str) and parent:
            return parent
    return None


def locate_round(
    messages: Sequence[Mapping[str, Any]], dispatch: OpenChamberDispatch
) -> tuple[int, str]:
    """Locate this dispatch round among the session's messages.

    Returns ``(user_index, error)`` with ``error`` one of ``"ok"``,
    ``"not_found"`` (the prompt was not recorded yet) or ``"ambiguous"``
    (cannot be uniquely attributed; the caller must fail, never guess).

    Attribution is by message id, not by time or position: user messages
    whose id is new relative to the pre-send snapshot are candidates;
    exactly one is required.  A dispatch whose snapshot failed cannot prove
    which user message is new and is ALWAYS ambiguous (never degraded to a
    guess).  A server-returned user message id (when the build provides
    one) takes precedence over the snapshot.
    """
    user_indexes = [
        index for index, message in enumerate(messages) if _role(message) == "user"
    ]

    if dispatch.user_message_id:
        for index in user_indexes:
            if _message_id(messages[index]) == dispatch.user_message_id:
                return index, "ok"
        return -1, "not_found"

    if not dispatch.pre_send_snapshot_ok:
        return -1, "ambiguous"

    candidates = [
        index
        for index in user_indexes
        if _message_id(messages[index]) not in dispatch.pre_send_message_ids
    ]
    if len(candidates) > 1 and dispatch.prompt_text:
        exact = [
            index
            for index in candidates
            if _message_text(messages[index]) == dispatch.prompt_text
        ]
        if len(exact) == 1:
            return exact[0], "ok"
    if len(candidates) > 1:
        return -1, "ambiguous"
    if not candidates:
        return -1, "not_found"
    return candidates[0], "ok"


def _message_text(message: Mapping[str, Any]) -> str:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return ""
    return "\n".join(
        str(part.get("text"))
        for part in parts
        if isinstance(part, Mapping)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ).strip()


def _message_created(message: Mapping[str, Any]) -> int | None:
    info = message.get("info")
    if isinstance(info, Mapping):
        created = (info.get("time") or {}).get("created")
        if isinstance(created, int) and not isinstance(created, bool) and created > 0:
            return created
    return None


def _chain_result(
    message: Mapping[str, Any],
    anchor_id: str,
    by_id: Mapping[str, Mapping[str, Any]],
    user_ids: set[str],
) -> str:
    """Resolve an assistant message's ``parentID`` chain.

    Returns ``"anchor"`` when the chain reaches the round's user message,
    ``"other"`` when it verifiably terminates at a different user message
    (history/another round -> not part of this round), or ``"unknown"``
    when the parent is missing, unresolvable or cyclic (cannot be proven:
    the round must fail, never include the message).
    """
    seen: set[str] = set()
    current = _message_id(message)
    while True:
        if not current:
            return "unknown"
        if current == anchor_id:
            return "anchor"
        if current in user_ids:
            return "other"
        if current in seen:
            return "unknown"
        seen.add(current)
        node = by_id.get(current)
        if node is None:
            return "unknown"
        parent = _parent_id(node)
        if not parent:
            return "unknown"
        current = parent


def _round_assistant_messages(
    messages: Sequence[Mapping[str, Any]],
    anchor_index: int,
    anchor_id: str | None,
) -> list[Mapping[str, Any]]:
    """Assistant messages that provably belong to the anchored round.

    Membership is decided by the verified ``parentID`` chain, NOT by array
    position: an assistant message is part of the round only when walking
    its ``parentID`` chain reaches the round's user message.  History and
    other rounds' messages chain to a different user and are excluded.
    Messages whose chain cannot be resolved make the round ambiguous.
    The round is returned ordered by ``time.created`` (strictly increasing
    across turns in real data); a message without a verified created
    timestamp or with a tie cannot be strictly ordered and the round fails.
    """
    if not anchor_id:
        raise OpenChamberSessionError(
            "cannot attribute this round without a verifiable user message "
            "id in the session"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    user_ids: set[str] = set()
    for message in messages:
        mid = _message_id(message)
        if mid:
            by_id[mid] = message
            if _role(message) == "user":
                user_ids.add(mid)

    round_messages: list[Mapping[str, Any]] = []
    unverifiable: list[str] = []
    for message in messages:
        if _role(message) != "assistant":
            continue
        result = _chain_result(message, anchor_id, by_id, user_ids)
        if result == "anchor":
            round_messages.append(message)
        elif result == "unknown":
            mid = _message_id(message)
            unverifiable.append(f"message {mid} has an unverifiable parent chain")
    if unverifiable:
        raise OpenChamberSessionError(
            "ambiguous: an assistant message in OpenChamber session "
            f"({'; '.join(unverifiable)}) cannot be proven to belong to "
            "this task round; the result was not guessed"
        )
    return _strictly_order_round(round_messages)


def _follow_continuation_chain(
    messages: Sequence[Mapping[str, Any]],
    round_messages: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """When the round's last assistant message has ``finish="tool-calls"``
    (an interrupted tool-calls round), look ahead for the NEXT user message
    in the session and include its chained assistant messages.

    This handles the case where the monitor auto-continues the session:
    the monitor's user message (U2) and assistant reply (A2) form a
    separate round that is invisible to ``_round_assistant_messages`` (which
    only follows ``parentID`` chains to the original anchor).  By detecting
    ``tool-calls`` and extending the round, ``wait_for_completion`` can
    find the final reply even when the monitor has continued the session."""
    if not round_messages:
        return round_messages
    last = round_messages[-1]
    info = last.get("info") if isinstance(last.get("info"), Mapping) else {}
    finish = info.get("finish") if isinstance(info, Mapping) else None
    if finish != "tool-calls":
        return round_messages
    # Find the position of the last assistant message in the session list.
    last_id = _message_id(last)
    last_index = -1
    for idx, msg in enumerate(messages):
        if _message_id(msg) == last_id:
            last_index = idx
            break
    if last_index < 0:
        return round_messages
    # Find the next user message after the last assistant message.
    next_user_index = -1
    for idx in range(last_index + 1, len(messages)):
        if _role(messages[idx]) == "user":
            next_user_index = idx
            break
    if next_user_index < 0:
        return round_messages
    # Collect assistant messages chained to this next user message.
    next_anchor_id = _message_id(messages[next_user_index])
    if not next_anchor_id:
        return round_messages
    by_id = {_message_id(m): m for m in messages if _message_id(m)}
    user_ids = {_message_id(m) for m in messages if _role(m) == "user"}
    extra: list[Mapping[str, Any]] = []
    for msg in messages[next_user_index + 1:]:
        if _role(msg) != "assistant":
            continue
        result = _chain_result(msg, next_anchor_id, by_id, user_ids)
        if result == "anchor":
            extra.append(msg)
        elif result == "unknown":
            # Unverifiable chain in continuation: include anyway
            # (the monitor's continuation is expected to chain cleanly).
            extra.append(msg)
    if extra:
        return _strictly_order_round(round_messages + extra)
    return round_messages


def _strictly_order_round(
    round_messages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if not round_messages:
        return []
    entries: list[tuple[int, Mapping[str, Any]]] = []
    for message in round_messages:
        created = _message_created(message)
        if created is None:
            raise OpenChamberSessionError(
                "ambiguous: an assistant message of this round has no "
                "verified created timestamp; the result was not guessed"
            )
        entries.append((created, message))
    keys = [created for created, _message in entries]
    if len(set(keys)) != len(keys):
        raise OpenChamberSessionError(
            "ambiguous: this round's assistant messages cannot be strictly "
            "ordered by their verified created timestamps; the result was "
            "not guessed"
        )
    return [message for _, message in sorted(entries, key=lambda item: item[0])]


def _parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return []
    return [part for part in parts if isinstance(part, Mapping)]


def has_pending_user_action(round_messages: Sequence[Mapping[str, Any]]) -> bool:
    """True while the round shows a candidate pending question/permission.

    A pending model question is a tool part ``{"type": "tool",
    "tool": "question", "state": {"status": "pending"}}`` (observed on this
    stack).  OTHER pending TOOL / PERMISSION parts are treated as the same
    CANDIDATE waiting signal: the relay keeps waiting and prompts the
    operator instead of failing.  The real permission-popup flow is not
    verified on this stack, so no claim is made that the permission path
    was validated.
    """
    for message in round_messages:
        for part in _parts(message):
            if part.get("type") not in ("tool", "permission"):
                continue
            state = part.get("state")
            if isinstance(state, Mapping) and state.get("status") == "pending":
                return True
    return False


def pending_user_action_prompt(session_id: str) -> str:
    return (
        "请在 OpenChamber 中处理：会话 "
        f"{session_id} 显示待处理的问题或权限提示"
        "（候选状态，真实权限流程待验证）；"
        "处理后中继将自动继续并回传"
    )


def _text_parts(message: Mapping[str, Any]) -> list[str]:
    texts = []
    for part in _parts(message):
        if part.get("type") != "text":
            continue
        if part.get("synthetic") is True:
            continue  # real marker for injected/internal text, not an answer
        state = part.get("state")
        if isinstance(state, Mapping) and state.get("status") == "ignored":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def _tool_call_names(message: Mapping[str, Any]) -> list[str]:
    names = []
    for part in _parts(message):
        if part.get("type") != "tool":
            continue
        name = part.get("tool")
        if isinstance(name, str) and name:
            names.append(name)
    return names


@dataclass(frozen=True, slots=True)
class MonitorScan:
    """One manual OpenChamber monitor probe.

    ``new_completed`` lists assistant replies that appeared after the
    monitor baseline and are NOT already wrapped: each is a
    ``(msg_id, text)`` pair, oldest first.  ``seen_ids`` is the full current
    set of assistant message ids seen in the session (the new baseline).
    ``completed_history_ids`` is the subset of those that are ALREADY fully
    completed replies (``time.completed`` + ``finish==stop`` + non-empty
    text); only those may seed the monitor's wrap baseline -- an assistant
    message that is still streaming when the monitor starts must stay
    eligible to wrap once it completes, so it must NOT enter the baseline.
    """

    new_completed: tuple[tuple[str, str], ...]
    seen_ids: frozenset[str]
    completed_history_ids: frozenset[str]


def _message_completed(message: Mapping[str, Any]) -> bool:
    """Whether a single assistant message is a fully completed reply.

    Mirrors the real-stack rules used by :func:`wait_for_completion`: a
    positive (non-bool) ``time.completed`` plus ``finish == "stop"``, no
    error, and non-empty final text.  Synthetic / ignored text parts are
    never answers.
    """
    info = message.get("info")
    if not isinstance(info, Mapping):
        return False
    if _role(message) != "assistant":
        return False
    finished = (info.get("time") or {}).get("completed")
    if not isinstance(finished, int) or isinstance(finished, bool) or finished <= 0:
        return False
    if info.get("finish") != "stop":
        return False
    if info.get("error"):
        return False
    return bool(_text_parts(message))


def _monitor_completed_replies(
    messages: Sequence[Mapping[str, Any]],
    baseline: frozenset[str],
    session_id: str = "",
) -> tuple[tuple[str, str], ...]:
    """Completed assistant replies newer than ``baseline``, oldest first.

    Shared by :func:`monitor_scan` and :func:`monitor_probe` so the manual
    monitor uses exactly one reply-extraction rule.  Already-wrapped replies
    never enter the list (the UI's runtime dedup is a second, independent
    guard).
    """
    ordered: list[tuple[int, str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        msg_id = _message_id(message)
        if not msg_id:
            continue
        if _role(message) != "assistant":
            continue
        if msg_id in baseline or is_message_wrapped(msg_id, session_id):
            continue
        if not _message_completed(message):
            continue
        created = _message_created(message) or 0
        ordered.append((created, msg_id, "\n".join(_text_parts(message)).strip()))
    ordered.sort(key=lambda item: item[0])
    return tuple((mid, text) for _created, mid, text in ordered if text)


def monitor_scan(
    client: "OpenChamberClient",
    session_id: str,
    directory: str,
    baseline: frozenset[str] = frozenset(),
) -> MonitorScan:
    """Scan a session for new completed assistant replies.

    Returns only replies that are newer than the ``baseline`` assistant ids
    and not already wrapped (see :func:`is_message_wrapped`), oldest first
    by recorded creation time.  ``seen_ids`` is every assistant message id
    currently in the session; ``completed_history_ids`` is the subset that is
    already a fully completed reply -- only that subset may seed the monitor
    baseline, so a message still streaming at start stays wrappable later.
    """
    messages = client.messages(session_id, directory)
    seen = set()
    completed_history = set()
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        msg_id = _message_id(message)
        if not msg_id or _role(message) != "assistant":
            continue
        seen.add(msg_id)
        if _message_completed(message):
            completed_history.add(msg_id)
    new_completed = _monitor_completed_replies(messages, baseline, session_id)
    return MonitorScan(
        new_completed, frozenset(seen), frozenset(completed_history)
    )


@dataclass(frozen=True, slots=True)
class MonitorMessageStat:
    """Compact per-message summary used by the manual monitor's round
    tracker.  ``text_length`` counts the real final text only (synthetic /
    ignored parts excluded), ``tool_count`` counts tool parts."""

    message_id: str
    role: str | None
    finish: str | None
    completed_ts: int | None
    text_length: int
    tool_count: int


@dataclass(frozen=True, slots=True)
class MonitorProbe:
    """One manual-monitor probe over a session.

    Like :class:`MonitorScan` it reports the new completed assistant replies
    (for wrapping), plus the whole current message set with role/finish/
    completion details and the session status -- everything the round
    tracker needs to tell "new activity" from "history idle" and to confirm
    an abnormal idle without a second HTTP round-trip.
    """

    new_completed: tuple[tuple[str, str], ...]
    seen_ids: frozenset[str]
    messages: tuple[MonitorMessageStat, ...]
    status: str


def monitor_probe(
    client: "OpenChamberClient",
    session_id: str,
    directory: str,
    baseline: frozenset[str] = frozenset(),
) -> MonitorProbe:
    """One manual-monitor probe: new completed replies + full message stats
    + current session status (one messages call + one status call)."""
    messages = client.messages(session_id, directory)
    status = client.session_status(session_id, directory)
    # Client-boundary normalization: only the exact statuses the tracker
    # understands may flow in.  A session vanished from the status map is
    # server-side idle (the tracker's own activity_observed precondition
    # guards against counting a never-seen session), so it maps to "idle";
    # anything else becomes "unknown" so the idle comparison can never
    # silently match (or fail to match) on exotic types.
    if status == MISSING_FROM_STATUS_MAP:
        status = "idle"
    elif not isinstance(status, str) or status not in ("idle", "busy", "retry"):
        status = "unknown"
    seen_ids: set[str] = set()
    stats: list[MonitorMessageStat] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        msg_id = _message_id(message)
        if not msg_id:
            continue
        seen_ids.add(msg_id)
        info = message.get("info")
        finish = info.get("finish") if isinstance(info, Mapping) else None
        completed = (info.get("time") or {}).get("completed") if isinstance(info, Mapping) else None
        if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
            completed = None
        finish = finish if isinstance(finish, str) else None
        stats.append(
            MonitorMessageStat(
                message_id=msg_id,
                role=_role(message),
                finish=finish,
                completed_ts=completed,
                text_length=len("\n".join(_text_parts(message)) or ""),
                tool_count=len(_tool_call_names(message)),
            )
        )
    new_completed = _monitor_completed_replies(messages, baseline, session_id)
    return MonitorProbe(new_completed, frozenset(seen_ids), tuple(stats), status)


def _latest_assistant_stat(probe: MonitorProbe) -> MonitorMessageStat | None:
    """The last assistant message in session order (the newest one)."""
    latest = None
    for stat in probe.messages:
        if stat.role == "assistant":
            latest = stat
    return latest


@dataclass(frozen=True, slots=True)
class MonitorStepOutcome:
    """What one tracker step concluded."""

    event: str  # "none" | "activity" | "idle_confirm" | "interrupted"
    idle_confirmations: int
    required_idle: int
    last_message_id: str | None
    last_finish: str | None
    reason: str | None


@dataclass(slots=True)
class MonitorRoundTracker:
    """Lightweight in-memory context for one manual "监听 OpenChamber" run.

    Lives entirely on the monitor polling thread (never persisted).  It
    distinguishes messages that appeared AFTER the monitor baseline from
    history, tracks the current round's latest assistant signature and
    confirms an abnormal idle only after ``required_idle`` consecutive,
    IDENTICAL suspicious polls (busy/retry, new messages, growing text or a
    changed finish all reset the counter).
    """

    session_id: str
    directory: str
    baseline_ids: set[str] = field(default_factory=set)
    required_idle: int = MONITOR_ONLY_IDLE_CONFIRMATIONS
    activity_observed: bool = False
    last_message_id: str | None = None
    last_text_length: int = 0
    last_finish: str | None = None
    last_tool_count: int = 0
    idle_confirmations: int = 0
    interruption_emitted: bool = False
    last_status: str | None = None
    last_seen_ids: set[str] = field(default_factory=set)
    reset_reason: str | None = None

    def reset_round(self, seen_ids: frozenset[str], reason: str | None = None) -> None:
        """A round ended (a final reply was wrapped, or the operator
        continued, or the monitor just started): every current message
        becomes history and the wake-up conditions start cold.

        ``reason`` is recorded for the ``relay.log`` audit trail so every
        reset is explainable (monitor start / a real final reply was wrapped
        / manual continue)."""
        self.reset_reason = reason
        self.baseline_ids = set(seen_ids)
        self.last_seen_ids = set(seen_ids)
        self.activity_observed = False
        self.last_message_id = None
        self.last_text_length = 0
        self.last_finish = None
        self.last_tool_count = 0
        self.idle_confirmations = 0
        self.interruption_emitted = False
        self.last_status = None

    def step(self, probe: MonitorProbe) -> MonitorStepOutcome:
        """Advance detection over one probe.

        ``event`` is ``"interrupted"`` once the abnormal idle is confirmed,
        ``"idle_confirm"`` while it is still counting up, ``"activity"``
        when the session produced fresh work, or ``"none"``.  A fully idle
        history session never triggers anything (``activity_observed``).
        """
        status = probe.status
        new_ids = probe.seen_ids - frozenset(self.last_seen_ids)
        self.last_seen_ids = set(probe.seen_ids)
        latest = _latest_assistant_stat(probe)
        if latest is not None:
            signature = (
                latest.message_id,
                latest.text_length,
                latest.finish or "",
                latest.tool_count,
            )
            previous = (
                self.last_message_id,
                self.last_text_length,
                self.last_finish or "",
                self.last_tool_count,
            )
            if signature != previous:
                changed_same_message = self.last_message_id == latest.message_id
                self.last_message_id = latest.message_id
                self.last_text_length = latest.text_length
                self.last_finish = latest.finish
                self.last_tool_count = latest.tool_count
                if changed_same_message and not new_ids:
                    # Same message, different content: streaming/text growth.
                    self.activity_observed = True
                    self.interruption_emitted = False
                    self.idle_confirmations = 0
                    return self._outcome("activity", reason="activity_same_message_text_growth")
        if new_ids:
            # A brand-new message implies activity; absorb it first so the
            # same block never counts as its own confirmation.
            self.activity_observed = True
            self.interruption_emitted = False
            self.idle_confirmations = 0
            return self._outcome("activity", reason="activity_new_message")
        if status in ("busy", "retry"):
            if status != self.last_status:
                self.activity_observed = True
            self.last_status = status
            self.idle_confirmations = 0
            self.interruption_emitted = False
            return self._outcome("activity", reason=f"activity_status_{status}")
        self.last_status = status
        if status != "idle":
            self.idle_confirmations = 0
            return self._outcome("none", reason=f"status_{status!r}")
        if not self.activity_observed:
            self.idle_confirmations = 0
            return self._outcome("none", reason="no_activity_since_baseline")
        round_messages = [
            stat for stat in probe.messages if stat.message_id not in self.baseline_ids
        ]
        round_assistants = [stat for stat in round_messages if stat.role == "assistant"]
        round_latest = round_assistants[-1] if round_assistants else None
        new_empty_assistant = any(
            stat.role == "assistant" and stat.text_length == 0
            for stat in round_assistants
        )
        tool_activity = any(stat.tool_count > 0 for stat in round_messages)
        suspicion = (
            (round_latest is not None and round_latest.finish == "tool-calls")
            or new_empty_assistant
            or (round_latest is not None and round_latest.completed_ts is None)
            or (round_latest is not None and round_latest.finish is None)
            or tool_activity
        )
        if not suspicion:
            self.idle_confirmations = 0
            return self._outcome("none", reason="idle_without_suspicion")
        if self.interruption_emitted:
            return self._outcome("none", reason="interruption_already_emitted")
        self.idle_confirmations += 1
        if self.idle_confirmations >= self.required_idle:
            self.interruption_emitted = True
            return self._outcome("interrupted", reason="monitor_idle_without_completed_reply")
        return self._outcome("idle_confirm", reason="idle_confirm")

    def _outcome(self, event: str, reason: str | None = None) -> MonitorStepOutcome:
        return MonitorStepOutcome(
            event=event,
            idle_confirmations=self.idle_confirmations,
            required_idle=self.required_idle,
            last_message_id=self.last_message_id,
            last_finish=self.last_finish,
            reason=reason,
        )


def extract_final_text(round_messages: Sequence[Mapping[str, Any]]) -> str:
    """Final assistant text of the round.

    Only the LAST message of the verified order contributes text: an
    earlier intermediate message's text is never a fallback.  Synthetic /
    ignored text parts are excluded per real-stack semantics.  An empty
    result means the final message genuinely carries no answer.
    """
    if not round_messages:
        return ""
    final = round_messages[-1]
    texts = _text_parts(final)
    return "\n".join(texts).strip()


def _sleep_interruptible(
    seconds: float, cancel_event: threading.Event | None
) -> None:
    """Sleep up to ``seconds`` in small chunks so cancellation is responsive.

    Raises :class:`OpenChamberCancelledError` as soon as ``cancel_event`` is
    set; ``cancel_event=None`` keeps the original single blocking sleep.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise OpenChamberCancelledError("OpenChamber wait cancelled by user")
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise OpenChamberCancelledError("OpenChamber wait cancelled by user")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def wait_for_completion(
    client: OpenChamberClient,
    dispatch: OpenChamberDispatch,
    timeout: float,
    poll_interval: float = 2.0,
    grace_seconds: float = 5.0,
    status_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> CompletionResult:
    """Wait until this dispatch round finished and verify the final answer.

    Completion requires ALL of: the session status is a confirmed idle
    (an explicit idle or a missing session id, which OpenCode's status
    service defines as idle; unrecognized status types, interface failures
    and malformed payloads are never converted into idle or success), the
    round is uniquely located by message id, the round's last assistant
    message is completed (``time.completed`` is a positive integer
    timestamp — 0, negative values and booleans are rejected, no
    ``info.error``), its
    ``finish`` is ``stop``, no candidate pending tool/permission part
    exists, and a non-empty final text exists.  Truncation
    (``finish == length``) and errors are reported as failures, never
    wrapped as success.

    While the round shows a candidate pending question/permission the relay
    keeps waiting (reporting "请在 OpenChamber 中处理") instead of failing:
    it never answers, approves or re-sends anything.  This candidate state
    is detected BOTH while the session reports busy/retry AND while it is
    idle — the round is left running and the operator is prompted until the
    interaction goes away.  The real permission-popup flow is not verified
    on this stack; the pending signal is a candidacy hint, not a claim that
    approvals were validated.  Only the overall deadline stops the relay;
    the session is kept and the backend keeps running.  ``timeout <= 0``
    means wait forever (no deadline); real errors (aborted, error, invalid
    session) still fail normally.

    A round that stops idle WITHOUT a usable answer (no reply, not
    completed, an absent ``finish``, or no final text) raises
    :class:`OpenChamberInterruptedError` — still an
    ``OpenChamberSessionError``, but signalling that the caller may safely
    ask the session to continue.  Ambiguous attribution, a never-recorded
    prompt, aborted/error details and ``finish`` values other than
    ``stop``/``tool-calls`` remain plain ``OpenChamberSessionError``
    failures that must not be silently continued.

    A session that DISAPPEARS from the status map (successful request,
    empty/other-only map → ``missing_from_status_map``) is handled like an
    idle stop ONLY after activity was observed first (busy/retry, a new
    assistant message, text growth, a changed finish/tool count, or the
    round being located): then a valid final reply completes the round
    normally, while an abnormal round state (finish=None, not completed,
    empty body, a final ``tool-calls`` without a valid follow-up) confirms
    through the same consecutive-poll mechanism with reason
    ``status_missing_after_activity``.  Before any activity is seen a
    vanished session is simply waited on (never a stall), and transport
    failures / unparseable responses stay ``unknown`` and never masquerade
    as a vanished session.  This also makes the stall detectable with
    ``timeout <= 0`` (no deadline).

    When ``cancel_event`` is set the wait loop raises
    :class:`OpenChamberCancelledError` (at the loop top or inside sleeps)
    without touching the session, so the operator can stop waiting and keep
    the session for manual monitoring.  Cancellation is a dedicated error,
    never a reuse of the timeout path.
    """
    update = status_callback or (lambda _status: None)
    deadline = (
        float("inf")
        if timeout <= 0
        else time.monotonic() + timeout
    )
    record_grace_deadline: float | None = None
    reply_grace_deadline: float | None = None
    incomplete_grace_deadline: float | None = None
    missing_tool_grace_deadline: float | None = None
    last_seen_message_id: str | None = None
    saw_pending_user_action = False
    # Activity precondition for counting a VANISHED session as a stall.
    # ``wait_for_completion`` is only ever called for a dispatch the service
    # accepted (a confirmed send: the prompt is recorded in the session), so
    # the round is, by definition, active from the first poll.  A status map
    # that never listed the session (service restart / status lag / degraded
    # status endpoint) must therefore NOT mask a genuinely stuck round.
    session_activity_observed = True
    last_activity_signature: tuple | None = None
    # Idle-interruption confirmation state.
    idle_confirmations = 0
    idle_signature: tuple | None = None
    # Limited transport retry state, reset on every successful read.
    transport_retry_count = 0
    # Fakes may only implement session_status (missing → idle already);
    # the real client provides the detailed vocabulary.
    status_fn = getattr(client, "session_status_detail", None)
    if status_fn is None:
        status_fn = client.session_status

    def _transport_failure(site: str) -> None:
        """Handle a transient transport error (SSE timeout / connection
        reset, surfaced as OpenChamberUnavailableError): retry up to
        MAX_TRANSPORT_RETRIES with cancel-aware delays, then escalate to an
        interrupted-continuation recovery.  Never re-POSTs the original task
        and never creates a new session."""
        nonlocal transport_retry_count
        transport_retry_count += 1
        if transport_retry_count <= MAX_TRANSPORT_RETRIES:
            delay = TRANSPORT_RETRY_DELAYS[transport_retry_count - 1]
            update(f"OpenChamber 连接暂时中断（{site}），正在重连（{transport_retry_count}/{MAX_TRANSPORT_RETRIES}）……")
            _LOG.info(
                "oc wait: transient connection error session=%s site=%s "
                "transport_retry_count=%d/%d",
                dispatch.session_id, site, transport_retry_count,
                MAX_TRANSPORT_RETRIES,
            )
            _sleep_interruptible(delay, cancel_event)
            return
        update("OpenChamber 连接多次中断，准备检查任务是否需要续接……")
        _LOG.warning(
            "oc wait: transport retries exhausted session=%s site=%s "
            "last_message_id=%s transport_retry_count=%d reason=transport_failure",
            dispatch.session_id, site, last_seen_message_id,
            transport_retry_count,
        )
        raise OpenChamberInterruptedError(
            "OpenCode 工具续接中断：OpenChamber 连接在传输层多次中断"
            f"（{site}），会话可能已进入 idle 但本轮没有有效最终回复",
            session_id=dispatch.session_id,
            last_message_id=last_seen_message_id,
            last_finish=None,
            reason="transport_failure",
        )

    def _note_idle_suspicion(reason: str, signature: tuple) -> None:
        """Increment the idle-interruption confirmation counter, resetting it
        when the observed abnormal-idle signature changes (busy, new
        message, growing text, completed reply or a changed finish all reset
        it).  Raises OpenChamberInterruptedError after REQUIRED
        consecutive identical abnormal-idle polls."""
        nonlocal idle_confirmations, idle_signature
        if idle_signature is None or idle_signature != signature:
            idle_signature = signature
            idle_confirmations = 1
            _LOG.info(
                "oc wait: idle-interruption suspicion session=%s "
                "reason=%s last_message_id=%s idle_confirmations=%d/%d",
                dispatch.session_id, reason,
                signature[1] if len(signature) > 1 else last_seen_message_id,
                idle_confirmations, REQUIRED_IDLE_CONFIRMATIONS,
            )
            update(f"检测到疑似工具续接中断；异常 idle 连续确认：1/{REQUIRED_IDLE_CONFIRMATIONS}")
            return
        idle_confirmations += 1
        _LOG.info(
            "oc wait: idle-interruption confirmation session=%s reason=%s "
            "idle_confirmations=%d/%d recovery_attempted=false",
            dispatch.session_id, reason, idle_confirmations,
            REQUIRED_IDLE_CONFIRMATIONS,
        )
        update(
            f"异常 idle 连续确认：{idle_confirmations}/{REQUIRED_IDLE_CONFIRMATIONS}"
        )
        if idle_confirmations >= REQUIRED_IDLE_CONFIRMATIONS:
            _LOG.warning(
                "oc wait: interrupted idle confirmed session=%s reason=%s "
                "last_message_id=%s last_finish=%s",
                dispatch.session_id, reason,
                signature[1] if len(signature) > 1 else last_seen_message_id,
                signature[2] if len(signature) > 2 else None,
            )
            raise OpenChamberInterruptedError(
                "OpenCode 工具续接中断：会话已进入 idle，但本轮没有有效最终回复"
                f"（reason={reason}）",
                session_id=dispatch.session_id,
                last_message_id=signature[1] if len(signature) > 1 else last_seen_message_id,
                last_finish=signature[2] if len(signature) > 2 else None,
                reason=reason,
            )

    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise OpenChamberCancelledError("OpenChamber wait cancelled by user")
        if deadline != float("inf") and time.monotonic() > deadline:
            detail = (
                "；若会话正等待你的问题/权限确认，仍可在 OpenChamber 中处理"
                if saw_pending_user_action
                else ""
            )
            raise OpenChamberTimeoutError(
                f"OpenChamber task did not finish within {timeout:.0f}s; "
                f"session {dispatch.session_id} may still be running"
                f"{detail}; open it in OpenChamber and check the result"
            )

        try:
            status_type = status_fn(dispatch.session_id, dispatch.directory)
        except OpenChamberModelRequestRejectedError:
            # A non-retryable model rejection must never become an infinite
            # wait; the relay handles it as a fresh-session retry instead.
            raise
        except OpenChamberUnavailableError:
            _transport_failure("session_status")
            continue
        except OpenChamberError:
            status_type = "unknown"
        transport_retry_count = 0

        if status_type in ("busy", "retry"):
            # The service reports this session as working: it is, by
            # definition, active.
            session_activity_observed = True
            idle_confirmations = 0
            idle_signature = None
            record_grace_deadline = None
            reply_grace_deadline = None
            incomplete_grace_deadline = None
            missing_tool_grace_deadline = None
            last_seen_message_id = None
            sleep_for = min(poll_interval, max(0.1, deadline - time.monotonic()))
            if client.round_has_pending_user_action(
                dispatch.session_id, dispatch.directory, dispatch
            ):
                # A pending question/permission is detected DURING busy/retry
                # too: the round is waiting for the operator, not executing.
                saw_pending_user_action = True
                update(pending_user_action_prompt(dispatch.session_id))
            else:
                saw_pending_user_action = False
                if timeout <= 0:
                    update("OpenChamber 正在工作中，无时间限制…")
                else:
                    update(
                        "OpenChamber 正在工作中（retry 为上游重试）…"
                        if status_type == "retry"
                        else "OpenChamber 正在工作中…"
                    )
            _sleep_interruptible(sleep_for, cancel_event)
            continue

        if status_type == MISSING_FROM_STATUS_MAP and not session_activity_observed:
            # The status request succeeded but the service does not report
            # this session at all, and the wait never saw it work: this is
            # NOT a stall (first polls, service restart, status lag) -- just
            # wait for the deadline, never start idle confirmations.
            idle_confirmations = 0
            idle_signature = None
            _sleep_interruptible(
                min(poll_interval, max(0.1, deadline - time.monotonic())),
                cancel_event,
            )
            update("OpenChamber status interface is not yet reporting this session, waiting…")
            continue

        if status_type not in ("idle", MISSING_FROM_STATUS_MAP):
            # Unrecognized status type or malformed payload (transport
            # failures raise earlier and retry at the transport layer): never
            # treat as idle, never as a vanished session, never convert to
            # success; keep waiting for the deadline.
            idle_confirmations = 0
            idle_signature = None
            _sleep_interruptible(
                min(poll_interval, max(0.1, deadline - time.monotonic())),
                cancel_event,
            )
            update("OpenChamber status unconfirmed, waiting…")
            continue

        # idle -- or a session that vanished from the status map AFTER the
        # wait observed it active: verify the message round before declaring
        # completion (a vanished session with a valid final reply completes
        # normally; one whose round is still abnormal confirms a stall).
        status_is_missing = status_type == MISSING_FROM_STATUS_MAP
        try:
            messages = client.messages(dispatch.session_id, dispatch.directory)
        except OpenChamberModelRequestRejectedError:
            raise
        except OpenChamberUnavailableError:
            _transport_failure("messages")
            continue
        except OpenChamberError:
            _sleep_interruptible(
                min(poll_interval, max(0.1, deadline - time.monotonic())),
                cancel_event,
            )
            update("OpenChamber 消息接口异常，等待中…")
            continue
        transport_retry_count = 0

        user_index, location_error = locate_round(messages, dispatch)
        if location_error == "ambiguous":
            raise OpenChamberSessionError(
                "ambiguous: this task round cannot be uniquely attributed "
                f"in OpenChamber session {dispatch.session_id} (multiple "
                "new user messages or a parentID mismatch; a message may "
                "have been typed into the same session manually); the "
                "result was not guessed"
            )
        if location_error == "not_found":
            # The dispatched prompt is not recorded yet (or the round never
            # started). Give it a short grace period before failing.
            if record_grace_deadline is None:
                record_grace_deadline = time.monotonic() + grace_seconds
            if time.monotonic() < record_grace_deadline:
                _sleep_interruptible(min(poll_interval, 0.5), cancel_event)
                update("等待 OpenChamber 记录本轮任务…")
                continue
            raise OpenChamberSessionError(
                "OpenChamber reports idle but the dispatched task was "
                f"never recorded in session {dispatch.session_id}; "
                "check the session"
            )
        record_grace_deadline = None
        # The round is located: the user message (and its assistant round)
        # exists in the session -- activity the vanished-status stall check
        # may build on from now on.
        session_activity_observed = True

        round_messages = _round_assistant_messages(
            messages, user_index, _message_id(messages[user_index])
        )
        # When the round ends with finish="tool-calls" (interrupted tool
        # execution), follow the continuation chain: the monitor may have
        # auto-continued the session, creating a new user→assistant round
        # that is invisible to _round_assistant_messages.  Extend the round
        # so wait_for_completion can detect the final reply.
        round_messages = _follow_continuation_chain(messages, round_messages)
        if not round_messages:
            if reply_grace_deadline is None:
                reply_grace_deadline = time.monotonic() + grace_seconds
            if time.monotonic() < reply_grace_deadline:
                _sleep_interruptible(min(poll_interval, 0.5), cancel_event)
                update("等待 OpenChamber 开始执行…")
                continue
            _note_idle_suspicion(
                "status_missing_after_activity"
                if status_is_missing
                else "no_assistant_reply",
                ("no_assistant_reply", last_seen_message_id, None, 0),
            )
            continue
        reply_grace_deadline = None

        last = round_messages[-1]
        info = last.get("info") if isinstance(last.get("info"), Mapping) else {}
        last_message_id = info.get("id") if isinstance(info, Mapping) else None
        if last_message_id != last_seen_message_id:
            # A new assistant message arrived: re-arm the incomplete grace.
            incomplete_grace_deadline = None
            last_seen_message_id = last_message_id

        # Any change of the round's observable state (new assistant message,
        # body growth, finish change, tool-call count change) is activity:
        # it re-arms the "曾经活动" precondition and zeroes the idle
        # confirmation counter (streaming / growth is not a stall).
        _finished_marker = (
            info.get("finish") if isinstance(info, Mapping) else None
        )
        _activity_signature = (
            last_message_id,
            _finished_marker,
            len(extract_final_text(round_messages)),
            sum(len(_tool_call_names(m)) for m in round_messages),
        )
        if (
            last_activity_signature is not None
            and _activity_signature != last_activity_signature
        ):
            session_activity_observed = True
            idle_confirmations = 0
            idle_signature = None
            _LOG.info(
                "oc wait: round activity change resets idle confirmations "
                "session=%s last_message_id=%s",
                dispatch.session_id, last_message_id,
            )
        last_activity_signature = _activity_signature

        if has_pending_user_action(round_messages):
            saw_pending_user_action = True
            update(pending_user_action_prompt(dispatch.session_id))
            _sleep_interruptible(
                min(poll_interval, max(0.1, deadline - time.monotonic())),
                cancel_event,
            )
            continue
        saw_pending_user_action = False

        error = info.get("error") if isinstance(info, Mapping) else None
        if error:
            detail = error_detail(error)
            if detail.strip().casefold() in {"bad request", "uri too long"}:
                raise OpenChamberModelRequestRejectedError(
                    "当前会话上下文过大，模型服务已拒绝请求: " + detail,
                    status_code=414 if detail.strip().casefold() == "uri too long" else 400,
                    is_retryable=False,
                )
            raise OpenChamberSessionError(
                f"OpenChamber task failed in session {dispatch.session_id}: "
                f"{detail}"
            )
        completed_ts = (
            (info.get("time") or {}).get("completed")
            if isinstance(info, Mapping)
            else None
        )
        finished_marker = info.get("finish") if isinstance(info, Mapping) else None
        completed = (
            isinstance(completed_ts, int)
            and not isinstance(completed_ts, bool)
            and completed_ts > 0
        )
        if not completed:
            if incomplete_grace_deadline is None:
                incomplete_grace_deadline = time.monotonic() + grace_seconds
            if time.monotonic() < incomplete_grace_deadline:
                _sleep_interruptible(min(poll_interval, 0.5), cancel_event)
                update("等待 OpenChamber 本轮执行结束…")
                continue
            final_len = len(extract_final_text(round_messages))
            _note_idle_suspicion(
                "status_missing_after_activity"
                if status_is_missing
                else "not_completed",
                ("not_completed", last_message_id, finished_marker, final_len),
            )
            continue

        finish = info.get("finish") if isinstance(info, Mapping) else None
        if finish == "length":
            raise OpenChamberSessionError(
                f"OpenChamber reply was truncated by the model output limit "
                f"(session {dispatch.session_id}); this is not a success"
            )
        if finish == "tool-calls":
            if status_is_missing:
                # The session vanished from the status map right after a
                # tool-calls message and NO continuation reply followed:
                # the tool continuation likely died with the session's
                # status entry.  Give the follow-up a short grace, then
                # confirm through the same 3-poll mechanism.
                if missing_tool_grace_deadline is None:
                    missing_tool_grace_deadline = time.monotonic() + grace_seconds
                if time.monotonic() < missing_tool_grace_deadline:
                    _sleep_interruptible(min(poll_interval, 0.5), cancel_event)
                    update("OpenChamber 状态消失，等待工具后续回复…")
                    continue
                final_len = len(extract_final_text(round_messages))
                _note_idle_suspicion(
                    "status_missing_after_activity",
                    ("missing_tool_calls", last_message_id, "tool-calls", final_len),
                )
                continue
            if timeout <= 0:
                update("OpenChamber 正在执行工具，无时间限制，等待后续回复…")
            else:
                update("OpenChamber 正在执行工具，等待后续回复…")
            _sleep_interruptible(
                min(poll_interval, max(0.1, deadline - time.monotonic())),
                cancel_event,
            )
            continue
        if finish != "stop":
            if finish is None:
                final_len = len(extract_final_text(round_messages))
                _note_idle_suspicion(
                    "status_missing_after_activity"
                    if status_is_missing
                    else "finish_none",
                    ("finish_none", last_message_id, None, final_len),
                )
                continue
            raise OpenChamberSessionError(
                f"OpenChamber round ended abnormally (finish={finish!r}) in "
                f"session {dispatch.session_id}"
            )

        tool_calls: list[str] = []
        for message in round_messages:
            tool_calls.extend(_tool_call_names(message))
        final_text = extract_final_text(round_messages)
        if not final_text:
            _note_idle_suspicion(
                "status_missing_after_activity"
                if status_is_missing
                else "no_final_text",
                ("no_final_text", last_message_id, "stop", 0),
            )
            continue

        actual_model = _model_from_message_info(info if isinstance(info, Mapping) else None)
        reference_model = (
            dispatch.resolved_model or dispatch.requested_model or actual_model
        )
        model_mismatch = (
            actual_model is not None
            and reference_model is not None
            and actual_model != reference_model
        )
        return CompletionResult(
            session_id=dispatch.session_id,
            final_text=final_text,
            finish=str(finish),
            actual_model=actual_model,
            tool_calls=tuple(tool_calls),
            requested_model=dispatch.requested_model,
            resolved_model=dispatch.resolved_model,
            model_mismatch=model_mismatch,
            round_message_ids=tuple(
                mid
                for mid in (_message_id(m) for m in round_messages)
                if mid
            ),
        )
