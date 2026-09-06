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

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import requests


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
            body["model"] = model.as_payload()
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
        )

    # ------------------------------------------------------------------ #
    # status and messages
    # ------------------------------------------------------------------ #

    def session_status(self, session_id: str, directory: str) -> str:
        """Return the session's status type.

        ``idle`` covers a MISSING session id: OpenCode's SessionStatus
        service deletes a session from the map exactly when it becomes idle
        (and falls back to idle for unknown ids), so the map only holds
        busy/retry entries.  A session id PRESENT with a null value is
        malformed data and is ``unknown``.  Anything else (unrecognized
        type, malformed payload) is ``unknown`` and must never be treated as
        idle or as success.
        """
        encoded = requests.utils.quote(directory, safe="")
        response = self._get_json(f"/api/session/status?directory={encoded}")
        if not isinstance(response, dict):
            return "unknown"
        if session_id not in response:
            return "idle"
        entry = response[session_id]
        if entry is None:
            return "unknown"
        if isinstance(entry, str) and entry:
            return entry if entry in ("idle", "busy", "retry") else "unknown"
        if isinstance(entry, Mapping):
            status_type = entry.get("type")
            if isinstance(status_type, str) and status_type:
                return status_type if status_type in ("idle", "busy", "retry") else "unknown"
        return "unknown"

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
    if len(candidates) > 1:
        return -1, "ambiguous"
    if not candidates:
        return -1, "not_found"
    return candidates[0], "ok"


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


def wait_for_completion(
    client: OpenChamberClient,
    dispatch: OpenChamberDispatch,
    timeout: float,
    poll_interval: float = 2.0,
    grace_seconds: float = 5.0,
    status_callback: Callable[[str], None] | None = None,
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
    the session is kept and the backend keeps running.
    """
    update = status_callback or (lambda _status: None)
    deadline = time.monotonic() + timeout
    record_grace_deadline: float | None = None
    reply_grace_deadline: float | None = None
    incomplete_grace_deadline: float | None = None
    last_seen_message_id: str | None = None
    saw_pending_user_action = False

    while True:
        if time.monotonic() > deadline:
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
            status_type = client.session_status(
                dispatch.session_id, dispatch.directory
            )
        except OpenChamberUnavailableError:
            status_type = "unknown"
        except OpenChamberError:
            status_type = "unknown"

        if status_type in ("busy", "retry"):
            record_grace_deadline = None
            reply_grace_deadline = None
            incomplete_grace_deadline = None
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
                update(
                    "OpenChamber 正在执行（retry，上游重试中）…"
                    if status_type == "retry"
                    else "OpenChamber 正在执行（busy）…"
                )
            time.sleep(sleep_for)
            continue

        if status_type != "idle":
            # Unrecognized status type or malformed payload: never treat as
            # idle and never convert to success; keep waiting for the
            # deadline.
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            update("OpenChamber 状态未确认，等待中…")
            continue

        # idle: verify the message round before declaring completion.
        try:
            messages = client.messages(dispatch.session_id, dispatch.directory)
        except OpenChamberUnavailableError:
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            update("OpenChamber 连接中断，等待恢复…")
            continue
        except OpenChamberError:
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            update("OpenChamber 消息接口异常，等待中…")
            continue

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
                time.sleep(min(poll_interval, 0.5))
                update("等待 OpenChamber 记录本轮任务…")
                continue
            raise OpenChamberSessionError(
                "OpenChamber reports idle but the dispatched task was "
                f"never recorded in session {dispatch.session_id}; "
                "check the session"
            )
        record_grace_deadline = None

        round_messages = _round_assistant_messages(
            messages, user_index, _message_id(messages[user_index])
        )
        if not round_messages:
            if reply_grace_deadline is None:
                reply_grace_deadline = time.monotonic() + grace_seconds
            if time.monotonic() < reply_grace_deadline:
                time.sleep(min(poll_interval, 0.5))
                update("等待 OpenChamber 开始执行…")
                continue
            raise OpenChamberSessionError(
                "OpenChamber reports idle but the session has no assistant "
                f"reply for this task (session {dispatch.session_id})"
            )
        reply_grace_deadline = None

        last = round_messages[-1]
        info = last.get("info") if isinstance(last.get("info"), Mapping) else {}
        last_message_id = info.get("id") if isinstance(info, Mapping) else None
        if last_message_id != last_seen_message_id:
            # A new assistant message arrived: re-arm the incomplete grace.
            incomplete_grace_deadline = None
            last_seen_message_id = last_message_id

        if has_pending_user_action(round_messages):
            saw_pending_user_action = True
            update(pending_user_action_prompt(dispatch.session_id))
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            continue
        saw_pending_user_action = False

        error = info.get("error") if isinstance(info, Mapping) else None
        if error:
            detail = error.get("message") if isinstance(error, Mapping) else error
            raise OpenChamberSessionError(
                f"OpenChamber task failed in session {dispatch.session_id}: "
                f"{detail!r}"
            )
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
        if not completed:
            if incomplete_grace_deadline is None:
                incomplete_grace_deadline = time.monotonic() + grace_seconds
            if time.monotonic() < incomplete_grace_deadline:
                time.sleep(min(poll_interval, 0.5))
                update("等待 OpenChamber 本轮执行结束…")
                continue
            raise OpenChamberSessionError(
                "OpenChamber reports idle but the last assistant message of "
                f"this round is not completed (session {dispatch.session_id})"
            )

        finish = info.get("finish") if isinstance(info, Mapping) else None
        if finish == "length":
            raise OpenChamberSessionError(
                f"OpenChamber reply was truncated by the model output limit "
                f"(session {dispatch.session_id}); this is not a success"
            )
        if finish != "stop":
            raise OpenChamberSessionError(
                f"OpenChamber round ended abnormally (finish={finish!r}) in "
                f"session {dispatch.session_id}"
            )

        tool_calls: list[str] = []
        for message in round_messages:
            tool_calls.extend(_tool_call_names(message))
        final_text = extract_final_text(round_messages)
        if not final_text:
            raise OpenChamberSessionError(
                f"OpenChamber round finished but has no final text "
                f"(session {dispatch.session_id})"
            )

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
        )