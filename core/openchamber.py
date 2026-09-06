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
  send (see :class:`OpenChamberDispatch`), never a time window.
* ``GET /api/session/status?directory=...`` ->
  ``{sessionId: {"type": "busy" | "retry" | ...}}``.  OpenChamber proxies
  this to OpenCode, whose ``SessionStatus.set()`` removes a session from the
  map the moment its status becomes ``idle`` (and ``get()`` falls back to
  ``{type: "idle"}`` for unknown ids).  A session id MISSING from the map
  therefore means idle; the map only ever contains busy/retry entries.
* ``GET /api/session/:id/message?directory=...`` -> message array; assistant
  ``info`` carries ``parentID`` (the user message that triggered it),
  ``finish`` / ``error`` / ``time.completed``.
* A model question is a part ``{"type": "tool", "tool": "question",
  "state": {"status": "pending"}}``; it completes (status ``completed``)
  only after the user answers in the OpenChamber UI.  Permission prompts are
  likewise pending tool/permission parts.

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
    error instead of being guessed.  ``pre_send_snapshot_ok`` is False when
    the pre-send snapshot request failed; the round is then located among
    ALL user messages, which stays safe but may over-report ambiguity.
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

    def _pre_send_snapshot(self, session_id: str, directory: str) -> tuple[frozenset[str], bool]:
        """Message ids already present in the session, taken before the send.

        Never fails the send: if the snapshot cannot be read, the round is
        later located among all user messages (``pre_send_snapshot_ok``
        False), which can only over-report ambiguity, never mis-attribute.
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

        ``idle`` covers both an explicit ``{"type": "idle"}`` entry and a
        missing session id: OpenCode's SessionStatus service deletes a
        session from the map exactly when it becomes idle (and falls back
        to idle for unknown ids), so the map only holds busy/retry entries.
        Anything else (unrecognized type, malformed payload) is ``unknown``
        and must never be treated as idle or as success.
        """
        encoded = requests.utils.quote(directory, safe="")
        response = self._get_json(f"/api/session/status?directory={encoded}")
        if not isinstance(response, dict):
            return "unknown"
        entry = response.get(session_id)
        if entry is None:
            return "idle"
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
    exactly one is required.  Assistant messages whose ``parentID`` points
    at a *different* user message make the attribution unprovable.  A
    server-returned user message id (when the build provides one) takes
    precedence over the snapshot.
    """
    user_indexes = [
        index for index, message in enumerate(messages) if _role(message) == "user"
    ]

    if dispatch.user_message_id:
        for index in user_indexes:
            if _message_id(messages[index]) == dispatch.user_message_id:
                return _check_parent_chain(
                    messages, index, dispatch.user_message_id
                )
        return -1, "not_found"

    if dispatch.pre_send_snapshot_ok:
        candidates = [
            index
            for index in user_indexes
            if _message_id(messages[index]) not in dispatch.pre_send_message_ids
        ]
    else:
        candidates = list(user_indexes)

    if len(candidates) > 1:
        return -1, "ambiguous"
    if not candidates:
        return -1, "not_found"
    return _check_parent_chain(
        messages, candidates[0], _message_id(messages[candidates[0]])
    )


def _check_parent_chain(
    messages: Sequence[Mapping[str, Any]], anchor_index: int, anchor_id: str | None
) -> tuple[int, str]:
    """Reject rounds whose assistant messages attach to another user message.

    Assistant messages after the anchor that carry a ``parentID`` pointing
    at a *user* message different from the anchor cannot be part of this
    round (history or a manually inserted message) and the round is then
    ambiguous.  A ``parentID`` pointing at another assistant message is a
    normal chain continuation and does not contradict the anchor.
    """
    known_user_ids = {
        message_id
        for message_id in (
            _message_id(message) for message in messages if _role(message) == "user"
        )
        if message_id
    }
    for index in range(anchor_index + 1, len(messages)):
        message = messages[index]
        if _role(message) != "assistant":
            continue
        parent = _parent_id(message)
        if parent is None or parent == anchor_id:
            continue
        if parent in known_user_ids:
            return -1, "ambiguous"
    return anchor_index, "ok"


def _parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return []
    return [part for part in parts if isinstance(part, Mapping)]


def _round_assistant_messages(
    messages: Sequence[Mapping[str, Any]],
    anchor_index: int,
    anchor_id: str | None,
) -> list[Mapping[str, Any]]:
    """Assistant messages belonging to the anchored round.

    Normally the anchor's user message precedes all assistant messages of
    the round in the array.  If the server returns them in another order,
    assistant messages that precede the anchor still belong to the round
    when their ``parentID`` points at the anchor.
    """
    by_index: dict[int, Mapping[str, Any]] = {}
    for index in range(anchor_index + 1, len(messages)):
        if _role(messages[index]) == "assistant":
            by_index[index] = messages[index]
    if anchor_id is not None:
        for index in range(0, anchor_index):
            if (
                _role(messages[index]) == "assistant"
                and _parent_id(messages[index]) == anchor_id
            ):
                by_index[index] = messages[index]
    return [by_index[index] for index in sorted(by_index)]


def has_pending_user_action(round_messages: Sequence[Mapping[str, Any]]) -> bool:
    """True while the round is blocked on a question or permission request.

    A pending model question is a tool part ``{"type": "tool",
    "tool": "question", "state": {"status": "pending"}}`` (observed on the
    local 1.22.2/1.18.29 stack); permission prompts surface the same way
    (pending tool/permission part) and through OpenCode's permission
    events.  Pending parts in ANY message of the round count: while one is
    pending the session is waiting for the operator, not executing.
    """
    for message in round_messages:
        for part in _parts(message):
            if part.get("type") == "tool" and part.get("tool") == "question":
                state = part.get("state")
                if isinstance(state, Mapping) and state.get("status") == "pending":
                    return True
            elif part.get("type") == "permission":
                state = part.get("state")
                if isinstance(state, Mapping) and state.get("status") == "pending":
                    return True
    return False


def _text_parts(message: Mapping[str, Any]) -> list[str]:
    texts = []
    for part in _parts(message):
        if part.get("type") != "text":
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
    """Final assistant text of the round (index-ordered, last real text)."""
    for message in reversed(round_messages):
        texts = _text_parts(message)
        if texts:
            return "\n".join(texts).strip()
    return ""


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
    message is completed (``time.completed`` set, no ``info.error``), its
    ``finish`` is ``stop``, no question/permission part is pending, and a
    non-empty final text exists.  Truncation (``finish == length``) and
    errors are reported as failures, never wrapped as success.

    While the round is blocked on a question or permission request the
    relay keeps waiting (reporting "请在 OpenChamber 中处理") instead of
    failing: it never answers, approves or re-sends anything.  Only the
    overall deadline stops the relay; the session is kept and the backend
    keeps running.
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
            saw_pending_user_action = False
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            update(
                "OpenChamber 正在执行（retry，上游重试中）…"
                if status_type == "retry"
                else "OpenChamber 正在执行（busy）…"
            )
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
            update(
                "请在 OpenChamber 中处理：会话 "
                f"{dispatch.session_id} 有未回答的问题或权限请求；"
                "处理完成后中继将自动继续并回传"
            )
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
        completed = (
            isinstance(info, Mapping)
            and isinstance((info.get("time") or {}).get("completed"), int)
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