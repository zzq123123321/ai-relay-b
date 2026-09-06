"""OpenChamber executor client.

All traffic is sent to the configured OpenChamber address (default
``http://127.0.0.1:57123``).  The ``/api`` prefix is proxied by OpenChamber
to its managed OpenCode instance; the proxy injects OpenCode's own
credentials, so this client never connects to the dynamic OpenCode port and
never handles OpenCode credentials itself.

Verified against the local OpenChamber 1.22.0 build (2026-09-06):

* ``POST /api/openchamber/sessions``  -> ``{"sessionId": ...}``
* ``POST /api/openchamber/sessions/:id/send`` -> ``baselineAssistantMessageId``
  and the resolved ``model`` / ``agent`` actually used
* ``GET  /api/session/status?directory=...`` -> ``{sessionId: {"type": ...}}``
  where type is ``idle`` / ``busy`` / ``retry``
* ``GET  /api/session/:id/message?directory=...`` -> message array whose
  assistant ``info`` carries ``finish`` / ``error`` / ``time.completed``
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
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

    The POST may already have started execution upstream; the caller must
    keep the session id and ask the operator to check the session instead of
    resending.
    """


class OpenChamberUserActionRequired(OpenChamberError):
    """The session is waiting for the user inside OpenChamber (question/permission)."""


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
    """What the send call confirmed on the server side."""

    session_id: str
    directory: str
    baseline_message_id: str | None
    baseline_message_time: int | None
    requested_model: ModelRef | None
    resolved_model: ModelRef | None
    agent: str | None
    prompt_dispatched: bool
    dispatched_as_command: bool
    prompt_error: str | None
    sent_at_ms: int


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
        session_id = payload.get("sessionId")
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
        body: dict[str, Any] = {"prompt": prompt, "directory": directory}
        if agent:
            body["agent"] = agent
        if model is not None:
            body["model"] = model.as_payload()
        payload = self._post_json(
            f"/api/openchamber/sessions/{session_id}/send", body
        )

        baseline_id = payload.get("baselineAssistantMessageId")
        if not isinstance(baseline_id, str) or not baseline_id:
            baseline_id = None
        baseline_time = self._baseline_time(session_id, directory, baseline_id)
        resolved = _model_from_dict(payload.get("model"))
        dispatched = payload.get("promptDispatched") is True
        prompt_error = payload.get("promptError")
        if not dispatched:
            detail = f": {prompt_error}" if prompt_error else ""
            raise OpenChamberSessionError(
                f"OpenChamber did not dispatch the prompt{detail} "
                f"(response: {payload!r}); session {session_id} kept for manual check"
            )
        return OpenChamberDispatch(
            session_id=session_id,
            directory=directory,
            baseline_message_id=baseline_id,
            baseline_message_time=baseline_time,
            requested_model=model,
            resolved_model=resolved,
            agent=payload.get("agent") if isinstance(payload.get("agent"), str) else None,
            prompt_dispatched=dispatched,
            dispatched_as_command=payload.get("dispatchedAsCommand") is True,
            prompt_error=str(prompt_error) if prompt_error else None,
            sent_at_ms=int(time.time() * 1000),
        )

    # ------------------------------------------------------------------ #
    # status and messages
    # ------------------------------------------------------------------ #

    def session_status(self, session_id: str, directory: str) -> str:
        """Return ``idle`` / ``busy`` / ``retry`` (or ``unknown``)."""
        encoded = requests.utils.quote(directory, safe="")
        response = self._get_json(f"/api/session/status?directory={encoded}")
        statuses = response.get(session_id) if isinstance(response, dict) else None
        if isinstance(statuses, Mapping):
            status_type = statuses.get("type")
            if isinstance(status_type, str) and status_type:
                return status_type
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

    def _baseline_time(
        self, session_id: str, directory: str, baseline_id: str | None
    ) -> int | None:
        if not baseline_id:
            return None
        try:
            for message in self.messages(session_id, directory):
                info = message.get("info")
                if not isinstance(info, dict) or info.get("id") != baseline_id:
                    continue
                created = (info.get("time") or {}).get("created")
                if isinstance(created, int):
                    return created
        except OpenChamberError:
            return None
        return None


def _assistant_messages_after(
    messages: Sequence[Mapping[str, Any]], anchor_index: int
) -> list[Mapping[str, Any]]:
    return [
        message
        for message in messages[anchor_index + 1 :]
        if _role(message) == "assistant"
    ]


def _role(message: Mapping[str, Any]) -> str | None:
    info = message.get("info")
    if isinstance(info, Mapping):
        role = info.get("role")
        if isinstance(role, str):
            return role
    return None


def _user_message_index(
    messages: Sequence[Mapping[str, Any]],
    dispatch: OpenChamberDispatch,
) -> int:
    """Index of the first user message that belongs to this dispatch round."""
    baseline_index = -1
    for index, message in enumerate(messages):
        info = message.get("info")
        if (
            isinstance(info, Mapping)
            and info.get("id") == dispatch.baseline_message_id
        ):
            baseline_index = index
            break

    if baseline_index >= 0:
        # The baseline anchors the round: everything before it is history.
        for index in range(baseline_index + 1, len(messages)):
            if _role(messages[index]) == "user":
                return index
        return -1

    # Baseline message unknown: fall back to the send-time window. A user
    # message must be created inside that window, otherwise it is history.
    lower_bound = (
        dispatch.baseline_message_time
        if dispatch.baseline_message_time is not None
        else dispatch.sent_at_ms - 5_000
    )
    for index, message in enumerate(messages):
        if _role(message) != "user":
            continue
        info = message.get("info")
        created = (info.get("time") or {}).get("created") if isinstance(info, Mapping) else None
        if isinstance(created, int) and created >= lower_bound:
            return index
    return -1


def _text_parts(message: Mapping[str, Any]) -> list[str]:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return []
    texts = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def _tool_call_names(message: Mapping[str, Any]) -> list[str]:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return []
    names = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") != "tool":
            continue
        name = part.get("tool")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _has_question_part(message: Mapping[str, Any]) -> bool:
    parts = message.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return False
    for part in parts:
        if isinstance(part, Mapping) and part.get("type") == "question":
            return True
    return False


def extract_final_text(
    messages: Sequence[Mapping[str, Any]], round_start_index: int
) -> str:
    """Final assistant text of the round, scanning backwards for real text."""
    for index in range(len(messages) - 1, round_start_index, -1):
        message = messages[index]
        if _role(message) != "assistant":
            continue
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

    Completion requires ALL of: the session state is no longer
    busy/retry, the round's last assistant message is completed
    (``time.completed`` set, no ``info.error``), its ``finish`` is ``stop``,
    it carries no pending question part, and a non-empty final text exists.
    Truncation (``finish == length``) and errors are reported as failures,
    never wrapped as success.
    """
    update = status_callback or (lambda _status: None)
    deadline = time.monotonic() + timeout
    record_grace_deadline: float | None = None
    reply_grace_deadline: float | None = None
    incomplete_grace_deadline: float | None = None
    last_seen_message_id: str | None = None

    while True:
        if time.monotonic() > deadline:
            raise OpenChamberTimeoutError(
                f"OpenChamber task did not finish within {timeout:.0f}s; "
                f"session {dispatch.session_id} may still be running, "
                "open it in OpenChamber and check the result"
            )

        try:
            status_type = client.session_status(
                dispatch.session_id, dispatch.directory
            )
        except OpenChamberUnavailableError:
            status_type = "unknown"

        if status_type in ("busy", "retry"):
            record_grace_deadline = None
            reply_grace_deadline = None
            incomplete_grace_deadline = None
            last_seen_message_id = None
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            update(f"OpenChamber 正在执行（{status_type}）…")
            continue

        # idle / unknown: the run may be over; verify the message round.
        try:
            messages = client.messages(dispatch.session_id, dispatch.directory)
        except OpenChamberUnavailableError as exc:
            if time.monotonic() > deadline:
                raise OpenChamberTimeoutError(
                    f"OpenChamber task did not finish within {timeout:.0f}s: {exc}"
                ) from exc
            time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
            update("OpenChamber 连接中断，等待恢复…")
            continue

        user_index = _user_message_index(messages, dispatch)
        if user_index < 0:
            # The dispatched prompt is not recorded yet (or the round never
            # started). Give it a short grace period before failing.
            if record_grace_deadline is None:
                record_grace_deadline = time.monotonic() + grace_seconds
            if time.monotonic() < record_grace_deadline:
                time.sleep(min(poll_interval, 0.5))
                update("等待 OpenChamber 记录本轮任务…")
                continue
            raise OpenChamberSessionError(
                "OpenChamber reports idle but the dispatched task was never "
                f"recorded in session {dispatch.session_id}; check the session"
            )
        record_grace_deadline = None

        round_messages = _assistant_messages_after(messages, user_index)
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

        if _has_question_part(last):
            raise OpenChamberUserActionRequired(
                "OpenChamber session is asking the user a question; answer it "
                f"in the OpenChamber window (session {dispatch.session_id}) "
                "before the task can complete"
            )

        error = info.get("error") if isinstance(info, Mapping) else None
        completed = (
            isinstance(info, Mapping)
            and isinstance((info.get("time") or {}).get("completed"), int)
        )
        if error:
            detail = error.get("message") if isinstance(error, Mapping) else error
            raise OpenChamberSessionError(
                f"OpenChamber task failed in session {dispatch.session_id}: "
                f"{detail!r}"
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
        final_text = extract_final_text(messages, user_index)
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