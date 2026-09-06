"""Fake OpenChamber HTTP transport and scripted client for tests."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import requests

from core.openchamber import (
    ModelRef,
    OpenChamberDispatch,
    OpenChamberSessionError,
    OpenChamberUnavailableError,
)


class FakeResponse:
    def __init__(
        self, status_code: int = 200, body: Any = None, text: str = ""
    ):
        self.status_code = status_code
        self._body = body
        self.text = text if body is None else json.dumps(body, ensure_ascii=False)

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


class FakeHttp:
    """requests.Session stand-in: records calls, returns queued responses."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.handlers: dict[tuple[str, str], Any] = {}
        self.raise_next: Exception | None = None

    def route(self, method: str, path: str, response: FakeResponse) -> None:
        self.handlers[(method.upper(), path)] = response

    def _dispatch(self, method: str, url: str, body: dict | None) -> FakeResponse:
        if self.raise_next is not None:
            exc, self.raise_next = self.raise_next, None
            raise exc
        path = url.split("http://fake.local", 1)[-1]
        handler = self.handlers.get((method.upper(), path))
        if handler is None:
            return FakeResponse(404, {"error": f"no route for {method} {path}"})
        return handler if isinstance(handler, FakeResponse) else handler()

    def get(self, url: str, timeout: float = 0, **_kw: Any) -> FakeResponse:
        self.calls.append(("GET", url.split("http://fake.local", 1)[-1], None))
        return self._dispatch("GET", url, None)

    def post(self, url: str, json: dict | None = None, timeout: float = 0, **_kw: Any) -> FakeResponse:
        self.calls.append(("POST", url.split("http://fake.local", 1)[-1], json))
        return self._dispatch("POST", url, json)

    def close(self) -> None:
        pass


def make_client(
    http: FakeHttp, base_url: str = "http://fake.local"
):
    from core.openchamber import OpenChamberClient

    return OpenChamberClient(base_url=base_url, timeout=1.0, transport=http)


def make_dispatch(
    session_id: str = "ses_test123",
    directory: str = "D:/proj",
    pre_ids: frozenset[str] | set[str] | None = None,
    snapshot_ok: bool = True,
    user_message_id: str | None = None,
    requested: ModelRef | None = None,
    resolved: ModelRef | None = None,
    agent: str | None = None,
    dispatched: bool = True,
    prompt_error: str | None = None,
) -> OpenChamberDispatch:
    return OpenChamberDispatch(
        session_id=session_id,
        directory=directory,
        requested_model=requested,
        resolved_model=resolved,
        agent=agent,
        prompt_dispatched=dispatched,
        dispatched_as_command=False,
        prompt_error=prompt_error,
        user_message_id=user_message_id,
        pre_send_message_ids=frozenset(pre_ids or ()),
        pre_send_snapshot_ok=snapshot_ok,
    )


def user_message(msg_id: str, text: str, created: int, session_id: str = "ses_test123") -> dict:
    return {
        "info": {
            "id": msg_id,
            "sessionID": session_id,
            "role": "user",
            "time": {"created": created},
        },
        "parts": [{"type": "text", "text": text}],
    }


def assistant_message(
    msg_id: str,
    created: int,
    completed: int | None = None,
    finish: str | None = None,
    error: Any = None,
    parts: Sequence[dict] | None = None,
    model: ModelRef | None = None,
    agent: str | None = None,
    session_id: str = "ses_test123",
    parent_id: str | None = None,
) -> dict:
    info: dict[str, Any] = {
        "id": msg_id,
        "sessionID": session_id,
        "role": "assistant",
        "time": {"created": created},
    }
    if parent_id is not None:
        info["parentID"] = parent_id
    if completed is not None:
        info["time"]["completed"] = completed
    if finish is not None:
        info["finish"] = finish
    if error is not None:
        info["error"] = error
    if model is not None:
        info["providerID"] = model.provider_id
        info["modelID"] = model.model_id
    if agent is not None:
        info["agent"] = agent
    return {"info": info, "parts": list(parts or [])}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_part(name: str, status: str = "completed", output: str = "") -> dict:
    return {
        "type": "tool",
        "tool": name,
        "state": {"status": status, "output": output},
    }


def question_part(status: str = "pending") -> dict:
    """Real OpenCode shape: a *tool* part named "question" with a state."""
    state: dict[str, Any] = {"status": status}
    if status == "pending":
        state.update({"input": {}, "raw": ""})
    else:
        state.update(
            {
                "input": {
                    "questions": [{"question": "继续吗？"}]
                },
                "output": "User has answered your questions.",
            }
        )
    return {"type": "tool", "tool": "question", "state": state}


class ScriptedOpenChamber:
    """OpenChamberClient stand-in with a scripted status/message timeline."""

    def __init__(self, directory: str = "D:/proj"):
        self.directory = directory
        self.call_log: list[str] = []
        self.status_timeline: list[str] = []
        self._status_index = 0
        self.message_timelines: list[list[dict]] = []
        self._message_index = 0
        self.next_session_id = "ses_test123"
        self.create_error: Exception | None = None
        self.open_error: Exception | None = None
        self.send_dispatch: OpenChamberDispatch | None = None
        self.send_error: Exception | None = None
        self.unavailable: bool = False
        self.messages_read = 0

    # -- recording helpers ------------------------------------------------

    def verify(self) -> None:
        self.call_log.append("verify")
        if self.unavailable:
            raise OpenChamberUnavailableError("unavailable")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        if self.create_error is not None:
            raise self.create_error
        return self.next_session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")
        if self.open_error is not None:
            raise self.open_error
        self.call_log.append("opened")

    def send(self, session_id, prompt, directory, agent=None, model=None) -> OpenChamberDispatch:
        self.call_log.append(f"send:{prompt!r}")
        if self.send_error is not None:
            raise self.send_error
        if self.send_dispatch is None:
            self.send_dispatch = make_dispatch(session_id=session_id, directory=directory)
        return self.send_dispatch

    # -- status / messages timeline ---------------------------------------

    def session_status(self, session_id: str, directory: str) -> str:
        if self.unavailable:
            raise OpenChamberUnavailableError("unavailable")
        if not self.status_timeline:
            return "idle"
        index = min(self._status_index, len(self.status_timeline) - 1)
        self._status_index += 1  # auto-advance per poll; last value sticks
        return self.status_timeline[index]

    def advance_status(self) -> None:
        self._status_index += 1

    def messages(self, session_id: str, directory: str) -> list[dict]:
        if self.unavailable:
            raise OpenChamberUnavailableError("unavailable")
        self.messages_read += 1
        if not self.message_timelines:
            return []
        index = min(self._message_index, len(self.message_timelines) - 1)
        self._message_index += 1
        return self.message_timelines[index]

    def advance_messages(self) -> None:
        self._message_index += 1

    def round_has_pending_user_action(
        self, session_id: str, directory: str, dispatch: OpenChamberDispatch
    ) -> bool:
        """Faithful busy-period pending check: reads the current session
        messages (same as the real client) and scans for any pending
        tool/permission part."""
        try:
            messages = self.messages(session_id, directory)
        except OpenChamberUnavailableError:
            return False
        for message in messages:
            for part in message.get("parts") or []:
                if part.get("type") not in ("tool", "permission"):
                    continue
                state = part.get("state")
                if isinstance(state, dict) and state.get("status") == "pending":
                    return True
        return False

    def close(self) -> None:
        pass