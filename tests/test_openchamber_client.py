"""OpenChamberClient behavior against the verified local API shapes."""

from __future__ import annotations

import os

import pytest
import requests

from core.openchamber import (
    ModelRef,
    OpenChamberAuthError,
    OpenChamberSessionError,
    OpenChamberUnavailableError,
)
from tests.fakes import (
    FakeHttp,
    FakeResponse,
    assistant_message,
    make_client,
    text_part,
    user_message,
)


def test_verify_ok():
    http = FakeHttp()
    http.route("GET", "/health", FakeResponse(200, {"status": "ok"}))
    client = make_client(http)
    assert client.verify()["status"] == "ok"


def test_verify_unhealthy_status():
    http = FakeHttp()
    http.route("GET", "/health", FakeResponse(200, {"status": "degraded"}))
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="unhealthy"):
        client.verify()


def test_verify_unavailable():
    http = FakeHttp()
    http.raise_next = requests.ConnectionError("connection refused")
    client = make_client(http)
    with pytest.raises(OpenChamberUnavailableError, match="cannot reach"):
        client.verify()


def test_auth_error_401_tells_operator_to_configure_auth():
    http = FakeHttp()
    http.route("GET", "/health", FakeResponse(401, {"error": "unauthorized"}))
    client = make_client(http)
    with pytest.raises(OpenChamberAuthError, match="authentication"):
        client.verify()


def test_api_error_500_carries_server_message():
    http = FakeHttp()
    http.route(
        "POST", "/api/openchamber/sessions", FakeResponse(500, {"error": "boom"})
    )
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="boom"):
        client.create_session("t", "D:/p")


def test_create_session_posts_title_and_directory():
    http = FakeHttp()
    http.route(
        "POST",
        "/api/openchamber/sessions",
        FakeResponse(200, {"sessionId": "ses_1", "directory": "D:/p"}),
    )
    client = make_client(http)
    assert client.create_session("AI Relay abcd1234", "D:/p") == "ses_1"
    method, path, body = http.calls[-1]
    assert method == "POST"
    assert path == "/api/openchamber/sessions"
    assert body == {"title": "AI Relay abcd1234", "directory": "D:/p"}


def test_create_session_without_id_fails():
    http = FakeHttp()
    http.route("POST", "/api/openchamber/sessions", FakeResponse(200, {}))
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="sessionId"):
        client.create_session("t", "D:/p")


def test_open_session_dispatches_native_deeplink(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(os, "startfile", lambda uri: seen.append(uri))
    client = make_client(FakeHttp())
    client.open_session("ses_abc")
    assert seen == ["openchamber://session/ses_abc"]


def test_open_session_failure_raises(monkeypatch):
    def boom(_uri):
        raise OSError("no protocol handler")

    monkeypatch.setattr(os, "startfile", boom)
    client = make_client(FakeHttp())
    with pytest.raises(OpenChamberSessionError, match="deep link"):
        client.open_session("ses_abc")


def test_send_posts_prompt_and_parses_dispatch():
    http = FakeHttp()
    http.route(
        "POST",
        "/api/openchamber/sessions/ses_1/send",
        FakeResponse(
            200,
            {
                "action": "send",
                "sessionId": "ses_1",
                "directory": "D:/p",
                "baselineAssistantMessageId": "msg_base",
                "model": {"providerID": "9router-new", "modelID": "9auto"},
                "agent": "orchestrator",
                "promptDispatched": True,
                "dispatchedAsCommand": False,
            },
        ),
    )
    http.route(
        "GET",
        "/api/session/ses_1/message?directory=D%3A%2Fp",
        FakeResponse(
            200,
            [
                user_message("u1", "old task", 500),
                assistant_message(
                    "msg_base",
                    600,
                    completed=700,
                    finish="stop",
                    parts=[text_part("old reply")],
                ),
            ],
        ),
    )
    client = make_client(http)
    dispatch = client.send(
        "ses_1",
        "do it",
        "D:/p",
        agent="build",
        model=ModelRef("4090", "qwen3.8-27b"),
    )
    assert dispatch.baseline_message_id == "msg_base"
    assert dispatch.baseline_message_time == 600
    assert dispatch.resolved_model == ModelRef("9router-new", "9auto")
    assert dispatch.requested_model == ModelRef("4090", "qwen3.8-27b")
    assert dispatch.agent == "orchestrator"
    assert dispatch.prompt_dispatched is True

    method, path, body = http.calls[0]
    assert method == "POST"
    assert path == "/api/openchamber/sessions/ses_1/send"
    assert body["prompt"] == "do it"
    assert body["directory"] == "D:/p"
    assert body["agent"] == "build"
    assert body["model"] == {"providerID": "4090", "modelID": "qwen3.8-27b"}


def test_send_without_dispatched_prompt_fails():
    http = FakeHttp()
    http.route(
        "POST",
        "/api/openchamber/sessions/ses_1/send",
        FakeResponse(
            200, {"promptDispatched": False, "promptError": "model offline"}
        ),
    )
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="did not dispatch"):
        client.send("ses_1", "do it", "D:/p")


def test_send_empty_prompt_rejected():
    client = make_client(FakeHttp())
    with pytest.raises(OpenChamberSessionError, match="prompt"):
        client.send("ses_1", "   ", "D:/p")


def test_invalid_model_format_rejected():
    with pytest.raises(OpenChamberSessionError, match="providerID/modelID"):
        ModelRef.parse("only-one-part")
    assert ModelRef.parse("") is None
    assert ModelRef.parse(None) is None


def test_session_status_lookup():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session/status?directory=D%3A%2Fp",
        FakeResponse(
            200, {"ses_1": {"type": "busy"}, "ses_2": {"type": "idle"}}
        ),
    )
    client = make_client(http)
    assert client.session_status("ses_1", "D:/p") == "busy"
    assert client.session_status("ses_2", "D:/p") == "idle"
    assert client.session_status("ses_missing", "D:/p") == "unknown"


def test_messages_non_list_fails():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session/ses_1/message?directory=D%3A%2Fp",
        FakeResponse(200, {"unexpected": True}),
    )
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="not a list"):
        client.messages("ses_1", "D:/p")