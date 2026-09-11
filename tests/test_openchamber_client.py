"""OpenChamberClient behavior against the verified local API shapes."""

from __future__ import annotations

import os

import pytest
import requests

from core.openchamber import (
    ModelRef,
    OpenChamberAuthError,
    OpenChamberBadRequestError,
    OpenChamberClient,
    OpenChamberModelRequestRejectedError,
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

MESSAGE_PATH = "/api/session/ses_1/message?directory=D%3A%2Fp"
STATUS_PATH = "/api/session/status?directory=D%3A%2Fp"


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


SEND_PATH = "/api/openchamber/sessions/ses_1/send"


def _send_rejecting_http(detail: str) -> FakeHttp:
    http = FakeHttp()
    http.route("GET", MESSAGE_PATH, FakeResponse(200, []))
    http.route("POST", SEND_PATH, FakeResponse(400, {"error": detail}))
    return http


def test_http_400_is_retryable_false_is_model_rejection():
    """statusCode=400 + isRetryable=false is classified by the STRUCTURED
    fields into a dedicated model-rejection error that keeps status_code,
    is_retryable and request_url."""
    detail = (
        "APIError\nstatusCode: 400\nisRetryable: false\nresponseBody: \n"
        "url: http://192.168.100.190:8080/v1/chat/completions"
    )
    client = make_client(_send_rejecting_http(detail))
    with pytest.raises(OpenChamberModelRequestRejectedError) as ei:
        client.send("ses_1", "do it", "D:/p")
    exc = ei.value
    assert exc.status_code == 400
    assert exc.is_retryable is False
    assert exc.request_url == "http://192.168.100.190:8080/v1/chat/completions"
    assert "statusCode: 400" in str(exc)


def test_http_400_is_retryable_true_stays_bad_request():
    """A structured isRetryable=true 400 is a plain retryable bad request,
    never a model rejection."""
    detail = (
        "APIError\nstatusCode: 400\nisRetryable: true\n"
        "url: http://192.168.100.190:8080/v1/chat/completions"
    )
    client = make_client(_send_rejecting_http(detail))
    with pytest.raises(OpenChamberBadRequestError) as ei:
        client.send("ses_1", "do it", "D:/p")
    assert not isinstance(ei.value, OpenChamberModelRequestRejectedError)


def test_http_400_without_upstream_fields_is_not_blindly_rejected():
    """An ordinary 400 with NO structured upstream fields stays a plain bad
    request — the relay must never downgrade every 400 to a model rejection."""
    client = make_client(_send_rejecting_http("bad prompt contents"))
    with pytest.raises(OpenChamberBadRequestError) as ei:
        client.send("ses_1", "do it", "D:/p")
    assert not isinstance(ei.value, OpenChamberModelRequestRejectedError)


def test_http_400_conservative_fallback_requires_full_upstream_signature():
    """Without the isRetryable field, a 400 is only a model rejection when
    the whole upstream APIError signature (marker + statusCode + url) is
    present; a partial signature stays a plain bad request."""
    full = "APIError\nstatusCode: 400\nurl: http://192.168.100.190:8080/v1/chat/completions"
    client = make_client(_send_rejecting_http(full))
    with pytest.raises(OpenChamberModelRequestRejectedError):
        client.send("ses_1", "do it", "D:/p")

    partial = "statusCode: 400\nurl: http://192.168.100.190:8080/v1/chat/completions"
    client2 = make_client(_send_rejecting_http(partial))
    with pytest.raises(OpenChamberBadRequestError):
        client2.send("ses_1", "do it", "D:/p")


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


def test_send_posts_prompt_parses_real_shape_and_snapshots_first():
    """The local 1.22.2 send response carries no baseline message id; the
    client must snapshot the session's message ids BEFORE the POST."""
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
                "model": {"providerID": "9router-new", "modelID": "9auto"},
                "agent": "orchestrator",
                "promptDispatched": True,
                "dispatchedAsCommand": False,
            },
        ),
    )
    http.route(
        "GET",
        MESSAGE_PATH,
        FakeResponse(
            200,
            [
                user_message("u1", "old task", 500),
                assistant_message(
                    "a1", 600, completed=700, finish="stop",
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
    # no baseline field in the response -> attribution relies on the
    # pre-send snapshot
    assert dispatch.user_message_id is None
    assert dispatch.pre_send_message_ids == frozenset({"u1", "a1"})
    assert dispatch.pre_send_snapshot_ok is True
    assert dispatch.resolved_model == ModelRef("9router-new", "9auto")
    assert dispatch.requested_model == ModelRef("4090", "qwen3.8-27b")
    assert dispatch.agent == "orchestrator"
    assert dispatch.prompt_dispatched is True

    # the snapshot (GET) strictly precedes the send (POST)
    paths = [call[1] for call in http.calls]
    assert paths.index(MESSAGE_PATH) < paths.index(
        "/api/openchamber/sessions/ses_1/send"
    )
    method, path, body = http.calls[-1]
    assert method == "POST"
    assert body["prompt"] == "do it"
    assert body["directory"] == "D:/p"
    assert body["agent"] == "build"
    assert body["model"] == "4090/qwen3.8-27b"


def test_send_snapshot_failure_aborts_before_sending():
    """A failed pre-send snapshot must STOP the send: with no trustworthy
    snapshot the round cannot be attributed, so guessing is forbidden and
    the task is never dispatched."""
    http = FakeHttp()
    http.route(
        "GET", MESSAGE_PATH, FakeResponse(500, {"error": "snapshot failed"})
    )
    http.route(
        "POST",
        "/api/openchamber/sessions/ses_1/send",
        FakeResponse(
            200,
            {
                "model": {"providerID": "9router-new", "modelID": "9auto"},
                "agent": "orchestrator",
                "promptDispatched": True,
            },
        ),
    )
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="was NOT sent"):
        client.send("ses_1", "do it", "D:/p")
    assert [call for call in http.calls if call[0] == "POST"] == []


def test_send_user_message_id_parsed_when_present():
    http = FakeHttp()
    # pre-send snapshot must be readable for the send to proceed
    http.route("GET", MESSAGE_PATH, FakeResponse(200, []))
    http.route(
        "POST",
        "/api/openchamber/sessions/ses_1/send",
        FakeResponse(
            200,
            {
                "model": {"providerID": "9router-new", "modelID": "9auto"},
                "promptDispatched": True,
                "userMessageId": "msg_u_new",
            },
        ),
    )
    client = make_client(http)
    dispatch = client.send("ses_1", "do it", "D:/p")
    assert dispatch.user_message_id == "msg_u_new"


def test_send_without_dispatched_prompt_fails():
    http = FakeHttp()
    # pre-send snapshot must be readable for the send to proceed
    http.route("GET", MESSAGE_PATH, FakeResponse(200, []))
    http.route(
        "POST",
        "/api/openchamber/sessions/ses_1/send",
        FakeResponse(200, {"promptDispatched": False, "promptError": "model offline"}),
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
        STATUS_PATH,
        FakeResponse(
            200, {"ses_1": {"type": "busy"}, "ses_2": {"type": "idle"}}
        ),
    )
    client = make_client(http)
    assert client.session_status("ses_1", "D:/p") == "busy"
    assert client.session_status("ses_2", "D:/p") == "idle"
    # a missing session id means idle: OpenCode's SessionStatus service
    # deletes sessions from the map exactly when they become idle
    assert client.session_status("ses_missing", "D:/p") == "idle"


def test_session_status_empty_map_is_idle():
    http = FakeHttp()
    http.route("GET", STATUS_PATH, FakeResponse(200, {}))
    client = make_client(http)
    assert client.session_status("ses_1", "D:/p") == "idle"


def test_session_status_unknown_type_stays_unknown():
    http = FakeHttp()
    http.route(
        "GET",
        STATUS_PATH,
        FakeResponse(200, {"ses_1": {"type": "paused"}}),
    )
    client = make_client(http)
    assert client.session_status("ses_1", "D:/p") == "unknown"


def test_session_status_null_entry_is_unknown():
    """A session id PRESENT in the map with a null value is malformed data
    (issue: the old code reported `idle`); only a MISSING id means idle
    per OpenCode's SessionStatus delete-on-idle semantics."""
    http = FakeHttp()
    http.route(
        "GET",
        STATUS_PATH,
        FakeResponse(200, {"ses_1": None}),
    )
    client = make_client(http)
    assert client.session_status("ses_1", "D:/p") == "unknown"
    # a session that the status map no longer lists is still idle
    assert client.session_status("ses_2", "D:/p") == "idle"


def test_session_status_malformed_payload_is_unknown():
    http = FakeHttp()
    http.route("GET", STATUS_PATH, FakeResponse(200, ["not", "a", "map"]))
    client = make_client(http)
    assert client.session_status("ses_1", "D:/p") == "unknown"


def test_messages_non_list_fails():
    http = FakeHttp()
    http.route(
        "GET",
        MESSAGE_PATH,
        FakeResponse(200, {"unexpected": True}),
    )
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="not a list"):
        client.messages("ses_1", "D:/p")


# ---------------------------------------------------------------------- #
# existing-session listing (fixed-session candidates, never auto-create)
# ---------------------------------------------------------------------- #


def test_list_sessions_directory_filtered_parses_ids_and_titles():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session?directory=D%3A%2Fp",
        FakeResponse(
            200,
            [
                {"id": "ses_1", "directory": "D:/p", "title": "One"},
                {"id": "ses_2", "directory": "D:/p", "title": ""},
                {"id": ""},  # malformed -> skipped
                {"title": "no id"},  # malformed -> skipped
            ],
        ),
    )
    client = make_client(http)
    assert client.list_sessions("D:/p") == [("ses_1", "One"), ("ses_2", "")]


def test_list_sessions_unfiltered():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session",
        FakeResponse(200, [{"id": "ses_1", "title": "A"}]),
    )
    client = make_client(http)
    assert client.list_sessions() == [("ses_1", "A")]
    assert client.list_sessions("   ") == [("ses_1", "A")]  # blank -> unfiltered


def test_list_sessions_non_list_fails():
    http = FakeHttp()
    http.route("GET", "/api/session", FakeResponse(200, {"x": 1}))
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="not a list"):
        client.list_sessions()


def test_session_exists_is_directory_membership():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session?directory=D%3A%2Fp",
        FakeResponse(200, [{"id": "ses_1", "title": "One"}]),
    )
    client = make_client(http)
    assert client.session_exists("ses_1", "D:/p") is True
    assert client.session_exists("ses_other", "D:/p") is False


def test_list_sessions_with_projects_returns_triples_unfiltered():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session",
        FakeResponse(
            200,
            [
                {"id": "ses_1", "directory": "D:/p", "title": "One"},
                {"id": "ses_2", "directory": None, "title": "Missing"},
            ],
        ),
    )
    client = make_client(http)
    assert client.list_sessions_with_projects() == [
        ("ses_1", "One", "D:/p"),
        ("ses_2", "Missing", ""),
    ]


def test_list_sessions_with_projects_filtered_quotes_directory():
    http = FakeHttp()
    http.route(
        "GET",
        "/api/session?directory=D%3A%2Fp",
        FakeResponse(200, [{"id": "ses_1", "directory": "D:/p", "title": "One"}]),
    )
    client = make_client(http)
    assert client.list_sessions_with_projects("D:/p") == [("ses_1", "One", "D:/p")]


def test_list_sessions_with_projects_non_list_fails():
    http = FakeHttp()
    http.route("GET", "/api/session", FakeResponse(200, {"x": 1}))
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="not a list"):
        client.list_sessions_with_projects()


def test_match_project_sessions_matches_normalized_paths():
    all_sessions = [
        ("ses_1", "One", r"D:\AIwork\跑跑卡丁车"),  # backslash form
        ("ses_2", "Two", "d:/aiwork/跑跑卡丁车/"),  # forward slash + trailing slash
        ("ses_3", "Other", "D:/aiwork/other"),
        ("ses_4", "NoDir", ""),
    ]
    matched = OpenChamberClient.match_project_sessions(
        "d:/AIWORK/跑跑卡丁车", all_sessions
    )
    assert matched == [("ses_1", "One"), ("ses_2", "Two")]


def test_match_project_sessions_blank_or_none_matches_nothing():
    all_sessions = [
        ("ses_1", "One", r"D:\AIwork\跑跑卡丁车"),
        ("ses_2", "NoDir", ""),
    ]
    assert OpenChamberClient.match_project_sessions("", all_sessions) == []
    assert OpenChamberClient.match_project_sessions("  ", all_sessions) == []


def test_extract_agent_model_sets_collects_agents_and_models():
    from core.openchamber import extract_agent_model_sets

    messages = [
        user_message("u1", "hello", 1000),
        assistant_message(
            "a1", 1100, completed=1200, finish="stop",
            parts=[text_part("ok")],
            model=ModelRef("opencode", "big-pickle"),
            agent="build",
            parent_id="u1",
        ),
        assistant_message(
            "a2", 1300, completed=1400, finish="stop",
            parts=[text_part("ok")],
            model=ModelRef("4090", "qwen3.8-27b"),
            agent="plan",
            parent_id="a1",
        ),
        assistant_message(
            "a3", 1500, completed=1600, finish="error",
            parts=[text_part("boom")],
            agent=None,
        ),
    ]
    agents, models = extract_agent_model_sets(messages)
    assert agents == {"build", "plan"}
    assert models == {"opencode/big-pickle", "4090/qwen3.8-27b"}


def test_extract_agent_model_sets_empty_for_no_info():
    from core.openchamber import extract_agent_model_sets

    assert extract_agent_model_sets([]) == (set(), set())
    assert extract_agent_model_sets([{"parts": []}]) == (set(), set())


def test_send_model_string_roundtrip_preserves_requested_and_resolved():
    """With the fix, send() posts model as the 'providerID/modelID' string
    OpenChamber's resolveRequestedModel() expects.  When the server reports
    the same model back, requested_model and resolved_model are identical."""
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
                "model": {"providerID": "4090", "modelID": "qwen3.8-27b"},
                "agent": "build",
                "promptDispatched": True,
                "dispatchedAsCommand": False,
            },
        ),
    )
    http.route(
        "GET",
        MESSAGE_PATH,
        FakeResponse(200, [user_message("u1", "old task", 500)]),
    )
    client = make_client(http)
    dispatch = client.send(
        "ses_1",
        "do it",
        "D:/p",
        agent="build",
        model=ModelRef("4090", "qwen3.8-27b"),
    )
    method, path, body = http.calls[-1]
    assert method == "POST"
    assert isinstance(body["model"], str)
    assert body["model"] == "4090/qwen3.8-27b"
    assert dispatch.requested_model == ModelRef("4090", "qwen3.8-27b")
    assert dispatch.resolved_model == ModelRef("4090", "qwen3.8-27b")


SNAPSHOT_PATH = "/api/permission-auto-accept"


def test_auto_accept_snapshot_parses_sessions():
    http = FakeHttp()
    http.route(
        "GET",
        SNAPSHOT_PATH,
        FakeResponse(
            200,
            {"sessions": {"ses_a": True, "ses_b": False, "ses_c": 1}, "revision": 3},
        ),
    )
    client = make_client(http)
    assert client.auto_accept_snapshot() == {
        "ses_a": True,
        "ses_b": False,
        "ses_c": True,
    }


def test_auto_accept_snapshot_without_sessions_fails():
    http = FakeHttp()
    http.route("GET", SNAPSHOT_PATH, FakeResponse(200, {"revision": 1}))
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="no sessions"):
        client.auto_accept_snapshot()


def test_auto_accept_snapshot_unavailable():
    http = FakeHttp()
    http.raise_next = requests.ConnectionError("connection refused")
    client = make_client(http)
    with pytest.raises(OpenChamberUnavailableError, match="cannot reach"):
        client.auto_accept_snapshot()


def test_set_session_auto_accept_puts_quoted_id_and_directory():
    http = FakeHttp()
    http.route(
        "PUT",
        "/api/permission-auto-accept/sessions/ses_a%20b",
        FakeResponse(200, {"sessions": {"ses_a b": True}}),
    )
    client = make_client(http)
    result = client.set_session_auto_accept("ses_a b", True, "D:/p")
    assert result == {"ses_a b": True}
    assert (
        "PUT",
        "/api/permission-auto-accept/sessions/ses_a%20b",
        {"enabled": True, "directory": "D:/p"},
    ) in http.calls


def test_set_session_auto_accept_without_directory_omits_it():
    http = FakeHttp()
    http.route(
        "PUT",
        "/api/permission-auto-accept/sessions/ses_1",
        FakeResponse(200, {"sessions": {"ses_1": False}}),
    )
    client = make_client(http)
    result = client.set_session_auto_accept("ses_1", False)
    assert result == {"ses_1": False}
    assert ("PUT", "/api/permission-auto-accept/sessions/ses_1", {"enabled": False}) in http.calls


def test_set_session_auto_accept_empty_id_fails():
    client = make_client(FakeHttp())
    with pytest.raises(OpenChamberSessionError, match="must not be empty"):
        client.set_session_auto_accept("", True)


def test_set_session_auto_accept_without_sessions_fails():
    http = FakeHttp()
    http.route(
        "PUT",
        "/api/permission-auto-accept/sessions/ses_1",
        FakeResponse(200, {"raw": "ok"}),
    )
    client = make_client(http)
    with pytest.raises(OpenChamberSessionError, match="no sessions"):
        client.set_session_auto_accept("ses_1", True, "D:/p")