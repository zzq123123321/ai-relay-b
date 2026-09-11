"""SessionRotation counting, reset rules and rotation orchestration."""

from __future__ import annotations

import json

import pytest

from core.openchamber import OpenChamberError, OpenChamberSessionError
from core.relay import directory_key
from core.relay_settings import RelaySettings, RelaySettingsError
from core.session_rotation import (
    DEFAULT_ROTATION_BASE_TITLE,
    SessionRotation,
    rotation_sequence_label,
    strip_rotation_suffix,
)
from tests.fakes import FakeHttp, FakeResponse, make_client

CREATE_PATH = "/api/openchamber/sessions"
SNAPSHOT_PATH = "/api/permission-auto-accept"


def make_rotation(settings=None, openchamber=None) -> SessionRotation:
    return SessionRotation(settings or RelaySettings(), openchamber)


def test_disabled_rotation_never_counts():
    rotation = make_rotation(
        RelaySettings(auto_rotate_enabled=False, auto_rotate_threshold=2)
    )
    assert rotation.note_auto_success("D:/proj") is False
    assert rotation.count("D:/proj") == 0


def test_counts_only_auto_successes_until_threshold():
    settings = RelaySettings(auto_rotate_enabled=True, auto_rotate_threshold=2)
    rotation = make_rotation(settings)
    assert rotation.note_auto_success("D:/proj") is False
    assert rotation.note_auto_success("D:/proj") is True
    # milestone kept: further successes keep reporting the threshold reached
    assert rotation.note_auto_success("D:/proj") is True
    assert rotation.count("D:/proj") == 3


def test_count_is_per_directory():
    settings = RelaySettings(auto_rotate_enabled=True, auto_rotate_threshold=2)
    rotation = make_rotation(settings)
    rotation.note_auto_success("D:/proj")
    assert rotation.count("D:/proj") == 1
    assert rotation.count("D:/other") == 0
    rotation.note_auto_success("D:/other")
    assert rotation.count("D:/other") == 1
    # resetting one directory leaves the other untouched
    rotation.reset("D:/proj")
    assert rotation.count("D:/proj") == 0
    assert rotation.count("D:/other") == 1


def test_reset_all_clears_every_directory():
    settings = RelaySettings(auto_rotate_enabled=True, auto_rotate_threshold=1)
    rotation = make_rotation(settings)
    rotation.note_auto_success("D:/a")
    rotation.note_auto_success("D:/b")
    rotation.reset_all()
    assert rotation.count("D:/a") == 0
    assert rotation.count("D:/b") == 0


def _rotate_http(new_id: str = "ses_test123") -> FakeHttp:
    http = FakeHttp()
    http.route("POST", CREATE_PATH, FakeResponse(200, {"sessionId": new_id}))
    return http


def test_rotate_creates_persists_and_returns_new_session():
    http = _rotate_http("ses_rotated1")
    client = make_client(http)
    settings = RelaySettings(auto_rotate_enabled=True, openchamber_directory="D:/proj")
    rotation = make_rotation(settings, openchamber=client)
    new_id = rotation.rotate("D:/proj", previous_session_id="ses_old")
    assert new_id == "ses_rotated1"
    key = directory_key("D:/proj")
    assert settings.openchamber_sessions[key] == "ses_rotated1"
    assert settings.openchamber_session_id == "ses_rotated1"
    # no persisted base name and no hint: the fallback base title is used
    # and the FIRST rotation is named "<base> - 第一个" (Chinese numeral,
    # never a digit sequence like "新会话-1").
    body = next(
        call[2] for call in http.calls if call[0] == "POST" and call[1] == CREATE_PATH
    )
    assert body["title"] == f"{DEFAULT_ROTATION_BASE_TITLE} - 第一个"
    assert body["directory"] == "D:/proj"
    # sequence and base title are committed (persisted in memory) only AFTER
    # the creation succeeded
    assert settings.auto_rotate_sequences[key] == 1
    assert settings.auto_rotate_base_titles[key] == DEFAULT_ROTATION_BASE_TITLE

def test_rotate_continues_persisted_sequence_after_sessions_deleted():
    """Earlier rotated sessions were DELETED from the service; the sequence
    must continue from the persisted value (never re-derived from the
    session list, never reuse 第一个/第二个)."""
    from tests.fakes import ScriptedOpenChamber

    client = ScriptedOpenChamber()
    client.next_session_id = "ses_rotated3"
    # the session list shows only the fixed session -- no rotated titles
    # to scan, so a list-scan implementation would restart at 第一个.
    client.sessions = [
        ("ses_old", "Task 14H", "D:/proj"),
    ]
    settings = RelaySettings(
        openchamber_directory="D:/proj",
        auto_rotate_sequences={directory_key("D:/proj"): 2},
        auto_rotate_base_titles={directory_key("D:/proj"): "Task 14H"},
    )
    rotation = make_rotation(settings, openchamber=client)

    rotation.rotate("D:/proj")

    assert "create:Task 14H - 第三个:D:/proj" in client.call_log
    key = directory_key("D:/proj")
    assert settings.auto_rotate_sequences[key] == 3
    assert settings.auto_rotate_base_titles[key] == "Task 14H"


def test_rotate_uses_current_session_title_then_persisted_base():
    """First rotation derives the base name from the current fixed
    session's title (via base_title_hint); every later rotation reuses the
    PERSISTED base even if the current title now carries a rotation
    suffix (no nested "Task 14H - 第一个 - 第二个" titles)."""
    from tests.fakes import ScriptedOpenChamber

    client = ScriptedOpenChamber()
    client.next_session_id = "ses_rot_a"
    rotation = make_rotation(
        RelaySettings(openchamber_directory="D:/proj"), openchamber=client
    )
    rotation.rotate("D:/proj", base_title_hint="Task 14H")
    assert "create:Task 14H - 第一个:D:/proj" in client.call_log
    key = directory_key("D:/proj")
    assert rotation.settings.auto_rotate_base_titles[key] == "Task 14H"

    # second rotation: the current session's title now carries the first
    # rotation's suffix -- the persisted base must win, not a re-derivation
    # that would nest suffixes.
    client.next_session_id = "ses_rot_b"
    rotation.rotate("D:/proj", base_title_hint="Task 14H - 第一个")
    assert "create:Task 14H - 第二个:D:/proj" in client.call_log
    assert rotation.settings.auto_rotate_sequences[key] == 2
    assert rotation.settings.auto_rotate_base_titles[key] == "Task 14H"


def test_rotate_hint_with_old_suffix_is_stripped():
    from tests.fakes import ScriptedOpenChamber

    client = ScriptedOpenChamber()
    client.next_session_id = "ses_rot_c"
    rotation = make_rotation(
        RelaySettings(openchamber_directory="D:/proj"), openchamber=client
    )
    # fresh settings (no persisted base) but the current session title
    # carries an old rotation suffix: the base must be the stripped title
    rotation.rotate("D:/proj", base_title_hint="Task 14H - 第二个")
    assert "create:Task 14H - 第一个:D:/proj" in client.call_log


def test_rotate_failed_creation_keeps_sequence_and_mappings():
    from tests.fakes import ScriptedOpenChamber

    client = ScriptedOpenChamber()
    client.next_session_id = "ses_never"
    client.create_error = OpenChamberSessionError("create exploded")
    settings = RelaySettings(
        openchamber_directory="D:/proj",
        openchamber_sessions={directory_key("D:/proj"): "ses_old"},
    )
    rotation = make_rotation(settings, openchamber=client)
    with pytest.raises(OpenChamberSessionError):
        rotation.rotate("D:/proj", base_title_hint="Task 14H")
    key = directory_key("D:/proj")
    # sequence and base title were NOT committed: the retry will still
    # create 第一个, and the session mapping is untouched
    assert settings.auto_rotate_sequences == {}
    assert settings.auto_rotate_base_titles == {}
    assert settings.openchamber_sessions[key] == "ses_old"
    assert settings.openchamber_session_id == ""


def test_rotation_sequence_persists_and_survives_reload(tmp_path):
    path = tmp_path / "relay_settings.json"
    settings = RelaySettings(openchamber_directory="D:/proj")
    settings._path = path
    client = make_client(_rotate_http("ses_new1"))
    rotation = make_rotation(settings, openchamber=client)
    rotation.rotate("D:/proj", base_title_hint="Task 14H")

    data = json.loads(path.read_text(encoding="utf-8"))
    key = directory_key("D:/proj")
    assert data["auto_rotate_sequences"][key] == 1
    assert data["auto_rotate_base_titles"][key] == "Task 14H"
    assert data["openchamber_sessions"][key] == "ses_new1"

    # a reloaded settings object (fresh process) must continue the SAME
    # sequence -- 第二个 -- not restart
    reloaded = RelaySettings.load(path)
    assert reloaded.auto_rotate_sequences[key] == 1
    assert reloaded.auto_rotate_base_titles[key] == "Task 14H"
    assert isinstance(reloaded.auto_rotate_sequences[key], int)
    assert isinstance(reloaded.auto_rotate_base_titles[key], str)

    client2 = make_client(_rotate_http("ses_new2"))
    rotation2 = make_rotation(reloaded, openchamber=client2)
    rotation2.rotate("D:/proj")
    data2 = json.loads(path.read_text(encoding="utf-8"))
    assert data2["auto_rotate_sequences"][key] == 2
    assert data2["openchamber_sessions"][key] == "ses_new2"


def test_rotation_sequence_label_covering_cases():
    # at least 1..100 must be exact Chinese numerals
    assert rotation_sequence_label(1) == "第一个"
    assert rotation_sequence_label(2) == "第二个"
    assert rotation_sequence_label(3) == "第三个"
    assert rotation_sequence_label(9) == "第九个"
    assert rotation_sequence_label(10) == "第十个"
    assert rotation_sequence_label(11) == "第十一个"
    assert rotation_sequence_label(19) == "第十九个"
    assert rotation_sequence_label(20) == "第二十个"
    assert rotation_sequence_label(21) == "第二十一个"
    assert rotation_sequence_label(99) == "第九十九个"
    assert rotation_sequence_label(100) == "第一百个"
    assert rotation_sequence_label(110) == "第一百一十个"
    for index in range(1, 101):
        label = rotation_sequence_label(index)
        assert label.startswith("第") and label.endswith("个")
    with pytest.raises(ValueError):
        rotation_sequence_label(0)
    with pytest.raises(ValueError):
        rotation_sequence_label(True)


def test_strip_rotation_suffix():
    assert strip_rotation_suffix("Task 14H - 第一个") == "Task 14H"
    assert strip_rotation_suffix("Task 14H - 第二个") == "Task 14H"
    assert strip_rotation_suffix("Task 14H - 第一百个") == "Task 14H"
    # no suffix: untouched (only trimmed)
    assert strip_rotation_suffix("  Task 14H  ") == "Task 14H"
    # a plain dash without the 第N个 grammar is NOT a rotation suffix
    assert strip_rotation_suffix("Task - 14H") == "Task - 14H"


def test_rotate_default_directory_syncs_session_id_only_for_default():
    http = _rotate_http()
    client = make_client(http)
    settings = RelaySettings(auto_rotate_enabled=True, openchamber_directory="D:/proj")
    rotation = make_rotation(settings, openchamber=client)
    # non-default project directory: mapping updated, default id untouched
    rotation.rotate("D:/other", previous_session_id="ses_old")
    assert settings.openchamber_sessions[directory_key("D:/other")] == "ses_test123"
    assert settings.openchamber_session_id == ""


def test_rotate_persists_settings_when_backed(tmp_path):
    path = tmp_path / "relay_settings.json"
    settings = RelaySettings(
        openchamber_directory="D:/proj", openchamber_session_id="ses_old"
    )
    settings._path = path
    client = make_client(_rotate_http("ses_new1"))
    rotation = make_rotation(settings, openchamber=client)
    rotation.rotate("D:/proj", previous_session_id="ses_old")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["openchamber_sessions"][directory_key("D:/proj")] == "ses_new1"
    assert data["openchamber_session_id"] == "ses_new1"


def test_rotate_inherits_previous_auto_accept_flag():
    http = _rotate_http("ses_new1")
    http.route(
        "GET",
        SNAPSHOT_PATH,
        FakeResponse(200, {"sessions": {"ses_old": True, "ses_other": False}}),
    )
    http.route(
        "PUT",
        "/api/permission-auto-accept/sessions/ses_new1",
        FakeResponse(200, {"sessions": {"ses_new1": True}}),
    )
    client = make_client(http)
    settings = RelaySettings(openchamber_directory="D:/proj")
    rotation = make_rotation(settings, openchamber=client)
    rotation.rotate(
        "D:/proj", previous_session_id="ses_old", inherit_auto_accept=True
    )
    assert (
        "PUT",
        "/api/permission-auto-accept/sessions/ses_new1",
        {"enabled": True, "directory": "D:/proj"},
    ) in http.calls


def test_rotate_inherit_defaults_off_when_old_session_missing():
    http = _rotate_http("ses_new1")
    http.route(
        "GET",
        SNAPSHOT_PATH,
        FakeResponse(200, {"sessions": {"ses_other": True}}),
    )
    http.route(
        "PUT",
        "/api/permission-auto-accept/sessions/ses_new1",
        FakeResponse(200, {"sessions": {"ses_new1": False}}),
    )
    client = make_client(http)
    settings = RelaySettings(openchamber_directory="D:/proj")
    rotation = make_rotation(settings, openchamber=client)
    rotation.rotate(
        "D:/proj", previous_session_id="ses_old", inherit_auto_accept=True
    )
    assert (
        "PUT",
        "/api/permission-auto-accept/sessions/ses_new1",
        {"enabled": False, "directory": "D:/proj"},
    ) in http.calls


def test_rotate_inherit_failure_does_not_abort_rotation():
    http = _rotate_http("ses_new1")
    http.route(
        "GET",
        SNAPSHOT_PATH,
        FakeResponse(500, {"error": "boom"}),
    )
    client = make_client(http)
    settings = RelaySettings(openchamber_directory="D:/proj")
    rotation = make_rotation(settings, openchamber=client)
    new_id = rotation.rotate(
        "D:/proj", previous_session_id="ses_old", inherit_auto_accept=True
    )
    assert new_id == "ses_new1"
    assert settings.openchamber_sessions[directory_key("D:/proj")] == "ses_new1"


def test_rotate_failure_leaves_settings_untouched():
    # no routes registered -> the create POST 404s
    client = make_client(FakeHttp())
    settings = RelaySettings(openchamber_directory="D:/proj")
    rotation = make_rotation(settings, openchamber=client)
    with pytest.raises(OpenChamberError):
        rotation.rotate("D:/proj")
    assert settings.openchamber_sessions == {}
    assert settings.openchamber_session_id == ""
    # a failed creation must not commit a sequence or a base title
    assert settings.auto_rotate_sequences == {}
    assert settings.auto_rotate_base_titles == {}


def test_settings_load_coerces_rotation_fields(tmp_path):
    path = tmp_path / "relay_settings.json"
    path.write_text(
        json.dumps(
            {
                "auto_rotate_enabled": True,
                "auto_rotate_threshold": 12,
                "auto_rotate_inherit_auto_accept": True,
            }
        ),
        encoding="utf-8",
    )
    settings = RelaySettings.load(path)
    assert settings.auto_rotate_enabled is True
    assert settings.auto_rotate_threshold == 12
    assert settings.auto_rotate_inherit_auto_accept is True


def test_settings_load_ignores_bool_as_threshold(tmp_path):
    path = tmp_path / "relay_settings.json"
    path.write_text(json.dumps({"auto_rotate_threshold": True}), encoding="utf-8")
    settings = RelaySettings.load(path)
    assert settings.auto_rotate_threshold == 5


def test_settings_load_coerces_mixed_dict_fields(tmp_path):
    """Both dict-valued fields share one coercion branch: string dicts
    (openchamber_sessions) keep str values, the int dict
    (auto_rotate_sequences) keeps ints, and non-string/non-numeric values
    are dropped."""
    path = tmp_path / "relay_settings.json"
    path.write_text(
        json.dumps(
            {
                "openchamber_sessions": {
                    "d1": "ses_a",
                    "d2": "7",  # numeric string stays a string
                },
                "auto_rotate_sequences": {
                    "d1": 2,
                    "d3": True,  # bool dropped
                    "d4": None,  # null dropped
                },
                "auto_rotate_base_titles": {"d1": "Task 14H"},
            }
        ),
        encoding="utf-8",
    )
    settings = RelaySettings.load(path)
    assert settings.openchamber_sessions == {"d1": "ses_a", "d2": "7"}
    assert settings.auto_rotate_sequences == {"d1": 2}
    assert isinstance(settings.auto_rotate_sequences["d1"], int)
    assert settings.auto_rotate_base_titles == {"d1": "Task 14H"}


def test_settings_validate_bounds_threshold():
    with pytest.raises(RelaySettingsError, match="auto_rotate_threshold"):
        RelaySettings(auto_rotate_threshold=0).validate()
    with pytest.raises(RelaySettingsError, match="auto_rotate_threshold"):
        RelaySettings(auto_rotate_threshold=101).validate()
    RelaySettings(auto_rotate_threshold=1).validate()
    RelaySettings(auto_rotate_threshold=100).validate()
