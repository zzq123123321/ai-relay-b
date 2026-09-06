"""Completion detection: only a fully verified round counts as success."""

from __future__ import annotations

import pytest

from core.openchamber import (
    ModelRef,
    OpenChamberSessionError,
    OpenChamberTimeoutError,
    wait_for_completion,
)
from tests.fakes import (
    ScriptedOpenChamber,
    assistant_message,
    make_dispatch,
    permission_part,
    question_part,
    text_part,
    tool_part,
    user_message,
)

# History of an EARLIER round (other task) in the same session.
HISTORY = [
    user_message("u1", "old task", 100),
    assistant_message(
        "a1", 200, completed=300, finish="stop",
        parts=[text_part("old reply")], parent_id="u1",
    ),
]
HISTORY_IDS = frozenset({"u1", "a1"})

MODEL = ModelRef("9router-new", "9auto")


def run(fake: ScriptedOpenChamber, dispatch, timeout: float = 5.0):
    return wait_for_completion(
        fake, dispatch, timeout, poll_interval=0.01, grace_seconds=0.05
    )


def new_round_dispatch(**kwargs):
    """Dispatch for a session whose pre-send snapshot knew the history."""
    kwargs.setdefault("pre_ids", HISTORY_IDS)
    kwargs.setdefault("requested", MODEL)
    kwargs.setdefault("resolved", MODEL)
    return make_dispatch(directory="D:/proj", **kwargs)


def test_success_after_busy_round_with_tool_call():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["busy", "busy", "idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, parts=[tool_part("bash", "completed", "relay-tool-check")],
                model=MODEL, parent_id="u2",
            ),
            assistant_message(
                "a3", 1200, completed=1300, finish="stop",
                parts=[text_part("relay-tool-check")], model=MODEL, parent_id="u2",
            ),
        ]
    ]
    result = run(fake, new_round_dispatch())
    assert result.final_text == "relay-tool-check"
    assert result.tool_calls == ("bash",)
    assert result.actual_model == MODEL
    assert result.model_mismatch is False
    assert result.finish == "stop"


def test_initial_idle_without_new_user_message_is_never_completion():
    fake = ScriptedOpenChamber()
    fake.status_timeline = []  # idle (session id missing from the status map)
    fake.message_timelines = [list(HISTORY)]
    with pytest.raises(OpenChamberSessionError, match="never recorded"):
        run(fake, new_round_dispatch())


def test_idle_with_user_but_no_assistant_reply_is_not_completion():
    fake = ScriptedOpenChamber()
    fake.status_timeline = []
    fake.message_timelines = [HISTORY + [user_message("u2", "new task", 1000)]]
    with pytest.raises(OpenChamberSessionError, match="no assistant reply"):
        run(fake, new_round_dispatch())


def test_truncated_finish_is_failure():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="length",
                parts=[text_part("partial answer")],
            ),
        ]
    ]
    with pytest.raises(OpenChamberSessionError, match="truncated"):
        run(fake, new_round_dispatch())


def test_run_error_is_failure_with_detail():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2",
                1100,
                completed=1200,
                error={"name": "ProviderError", "message": "provider down"},
                parts=[text_part("")],
            ),
        ]
    ]
    with pytest.raises(OpenChamberSessionError, match="provider down"):
        run(fake, new_round_dispatch())


def test_indefinite_busy_times_out():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["busy"]
    fake.message_timelines = [list(HISTORY)]
    with pytest.raises(OpenChamberTimeoutError, match="session"):
        run(fake, new_round_dispatch(), timeout=0.1)


def test_model_mismatch_detected_between_actual_and_resolved():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    requested = ModelRef("4090", "qwen3.8-27b")
    resolved = ModelRef("9router-new", "9auto")
    other = ModelRef("volcengine", "doubao-x")
    dispatch = make_dispatch(
        directory="D:/proj",
        pre_ids=HISTORY_IDS,
        requested=requested,
        resolved=resolved,
    )
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("done")], model=resolved,
            ),
        ]
    ]
    result = run(fake, dispatch)
    assert result.model_mismatch is False  # actual == resolved

    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("done")], model=other,
            ),
        ]
    ]
    fake._status_index = 0
    fake._message_index = 0
    result2 = run(fake, dispatch)
    assert result2.model_mismatch is True
    assert result2.actual_model == other


def test_final_text_falls_back_to_earlier_assistant_message():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, completed=1150, finish="tool-calls",
                parts=[tool_part("bash", "completed", "x"), text_part("part1")],
            ),
            assistant_message(
                "a3", 1200, completed=1300, finish="stop",
                parts=[tool_part("write", "completed", "y")],
            ),
        ]
    ]
    result = run(fake, new_round_dispatch())
    assert result.final_text == "part1"
    assert result.tool_calls == ("bash", "write")


# ---------------------------------------------------------------------- #
# round association: message ids, not time windows
# ---------------------------------------------------------------------- #


def test_manually_inserted_message_makes_round_ambiguous():
    """Two NEW user messages (ours + one typed manually) -> never guess."""
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            user_message("u_manual", "人工插入的消息", 1001),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    with pytest.raises(OpenChamberSessionError, match="ambiguous"):
        run(fake, new_round_dispatch())


def test_manual_message_before_send_is_history_not_round():
    """A manual message already in the pre-send snapshot is history."""
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u_manual", "人工插入的消息", 900),
            assistant_message(
                "a_manual", 950, completed=990, finish="stop",
                parts=[text_part("manual reply")], parent_id="u_manual",
            ),
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    dispatch = new_round_dispatch(
        pre_ids=HISTORY_IDS | frozenset({"u_manual", "a_manual"})
    )
    result = run(fake, dispatch)
    assert result.final_text == "final"  # not the manual round's reply


def test_same_message_timestamps_still_attributed_by_id():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1000, completed=1000, finish="stop",  # identical time
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    result = run(fake, new_round_dispatch())
    assert result.final_text == "final"


def test_reordered_message_list_still_attributed_by_id():
    """The assistant message is returned BEFORE its user message."""
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
            user_message("u2", "relay task", 1000),
        ]
    ]
    result = run(fake, new_round_dispatch())
    assert result.final_text == "final"


def test_assistant_attached_to_other_user_message_is_ambiguous():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u1",  # points at history!
            ),
        ]
    ]
    with pytest.raises(OpenChamberSessionError, match="ambiguous"):
        run(fake, new_round_dispatch())


def test_history_reply_never_used_for_new_round():
    """Even though history has a finished stop round, the new round must
    produce its own result or the task fails."""
    fake = ScriptedOpenChamber()
    fake.status_timeline = []  # idle
    fake.message_timelines = [list(HISTORY)]
    with pytest.raises(OpenChamberSessionError, match="never recorded"):
        run(fake, new_round_dispatch())


def test_user_message_id_from_send_response_wins():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    # server-reported user message id is authoritative over the snapshot
    dispatch = new_round_dispatch(user_message_id="u2", pre_ids=frozenset())
    result = run(fake, dispatch)
    assert result.final_text == "final"


# ---------------------------------------------------------------------- #
# strict completion: status semantics
# ---------------------------------------------------------------------- #


def test_missing_status_id_means_idle_and_completes():
    """An empty status map (session id absent) is idle per OpenCode's
    SessionStatus service, so a fully verified round completes."""
    fake = ScriptedOpenChamber()
    fake.status_timeline = []  # -> "idle" (missing from map)
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    result = run(fake, new_round_dispatch())
    assert result.final_text == "final"


def test_unknown_status_type_is_never_success():
    """An unrecognized status type must not be converted into idle."""
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["paused"]  # not busy/retry/idle
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    with pytest.raises(OpenChamberTimeoutError):
        run(fake, new_round_dispatch(), timeout=0.2)


def test_status_interface_failure_is_never_success():
    """Interface failures keep the relay waiting; the round may be perfect
    but without a confirmed idle status it must not complete."""
    fake = ScriptedOpenChamber()
    fake.unavailable = True
    fake.status_timeline = []
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("final")], parent_id="u2",
            ),
        ]
    ]
    with pytest.raises(OpenChamberTimeoutError):
        run(fake, new_round_dispatch(), timeout=0.2)


# ---------------------------------------------------------------------- #
# questions and permission prompts: wait, never auto-answer
# ---------------------------------------------------------------------- #


def test_pending_question_keeps_waiting_then_completes_after_answer():
    """Model asks a question -> relay waits (does not fail, does not
    answer); the user answers in the OpenChamber UI -> the round continues
    and completes with the continuation's final text."""
    fake = ScriptedOpenChamber()
    # busy while the model runs; the question goes pending while the
    # session leaves the status map (idle); the user's answer re-triggers
    # busy; then idle with the final message.
    fake.status_timeline = ["busy", "idle", "idle", "busy", "idle"]
    pending_round = (
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100,
                parts=[question_part("pending")],  # waiting for the user
            ),
        ]
    )
    answered_round = (
        HISTORY
        + [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2",
                1100,
                completed=1200,
                finish="tool-calls",
                parts=[question_part("completed")],
            ),
            assistant_message(
                "a3",
                1201,
                completed=1300,
                finish="stop",
                parts=[text_part("final after answer")],
                parent_id="u2",
            ),
        ]
    )
    # messages() is only called on the idle polls:
    # idle#1 -> pending, idle#2 -> pending, busy (no call), idle#3 -> done
    fake.message_timelines = [pending_round, pending_round, answered_round]
    result = run(fake, new_round_dispatch())
    assert result.final_text == "final after answer"
    assert result.tool_calls == ("question",)


def test_unanswered_question_times_out_instead_of_failing_or_guessing():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    fake.message_timelines = [
        [
            user_message("u2", "relay task", 1000),
            assistant_message(
                "a2", 1100, parts=[question_part("pending")]
            ),
        ]
    ]
    with pytest.raises(OpenChamberTimeoutError, match="ses_test123"):
        run(fake, new_round_dispatch(pre_ids=frozenset()), timeout=0.3)


def test_pending_permission_prompt_keeps_waiting():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["busy", "idle", "idle", "busy", "idle"]
    pending_round = [
        user_message("u2", "relay task", 1000),
        assistant_message(
            "a2", 1100,
            parts=[tool_part("bash", "pending"), permission_part("pending")],
        ),
    ]
    answered_round = [
        user_message("u2", "relay task", 1000),
        assistant_message(
            "a2", 1100, completed=1200, finish="tool-calls",
            parts=[tool_part("bash", "completed", "ok")],
        ),
        assistant_message(
            "a3", 1201, completed=1300, finish="stop",
            parts=[text_part("final")], parent_id="u2",
        ),
    ]
    fake.message_timelines = [pending_round, pending_round, answered_round]
    dispatch = new_round_dispatch(pre_ids=frozenset())
    result = run(fake, dispatch)
    assert result.final_text == "final"