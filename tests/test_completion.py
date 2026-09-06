"""Completion detection: only a fully verified round counts as success."""

from __future__ import annotations

import pytest

from core.openchamber import (
    ModelRef,
    OpenChamberSessionError,
    OpenChamberTimeoutError,
    OpenChamberUserActionRequired,
    wait_for_completion,
)
from tests.fakes import (
    ScriptedOpenChamber,
    assistant_message,
    make_dispatch,
    question_part,
    text_part,
    tool_part,
    user_message,
)

HISTORY = [
    user_message("u1", "old task", 100),
    assistant_message(
        "a1", 200, completed=300, finish="stop", parts=[text_part("old reply")]
    ),
]


def run(fake: ScriptedOpenChamber, dispatch, timeout: float = 5.0):
    return wait_for_completion(
        fake, dispatch, timeout, poll_interval=0.01, grace_seconds=0.05
    )


def test_success_after_busy_round_with_tool_call():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["busy", "busy", "idle"]
    model = ModelRef("9router-new", "9auto")
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, parts=[tool_part("bash", "completed", "relay-tool-check")],
                model=model,
            ),
            assistant_message(
                "a3", 1200, completed=1300, finish="stop",
                parts=[text_part("relay-tool-check")], model=model,
            ),
        ]
    ]
    dispatch = make_dispatch(
        directory=fake.directory,
        baseline_id="a1",
        baseline_time=300,
        requested=model,
        resolved=model,
        sent_at_ms=1000,
    )
    result = run(fake, dispatch)
    assert result.final_text == "relay-tool-check"
    assert result.tool_calls == ("bash",)
    assert result.actual_model == model
    assert result.model_mismatch is False
    assert result.finish == "stop"


def test_initial_idle_without_new_user_message_is_never_completion():
    fake = ScriptedOpenChamber()
    fake.status_timeline = []  # idle from the start
    fake.message_timelines = [list(HISTORY)]
    dispatch = make_dispatch(
        directory=fake.directory,
        baseline_id="a1",
        baseline_time=300,
        sent_at_ms=999999,
    )
    with pytest.raises(OpenChamberSessionError, match="never recorded"):
        run(fake, dispatch)


def test_idle_with_user_but_no_assistant_reply_is_not_completion():
    fake = ScriptedOpenChamber()
    fake.status_timeline = []
    fake.message_timelines = [HISTORY + [user_message("u2", "new task", 1000)]]
    dispatch = make_dispatch(
        directory=fake.directory,
        baseline_id="a1",
        baseline_time=300,
        sent_at_ms=1000,
    )
    with pytest.raises(OpenChamberSessionError, match="no assistant reply"):
        run(fake, dispatch)


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
    dispatch = make_dispatch(
        directory=fake.directory, baseline_id="a1", baseline_time=300, sent_at_ms=1000
    )
    with pytest.raises(OpenChamberSessionError, match="truncated"):
        run(fake, dispatch)


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
    dispatch = make_dispatch(
        directory=fake.directory, baseline_id="a1", baseline_time=300, sent_at_ms=1000
    )
    with pytest.raises(OpenChamberSessionError, match="provider down"):
        run(fake, dispatch)


def test_question_part_defers_to_user():
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
                finish="stop",
                parts=[question_part(), text_part("")],
            ),
        ]
    ]
    dispatch = make_dispatch(
        directory=fake.directory, baseline_id="a1", baseline_time=300, sent_at_ms=1000
    )
    with pytest.raises(OpenChamberUserActionRequired, match="OpenChamber"):
        run(fake, dispatch)


def test_indefinite_busy_times_out():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["busy"]
    fake.message_timelines = [list(HISTORY)]
    dispatch = make_dispatch(
        directory=fake.directory, baseline_id="a1", baseline_time=300, sent_at_ms=1000
    )
    with pytest.raises(OpenChamberTimeoutError, match="session"):
        run(fake, dispatch, timeout=0.1)


def test_model_mismatch_detected_between_actual_and_resolved():
    fake = ScriptedOpenChamber()
    fake.status_timeline = ["idle"]
    requested = ModelRef("4090", "qwen3.8-27b")
    resolved = ModelRef("9router-new", "9auto")
    actual = ModelRef("9router-new", "9auto")
    fake.message_timelines = [
        HISTORY
        + [
            user_message("u2", "new task", 1000),
            assistant_message(
                "a2", 1100, completed=1200, finish="stop",
                parts=[text_part("done")], model=actual,
            ),
        ]
    ]
    dispatch = make_dispatch(
        directory=fake.directory,
        baseline_id="a1",
        baseline_time=300,
        requested=requested,
        resolved=resolved,
        sent_at_ms=1000,
    )
    result = run(fake, dispatch)
    assert result.model_mismatch is False  # actual == resolved
    # Now a genuinely different actual model:
    other = ModelRef("volcengine", "doubao-x")
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
    dispatch = make_dispatch(
        directory=fake.directory, baseline_id="a1", baseline_time=300, sent_at_ms=1000
    )
    result = run(fake, dispatch)
    assert result.final_text == "part1"
    assert result.tool_calls == ("bash", "write")


def test_unknown_baseline_never_uses_history_reply():
    fake = ScriptedOpenChamber()
    fake.status_timeline = []  # idle
    fake.message_timelines = [list(HISTORY)]
    dispatch = make_dispatch(
        directory=fake.directory,
        baseline_id=None,
        baseline_time=None,
        sent_at_ms=999999,  # far in the future: history predates the window
    )
    with pytest.raises(OpenChamberSessionError, match="never recorded"):
        run(fake, dispatch)