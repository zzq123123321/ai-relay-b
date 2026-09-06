"""RelayWorkflow routing, ordering, failure semantics and persistence."""

from __future__ import annotations

import pytest

from core.openchamber import (
    ModelRef,
    OpenChamberSessionError,
    OpenChamberTimeoutError,
)
from core.protocol import ProtocolFormat, parse_message
from core.relay import RelayWorkflow, resolve_executor_kind
from core.relay_settings import (
    TARGET_EXECUTOR,
    TARGET_OPENCHAMBER,
    TARGET_REASONIX,
    RelaySettings,
)
from core.task_registry import TaskRegistry
from tests.fakes import (
    ScriptedOpenChamber,
    assistant_message,
    make_dispatch,
    text_part,
    user_message,
)


class FakeReasonix:
    def __init__(self, reply: str = "reasonix-reply", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.executed: list[str] = []

    def self_check(self):
        return {"window": True, "composer": True, "composer_writable": True,
                "send_button": True}

    def execute(self, task: str) -> str:
        self.executed.append(task)
        if self.fail:
            raise RuntimeError("Reasonix window was not found")
        return f"{self.reply}: {task}"


def v1_task(target: str, body: str, message_id: str = "task-001") -> str:
    return "\n".join(
        (
            "AI_RELAY/1",
            f"MESSAGE_ID: {message_id}",
            "SOURCE: CHATGPT",
            f"TARGET: {target}",
            "TYPE: TASK",
            "ROUND: 1",
            "MAX_ROUNDS: 3",
            "",
            body,
        )
    )


def legacy_task(body: str, task_id: str = "legacy-001") -> str:
    return "\n".join(
        (
            "----- AI_RELAY_BEGIN -----",
            "SOURCE: CHATGPT",
            f"TARGET: {TARGET_REASONIX}",
            "TYPE: TASK",
            f"TASK_ID: {task_id}",
            "ROUND: 1",
            "MAX_ROUNDS: 3",
            "CONTENT:",
            body,
            "----- AI_RELAY_END -----",
        )
    )


def make_workflow(tmp_path, reasonix=None, oc=None, settings=None, default=TARGET_REASONIX):
    settings = settings or RelaySettings(
        default_target=default,
        openchamber_directory="D:/proj",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    reasonix = reasonix or FakeReasonix()
    return RelayWorkflow(
        reasonix=reasonix,
        registry=TaskRegistry(tmp_path / "tasks.json"),
        settings=settings,
        openchamber=oc,
        replies_dir=tmp_path / "replies",
    )


def scripted_oc(directory: str = "D:/proj") -> ScriptedOpenChamber:
    oc = ScriptedOpenChamber(directory=directory)
    oc.status_timeline = ["idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")],
                model=ModelRef("9router-new", "9auto"),
            ),
        ]
    ]
    return oc


# ---------------------------------------------------------------------- #
# routing
# ---------------------------------------------------------------------- #


def test_explicit_targets_map_to_themselves():
    settings = RelaySettings(default_target=TARGET_OPENCHAMBER)
    assert resolve_executor_kind(TARGET_REASONIX, settings) == TARGET_REASONIX
    assert resolve_executor_kind(TARGET_OPENCHAMBER, settings) == TARGET_OPENCHAMBER


def test_executor_target_follows_default():
    assert resolve_executor_kind(TARGET_EXECUTOR, RelaySettings()) == TARGET_REASONIX
    assert (
        resolve_executor_kind(
            TARGET_EXECUTOR, RelaySettings(default_target=TARGET_OPENCHAMBER)
        )
        == TARGET_OPENCHAMBER
    )


def test_explicit_target_not_overridden_by_default(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    response = wf.process(v1_task(TARGET_REASONIX, "hi"))
    assert "reasonix-reply" in response
    assert oc.call_log == []  # OpenChamber must not be touched


def test_route_reasonix_explicit(tmp_path):
    reasonix = FakeReasonix()
    wf = make_workflow(tmp_path, reasonix=reasonix)
    response = wf.process(v1_task(TARGET_REASONIX, "say hi"))
    assert reasonix.executed == ["say hi"]
    message = parse_message(response)
    assert message.protocol_format is ProtocolFormat.V1
    assert message.in_reply_to == "task-001"
    assert message.target == "CHATGPT"
    assert message.message_type.value == "RESPONSE"
    assert "say hi" in message.body


def test_route_reasonix_legacy_format(tmp_path):
    reasonix = FakeReasonix()
    wf = make_workflow(tmp_path, reasonix=reasonix)
    response = wf.process(legacy_task("say hi"))
    message = parse_message(response)
    assert message.protocol_format is ProtocolFormat.LEGACY_WEB
    assert message.message_id == "legacy-001"
    assert "----- AI_RELAY_BEGIN -----" in response


def test_route_openchamber_explicit(tmp_path):
    oc = scripted_oc()
    reasonix = FakeReasonix()
    wf = make_workflow(tmp_path, reasonix=reasonix, oc=oc)
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert reasonix.executed == []
    message = parse_message(response)
    assert message.body == "final answer"
    assert message.in_reply_to == "task-001"


def test_route_executor_default_openchamber(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    response = wf.process(v1_task(TARGET_EXECUTOR, "do it"))
    assert "final answer" in response
    assert any(call.startswith("send:") for call in oc.call_log)


def test_reasonix_offline_does_not_block_openchamber(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(
        tmp_path, reasonix=FakeReasonix(fail=True), oc=oc
    )
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert "final answer" in response


def test_unknown_target_rejected(tmp_path):
    wf = make_workflow(tmp_path)
    with pytest.raises(RuntimeError, match="not executable"):
        wf.process(v1_task("SOMEONE_ELSE", "x"))


def test_response_message_type_rejected(tmp_path):
    wf = make_workflow(tmp_path)
    with pytest.raises(RuntimeError, match="not a TASK"):
        wf.process(
            "AI_RELAY/1\nMESSAGE_ID: t1\nSOURCE: CHATGPT\nTARGET: REASONIX\n"
            "TYPE: RESPONSE\n\nhello"
        )


# ---------------------------------------------------------------------- #
# OpenChamber flow
# ---------------------------------------------------------------------- #


def test_openchamber_flow_order_create_open_send(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    kinds = [call.split(":")[0] for call in oc.call_log if call != "opened"]
    assert kinds == ["verify", "create", "open", "send"]
    # the deep-link request strictly precedes the send
    assert oc.call_log.index("open:ses_test123") < oc.call_log.index(
        "send:'do it'"
    )
    # the session was created WITHOUT a prompt
    create_call = next(c for c in oc.call_log if c.startswith("create:"))
    assert "do it" not in create_call


def test_openchamber_task_persisted_and_reply_stored(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "COMPLETED"
    assert record["executor"] == "OPENCHAMBER"
    assert record["session_id"] == "ses_test123"
    assert record["directory"] == "D:/proj"
    assert (tmp_path / "replies" / "task-001.response.txt").exists()
    assert wf.load_reply("task-001") == response
    assert wf.outcome is not None
    assert wf.outcome.session_id == "ses_test123"
    assert wf.outcome.executor == "OPENCHAMBER"


def test_deeplink_failure_prevents_send(tmp_path):
    oc = scripted_oc()
    oc.open_error = OpenChamberSessionError("no protocol handler")
    wf = make_workflow(tmp_path, oc=oc)
    with pytest.raises(RuntimeError, match="no protocol handler"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert not any(call.startswith("send:") for call in oc.call_log)
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_test123"  # kept for manual check
    assert not (tmp_path / "replies" / "task-001.response.txt").exists()


def test_send_failure_keeps_session_and_fails(tmp_path):
    oc = scripted_oc()
    oc.send_error = OpenChamberSessionError("prompt not dispatched")
    wf = make_workflow(tmp_path, oc=oc)
    with pytest.raises(RuntimeError, match="prompt not dispatched"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["session_id"] == "ses_test123"


def test_timeout_fails_and_keeps_session(tmp_path):
    oc = scripted_oc()
    oc.status_timeline = ["busy"]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        completion_timeout=0.05,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(OpenChamberTimeoutError, match="ses_test123"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert "timeout" in record["error"].lower() or "did not finish" in record["error"]
    assert record["session_id"] == "ses_test123"


def test_model_mismatch_reported_in_outcome(tmp_path):
    oc = scripted_oc()
    requested = ModelRef("4090", "qwen3.8-27b")
    oc.send_dispatch = make_dispatch(
        session_id="ses_test123",
        directory="D:/proj",
        baseline_id=None,
        baseline_time=None,
        requested=requested,
        resolved=requested,
        sent_at_ms=1000,
    )
    other = ModelRef("9router-new", "9auto")
    oc.message_timelines = [
        [
            user_message("u_new", "task body", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")], model=other,
            ),
        ]
    ]
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        openchamber_model="4090/qwen3.8-27b",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert wf.outcome is not None
    assert wf.outcome.note is not None
    assert "不一致" in wf.outcome.note


def test_missing_directory_fails_clearly(tmp_path):
    oc = scripted_oc()
    settings = RelaySettings(default_target=TARGET_REASONIX, openchamber_directory="  ")
    wf = make_workflow(tmp_path, oc=oc, settings=settings)
    with pytest.raises(RuntimeError, match="项目目录未配置"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert oc.call_log == []


def test_duplicate_task_rejected(tmp_path):
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    with pytest.raises(RuntimeError, match="already processed"):
        wf.process(v1_task(TARGET_OPENCHAMBER, "do it again"))
    assert sum(call.startswith("send:") for call in oc.call_log) == 1


def test_reasonix_failure_marks_failed(tmp_path):
    wf = make_workflow(tmp_path, reasonix=FakeReasonix(fail=True))
    with pytest.raises(RuntimeError, match="Reasonix window"):
        wf.process(v1_task(TARGET_REASONIX, "hi"))
    record = wf.registry.record("task-001")
    assert record["state"] == "FAILED"
    assert record["executor"] == "REASONIX"