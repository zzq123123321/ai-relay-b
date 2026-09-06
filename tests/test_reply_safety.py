"""Reply file naming: untrusted task ids must never affect the file path."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.protocol import parse_message
from core.relay import RelayWorkflow
from core.relay_settings import RelaySettings, TARGET_REASONIX
from core.task_registry import TaskRegistry
from tests.test_relay_workflow import FakeReasonix


def v1_task(target: str, body: str, message_id: str) -> str:
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


def make_workflow(tmp_path: Path, message_id: str):
    settings = RelaySettings(
        default_target=TARGET_REASONIX,
        openchamber_directory="D:/proj",
        completion_timeout=5.0,
        poll_interval=0.01,
    )
    return RelayWorkflow(
        reasonix=FakeReasonix(),
        registry=TaskRegistry(tmp_path / "tasks.json"),
        settings=settings,
        replies_dir=tmp_path / "replies",
    ), message_id


# Hostile MESSAGE_ID values taken verbatim from the untrusted clipboard task.
HOSTILE_IDS = [
    "..\\..\\..\\windows\\system32\\evil",
    "../evil",
    "..\\evil",
    "C:\\temp\\evil",
    "D:/evil/response.txt",
    "/absolute/evil",
    "C:",
    "CON",
    "NUL",
    "aux",
    "prn",
    "COM1",
    "a*b?c|d:e",
    "中文任务ID",
    "tab\there",
    "trail\\",
    "a\\b\\c",
    "x" * 500,
    "\x00null",
    "..",
]


@pytest.mark.parametrize("task_id", HOSTILE_IDS)
def test_save_reply_never_escapes_replies_dir(tmp_path: Path, task_id: str):
    """Every hostile id collapses to a hash name inside data/replies."""
    wf, message_id = make_workflow(tmp_path, task_id)
    wf.process(v1_task(TARGET_REASONIX, "hello", message_id))

    replies = tmp_path / "replies"
    files = list(replies.iterdir()) if replies.exists() else []
    assert len(files) == 1, f"unexpected files: {files}"
    only = files[0]
    assert only.parent == replies.resolve() or only.parent == replies
    assert only.name.startswith("rel_") and only.name.endswith(".response.txt")
    # the file content round-trips and is retrievable by the ORIGINAL id
    stored = only.read_text(encoding="utf-8")
    assert "hello" in stored
    assert wf.load_reply(task_id) == stored
    # and the registry keeps the original id -> file mapping
    record = wf.registry.record(task_id)
    assert record is not None
    assert record["reply_file"] == str(only)


@pytest.mark.parametrize("task_id", HOSTILE_IDS)
def test_hostile_id_saved_file_is_not_the_raw_name(tmp_path: Path, task_id: str):
    wf, message_id = make_workflow(tmp_path, task_id)
    wf.process(v1_task(TARGET_REASONIX, "hello", message_id))
    replies = tmp_path / "replies"
    names = {p.name for p in replies.iterdir()}
    assert f"{task_id}.response.txt" not in names


def test_distinct_ids_get_distinct_files(tmp_path: Path):
    wf, _ = make_workflow(tmp_path, "id-a")
    wf.process(v1_task(TARGET_REASONIX, "a", "id-a"))
    wf.process(v1_task(TARGET_REASONIX, "b", "id-b"))
    replies = tmp_path / "replies"
    assert len(list(replies.iterdir())) == 2
    assert wf.load_reply("id-a") != wf.load_reply("id-b")
    assert "a" in (wf.load_reply("id-a") or "")
    assert "b" in (wf.load_reply("id-b") or "")


def test_load_reply_unknown_task_returns_none(tmp_path: Path):
    wf, _ = make_workflow(tmp_path, "id-a")
    assert wf.load_reply("never-processed") is None
    # even a hostile unknown id must not escape or raise
    assert wf.load_reply("..\\..\\secrets") is None
    assert wf.load_reply("C:\\temp\\x") is None


def test_recopy_uses_original_task_mapping_not_rerun(tmp_path: Path):
    wf, message_id = make_workflow(tmp_path, "recopy-1")
    response = wf.process(v1_task(TARGET_REASONIX, "do once", message_id))
    executed = wf.reasonix.executed  # type: ignore[attr-defined]
    reloaded = wf.load_reply("recopy-1")
    assert reloaded == response
    assert executed == ["do once"]  # re-copy must not re-execute the task


def test_legacy_reply_name_still_readable_inside_dir(tmp_path: Path):
    """Replies written before the hash-naming change remain re-copyable as
    long as they live inside the replies dir."""
    wf, message_id = make_workflow(tmp_path, "legacy-1")
    response = wf.process(v1_task(TARGET_REASONIX, "legacy body", message_id))
    replies = tmp_path / "replies"
    legacy = replies / "legacy-1.response.txt"
    legacy.write_text(response, encoding="utf-8")
    # remove the hash copy so only the legacy name can satisfy the load
    for other in replies.iterdir():
        if other != legacy:
            other.unlink()
    assert wf.load_reply("legacy-1") == response


def test_protocol_task_id_passes_through_untouched(tmp_path: str):
    """The original id stays in the protocol/registry even when the file
    name is a hash."""
    wf, message_id = make_workflow(tmp_path, "中文*任务|id")
    response = wf.process(v1_task(TARGET_REASONIX, "x", message_id))
    parsed = parse_message(response)
    assert parsed.in_reply_to == "中文*任务|id"