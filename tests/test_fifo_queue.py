"""Persistent FIFO queue: A-side tasks are never lost while B is busy.

Pins the queue contract added on top of the A-side priority invariants:

A. A new valid TASK received while a task is running is CLAIMED and persisted
   as QUEUED (never rejected) and the UI reports the queue position.
B. When the running task reaches ANY terminal state the queued task starts
   automatically, in FIFO order, and both land in terminal states.
C. A restart restores QUEUED tasks from tasks.json and executes them in
   order; stale RECEIVED/PROCESSING records from a dead run are reconciled
   (COMPLETED / FAILED / RECOVERY_REQUIRED) without re-sending anything.
D. A repeated TASK_ID that is already QUEUED is not re-queued and not
   re-executed: side A gets the existing state, bound to the original id.
E. While a queued task is the one running, the manual monitor still yields
   to its session (ownership now includes the owning task id).
F. Stopping the manual monitor never cancels the A-side task's own
   auto-recovery: the task still completes and answers under its TASK_ID.
G. Closing the window never starts a queued task (it stays persisted for
   the next run) and never crashes.
"""

from __future__ import annotations

import time

import core.relay as relay_mod
from core.openchamber import (
    OpenChamberUnavailableError,
    is_message_wrapped,
)
from core.protocol import parse_message
from core.relay import directory_key
from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
from core.task_registry import TaskRegistry
from PySide6.QtTest import QTest
from tests.fakes import (
    assistant_message,
    make_dispatch,
    text_part,
    user_message,
)
from tests.test_oc_monitor import MutableMessages, completed_reply, manual_completed_ids
from tests.test_relay_workflow import RecoveryOpenChamber, v1_task
from tests.test_ui_startup import build_window, shutdown_window, wait_until

import ui as ui_mod


class ControllableRoundOc:
    """An OpenChamber fake that completes each task round on demand.

    Every ``send`` records a new round (user message with a unique id) that
    stays "busy" until the test flips it via :meth:`complete_last`, at which
    point the round gets its scripted reply (completed, ``finish=stop``).
    Sequential rounds are fully attributable by user message id.
    """

    def __init__(self, session_id: str, directory: str, replies: list[str]):
        self.session_id = session_id
        self.directory = directory
        self.replies = list(replies)
        self.sessions = [(session_id, "FIFO session")]
        self.rounds: list[dict] = []
        self.messages_all: list[dict] = []
        self.call_log: list[str] = []
        self.sent_prompts: list[str] = []

    def verify(self) -> None:
        self.call_log.append("verify")

    def create_session(self, title: str, directory: str) -> str:
        self.call_log.append(f"create:{title}:{directory}")
        return self.session_id

    def open_session(self, session_id: str) -> None:
        self.call_log.append(f"open:{session_id}")

    def list_sessions(self, directory: str | None = None) -> list[tuple[str, str]]:
        return [(self.session_id, "FIFO session")]

    def session_exists(self, session_id: str, directory: str) -> bool:
        return session_id == self.session_id and directory == self.directory

    def send(self, session_id, prompt, directory, agent=None, model=None):
        self.call_log.append(f"send:{prompt!r}")
        self.sent_prompts.append(prompt)
        user_id = f"u_fifo{len(self.rounds)}"
        self.rounds.append({"user_id": user_id, "done": False})
        self.messages_all.append(
            user_message(
                user_id, prompt, 1000 + 1000 * len(self.rounds),
                session_id=session_id,
            )
        )
        return make_dispatch(
            session_id=session_id,
            directory=directory,
            user_message_id=user_id,
            prompt_text=prompt,
        )

    def complete_last(self) -> None:
        """Flip the newest round to a verified completed reply."""
        if not self.rounds or self.rounds[-1]["done"]:
            return
        self.rounds[-1]["done"] = True
        reply = self.replies.pop(0) if self.replies else "ok"
        index = len(self.rounds) - 1
        self.messages_all.append(
            assistant_message(
                f"a_fifo{index}", 2000 + 1000 * index,
                completed=3000 + 1000 * index,
                finish="stop",
                parts=[text_part(reply)],
                parent_id=self.rounds[-1]["user_id"],
                session_id=self.session_id,
            )
        )

    def session_status(self, session_id: str, directory: str) -> str:
        if not self.rounds or self.rounds[-1]["done"]:
            return "idle"
        return "busy"

    def messages(self, session_id: str, directory: str) -> list[dict]:
        return list(self.messages_all)

    def round_has_pending_user_action(self, session_id, directory, dispatch) -> bool:
        return False

    def close(self) -> None:
        pass


def build_fifo_window(qapp, monkeypatch, tmp_path, task_oc, monitor_fake=None):
    """A window whose A-side relay runs against ``task_oc``; the manual
    monitor (when started) polls ``monitor_fake`` on the SAME session id and
    project directory so the ownership guards match."""
    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory=str(tmp_path),
        openchamber_session_id="ses_test123",
        openchamber_sessions={directory_key(str(tmp_path)): "ses_test123"},
        poll_interval=0.05,
        completion_timeout=0,  # no timeout: a busy round waits until completed
    )
    monitor_fake = monitor_fake or MutableMessages(
        [completed_reply("a_fq_hist", created=1000, text="history")]
    )
    monkeypatch.setattr(ui_mod, "OpenChamberClient", lambda url: monitor_fake)
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=task_oc
    )
    window._startup_check_pending = False
    return window


def _idle(qapp, window):
    assert wait_until(lambda: window._listener.enabled), "listener never started"
    assert wait_until(lambda: not window._busy), "startup tasks never settled"


def test_busy_task_is_queued_not_rejected_and_runs_in_fifo_order(
    qapp, monkeypatch, tmp_path
):
    """A+B+D: a TASK arriving while busy is QUEUED (persisted, position
    reported) and, once the running task completes, it starts automatically
    and completes too -- strictly in FIFO order, nothing re-sent."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply one", "reply two"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)

        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "one", "task-fifo-001", str(tmp_path)))
        assert wait_until(lambda: window._busy), "first task did not start"
        assert wait_until(
            lambda: [c for c in oc.call_log if c.startswith("send:")]
        ), "first task was not sent"

        # a second task arrives while the first is running: it must be
        # QUEUED, never rejected
        clipboard.setText(v1_task("OPENCHAMBER", "two", "task-fifo-002", str(tmp_path)))
        assert wait_until(
            lambda: (window._workflow.registry.record("task-fifo-002") or {}).get(
                "state"
            )
            == "QUEUED"
        ), "busy task was not accepted into the queue"
        assert window._queue_status_label.text() == "等待任务：1"
        # it was not executed yet
        assert len(oc.sent_prompts) == 1

        # the first task completes: its reply goes to side A under its id...
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-001" in (clipboard.text() or "")
        ), "first task reply was not returned to side A"
        assert window._workflow.registry.record("task-fifo-001")["state"] == "COMPLETED"

        # ...and the queued task starts automatically and completes too
        assert wait_until(
            lambda: len(oc.sent_prompts) == 2,
        ), "queued task did not start"
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-002" in (clipboard.text() or "")
        ), "queued task reply was not returned to side A"
        assert window._workflow.registry.record("task-fifo-002")["state"] == "COMPLETED"
        assert window._queue_status_label.text() == "等待任务：0"

        # strict FIFO order: "one" was sent before "two", never re-sent
        assert [p.split("\n\n")[0] for p in oc.sent_prompts] == ["one", "two"]
    finally:
        shutdown_window(window)


def test_restart_restores_queued_tasks_and_runs_them_in_order(
    qapp, monkeypatch, tmp_path
):
    """C: QUEUED records persisted in tasks.json by a previous run are
    picked up on startup and executed in FIFO order from their persisted
    raw protocol text (no clipboard re-copy needed)."""
    qapp.clipboard().clear()
    registry = TaskRegistry(tmp_path / "tasks.json")
    registry.mark(
        "task-fifo-101",
        "QUEUED",
        raw_message=v1_task("OPENCHAMBER", "queued one", "task-fifo-101", str(tmp_path)),
        sequence="1",
        received_at="2026-09-09T10:00:00",
    )
    registry.mark(
        "task-fifo-102",
        "QUEUED",
        raw_message=v1_task("OPENCHAMBER", "queued two", "task-fifo-102", str(tmp_path)),
        sequence="2",
        received_at="2026-09-09T10:00:01",
    )
    oc = ControllableRoundOc(
        "ses_test123", str(tmp_path), ["queued reply one", "queued reply two"]
    )
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        assert wait_until(lambda: window._listener.enabled)
        # NOTE: with a pre-seeded queue the head may start DURING the
        # startup self-check, so the window is not necessarily idle here;
        # wait for the head's send instead (a pre-seeded queue can only
        # shrink, so the queue readout is never asserted).
        assert wait_until(
            lambda: len(oc.sent_prompts) == 1
        ), "queued task did not auto-start"
        oc.complete_last()
        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-101" in (clipboard.text() or "")
        ), "first queued task was not completed after restart"

        # ...then the second one, in order
        assert wait_until(
            lambda: len(oc.sent_prompts) == 2
        ), "second queued task did not start"
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-102" in (clipboard.text() or "")
        ), "second queued task was not completed after restart"

        assert window._workflow.registry.record("task-fifo-101")["state"] == "COMPLETED"
        assert window._workflow.registry.record("task-fifo-102")["state"] == "COMPLETED"
        assert window._queue_status_label.text() == "等待任务：0"
        assert [p.split("\n\n")[0] for p in oc.sent_prompts] == [
            "queued one",
            "queued two",
        ]
    finally:
        shutdown_window(window)


class _ReconcileClient:
    """OpenChamberClient stand-in for the startup reconciliation query."""

    def __init__(self, url: str, task_id: str = "task-stale-001", fail: bool = False):
        self.url = url
        self.fail = fail
        self.messages_payload = [
            user_message(
                "u_stale",
                f"stale body\n\n[AI_RELAY_TASK_ID: {task_id}]",
                1000,
                session_id="ses_test123",
            ),
            assistant_message(
                "a_stale",
                1100,
                completed=1200,
                finish="stop",
                parts=[text_part("stale final answer")],
                parent_id="u_stale",
                session_id="ses_test123",
            ),
        ]

    def verify(self) -> None:
        if self.fail:
            raise OpenChamberUnavailableError("service unreachable")

    def messages(self, session_id: str, directory: str) -> list[dict]:
        if self.fail:
            raise OpenChamberUnavailableError("service unreachable")
        return list(self.messages_payload)

    def close(self) -> None:
        pass


def _seed_stale_processing(tmp_path) -> None:
    """One PROCESSING record left behind by a dead run: it carried a
    configured OpenChamber session, so it can be reconciled for real."""
    registry = TaskRegistry(tmp_path / "tasks.json")
    registry.mark(
        "task-stale-001",
        "PROCESSING",
        executor=TARGET_OPENCHAMBER,
        session_id="ses_test123",
        directory=str(tmp_path),
        raw_message=v1_task(
            "OPENCHAMBER", "stale body", "task-stale-001", str(tmp_path)
        ),
    )


def test_stale_processing_reconciled_to_completed_on_startup(
    qapp, monkeypatch, tmp_path
):
    """C: a stale PROCESSING record whose OpenChamber round verifiably
    completed is restored to COMPLETED with the final reply wrapped under
    the ORIGINAL task id (never a manual-* id) and handed to side A."""
    qapp.clipboard().clear()
    _seed_stale_processing(tmp_path)
    monkeypatch.setattr(relay_mod, "OpenChamberClient", lambda url: _ReconcileClient(url))
    oc = ControllableRoundOc("ses_test123", str(tmp_path), [])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(
            lambda: (window._workflow.registry.record("task-stale-001") or {}).get(
                "state"
            )
            == "COMPLETED"
        ), "stale PROCESSING record was not reconciled to COMPLETED"
        record = window._workflow.registry.record("task-stale-001")
        assert record.get("reply_file"), "recovered reply was not saved"
        # the message is marked wrapped so the monitor can never re-wrap it
        assert is_message_wrapped("a_stale", "ses_test123")
        # side A gets the recovered reply under the ORIGINAL task id
        clipboard = qapp.clipboard()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-stale-001" in (clipboard.text() or "")
        ), "recovered reply was not returned to side A"
        recovered = parse_message(clipboard.text())
        assert recovered.in_reply_to == "task-stale-001"
        assert "stale final answer" in recovered.body
    finally:
        shutdown_window(window)


def test_stale_processing_unreachable_becomes_recovery_required(
    qapp, monkeypatch, tmp_path
):
    """C: when the OpenChamber service cannot be queried, the stale record
    must not be guessed as FAILED/COMPLETED: it becomes RECOVERY_REQUIRED
    (terminal, human-checked) and the queue still drains."""
    qapp.clipboard().clear()
    _seed_stale_processing(tmp_path)
    monkeypatch.setattr(
        relay_mod, "OpenChamberClient", lambda url: _ReconcileClient(url, fail=True)
    )
    oc = ControllableRoundOc("ses_test123", str(tmp_path), [])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(
            lambda: (window._workflow.registry.record("task-stale-001") or {}).get(
                "state"
            )
            == "RECOVERY_REQUIRED"
        ), "unreconcilable stale record was not parked as RECOVERY_REQUIRED"
        record = window._workflow.registry.record("task-stale-001")
        assert record.get("error"), "RECOVERY_REQUIRED must record the reason"
        # no reply may have been produced for the unknown task
        assert window._workflow.load_reply("task-stale-001") is None
    finally:
        shutdown_window(window)


def test_duplicate_queued_task_id_is_not_requeued_or_reexecuted(
    qapp, monkeypatch, tmp_path
):
    """D: re-sending the TASK_ID of a task that is already QUEUED never
    adds a second queue entry and never executes it; side A is answered
    with the existing state, bound to the original id.  The task then runs
    exactly once when its turn comes."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply one", "reply two"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        task_two = v1_task("OPENCHAMBER", "two", "task-fifo-002", str(tmp_path))

        clipboard.setText(v1_task("OPENCHAMBER", "one", "task-fifo-001", str(tmp_path)))
        assert wait_until(lambda: window._busy)
        clipboard.setText(task_two)
        assert wait_until(
            lambda: (window._workflow.registry.record("task-fifo-002") or {}).get(
                "state"
            )
            == "QUEUED"
        )

        # re-send the SAME queued task id (via a different text first so the
        # listener sees a change): it must be answered, not re-queued
        clipboard.setText("temporary notes that are not a relay message")
        QTest.qWait(100)
        clipboard.setText(task_two)
        assert wait_until(
            lambda: "重复任务" in (window.detail_label.text() or "")
        ), "duplicate of a QUEUED task was not answered"
        answered = parse_message(clipboard.text())
        assert answered.in_reply_to == "task-fifo-002"
        assert "QUEUED" in answered.body
        # still exactly one queue entry
        assert len(window._workflow.registry.queued_records()) == 1

        # when its turn comes it runs exactly once
        assert wait_until(lambda: len(oc.sent_prompts) >= 1), "task-001 never sent"
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-001" in (clipboard.text() or "")
        )
        assert wait_until(lambda: len(oc.sent_prompts) == 2)
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-002" in (clipboard.text() or "")
        )
        assert len(oc.sent_prompts) == 2
        assert [p.split("\n\n")[0] for p in oc.sent_prompts] == ["one", "two"]
    finally:
        shutdown_window(window)


def test_monitor_yields_to_running_queued_task_with_owner_task_id(
    qapp, monkeypatch, tmp_path
):
    """E: once the QUEUED task becomes the running task it owns its session
    (the owner map carries the task id); the manual monitor still yields for
    it and never wraps its replies as manual-*."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply one", "reply two"])
    monitor_fake = MutableMessages(
        [completed_reply("a_fq_hist", created=1000, text="history")]
    )
    window = build_fifo_window(
        qapp, monkeypatch, tmp_path, oc, monitor_fake=monitor_fake
    )
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "one", "task-fifo-001", str(tmp_path)))
        assert wait_until(lambda: window._busy)
        clipboard.setText(v1_task("OPENCHAMBER", "two", "task-fifo-002", str(tmp_path)))
        assert wait_until(
            lambda: (window._workflow.registry.record("task-fifo-002") or {}).get(
                "state"
            )
            == "QUEUED"
        )
        assert wait_until(lambda: len(oc.sent_prompts) >= 1)
        oc.complete_last()
        assert wait_until(lambda: "IN_REPLY_TO: task-fifo-001" in (clipboard.text() or ""))
        # the monitor starts while the SECOND (ex-queued) task is running
        assert wait_until(lambda: len(oc.sent_prompts) >= 2)
        window._toggle_monitor()
        assert window._oc_monitor_active
        assert wait_until(
            lambda: window._a_side_session_owner.get("ses_test123")
            == "task-fifo-002"
        ), "session ownership did not record the owning task id"

        # a final reply "arrives" on the monitored session while the
        # ex-queued task owns it: the monitor observes but must not wrap it
        monitor_fake.add(
            completed_reply("a_fq_task", created=4000, text="owned final")
        )
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            QTest.qWait(40)
        assert manual_completed_ids(window) == set(), (
            "monitor wrapped a reply owned by the running (ex-queued) task"
        )
        assert not is_message_wrapped("a_fq_task", "ses_test123")
    finally:
        shutdown_window(window)


def test_stopping_monitor_does_not_cancel_a_side_auto_recovery(
    qapp, monkeypatch, tmp_path
):
    """F: the A-side task's own auto-recovery (interrupted round -> continue
    prompt in the same session -> completed) is driven by the task worker
    with its OWN cancel event; stopping the manual monitor only stops the
    monitor.  The task still completes and answers under its TASK_ID."""
    qapp.clipboard().clear()
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 0.0)
    oc = RecoveryOpenChamber(
        session_id="ses_test123", directory=str(tmp_path), recover_at=1
    )
    monitor_fake = MutableMessages(
        [completed_reply("a_fq_hist", created=1000, text="history")]
    )
    window = build_fifo_window(
        qapp, monkeypatch, tmp_path, oc, monitor_fake=monitor_fake
    )
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(
            v1_task("OPENCHAMBER", "recover me", "task-fifo-201", str(tmp_path))
        )
        assert wait_until(lambda: window._busy), "task did not start"
        # the monitor watches the same session; it yields while the task runs
        window._toggle_monitor()
        assert window._oc_monitor_active
        # now STOP the monitor mid-task: the A-side task must keep going
        window._stop_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "monitor stop did not drain"
        assert not window._oc_monitor_active
        # the task's auto-recovery still fires and completes the round
        # (idle recovery needs the 5s completion grace + confirmation polls,
        # so allow a generous window)
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-201" in (clipboard.text() or ""),
            timeout=20.0,
        ), "stopping the monitor cancelled the A-side task's auto-recovery"
        record = window._workflow.registry.record("task-fifo-201")
        assert record is not None and record["state"] == "COMPLETED"
        response = parse_message(clipboard.text())
        assert response.in_reply_to == "task-fifo-201"
        assert response.message_id != "task-fifo-201"
    finally:
        shutdown_window(window)


def test_close_never_starts_a_queued_task_and_persists_it(
    qapp, monkeypatch, tmp_path
):
    """G: while the window is closing, queued tasks are never started (the
    lane is blocked) and their QUEUED records persist in tasks.json for the
    next run; no crash and no half-launched task."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply one", "reply two"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "one", "task-fifo-001", str(tmp_path)))
        assert wait_until(lambda: window._busy)
        clipboard.setText(v1_task("OPENCHAMBER", "two", "task-fifo-002", str(tmp_path)))
        assert wait_until(
            lambda: (window._workflow.registry.record("task-fifo-002") or {}).get(
                "state"
            )
            == "QUEUED"
        )
        assert wait_until(lambda: len(oc.sent_prompts) >= 1)

        # simulate the close handshake: no new work may start from here on
        window._close_requested = True
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-fifo-001" in (clipboard.text() or "")
        ), "running task must still finish its reply while closing"
        # the queued task must NOT have started (its send never happened)
        assert len(oc.sent_prompts) == 1
        assert (
            window._workflow.registry.record("task-fifo-002")["state"] == "QUEUED"
        ), "queued task was started during window close"
        # and the record is persisted on disk for the next run
        persisted = TaskRegistry(tmp_path / "tasks.json").record("task-fifo-002")
        assert persisted is not None and persisted["state"] == "QUEUED"
    finally:
        shutdown_window(window)


# -------------------------------------------------------------------- #
# manual 包装内容 of the RUNNING A-side task: bind to the original TASK_ID,
# end the round (COMPLETED / completion_source=manual_wrap), invalidate the
# worker, and advance the FIFO queue without deleting it.
# -------------------------------------------------------------------- #


def test_wrap_running_task_binds_original_task_id(qapp, monkeypatch, tmp_path):
    """1: a PROCESSING A-side task + 包装内容 wraps under the ORIGINAL
    TASK_ID (IN_REPLY_TO=task-A), never a manual-* id."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["unused"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "task A", "task-A", str(tmp_path)))
        assert wait_until(lambda: window._busy), "task-A did not start"
        assert wait_until(
            lambda: (window._workflow.registry.record("task-A") or {}).get(
                "state"
            )
            == "PROCESSING"
        ), "task-A is not PROCESSING"

        clipboard.setText("人工确认的最终回复")
        window.wrap_button.click()

        message = parse_message(clipboard.text() or "")
        assert message.in_reply_to == "task-A"
        assert "人工确认的最终回复" in message.body
        assert not message.in_reply_to.startswith("manual-")
    finally:
        shutdown_window(window)


def test_wrap_running_task_marks_completed_manual_wrap(qapp, monkeypatch, tmp_path):
    """2+4: wrapping the running task marks it COMPLETED with
    completion_source=manual_wrap, frees the lane, and the superseded worker's
    late (cancelled) result can neither re-write the clipboard nor clobber the
    COMPLETED state."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["unused"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "task A", "task-A", str(tmp_path)))
        assert wait_until(lambda: window._busy)
        assert wait_until(
            lambda: (window._workflow.registry.record("task-A") or {}).get(
                "state"
            )
            == "PROCESSING"
        )

        clipboard.setText("人工确认的最终回复")
        window.wrap_button.click()

        wrapped = clipboard.text() or ""
        record = window._workflow.registry.record("task-A")
        assert record["state"] == "COMPLETED"
        assert record.get("completion_source") == "manual_wrap"
        assert window._busy is False
        assert window._current_a_task_id is None

        # let the just-cancelled worker drain; its late result must be dropped
        assert wait_until(
            lambda: not window._general_workers, timeout=5.0
        ), "superseded worker did not drain"
        assert (clipboard.text() or "") == wrapped
        assert window._workflow.registry.record("task-A")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


def test_mark_if_state_keeps_manual_completion(tmp_path):
    """3+4 (registry guard): once a task is COMPLETED via a manual wrap, a
    late worker's COMPLETED or FAILED write is a no-op (completion_source is
    preserved); a fresh RECEIVED task still transitions normally."""
    from core.task_registry import TaskRegistry

    registry = TaskRegistry(tmp_path / "tasks.json")
    registry.mark(
        "task-A",
        "PROCESSING",
        executor=TARGET_OPENCHAMBER,
        session_id="ses_test123",
        directory=str(tmp_path),
    )
    registry.mark("task-A", "COMPLETED", completion_source="manual_wrap")

    # late worker failure: the task is no longer RECEIVED/PROCESSING
    assert (
        registry.mark_if_state(
            "task-A", ("RECEIVED", "PROCESSING"), "FAILED", "boom"
        )
        is False
    )
    # late worker success: also a no-op, must not rewrite completion_source
    assert (
        registry.mark_if_state(
            "task-A",
            ("RECEIVED", "PROCESSING"),
            "COMPLETED",
            completion_source="auto_relay",
        )
        is False
    )
    record = registry.record("task-A")
    assert record["state"] == "COMPLETED"
    assert record.get("completion_source") == "manual_wrap"

    # a live task still transitions normally
    registry.mark("task-B", "RECEIVED")
    assert registry.mark_if_state(
        "task-B", ("RECEIVED", "PROCESSING"), "PROCESSING"
    )
    assert registry.record("task-B")["state"] == "PROCESSING"


def test_wrap_with_only_queued_task_uses_manual_id(qapp, monkeypatch, tmp_path):
    """5: with NO running task (only a QUEUED task-B), 包装内容 wraps a
    standalone manual-* response and never binds the clipboard text to
    task-B, which stays QUEUED."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["unused"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        window._workflow.registry.mark(
            "task-B",
            "QUEUED",
            raw_message=v1_task("OPENCHAMBER", "B", "task-B", str(tmp_path)),
            sequence="1",
            received_at="2026-09-09T10:00:00",
        )
        window._update_queue_label()
        assert window._current_a_task_id is None

        clipboard = qapp.clipboard()
        clipboard.setText("普通剪贴板内容")
        window.wrap_button.click()

        message = parse_message(clipboard.text() or "")
        assert message.in_reply_to.startswith("manual-")
        assert window._workflow.registry.record("task-B")["state"] == "QUEUED"
    finally:
        shutdown_window(window)


def test_wrap_running_task_advances_fifo(qapp, monkeypatch, tmp_path):
    """6: wrapping the running task-A (with task-B queued) completes A and
    automatically starts task-B from the queue."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply B"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "A", "task-A", str(tmp_path)))
        assert wait_until(lambda: window._busy)
        assert wait_until(
            lambda: [c for c in oc.call_log if c.startswith("send:")]
        )
        clipboard.setText(v1_task("OPENCHAMBER", "B", "task-B", str(tmp_path)))
        assert wait_until(
            lambda: (window._workflow.registry.record("task-B") or {}).get(
                "state"
            )
            == "QUEUED"
        )

        clipboard.setText("人工最终回复")
        window.wrap_button.click()
        assert window._workflow.registry.record("task-A")["state"] == "COMPLETED"

        assert wait_until(lambda: len(oc.sent_prompts) == 2), "task-B did not auto-start"
        oc.complete_last()
        assert wait_until(
            lambda: "IN_REPLY_TO: task-B" in (clipboard.text() or "")
        ), "task-B reply was not returned to side A"
        assert window._workflow.registry.record("task-B")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)


def test_wrap_preserves_subsequent_queue(qapp, monkeypatch, tmp_path):
    """7: wrapping task-A never deletes the queued tasks behind it: task-B and
    task-C both remain, task-B runs next and task-C stays queued."""
    qapp.clipboard().clear()
    oc = ControllableRoundOc("ses_test123", str(tmp_path), ["reply B", "reply C"])
    window = build_fifo_window(qapp, monkeypatch, tmp_path, oc)
    try:
        _idle(qapp, window)
        clipboard = qapp.clipboard()
        clipboard.setText(v1_task("OPENCHAMBER", "A", "task-A", str(tmp_path)))
        assert wait_until(lambda: window._busy)
        assert wait_until(lambda: [c for c in oc.call_log if c.startswith("send:")])
        clipboard.setText(v1_task("OPENCHAMBER", "B", "task-B", str(tmp_path)))
        assert wait_until(
            lambda: (window._workflow.registry.record("task-B") or {}).get(
                "state"
            )
            == "QUEUED"
        )
        clipboard.setText(v1_task("OPENCHAMBER", "C", "task-C", str(tmp_path)))
        assert wait_until(
            lambda: (window._workflow.registry.record("task-C") or {}).get(
                "state"
            )
            == "QUEUED"
        )

        clipboard.setText("人工最终回复")
        window.wrap_button.click()
        # A resolved by manual wrap; the queued entries behind it survive.
        assert window._workflow.registry.record("task-B") is not None
        assert window._workflow.registry.record("task-C") is not None
        record_a = window._workflow.registry.record("task-A")
        assert record_a["state"] == "COMPLETED"
        assert record_a.get("completion_source") == "manual_wrap"

        # FIFO: B runs next while C stays queued behind it.
        assert wait_until(lambda: len(oc.sent_prompts) == 2), "task-B did not start"
        assert window._workflow.registry.record("task-C")["state"] == "QUEUED"

        # Drive B then C to a clean COMPLETED so every worker drains before
        # shutdown (a busy round left at shutdown races the cancel and leaks).
        oc.complete_last()
        assert wait_until(
            lambda: window._workflow.registry.record("task-B")["state"] == "COMPLETED"
        )
        assert wait_until(lambda: len(oc.sent_prompts) == 3), "task-C did not start"
        oc.complete_last()
        assert wait_until(
            lambda: window._workflow.registry.record("task-C")["state"] == "COMPLETED"
        )
        assert wait_until(lambda: not window._general_workers, timeout=5.0)
        assert window._workflow.registry.record("task-C")["state"] == "COMPLETED"
    finally:
        shutdown_window(window)