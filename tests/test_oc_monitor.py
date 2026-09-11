"""Manual OpenChamber monitor: scan semantics, auto-relay dedup and the
UI button flow (start → wrap once → stop)."""

from __future__ import annotations

import logging
import threading
import time

import pytest
from PySide6.QtCore import QObject, Slot

from core.openchamber import (
    MONITOR_CONTINUE_PROMPT,
    MONITOR_MAX_TRANSPORT_RETRIES,
    MonitorMessageStat,
    MonitorProbe,
    MonitorRoundTracker,
    OpenChamberBadRequestError,
    OpenChamberCancelledError,
    OpenChamberClient,
    OpenChamberUnavailableError,
    is_message_wrapped,
    mark_message_wrapped,
    monitor_probe,
    monitor_scan,
    reset_wrapped_message_ids,
)
from core.relay import OpenChamberContinue
from core.relay_settings import TARGET_OPENCHAMBER, RelaySettings
from tests.fakes import (
    assistant_message,
    make_dispatch,
    question_part,
    text_part,
    tool_part,
    user_message,
)
from tests.test_relay_workflow import (
    BusyOpenChamber,
    make_workflow,
    scripted_oc,
    v1_task,
)
from tests.test_ui_startup import shutdown_window, wait_until


def _auto_workers_drained(window) -> bool:
    """True when every ``auto-N`` continuation worker has delivered its
    ``finished`` signal; the poll worker legitimately stays registered while
    the monitor is running, so the whole set must NOT be awaited here."""
    for token in window._monitor_workers:
        if token.split(":")[1].startswith("auto-"):
            return False
    return True


class MutableMessages:
    """A fake OpenChamber client whose message list is edited by the test
    between polls, exactly like a real session growing over time."""

    def __init__(self, initial=None):
        self.messages_list = list(initial or [])
        self.exists = True
        self.fail_all = False
        self.status = "idle"
        self.sent: list[tuple] = []
        self.send_error: BaseException | None = None

    def add(self, message):
        self.messages_list.append(message)

    def session_exists(self, session_id, directory):
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")
        return self.exists

    def messages(self, session_id, directory):
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")
        return [dict(m) for m in self.messages_list]

    def session_status(self, session_id, directory):
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")
        return self.status

    def send(self, session_id, prompt, directory, agent=None, model=None):
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")
        if self.send_error is not None:
            raise self.send_error
        self.sent.append((session_id, directory, prompt, agent, model))
        return None


def completed_reply(msg_id: str, created: int = 3000, text: str = "freshest") -> dict:
    return assistant_message(
        msg_id,
        created,
        completed=created + 100,
        finish="stop",
        parts=[text_part(text)],
    )


def in_progress_reply(msg_id: str, created: int = 3000) -> dict:
    return assistant_message(
        msg_id,
        created,
        completed=None,
        finish=None,
        parts=[text_part("still working")],
    )


# -------------------------------------------------------------------- #
# monitor_scan: baseline, completion rules and dedup
# -------------------------------------------------------------------- #


def test_monitor_scan_skips_baseline_history():
    old = completed_reply("a_old", created=1000)
    client = MutableMessages([old])
    scan = monitor_scan(client, "ses_test123", "D:/proj", frozenset({"a_old"}))
    assert scan.new_completed == ()
    assert scan.seen_ids == frozenset({"a_old"})


def test_monitor_scan_returns_newest_completed_only():
    old = completed_reply("a_old", created=1000, text="history")
    fresh = completed_reply("a_new", created=3000, text="freshest")
    client = MutableMessages([old, fresh])
    scan = monitor_scan(client, "ses_test123", "D:/proj", frozenset({"a_old"}))
    assert scan.new_completed == (("a_new", "freshest"),)
    assert scan.seen_ids == frozenset({"a_old", "a_new"})


def test_monitor_scan_ignores_incomplete_and_non_stop_messages():
    client = MutableMessages(
        [
            in_progress_reply("a_running"),
            completed_reply("a_stopped", created=3000, text="done here"),
        ]
    )
    scan = monitor_scan(client, "ses_test123", "D:/proj", frozenset())
    assert scan.new_completed == (("a_stopped", "done here"),)


def test_monitor_scan_orders_oldest_first():
    second = completed_reply("a_later", created=5000, text="later")
    first = completed_reply("a_earlier", created=4000, text="earlier")
    client = MutableMessages([first, second])
    scan = monitor_scan(client, "ses_test123", "D:/proj", frozenset())
    assert scan.new_completed == (("a_earlier", "earlier"), ("a_later", "later"))


def test_monitor_scan_skips_already_wrapped():
    mark_message_wrapped("a_wrapped", "ses_test123")
    fresh = completed_reply("a_fresh", created=3000, text="freshest")
    client = MutableMessages(
        [
            completed_reply("a_wrapped", created=1000, text="already packaged"),
            fresh,
        ]
    )
    try:
        scan = monitor_scan(client, "ses_test123", "D:/proj", frozenset())
        assert scan.new_completed == (("a_fresh", "freshest"),)
        assert scan.seen_ids == frozenset({"a_wrapped", "a_fresh"})
    finally:
        # module-level registry must not leak into other tests
        reset_wrapped_message_ids()


def test_wrapped_registry_scoped_by_session():
    """The same message id in TWO sessions must not affect each other: only
    ``(session_id, message_id)`` pairs are consumed, never bare ids."""
    mark_message_wrapped("m_shared", "ses_a")
    assert is_message_wrapped("m_shared", "ses_a")
    assert not is_message_wrapped("m_shared", "ses_b")
    # the OTHER session still sees its own message as wrappable
    client_b = MutableMessages([completed_reply("m_shared", created=1000, text="b done")])
    scan_b = monitor_scan(client_b, "ses_b", "D:/proj", frozenset())
    assert scan_b.new_completed == (("m_shared", "b done"),)
    # and the wrapped session skips it
    client_a = MutableMessages([completed_reply("m_shared", created=1000, text="a done")])
    scan_a = monitor_scan(client_a, "ses_a", "D:/proj", frozenset())
    assert scan_a.new_completed == ()


def test_wrapped_registry_same_session_same_id_wrapped_once():
    """Marking the same (session, message-id) pair more than once is a
    no-op: a reply is consumed exactly once within a session."""
    mark_message_wrapped("m_once", "ses_a")
    mark_message_wrapped("m_once", "ses_a")
    client = MutableMessages([completed_reply("m_once", created=1000, text="once done")])
    scan = monitor_scan(client, "ses_a", "D:/proj", frozenset())
    assert scan.new_completed == ()
    assert is_message_wrapped("m_once", "ses_a")


def test_wrapped_registry_does_not_leak_between_tests():
    """The autouse conftest fixture clears the process-global registry before
    and after every test; the explicit reset is its primitive.  This proves
    the reset alone removes ALL sessions' marks (cross-test leak prevention),
    and the fixture guarantees no mark can survive into the next test
    regardless of file/test execution order."""
    mark_message_wrapped("m_leak", "ses_a")
    mark_message_wrapped("m_leak", "ses_b")
    assert is_message_wrapped("m_leak", "ses_a")
    assert is_message_wrapped("m_leak", "ses_b")
    reset_wrapped_message_ids()
    assert not is_message_wrapped("m_leak", "ses_a")
    assert not is_message_wrapped("m_leak", "ses_b")


def test_monitor_scan_rejects_user_messages_and_question_tools():
    client = MutableMessages(
        [
            user_message("u_1", "task", 1000),
            assistant_message(
                "a_tool",
                2000,
                completed=2100,
                finish="stop",
                parts=[question_part("completed")],
            ),
        ]
    )
    scan = monitor_scan(client, "ses_test123", "D:/proj", frozenset())
    assert scan.new_completed == ()


def test_relay_package_marks_round_ids_wrapped(tmp_path):
    """The auto relay must mark its verified round as wrapped so the manual
    monitor never re-packages the same reply."""
    oc = scripted_oc()
    wf = make_workflow(tmp_path, oc=oc)
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert "final answer" in response
    assert is_message_wrapped("a_new", "ses_test123")
    # a monitor scanning the same session with an empty baseline must skip
    # the reply the auto relay already packaged
    scan = monitor_scan(oc, "ses_test123", "D:/proj", frozenset())
    assert scan.new_completed == ()


# -------------------------------------------------------------------- #
# monitor_probe + MonitorRoundTracker: abnormal-idle detection
# -------------------------------------------------------------------- #


def tool_call_message(
    msg_id: str, created: int = 4000, completed: int | None = None, text: str = ""
) -> dict:
    parts = []
    if text:
        parts.append(text_part(text))
    parts.append(tool_part("bash", status="completed"))
    return assistant_message(
        msg_id, created, completed=completed, finish="tool-calls", parts=parts
    )


def _fresh_tracker() -> MonitorRoundTracker:
    tracker = MonitorRoundTracker("ses_test123", "D:/proj")
    tracker.reset_round(frozenset())
    return tracker


def _probe(client, tracker=None):
    baseline = frozenset(tracker.baseline_ids) if tracker else frozenset()
    return monitor_probe(client, "ses_test123", "D:/proj", baseline)


def test_monitor_probe_reports_status_new_replies_and_stats():
    client = MutableMessages(
        [
            completed_reply("a_old", created=1000, text="history"),
            completed_reply("a_fresh", created=3000, text="freshest"),
        ]
    )
    client.status = "busy"
    probe = monitor_probe(client, "ses_test123", "D:/proj", frozenset({"a_old"}))
    assert isinstance(probe, MonitorProbe)
    assert probe.status == "busy"
    assert probe.new_completed == (("a_fresh", "freshest"),)
    assert probe.seen_ids == frozenset({"a_old", "a_fresh"})
    last = probe.messages[-1]
    assert isinstance(last, MonitorMessageStat)
    assert last.message_id == "a_fresh"
    assert last.role == "assistant"
    assert last.finish == "stop"
    assert last.completed_ts is not None
    assert last.text_length == len("freshest")
    assert last.tool_count == 0


def test_monitor_probe_distinguishes_tool_call_message():
    client = MutableMessages([tool_call_message("a_tool", 4000)])
    probe = _probe(client)
    last = probe.messages[-1]
    assert last.finish == "tool-calls"
    assert last.completed_ts is None
    assert last.tool_count == 1
    # a tool-calls message carries no completed reply text
    assert probe.new_completed == ()


def test_tracker_history_idle_never_triggers():
    client = MutableMessages([completed_reply("a_old", created=1000)])
    tracker = _fresh_tracker()
    tracker.reset_round(frozenset({"a_old"}))
    for _ in range(30):
        outcome = tracker.step(_probe(client, tracker))
        assert outcome.event not in ("idle_confirm", "interrupted")
        assert outcome.idle_confirmations == 0


def test_tracker_activity_start_stops_initial_idle():
    """A session that is idle BEFORE the monitor rounds does not trigger."""
    client = MutableMessages([completed_reply("a_old", created=1000)])
    tracker = _fresh_tracker()
    tracker.reset_round(frozenset({"a_old"}))  # history only
    for _ in range(10):
        assert tracker.step(_probe(client, tracker)).event != "interrupted"


def test_tracker_interrupts_after_three_identical_idle_confirmations():
    client = MutableMessages()
    tracker = _fresh_tracker()
    client.add(tool_call_message("a_tool", 4000))
    events = [tracker.step(_probe(client, tracker)).event for _ in range(5)]
    assert events == ["activity", "idle_confirm", "idle_confirm", "interrupted", "none"]
    assert tracker.interruption_emitted is True
    assert tracker.last_finish == "tool-calls"


def test_tracker_requires_three_confirmations_not_two():
    client = MutableMessages([tool_call_message("a_tool", 4000)])
    tracker = _fresh_tracker()
    first = tracker.step(_probe(client, tracker))
    second = tracker.step(_probe(client, tracker))
    third = tracker.step(_probe(client, tracker))
    assert (first.event, first.idle_confirmations) == ("activity", 0)
    assert (second.event, second.idle_confirmations) == ("idle_confirm", 1)
    assert (third.event, third.idle_confirmations) == ("idle_confirm", 2)
    assert tracker.step(_probe(client, tracker)).event == "interrupted"


def test_tracker_busy_status_resets_confirmation_counter():
    client = MutableMessages([tool_call_message("a_tool", 4000)])
    tracker = _fresh_tracker()
    assert tracker.step(_probe(client, tracker)).event == "activity"
    assert tracker.step(_probe(client, tracker)).event == "idle_confirm"
    client.status = "busy"
    busy = tracker.step(_probe(client, tracker))
    assert busy.event == "activity"
    assert busy.idle_confirmations == 0
    client.status = "idle"
    again = tracker.step(_probe(client, tracker))
    assert again.event == "idle_confirm"
    assert again.idle_confirmations == 1


def test_tracker_new_message_resets_confirmation_counter():
    client = MutableMessages([tool_call_message("a_tool", 4000)])
    tracker = _fresh_tracker()
    assert tracker.step(_probe(client, tracker)).event == "activity"
    assert tracker.step(_probe(client, tracker)).event == "idle_confirm"
    client.add(tool_call_message("a_tool2", 4100))
    fresh = tracker.step(_probe(client, tracker))
    assert fresh.event == "activity"
    assert fresh.idle_confirmations == 0


def test_tracker_text_growth_resets_confirmation_counter():
    client = MutableMessages()
    tracker = _fresh_tracker()
    client.add(in_progress_reply("a_text", created=4000))
    assert tracker.step(_probe(client, tracker)).event == "activity"
    assert tracker.step(_probe(client, tracker)).event == "idle_confirm"
    client.messages_list[0]["parts"] = [text_part("still working much longer")]
    growth = tracker.step(_probe(client, tracker))
    assert growth.event == "activity"
    assert growth.idle_confirmations == 0


def test_tracker_completed_final_reply_never_interrupts():
    client = MutableMessages()
    tracker = _fresh_tracker()
    client.add(completed_reply("a_done", created=3000, text="answer here"))
    # first the new reply is activity (the poller wraps it separately)...
    assert tracker.step(_probe(client, tracker)).event == "activity"
    # ...then the idle session is genuinely harmless
    outcome = tracker.step(_probe(client, tracker))
    assert outcome.event == "none"
    assert outcome.idle_confirmations == 0


def test_tracker_empty_new_assistant_counts_as_suspicion():
    client = MutableMessages()
    tracker = _fresh_tracker()
    client.add(
        assistant_message("a_empty", 4000, completed=None, finish=None, parts=[])
    )
    assert tracker.step(_probe(client, tracker)).event == "activity"
    assert tracker.step(_probe(client, tracker)).event == "idle_confirm"
    assert tracker.step(_probe(client, tracker)).event == "idle_confirm"
    assert tracker.step(_probe(client, tracker)).event == "interrupted"


def test_tracker_prebaseline_tool_calls_do_not_trigger():
    client = MutableMessages([tool_call_message("a_old", created=1000)])
    tracker = _fresh_tracker()
    tracker.reset_round(frozenset({"a_old"}))
    for _ in range(20):
        outcome = tracker.step(_probe(client, tracker))
        assert outcome.event not in ("idle_confirm", "interrupted")


def test_tracker_outcome_exposes_required_idle_and_last_finish():
    client = MutableMessages([tool_call_message("a_tool", 4000)])
    tracker = _fresh_tracker()
    outcome = tracker.step(_probe(client, tracker))
    assert outcome.required_idle == 3
    assert outcome.last_message_id == "a_tool"
    assert outcome.last_finish == "tool-calls"
    assert outcome.reason == "activity_new_message"


def test_client_session_status_normalizes_object_shapes():
    """Requirement 6: session/status payloads come back as objects from the
    desktop fork (``{"type": "busy"}``); tolerate ``status`` / ``state``
    keys too so an object-shaped idle is never misread as ``unknown`` and
    silently compared away.  Non-whitelisted types stay ``unknown``."""
    client = OpenChamberClient("http://127.0.0.1:57123")

    def with_payload(payload):
        client._get_json = lambda path: payload
        return client.session_status("ses_x", "D:/proj")

    assert with_payload({"ses_x": {"type": "busy"}}) == "busy"
    assert with_payload({"ses_x": {"type": "idle"}}) == "idle"
    assert with_payload({"ses_x": {"status": "idle"}}) == "idle"
    assert with_payload({"ses_x": {"state": "busy"}}) == "busy"
    assert with_payload({"ses_x": {"status": "retry"}}) == "retry"
    assert with_payload({"ses_x": {"type": "sleep"}}) == "unknown"
    assert with_payload({"ses_x": {"status": "SLEEPING"}}) == "unknown"
    assert with_payload({"ses_x": {"state": {"type": "idle"}}}) == "unknown"
    assert with_payload({"ses_x": None}) == "unknown"
    assert with_payload({"ses_x": "busy"}) == "busy"
    assert with_payload({}) == "idle"  # session dropped from the map == idle


def test_monitor_probe_normalizes_exotic_status_to_unknown_and_tracker_wont_count():
    """Exotic statuses (sleep/starting/...) become ``unknown`` at the probe
    boundary and may never accumulate idle confirmations."""
    client = MutableMessages([tool_call_message("a_tool", 4000)])
    client.status = "sleep"
    tracker = _fresh_tracker()
    assert tracker.step(_probe(client, tracker)).event == "activity"
    for _ in range(10):
        outcome = tracker.step(_probe(client, tracker))
        assert outcome.event == "none"
        assert outcome.idle_confirmations == 0
        assert outcome.reason == "status_'unknown'"
    client.status = "idle"
    confirming = tracker.step(_probe(client, tracker))
    assert confirming.event in ("idle_confirm", "interrupted")


def test_tracker_staged_progress_then_tool_then_empty_then_idle_interrupts():
    """Requirement 8: completed staged assistant progress text (finish
    ``tool-calls``) → real tool activity → a batch of EMPTY assistant
    messages → idle MUST trip the interruption after three confirmations.
    Staged progress must never be packaged (``new_completed`` stays empty)
    and must never clear ``activity_observed``."""
    client = MutableMessages(
        [
            tool_call_message("s1", 3000, text="staged text one"),
            tool_call_message("s2", 3400, text="staged text two"),
        ]
    )
    tracker = _fresh_tracker()
    first = tracker.step(_probe(client, tracker))
    assert (first.event, first.reason) == ("activity", "activity_new_message")
    assert tracker.activity_observed is True

    client.add(tool_call_message("tool1", 4000, text="still working"))
    second = tracker.step(_probe(client, tracker))
    assert (second.event, second.reason) == ("activity", "activity_new_message")
    assert tracker.activity_observed is True

    for i, mid in enumerate(["e1", "e2", "e3"]):
        client.add(
            assistant_message(mid, 5000 + i, completed=None, finish=None, parts=[])
        )
    assert tracker.step(_probe(client, tracker)).event == "activity"

    o1 = tracker.step(_probe(client, tracker))
    o2 = tracker.step(_probe(client, tracker))
    o3 = tracker.step(_probe(client, tracker))
    assert (o1.event, o1.idle_confirmations, o1.reason) == (
        "idle_confirm", 1, "idle_confirm",
    )
    assert (o2.event, o2.idle_confirmations) == ("idle_confirm", 2)
    assert o3.event == "interrupted"
    assert o3.reason == "monitor_idle_without_completed_reply"
    assert o3.last_finish is None

    # staged progress and empty assistants are never wrap candidates
    assert _probe(client, tracker).new_completed == ()


# -------------------------------------------------------------------- #
# UI monitor: button flow
# -------------------------------------------------------------------- #


def build_monitor_window(
    qapp,
    monkeypatch,
    tmp_path,
    initial_messages=None,
    poll_interval=0.05,
    openchamber_session_id="ses_test123",
):
    import ui as ui_mod
    from tests.test_ui_startup import build_window

    fake = MutableMessages(initial_messages)
    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory=str(tmp_path),
        openchamber_session_id=openchamber_session_id,
        poll_interval=poll_interval,
        completion_timeout=5.0,
    )
    monkeypatch.setattr(ui_mod, "OpenChamberClient", lambda url: fake)
    window = build_window(qapp, monkeypatch, tmp_path, settings=settings)
    window._startup_check_pending = False
    return window, fake


def manual_completed_ids(window):
    return {
        record["task_id"]
        for record in window._workflow.registry.completed_records()
        if record["task_id"].startswith("manual-")
    }


def test_ui_monitor_start_ignores_history(qapp, monkeypatch, tmp_path):
    """Scenario 1: starting the monitor must not wrap replies that already
    existed when it started."""
    qapp.clipboard().clear()
    old = completed_reply("a_old", created=1000, text="pre-existing reply")
    window, fake = build_monitor_window(
        qapp, monkeypatch, tmp_path, initial_messages=[old]
    )
    try:
        window._toggle_monitor()
        assert window.monitor_button.text() == "停止监控 OpenChamber"
        assert "正在监听 OpenChamber" in window.status_label.text()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            from PySide6.QtTest import QTest
            QTest.qWait(40)
        assert manual_completed_ids(window) == set()
    finally:
        drain_monitor(window)


def test_ui_monitor_wraps_new_reply_once(qapp, monkeypatch, tmp_path):
    """Scenarios 2/5/6: a new completed reply is wrapped exactly once with a
    manual-{uuid} TASK_ID and a fresh RESPONSE_ID; repeated polls never
    re-package it."""
    from PySide6.QtTest import QTest
    from core.protocol import ProtocolFormat, parse_message

    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        fake.add(completed_reply("a_fresh", created=3000, text="brand new"))
        deadline = time.monotonic() + 3.0
        text = ""
        while time.monotonic() < deadline:
            QTest.qWait(40)
            text = qapp.clipboard().text()
            if "brand new" in text:
                break
        assert "brand new" in text, "new reply was never wrapped"
        message = parse_message(text)
        assert message.protocol_format is ProtocolFormat.V1
        assert message.in_reply_to.startswith("manual-")
        assert "brand new" in message.body

        # leave some polls running: the same reply must not be re-wrapped
        QTest.qWait(400)
        assert manual_completed_ids(window) == {message.in_reply_to}

        # the reply was persisted for re-copy
        assert window._workflow.load_reply(message.in_reply_to) == text
    finally:
        drain_monitor(window)


def test_ui_monitor_wraps_message_that_completes_after_streaming(
    qapp, monkeypatch, tmp_path
):
    """Regression (MonitorPollTask seen handling): a streaming assistant
    message with no time.completed must NOT be added to the runtime baseline
    on the poll that first sees it.  Once it completes on a later poll it is
    wrapped exactly once; later polls never re-wrap it."""
    from PySide6.QtTest import QTest
    from core.protocol import ProtocolFormat, parse_message

    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        assert manual_completed_ids(window) == set()

        # first poll sees a brand-new assistant message that is still
        # streaming (created present, completed absent)
        streaming = in_progress_reply("a_stream", created=3000)
        fake.add(streaming)
        QTest.qWait(300)
        assert manual_completed_ids(window) == set()
        assert qapp.clipboard().text() == ""

        # the same message id completes with text
        streaming["info"]["time"]["completed"] = 3100
        streaming["info"]["finish"] = "stop"
        deadline = time.monotonic() + 3.0
        text = ""
        while time.monotonic() < deadline:
            QTest.qWait(40)
            text = qapp.clipboard().text()
            if "still working" in text:
                break
        assert "still working" in text, "completed reply was never wrapped"
        message = parse_message(text)
        assert message.protocol_format is ProtocolFormat.V1
        assert message.in_reply_to.startswith("manual-")
        assert "still working" in message.body

        # later polls on the same message id must not re-wrap it
        QTest.qWait(400)
        assert manual_completed_ids(window) == {message.in_reply_to}
    finally:
        drain_monitor(window)


def test_ui_monitor_wraps_consecutive_replies_separately(qapp, monkeypatch, tmp_path):
    """Scenario 3/6: two consecutive replies each get their own manual task
    and a distinct RESPONSE_ID."""
    from PySide6.QtTest import QTest
    from core.protocol import parse_message

    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        fake.add(completed_reply("a_one", created=3000, text="first"))
        fake.add(completed_reply("a_two", created=3200, text="second"))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            QTest.qWait(40)
            if len(manual_completed_ids(window)) >= 2:
                break

        assert len(manual_completed_ids(window)) == 2
        responses = {}
        for record in window._workflow.registry.completed_records():
            if not record["task_id"].startswith("manual-"):
                continue
            saved = window._workflow.load_reply(record["task_id"])
            responses[saved] = record["task_id"]
        assert len(responses) == 2
        bodies = {parse_message(saved).body for saved in responses}
        assert bodies == {"first", "second"}
        ids = [parse_message(saved).message_id for saved in responses]
        assert len(set(ids)) == 2, "each wrap must have a distinct RESPONSE_ID"
    finally:
        drain_monitor(window)


def test_ui_monitor_stop_prevents_further_wraps(qapp, monkeypatch, tmp_path):
    """Scenario 4: after stopping, new replies are no longer read/wrapped."""
    from PySide6.QtTest import QTest

    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        assert window.monitor_button.text() == "停止监控 OpenChamber"
        window._toggle_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        assert window.monitor_button.text() == "监控 OpenChamber"
        assert not window._oc_monitor_active

        fake.add(completed_reply("a_quiet", created=3000, text="ignored"))
        QTest.qWait(300)
        assert manual_completed_ids(window) == set()
    finally:
        drain_monitor(window)


def test_ui_monitor_failure_does_not_freeze_and_can_restart(qapp, monkeypatch, tmp_path):
    """Scenario 8: transport failures are retried (re-query only, never any
    send), then reported as unusable; the monitor stays stoppable and can be
    restarted once the server recovers."""
    from PySide6.QtTest import QTest
    from core.protocol import parse_message

    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    window._monitor_transport_delays = (0.3, 0.3)
    try:
        window._toggle_monitor()
        fake.fail_all = True
        # retries first, then the connection is declared unusable
        retry_label = (
            f"正在重连（{MONITOR_MAX_TRANSPORT_RETRIES}/"
            f"{MONITOR_MAX_TRANSPORT_RETRIES}）"
        )
        assert wait_until(
            lambda: retry_label in window.status_label.text()
        ), "transport retry status never appeared"
        assert wait_until(
            lambda: "连接多次中断" in window.status_label.text()
        ), "transport exhaustion never reported"
        assert window._oc_monitor_transport_down is True
        assert window.continue_button.isEnabled() is False
        assert window.stop_button.isEnabled() is True
        assert window.monitor_button.text() == "停止监控 OpenChamber"

        # still stoppable
        window._toggle_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        assert window.monitor_button.text() == "监控 OpenChamber"
        assert not window._oc_monitor_active

        # and restartable once the server recovers
        fake.fail_all = False
        window._toggle_monitor()
        assert window.monitor_button.text() == "停止监控 OpenChamber"
        fake.add(completed_reply("a_again", created=3000, text="recovered"))
        assert wait_until(lambda: "recovered" in qapp.clipboard().text())
        assert "recovered" in parse_message(qapp.clipboard().text()).body
    finally:
        drain_monitor(window)


def test_ui_monitor_needs_configured_session(qapp, monkeypatch, tmp_path):
    from PySide6.QtTest import QTest

    window, fake = build_monitor_window(
        qapp, monkeypatch, tmp_path, openchamber_session_id=""
    )
    try:
        window._toggle_monitor()
        assert window.monitor_button.text() == "监控 OpenChamber"
        assert "会话 ID" in window.detail_label.text()
        assert not window._oc_monitor_active
    finally:
        drain_monitor(window)


def test_stopped_task_keeps_late_reply_wrappable(tmp_path):
    """After the operator stops the wait, a LATE completed reply that arrives
    afterwards is not marked wrapped and stays wrappable by the manual
    OpenChamber monitor (which the requirement guarantees)."""
    oc = BusyOpenChamber()
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    cancel = threading.Event()
    errors: list[BaseException] = []

    def run():
        try:
            wf.process(v1_task(TARGET_OPENCHAMBER, "do it"), cancel_event=cancel)
        except BaseException as exc:  # noqa: BLE001 - collected for asserts
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.2)
    cancel.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(errors[0], OpenChamberCancelledError)

    # the session still exists; a late reply appears AFTER the stop
    assert oc.session_exists("ses_test123", "D:/proj")
    oc.history.append(
        assistant_message(
            "a_late", 5000, completed=5100, finish="stop",
            parts=[text_part("late answer")],
        )
    )
    assert is_message_wrapped("a_late", "ses_test123") is False
    scan = monitor_scan(oc, "ses_test123", "D:/proj", frozenset())
    assert ("a_late", "late answer") in scan.new_completed


# -------------------------------------------------------------------- #
# manual monitor interruption: continue/stop flow and transport handling
# -------------------------------------------------------------------- #


def build_tiny_delay_monitor(qapp, monkeypatch, tmp_path, **kw):
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path, **kw)
    window._monitor_transport_delays = (0.05, 0.05)
    return window, fake


def drain_monitor(window):
    """Stop the manual monitor and WAIT until every monitor worker emitted
    ``finished`` before closing, so a running poll / continue / recovery
    thread can never outlive the window (the access-violation race)."""
    shutdown_window(window)


def test_ui_monitor_interrupted_enables_continue_and_sends_fixed_prompt_once(
    qapp, monkeypatch, tmp_path
):
    from PySide6.QtTest import QTest

    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        # Manual-only mode (auto-recovery disabled): the MANUAL "继续当前任务"
        # button flow exercises the exact same one-shot fixed-prompt send.
        window._oc_monitor_auto_disabled = True
        fake.add(tool_call_message("a_tool", 4000))
        assert wait_until(lambda: window._oc_monitor_interrupted)
        assert "异常停顿" in window.status_label.text()
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True

        # one click -> one fixed-prompt send into the ORIGINAL session
        window._continue_task()
        assert wait_until(lambda: len(fake.sent) == 1), "monitor continue never sent"
        session_id, directory, prompt, agent, model = fake.sent[0]
        assert session_id == "ses_test123"
        assert directory == str(tmp_path)
        assert prompt == MONITOR_CONTINUE_PROMPT

        # no further send on the poller's own subsequent polls
        QTest.qWait(400)
        assert len(fake.sent) == 1

        # continuation created NO TASK and did NOT touch auto rotation
        assert not (tmp_path / "tasks.json").exists()
        assert window._rotation.count(str(tmp_path)) == 0
    finally:
        drain_monitor(window)


def test_ui_monitor_auto_relay_claim_beats_monitor_continue(
    qapp, monkeypatch, tmp_path
):
    """When BOTH an auto-relay interrupted task and a monitor interruption
    exist, "继续当前任务" serves the auto relay; the manual monitor send is
    never triggered."""
    import ui as ui_mod
    from core.protocol import parse_message
    from PySide6.QtCore import QRunnable

    class StubContinueTask(QRunnable):
        def __init__(self, workflow, cancel_event):
            super().__init__()
            self.workflow = workflow
            self.cancel_event = cancel_event
            self.signals = ui_mod.WorkerSignals()
            constructed.append(self)

        def run(self):
            # A real ContinueTask emits ``finished`` at every exit path; the
            # strict worker drain relies on it, so the stub must too.
            self.signals.finished.emit()

    constructed = []
    monkeypatch.setattr(ui_mod, "ContinueTask", StubContinueTask)
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        # Manual-only mode: auto-recovery disabled, so the stall lands on the
        # plain interrupted state (the auto-relay claim is what is under test).
        window._oc_monitor_auto_disabled = True
        fake.add(tool_call_message("a_tool", 4000))
        assert wait_until(lambda: window._oc_monitor_interrupted)

        message = parse_message(v1_task("OPENCHAMBER", "do it"))
        window._workflow.pending_continue = OpenChamberContinue(
            message=message, session_id="ses_auto", directory="D:/proj",
        )
        window._set_controls_enabled(True)
        window._continue_task()

        assert len(constructed) == 1, "auto-relay ContinueTask was not created"
        assert window._current_task_is_auto is False  # manual continue path
        assert fake.sent == [], "monitor continue must not fire when auto wins"
    finally:
        window._busy = False
        window._task_cancel_event = None
        window._workflow.pending_continue = None
        drain_monitor(window)


def test_ui_monitor_stop_after_interrupted_clears_state_keeps_session(
    qapp, monkeypatch, tmp_path
):
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        # manual-only mode for this flow
        window._oc_monitor_auto_disabled = True
        fake.add(tool_call_message("a_tool", 4000))
        assert wait_until(lambda: window._oc_monitor_interrupted)
        assert window.continue_button.isEnabled() is True

        window._toggle_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        assert window.monitor_button.text() == "监控 OpenChamber"
        assert not window._oc_monitor_active
        assert not window._oc_monitor_interrupted
        assert window.continue_button.isEnabled() is False
        assert "已停止监听 OpenChamber，会话和历史记录均已保留。" in window.status_label.text()

        # late replies after stop are never wrapped
        from PySide6.QtTest import QTest

        fake.add(completed_reply("a_late", created=6000, text="ignored"))
        QTest.qWait(300)
        assert manual_completed_ids(window) == set()
    finally:
        drain_monitor(window)


def test_ui_monitor_stale_interruption_after_stop_is_ignored(
    qapp, monkeypatch, tmp_path
):
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        stopped_generation = window._monitor_generation  # monitor run's gen
        window._toggle_monitor()  # stopped (generation bumped again)
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        # a stale signal from the previous generation must be ignored entirely
        window._on_monitor_interrupted(
            {
                "generation": stopped_generation,
                "session_id": "ses_test123",
                "directory": str(tmp_path),
                "last_message_id": "a_tool",
                "last_finish": "tool-calls",
                "reason": "monitor_idle_without_completed_reply",
            }
        )
        assert window._oc_monitor_interrupted is False
        assert window.continue_button.isEnabled() is False
        assert window.stop_button.isEnabled() is False
    finally:
        drain_monitor(window)


def test_ui_session_manual_switch_stops_monitor(qapp, monkeypatch, tmp_path):
    from PySide6.QtTest import QTest

    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        assert window._oc_monitor_active
        window._on_session_manual_switch(0)
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "session switch did not drain the monitor"
        assert not window._oc_monitor_active
        assert window.monitor_button.text() == "监控 OpenChamber"
        fake.add(completed_reply("a_new", created=3000, text="ignored"))
        QTest.qWait(300)
        assert manual_completed_ids(window) == set()
    finally:
        drain_monitor(window)


def test_ui_monitor_wrap_after_continue_resets_abnormal_state(
    qapp, monkeypatch, tmp_path
):
    from PySide6.QtTest import QTest

    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        # Manual-only mode (auto-recovery disabled for this regression).
        window._oc_monitor_auto_disabled = True
        fake.add(tool_call_message("a_tool", 4000))
        assert wait_until(lambda: window._oc_monitor_interrupted)
        window._continue_task()
        assert wait_until(lambda: len(fake.sent) == 1)

        # the round later completes with a real reply: it is wrapped and the
        # abnormal state is cleared (continue no longer offered)
        fake.add(completed_reply("a_done", created=5000, text="the answer"))
        assert wait_until(lambda: "the answer" in qapp.clipboard().text())
        assert not window._oc_monitor_interrupted
        assert not window._oc_monitor_continuing
        assert window.continue_button.isEnabled() is False
        # still wrapping once only
        QTest.qWait(300)
        assert len(manual_completed_ids(window)) == 1
    finally:
        drain_monitor(window)


def test_ui_monitor_staged_progress_never_wraps_then_empty_assistant_interrupts(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement 8 regression at the UI level: completed staged assistant
    text (finish ``tool-calls``) and a flush of EMPTY assistant messages must
    ONLY drive the tracker (no wrapping), and the abnormal idle must be
    reported with idle-1/3..3/3 audit lines in relay.log.  Auto-recovery is
    disabled so the stall is observed in the plain interrupted state."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        # manual-only regression
        window._oc_monitor_auto_disabled = True
        fake.add(tool_call_message("s1", 3000, text="staged text one"))
        fake.add(tool_call_message("s2", 3400, text="staged text two"))
        fake.add(tool_call_message("tool1", 4000, text="still working"))
        for i, mid in enumerate(["e1", "e2", "e3", "e4"]):
            fake.add(
                assistant_message(mid, 5000 + i, completed=None, finish=None, parts=[])
            )
        assert wait_until(lambda: window._oc_monitor_interrupted)
        assert "异常停顿" in window.status_label.text()
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True

        # staged progress and empty assistants are never wrapped / packaged
        assert window._oc_monitor_seen == set()
        assert not is_message_wrapped("s1", "ses_test123")
        assert not is_message_wrapped("tool1", "ses_test123")
        assert manual_completed_ids(window) == set()

        # audit trail exists with the required milestones (req 7)
        lines = [f"{r.levelname} {r.message}" for r in caplog.records]
        assert any("event=idle_confirm idle=1/3" in line for line in lines)
        assert any("event=idle_confirm idle=2/3" in line for line in lines)
        assert any("event=interrupted idle=3/3" in line for line in lines)
        assert any("reason=activity_new_message" in line for line in lines)

        # the interruption was reached in roughly 3 poll intervals (req 5)
        assert window._oc_monitor_interrupted
        QTest.qWait(300)
        assert manual_completed_ids(window) == set()

        # no tasks.json was written on the detection path
        assert not (tmp_path / "tasks.json").exists()
    finally:
        drain_monitor(window)


def test_ui_monitor_busy_without_progress_logs_evidence_but_never_resumes(
    qapp, monkeypatch, tmp_path, caplog
):
    """Evidence-only "busy but no progress" line: after the session keeps
    reporting ``busy`` with an unchanged message signature, relay.log gets a
    periodic marker; it must NOT auto-send a continuation (tools may still be
    running)."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_stale_log_interval = 0.1
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(
            lambda: any(
                "会话仍报告 busy，但消息没有进展" in r.message
                for r in caplog.records
            )
        )
        # behaviour-neutral: no auto-continue, no wrap, still listening
        assert fake.sent == []
        assert window._oc_monitor_active
        assert manual_completed_ids(window) == set()
        assert not window._oc_monitor_interrupted
    finally:
        drain_monitor(window)


def test_ui_monitor_busy_with_activity_never_logs_stale(
    qapp, monkeypatch, tmp_path, caplog
):
    """A genuinely working busy session produces no "no progress" marker
    while messages keep arriving; once it truly stalls the marker appears,
    proving the reset-on-activity behaviour without auto-resuming."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_stale_log_interval = 0.3
    try:
        window._toggle_monitor()
        fake.status = "busy"
        for i in range(20):
            fake.add(tool_call_message(f"t{i}", 4000 + i * 10, text=f"step {i}"))
            QTest.qWait(50)
        for r in caplog.records:
            assert "会话仍报告 busy，但消息没有进展" not in r.message
        # once activity stops, the stale marker appears (interval 0.3s)
        assert wait_until(
            lambda: any(
                "会话仍报告 busy，但消息没有进展" in r.message
                for r in caplog.records
            )
        )
        assert fake.sent == []  # never auto-resumes
        assert not window._oc_monitor_interrupted
    finally:
        drain_monitor(window)


# -------------------------------------------------------------------- #
# busy-without-progress trigger: 120s (configurable for tests) of a
# COMPLETELY unchanged busy signature fires the listening interruption
# exactly once per stuck period, without ever auto-continuing.
# -------------------------------------------------------------------- #


class _InterruptionSpy(QObject):
    """Main-thread QObject slot capturing the ``interrupted`` signal.

    Plain cross-thread Python lambdas are not delivered reliably by PySide6
    in this pytest/Qt environment; a QObject receiver is, exactly like the
    window's own ``_on_monitor_interrupted`` slot."""

    def __init__(self):
        super().__init__()
        self.fired: list[dict] = []

    @Slot(object)
    def capture(self, payload):
        self.fired.append(payload)


def test_ui_monitor_busy_stuck_below_threshold_never_triggers(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: 119s of unchanged busy does NOT trigger; with the test
    threshold set well above the probing window the monitor keeps listening
    and never auto-sends."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 5.0
    window._monitor_busy_stale_log_interval = 0.1
    spy = _InterruptionSpy()
    try:
        window._toggle_monitor()
        window._oc_monitor_task.signals.interrupted.connect(spy.capture)
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        QTest.qWait(400)  # far below the 5s threshold
        assert spy.fired == []
        assert not window._oc_monitor_interrupted
        assert window.continue_button.isEnabled() is False
        assert fake.sent == []  # never auto-sends below the threshold
    finally:
        drain_monitor(window)


def test_ui_monitor_busy_stuck_at_threshold_triggers_once(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement (manual-only mode): when the unchanged busy signature
    reaches the 120s threshold (compressed here), the listening interruption
    fires with the dedicated reason, the exact UI copy, continue/stop
    enabled, new-session disabled, NO send -- and later identical polls do
    NOT repeat the interruption.  Auto-recovery is disabled so the MANUAL
    stall offer is observed; the auto path is covered by its own tests."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.15
    window._monitor_busy_stale_log_interval = 0.1
    spy = _InterruptionSpy()
    try:
        window._toggle_monitor()
        window._oc_monitor_auto_disabled = True  # manual-only mode
        window._oc_monitor_task.signals.interrupted.connect(spy.capture)
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(
            lambda: (
                window._oc_monitor_interrupted
                and len(spy.fired) == 1
                and "长时间没有进展" in window.status_label.text()
            )
        )
        assert len(spy.fired) == 1
        assert spy.fired[0]["reason"] == "monitor_busy_without_progress"
        assert spy.fired[0]["last_message_id"] == "a_tool"
        assert spy.fired[0]["last_finish"] == "tool-calls"

        assert "长时间没有进展" in window.status_label.text()
        assert "超过120秒没有新消息或内容变化" in window.detail_label.text()
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True
        assert window.new_session_button.isEnabled() is False
        assert fake.sent == []  # never auto-continues

        # once emitted, identical stuck polls must NOT re-emit (no 30s spam)
        QTest.qWait(300)
        assert len(spy.fired) == 1
        assert window._oc_monitor_interrupted

        # the dedicated audit line carries the reason
        assert any(
            "reason=monitor_busy_without_progress" in r.message
            and r.levelname == "WARNING"
            for r in caplog.records
        )
    finally:
        drain_monitor(window)


def test_ui_monitor_busy_stuck_activity_resets_threshold(
    qapp, monkeypatch, tmp_path
):
    """Requirement: any message change (new id / text growth / finish / tool
    count) while busy restarts the 120s window, so a genuinely producing long
    tool run never trips.  Once the activity stops, the unchanged window
    starts and the interruption eventually fires."""
    from PySide6.QtTest import QTest

    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.25
    window._monitor_busy_stale_log_interval = 0.1
    try:
        window._toggle_monitor()
        window._oc_monitor_auto_disabled = True  # manual-only mode
        fake.status = "busy"
        # continuous progress: never allow an unchanged window to accumulate
        for i in range(8):
            if not window._oc_monitor_interrupted:
                fake.add(tool_call_message(f"t{i}", 4000 + i * 10, text=f"step {i}"))
            QTest.qWait(60)
        assert not window._oc_monitor_interrupted
        assert fake.sent == []
        # once additions stop, the unchanged busy window trips after ~0.25s
        assert wait_until(lambda: window._oc_monitor_interrupted)
        assert window.continue_button.isEnabled() is True
    finally:
        drain_monitor(window)


def test_ui_monitor_busy_stuck_continue_sends_once_and_resets_timer(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement (manual-only mode): after the stuck interruption the
    operator's single "继续当前任务" click sends the fixed prompt exactly once
    (no TASK, no tasks.json, no rotation); the successful send zeroes the
    busy timer and listening continues, so only a NEW full stuck window may
    re-alert.  Auto-recovery is disabled to isolate the manual click path."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.1
    window._monitor_busy_stale_log_interval = 0.1
    spy = _InterruptionSpy()
    try:
        window._toggle_monitor()
        window._oc_monitor_auto_disabled = True  # manual-only mode
        window._oc_monitor_task.signals.interrupted.connect(spy.capture)
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(
            lambda: window._oc_monitor_interrupted and len(spy.fired) == 1
        )
        assert len(spy.fired) == 1

        # continuing is a manual one-shot: exactly one fixed-prompt send
        window._continue_task()
        assert wait_until(lambda: len(fake.sent) == 1)
        session_id, directory, prompt, agent, model = fake.sent[0]
        assert session_id == "ses_test123"
        assert prompt == MONITOR_CONTINUE_PROMPT
        QTest.qWait(200)
        assert len(fake.sent) == 1  # the poller never auto-sends
        assert not (tmp_path / "tasks.json").exists()
        assert window._rotation.count(str(tmp_path)) == 0
        # the successful continue zeroed the busy timer (the reset poll is
        # asynchronous: wait for its evidence line)
        assert wait_until(
            lambda: any(
                "监听续接后清零 busy 无进展计时" in r.message
                for r in caplog.records
            )
        )

        # a single busy-stuck period fires at most once; only a NEW full
        # unchanged window (0.1s here) can re-alert after the reset
        assert wait_until(
            lambda: len(spy.fired) == 2 and window._oc_monitor_interrupted
        )
    finally:
        drain_monitor(window)


def test_ui_monitor_busy_with_activity_never_misjudges(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: a long tool execution that continuously reports busy AND
    keeps producing message/status changes must never be flagged as stuck,
    even with a very small threshold."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.2
    window._monitor_busy_stale_log_interval = 0.1
    spy = _InterruptionSpy()
    try:
        window._toggle_monitor()
        window._oc_monitor_task.signals.interrupted.connect(spy.capture)
        fake.status = "busy"
        for i in range(12):
            fake.add(tool_call_message(f"m{i}", 4000 + i * 10, text=f"progress {i}"))
            QTest.qWait(50)
        assert spy.fired == []
        assert not window._oc_monitor_interrupted
        assert window.continue_button.isEnabled() is False
        assert fake.sent == []
    finally:
        drain_monitor(window)


def test_busy_without_progress_feature_leaves_auto_relay_untouched(tmp_path):
    """Requirement: the manual-monitor busy-stuck feature lives only inside
    MonitorPollTask; the AUTO relay (wait_for_completion) keeps completing
    busy-then-idle rounds and marking them wrapped as before."""
    from tests.fakes import ScriptedOpenChamber
    from tests.test_relay_workflow import make_workflow, v1_task

    oc = ScriptedOpenChamber(directory="D:/proj")
    oc.status_timeline = ["busy", "busy", "retry", "idle"]
    oc.message_timelines = [
        [
            user_message("u_new", "do it", 1000),
            assistant_message(
                "a_new", 1100, completed=1200, finish="stop",
                parts=[text_part("final answer")],
                parent_id="u_new",
            ),
        ],
    ]
    wf = make_workflow(tmp_path, oc=oc, default=TARGET_OPENCHAMBER)
    response = wf.process(v1_task(TARGET_OPENCHAMBER, "do it"))
    assert "final answer" in response
    assert is_message_wrapped("a_new", "ses_test123")
    # a manual monitor scanning the same session must never re-wrap it
    scan = monitor_scan(oc, "ses_test123", "D:/proj", frozenset())
    assert scan.new_completed == ()


# -------------------------------------------------------------------- #
# auto-recovery: a detected stall (busy 120s / idle 3x) auto-continues the
# ORIGINAL session, at most 3 attempts, 10s-ish interval (test-compressed);
# real final reply or stop resets the budget; transport failures re-query
# before counting; non-retryable failures stop the auto path entirely.
# -------------------------------------------------------------------- #


def test_auto_continue_first_stall_sends_once_after_interval(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: the FIRST stall (busy-without-progress here) triggers ONE
    auto continuation into the ORIGINAL session after the short interval --
    fixed continue prompt, current session/Agent/Model, no TASK, no
    tasks.json, no rotation; the per-round counter becomes 1."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.6
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.05
    spy = _InterruptionSpy()
    try:
        window._toggle_monitor()
        window._oc_monitor_task.signals.interrupted.connect(spy.capture)
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        # the stall is claimed by the AUTO path: pending, attempt 1, counter 1
        assert wait_until(
            lambda: (
                len(spy.fired) == 1
                and window._oc_monitor_auto_pending
                and not window._oc_monitor_interrupted
                and window._oc_monitor_auto_count == 1
                and "自动续接 1/3" in window.status_label.text()
            )
        )
        assert wait_until(lambda: len(fake.sent) == 1)
        # the auto worker emitted ``finished``: the set is stable again
        assert wait_until(lambda: _auto_workers_drained(window))
        session_id, directory, prompt, agent, model = fake.sent[0]
        assert session_id == "ses_test123"
        assert directory == str(tmp_path)
        assert prompt == MONITOR_CONTINUE_PROMPT
        assert window._oc_monitor_auto_count == 1
        # The success handler clears ``_oc_monitor_auto_pending`` and writes
        # the durable detail label.  Do NOT wait on the transient status label:
        # the running poller overwrites it with busy-progress text, so that
        # text is only guaranteed at the very first success transition above.
        assert wait_until(
            lambda: (
                not window._oc_monitor_auto_pending
                and "已发送续接提示" in window.detail_label.text()
            )
        )
        # the successful send zeroed the busy timer: the window-owned reset
        # event is armed by the success handler and consumed (cleared) by the
        # next poll that observes the still-busy state.  Waiting for the
        # cleared event -- not a transient log line -- is the deterministic
        # proof the reset poll happened and no TASK / rotation is produced.
        assert wait_until(
            lambda: (
                not window._oc_monitor_busy_reset.is_set()
                and not window._oc_monitor_auto_pending
            )
        )
        assert any("自动续接 1/3 发送成功" in r.message for r in caplog.records)
        assert any("自动续接 1/3 开始" in r.message for r in caplog.records)
        assert not (tmp_path / "tasks.json").exists()
        assert window._rotation.count(str(tmp_path)) == 0
        # stop before the same signature can stall a SECOND time, then wait
        # until every monitor worker drained (no late send can be added)
        window._toggle_monitor()
        assert wait_until(
            lambda: (
                not window._monitor_workers
                and not window._oc_monitor_stopping
            )
        )
        assert len(fake.sent) == 1
    finally:
        drain_monitor(window)


def test_auto_continue_second_stall_triggers_second_send(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: after the first auto continuation succeeds, a SECOND full
    stall fires attempt 2/3 -- the counter is KEPT across stalls (only a real
    final reply or a stop zeroes it)."""
    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.25
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.05
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(
            lambda: (
                len(fake.sent) == 2
                and window._oc_monitor_auto_count == 2
                and not window._oc_monitor_auto_pending
                and "已自动续接 2/3" in window.status_label.text()
            )
        )
        session_id, directory, prompt, agent, model = fake.sent[1]
        assert session_id == "ses_test123"
        assert prompt == MONITOR_CONTINUE_PROMPT
        assert any("自动续接 2/3 发送成功" in r.message for r in caplog.records)
        # the second auto worker emitted ``finished`` and the set is stable
        assert wait_until(lambda: _auto_workers_drained(window))
        # a third stall would begin attempt 3: stop before it lands, then wait
        # until every monitor worker drained (no late send can be added)
        window._toggle_monitor()
        assert wait_until(
            lambda: (
                not window._monitor_workers
                and not window._oc_monitor_stopping
            )
        )
        assert len(fake.sent) == 2
    finally:
        drain_monitor(window)


def test_auto_continue_max_three_attempts(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: at most 3 auto continuations per round.  All three
    stalls are served automatically, then the state lands in the manual
    exhausted copy "自动恢复3次仍未完成，请人工继续或停止" with continue/stop
    restored."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.1
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.02
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(
            lambda: (
                window._oc_monitor_auto_count == 3
                and window._oc_monitor_auto_disabled
                and window._oc_monitor_interrupted
                and len(fake.sent) == 3
                and "自动恢复3次仍未完成" in window.status_label.text()
            )
        )
        assert not window._oc_monitor_auto_pending
        # all three auto workers emitted ``finished`` and the set is stable
        assert wait_until(lambda: _auto_workers_drained(window))
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True
        assert "已自动续接3次" in window.detail_label.text()
        assert any("自动续接 3/3 发送成功" in r.message for r in caplog.records)
        assert any("自动恢复次数已达3次，转入人工等待" in r.message for r in caplog.records)
    finally:
        drain_monitor(window)


def test_auto_continue_no_fourth_send(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: after 3 attempts are used up a FOURTH (and any further)
    stall never sends again -- the detached manual state is kept, no matter
    how many identical stuck periods pass."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.1
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.02
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        # run through all 3 attempts to the exhausted manual state
        assert wait_until(
            lambda: (
                window._oc_monitor_auto_disabled
                and window._oc_monitor_interrupted
                and len(fake.sent) == 3
            )
        )
        # wait for the last auto worker's ``finished`` so the observation
        # below starts from a stable (fully drained) state
        assert wait_until(lambda: _auto_workers_drained(window))
        # keep many MORE identical busy stalls running: no fourth send ever
        QTest.qWait(600)
        assert len(fake.sent) == 3
        assert window._oc_monitor_auto_count == 3
        assert window._oc_monitor_auto_disabled
        assert window.continue_button.isEnabled() is True
    finally:
        drain_monitor(window)


def test_auto_continue_budget_resets_after_wrapped_final_reply(
    qapp, monkeypatch, tmp_path
):
    """Requirement: a REAL final reply (completed + finish=stop + non-empty
    text) resets the auto-recovery budget; a fresh stall afterwards starts a
    NEW round from attempt 1/3, not 2/3."""
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.2
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.05
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_auto_tool5", 4000, text="working hard"))
        assert wait_until(lambda: len(fake.sent) == 1)
        assert window._oc_monitor_auto_count == 1
        # the session finally completes: the reply is wrapped once
        fake.add(
            completed_reply("a_auto_final", created=5000, text="final answer")
        )
        assert wait_until(lambda: "final answer" in qapp.clipboard().text())
        assert not window._oc_monitor_auto_pending
        # the wrapped final reply zeroes the budget and re-arms listening
        assert wait_until(lambda: window._oc_monitor_auto_count == 0)
        assert not window._oc_monitor_auto_disabled
        # a fresh full stall starts attempt 1/3 AGAIN (not 2/3)
        assert wait_until(
            lambda: (
                len(fake.sent) == 2
                and window._oc_monitor_auto_count == 1
                and "已自动续接 1/3" in window.status_label.text()
            )
        )
        window._toggle_monitor()
        assert len(fake.sent) == 2
    finally:
        drain_monitor(window)


def test_auto_continue_stop_aborts_pending_send(qapp, monkeypatch, tmp_path):
    """Requirement: "停止当前任务" cancels a waiting auto continuation: the
    stop event aborts the in-flight interval wait and NO send may ever land;
    the monitor is stopped with all state reset."""
    from PySide6.QtTest import QTest

    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.15
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 1.0  # long wait, easy to cancel
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        # the pending auto continuation entered its wait window
        assert wait_until(
            lambda: window._oc_monitor_auto_pending and window._oc_monitor_auto_count == 1
        )
        window._stop_task()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        assert not window._oc_monitor_active
        assert not window._oc_monitor_auto_pending
        assert fake.sent == []
        QTest.qWait(300)
        assert fake.sent == []
        assert window.monitor_button.text() == "监控 OpenChamber"
    finally:
        drain_monitor(window)


def test_auto_continue_http400_stops_auto_recovery(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: a non-retryable send failure (HTTP 400) is NEVER
    auto-continued: the failed attempt stops the auto recovery, restores the
    manual offer and later stalls never re-send."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.15
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.05
    fake.send_error = OpenChamberBadRequestError(
        "OpenChamber rejected the request (HTTP 400): malformed"
    )
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(
            lambda: (
                window._oc_monitor_auto_disabled
                and window._oc_monitor_interrupted
                and not window._oc_monitor_auto_pending
                and "已停止自动恢复" in window.status_label.text()
            )
        )
        assert window._oc_monitor_auto_count == 1  # the failed attempt counted
        assert window.continue_button.isEnabled() is True
        assert window.stop_button.isEnabled() is True
        assert any("不可重试" in r.message for r in caplog.records)
        # further identical stalls never auto-send again
        QTest.qWait(400)
        assert fake.sent == []
        assert window._oc_monitor_auto_disabled
    finally:
        drain_monitor(window)


def test_monitor_worker_registry_emptied_after_done(qapp, monkeypatch, tmp_path):
    """Requirement: the monitor worker registry (and the strong task refs held
    to keep ``finished`` deliveries safe) is empty once the monitor stopped."""
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        assert window._monitor_workers, "poll worker must be tracked"
        assert window._monitor_task_refs
        window._toggle_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        assert window._monitor_workers == set()
        assert window._monitor_task_refs == {}
        assert window.monitor_button.text() == "监控 OpenChamber"
    finally:
        drain_monitor(window)


def test_monitor_stop_during_recovery_query_drains(
    qapp, monkeypatch, tmp_path,
):
    """Requirement: stopping while a recovery-query (transport-failed re-check)
    is in flight aborts the query wait via the stop event and the worker
    drains cleanly -- no hang, no crash, state fully reset."""
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.15
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.05
    window._monitor_transport_delays = (0.5, 0.5)  # keep the query in flight
    fake.send_error = OpenChamberUnavailableError("connection reset")
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool2", 4000, text="working hard"))
        # the auto send failed with a transport error and a recovery-query
        # worker was dispatched (delays keep it pending)
        assert wait_until(
            lambda: "发送失败（传输）" in window.status_label.text()
        )
        window._toggle_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "recovery worker did not drain"
        assert window.monitor_button.text() == "监控 OpenChamber"
        assert not window._oc_monitor_active
    finally:
        drain_monitor(window)


def test_window_close_during_auto_continue_drains_and_closes(
    qapp, monkeypatch, tmp_path,
):
    """Requirement: closing the window while an auto-continue worker is in
    flight must not destroy it under a live worker -- the close is deferred,
    the monitor is asked to stop, and the window only hides once every
    worker emitted ``finished`` (bounded by the close-check timer)."""
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.2
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.5  # still waiting when we close
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool4", 4000, text="working hard"))
        assert wait_until(lambda: window._oc_monitor_auto_pending)
        window.close()
        assert window.isVisible(), "close must be deferred while workers live"
        assert wait_until(
            lambda: not window.isVisible()
        ), "window never closed after the workers drained"
        assert not window._oc_monitor_active
        assert window.monitor_button.text() == "监控 OpenChamber"
    finally:
        drain_monitor(window)


def test_monitor_fast_start_stop_switch_session_restart(
    qapp, monkeypatch, tmp_path,
):
    """Requirement: a fast start -> stop -> manual session switch -> restart
    cycle never overlaps two monitor generations and every stop drains fully."""
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        assert window._oc_monitor_active
        window._toggle_monitor()
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "first stop did not drain"
        window._on_session_manual_switch(0)
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "session switch did not drain"
        window._toggle_monitor()
        assert window._oc_monitor_active
        fake.add(completed_reply("a_cycle", created=3000, text="cycle done"))
        assert wait_until(lambda: "cycle done" in qapp.clipboard().text())
    finally:
        drain_monitor(window)


def test_monitor_stale_generation_signals_do_not_update_new_ui(
    qapp, monkeypatch, tmp_path,
):
    """Requirement: a late signal from an OLD monitor generation arriving after
    a NEW run started must leave the new UI (and its state) untouched."""
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    try:
        window._toggle_monitor()
        old_gen = window._monitor_generation
        window._toggle_monitor()  # stop bumps the generation again
        assert wait_until(
            lambda: not window._monitor_workers and not window._oc_monitor_stopping
        ), "stop did not drain"
        window._toggle_monitor()  # a fresh run bumps it once more
        assert window._monitor_generation > old_gen

        # a stale interruption from the old generation is ignored entirely
        window._on_monitor_interrupted(
            {
                "generation": old_gen,
                "session_id": "ses_test123",
                "directory": str(tmp_path),
                "last_message_id": "a_tool",
                "last_finish": "tool-calls",
                "reason": "monitor_idle_without_completed_reply",
            }
        )
        assert window._oc_monitor_interrupted is False
        assert window.continue_button.isEnabled() is False
        # stale auto-continue success/failure and stale status updates are
        # dropped by the generation guard
        window._guard(old_gen, window._on_auto_continue_succeeded, "x")
        assert "已发送续接提示" not in window.status_label.text()
        window._guard(old_gen, window._on_auto_continue_failed, "boom", "non_retryable")
        assert window._oc_monitor_auto_disabled is False
        window._guard(old_gen, window._set_status, "STALE")
        assert "STALE" not in window.status_label.text()
        # the live (new) run is unaffected
        fake.add(completed_reply("a_live2", created=3000, text="live answer"))
        assert wait_until(lambda: "live answer" in qapp.clipboard().text())
    finally:
        drain_monitor(window)


def test_auto_continue_duplicate_interrupts_do_not_concur(
    qapp, monkeypatch, tmp_path, caplog
):
    """Requirement: a continuation already waiting must never be stacked with
    a concurrent one -- duplicate interruption signals arriving during the
    wait are dropped, and exactly one send lands afterwards."""
    from PySide6.QtTest import QTest

    caplog.set_level(logging.INFO, logger="ai_relay_b")
    qapp.clipboard().clear()
    window, fake = build_tiny_delay_monitor(qapp, monkeypatch, tmp_path)
    window._monitor_busy_without_progress_threshold = 0.2
    window._monitor_busy_stale_log_interval = 0.1
    window._monitor_auto_continue_interval = 0.3  # long wait to inject dups
    try:
        window._toggle_monitor()
        fake.status = "busy"
        fake.add(tool_call_message("a_tool", 4000, text="working hard"))
        assert wait_until(lambda: window._oc_monitor_auto_pending)
        duplicate = {
            "generation": window._monitor_generation,
            "session_id": "ses_test123",
            "directory": str(tmp_path),
            "last_message_id": "a_tool",
            "last_finish": "tool-calls",
            "reason": "monitor_busy_without_progress",
        }
        window._on_monitor_interrupted(dict(duplicate))
        window._on_monitor_interrupted(dict(duplicate))
        QTest.qWait(120)
        assert window._oc_monitor_auto_count == 1
        assert len(fake.sent) == 0  # still waiting, nothing stacked
        assert any("忽略重复中断信号" in r.message for r in caplog.records)
        # only the single originally scheduled send lands afterwards
        assert wait_until(lambda: len(fake.sent) == 1)
        assert window._oc_monitor_auto_count == 1
        window._toggle_monitor()
        assert len(fake.sent) == 1
    finally:
        drain_monitor(window)


# -------------------------------------------------------------------- #
# 任务接收 vs OpenChamber monitor: pause independence and resume
# -------------------------------------------------------------------- #


def test_pause_does_not_stop_running_monitor(qapp, monkeypatch, tmp_path):
    """Scenario 8: pausing 任务接收 stops ONLY the clipboard listener; an
    OpenChamber monitor already running keeps polling and wrapping, and new
    clipboard tasks are still ignored while paused."""
    from PySide6.QtTest import QTest

    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: not window._busy)
        window._toggle_monitor()
        assert window._oc_monitor_active

        window._pause()
        assert not window._listener.enabled
        assert window._clipboard_status_label.text() == "任务接收：已暂停"
        assert window._monitor_status_label.text() == "OpenChamber监控：运行中"

        fake.add(completed_reply("a_paused", created=3000, text="while paused"))
        deadline = time.monotonic() + 3.0
        text = ""
        while time.monotonic() < deadline:
            QTest.qWait(40)
            text = qapp.clipboard().text()
            if "while paused" in text:
                break
        assert "while paused" in text, "monitor stopped while clipboard paused"

        qapp.clipboard().setText(
            v1_task("REASONIX", "new while paused", "task-mon-pause-001")
        )
        QTest.qWait(300)
        assert window._workflow.registry.record("task-mon-pause-001") is None
        assert window.monitor_button.text() == "停止监控 OpenChamber"
    finally:
        drain_monitor(window)


def test_monitor_click_resumes_paused_a_side_and_hints(qapp, monkeypatch, tmp_path):
    """Scenario 9: after an explicit pause, clicking 监控 OpenChamber resumes
    任务接收, starts the monitor and shows the resume hint exactly once."""
    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: window._listener.enabled)
        assert wait_until(lambda: not window._busy)
        window._pause()
        assert not window._listener.enabled

        window._toggle_monitor()
        assert window._listener.enabled
        assert window._oc_monitor_active
        assert window._clipboard_status_label.text() == "任务接收：运行中"
        assert window._monitor_status_label.text() == "OpenChamber监控：运行中"
        assert window.monitor_button.text() == "停止监控 OpenChamber"
        assert (
            "任务接收当前已暂停，已同时重新启动。" in window.status_label.text()
        )
    finally:
        drain_monitor(window)


def test_runtime_monitor_label_tracks_start_and_stop(qapp, monkeypatch, tmp_path):
    """Scenario 11: the OpenChamber monitor label turns running on start and
    back to 未启动 after the asynchronous stop finalises."""
    qapp.clipboard().clear()
    window, fake = build_monitor_window(qapp, monkeypatch, tmp_path)
    try:
        assert wait_until(lambda: not window._busy)
        assert window._monitor_status_label.text() == "OpenChamber监控：未启动"

        window._toggle_monitor()
        assert window._monitor_status_label.text() == "OpenChamber监控：运行中"

        window._toggle_monitor()
        assert wait_until(lambda: not window._oc_monitor_active)
        assert window._monitor_status_label.text() == "OpenChamber监控：未启动"
        assert window.monitor_button.text() == "监控 OpenChamber"
    finally:
        drain_monitor(window)


def test_monitor_wraps_preexisting_message_that_completes_later(
    qapp, monkeypatch, tmp_path
):
    """Baseline fix: an assistant message that ALREADY exists (still
    streaming) when the monitor starts must NOT seed the wrap baseline, so the
    SAME message id is wrapped once it completes afterwards.  (Regression for
    ``baseline = seen_ids`` permanently skipping such a message.)"""
    from PySide6.QtTest import QTest

    from core.protocol import parse_message

    qapp.clipboard().clear()
    pre = in_progress_reply("a_pre", created=1000)  # streaming, not completed
    window, fake = build_monitor_window(
        qapp, monkeypatch, tmp_path, initial_messages=[pre]
    )
    fake.status = "busy"  # a streaming message implies a busy session
    try:
        window._toggle_monitor()
        assert manual_completed_ids(window) == set()
        QTest.qWait(150)
        assert manual_completed_ids(window) == set(), "streaming msg wrapped early"

        # the SAME message id completes with stop + non-empty text
        pre["info"]["time"]["completed"] = 1100
        pre["info"]["finish"] = "stop"
        deadline = time.monotonic() + 3.0
        text = ""
        while time.monotonic() < deadline:
            QTest.qWait(40)
            text = qapp.clipboard().text() or ""
            if "still working" in text:
                break
        assert "still working" in text, "pre-existing message was never wrapped"
        message = parse_message(text)
        assert message.in_reply_to.startswith("manual-")
        assert "still working" in message.body

        # later polls on the same id must not re-wrap it
        QTest.qWait(300)
        assert manual_completed_ids(window) == {message.in_reply_to}
    finally:
        drain_monitor(window)


def test_monitor_completed_history_not_rewrapped(qapp, monkeypatch, tmp_path):
    """A message that was ALREADY a completed reply when the monitor starts is
    history: it is never wrapped, while a genuinely new reply still is."""
    qapp.clipboard().clear()
    window, fake = build_monitor_window(
        qapp,
        monkeypatch,
        tmp_path,
        initial_messages=[completed_reply("a_hist", created=1000, text="history")],
    )
    try:
        window._toggle_monitor()
        from PySide6.QtTest import QTest

        QTest.qWait(300)
        assert manual_completed_ids(window) == set(), "history was re-wrapped"

        fake.add(completed_reply("a_new", created=3000, text="brand new"))
        assert wait_until(lambda: "brand new" in (qapp.clipboard().text() or ""))
    finally:
        drain_monitor(window)


def test_monitor_yields_to_a_side_task_in_same_session(qapp, monkeypatch, tmp_path):
    """10: while an A-side task owns the monitored session, a final reply
    arriving on that session is observed but never wrapped by the manual
    monitor (the auto relay owns it)."""
    from PySide6.QtTest import QTest

    qapp.clipboard().clear()
    window, fake = build_monitor_window(
        qapp,
        monkeypatch,
        tmp_path,
        initial_messages=[completed_reply("a_hist", created=1000, text="history")],
    )
    fake.status = "busy"
    try:
        # Let startup settle first: the self-check is a general worker whose
        # completion funnels through _finish_task, which CLEARS A-side session
        # ownership.  If it were still pending when we simulated an in-flight
        # task below, its late _finish_task could clear our ownership and race
        # the poller's reply signal (flaky yield).  Settle it before we go on.
        assert wait_until(
            lambda: not window._general_workers and not window._busy,
            timeout=5.0,
        ), "startup self-check did not settle"
        window._toggle_monitor()
        assert window._oc_monitor_active
        # an A-side task owns this exact session
        window._a_side_inflight_sessions.add("ses_test123")
        window._a_side_session_owner["ses_test123"] = "task-X"

        fake.add(completed_reply("a_owned", created=3000, text="owned final"))
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            QTest.qWait(40)
        assert manual_completed_ids(window) == set(), (
            "monitor wrapped a reply owned by an in-flight A-side task"
        )
    finally:
        drain_monitor(window)

# -------------------------------------------------------------------- #
# B-side task + monitor: status map vanishing -> auto-continue -> task
# completes, never wrapped as a manual-* task
# -------------------------------------------------------------------- #


def _task_history() -> list:
    """A previous, fully completed round: the parent links let the wait loop
    prove which assistant message belongs to which task round."""
    return [
        user_message("u0", "old task", 900),
        assistant_message(
            "a_hist", 1000, completed=1100, finish="stop",
            parts=[text_part("history")], parent_id="u0",
        ),
    ]


class TaskFake(MutableMessages):
    """MutableMessages whose send() behaves like the real client: it
    records the pre-send snapshot, appends the user message and an
    in-progress assistant reply (parent-linked), and returns a real
    dispatch (so a B-side task wait can locate its round)."""

    def __init__(self, initial=None):
        super().__init__(initial)
        self.user_ids: list[str] = [
            m["info"]["id"]
            for m in (initial or [])
            if m.get("info", {}).get("role") == "user"
        ]

    def verify(self) -> None:
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")

    def create_session(self, title: str, directory: str) -> str:
        return "ses_test123"

    def open_session(self, session_id: str) -> None:
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")

    def list_sessions(self, directory: str | None = None) -> list:
        if directory is not None and directory != "D:/proj":
            return []
        return [("ses_test123", "task session")]

    def send(self, session_id, prompt, directory, agent=None, model=None):
        if self.fail_all:
            raise OpenChamberUnavailableError("server exploded")
        if self.send_error is not None:
            raise self.send_error
        self.sent.append((session_id, directory, prompt, agent, model))
        pre_ids = frozenset(
            m["info"]["id"]
            for m in self.messages_list
            if m.get("info", {}).get("id")
        )
        n_user = len(self.user_ids)
        user_id = f"u{n_user + 1}"
        self.user_ids.append(user_id)
        self.messages_list.append(
            user_message(user_id, prompt, 2000 + n_user * 10, session_id=session_id)
        )
        self.messages_list.append(
            assistant_message(
                f"a{user_id}", 2050 + n_user * 10, completed=None, finish=None,
                parts=[text_part("still working")], parent_id=user_id,
            )
        )
        return make_dispatch(
            session_id=session_id,
            directory=directory,
            pre_ids=pre_ids,
            user_message_id=user_id,
        )


def test_ui_b_side_task_status_missing_auto_continue_completes_task(
    qapp, monkeypatch, tmp_path
):
    """B-side OpenChamber task + running monitor: the status map loses the
    session.  The monitor's automatic continuation (1/3) resumes the round
    before the relay's own 1/1 re-check (kept slow so the monitor is the
    one sending); the B-side worker's re-check sees the monitor's prompt
    and consumes the resulting reply, so the task COMPLETES under the
    ORIGINAL TASK_ID and the monitor never wraps the reply as a manual-*
    task (no manual-* registry records, no manual-* worker tokens)."""
    from PySide6.QtTest import QTest

    import core.relay as relay_mod

    # The relay's one-shot auto-recovery re-check is kept well SLOWER than
    # the monitor's auto-continue, so the monitor's 1/3 prompt is what the
    # B-side worker's re-check observes (the "monitor already continued"
    # branch) and the relay never sends its own prompt into the round.
    monkeypatch.setattr(relay_mod, "RECOVERY_DELAY_SECONDS", 2.0)
    monkeypatch.setattr(relay_mod, "COMPLETION_GRACE_SECONDS", 0.05)
    qapp.clipboard().clear()

    from tests.test_ui_startup import build_window

    fake = TaskFake(_task_history())
    fake.status = "missing_from_status_map"
    import ui as ui_mod

    monkeypatch.setattr(ui_mod, "OpenChamberClient", lambda url: fake)
    settings = RelaySettings(
        default_target=TARGET_OPENCHAMBER,
        openchamber_directory=str(tmp_path),
        openchamber_session_id="ses_test123",
        poll_interval=0.05,
        completion_timeout=5.0,
    )
    window = build_window(
        qapp, monkeypatch, tmp_path, settings=settings, openchamber=fake
    )
    window._startup_check_pending = False
    try:
        assert wait_until(
            lambda: not window._general_workers and not window._busy,
            timeout=5.0,
        ), "startup self-check did not settle"
        # start the B-side task, then the manual monitor on the same session
        window._toggle_monitor()
        assert window._oc_monitor_active
        # the monitor toggle auto-started the clipboard listener; the test
        # delivers the task manually, so pause it again to keep the same
        # task text from being re-delivered (duplicate answer)
        window._listener.pause()
        qapp.clipboard().setText(v1_task("OPENCHAMBER", "do it", "task-miss-001"))
        window._on_clipboard_text(qapp.clipboard().text())

        # Somebody (monitor auto-continue 1/3 or relay auto-recovery 1/1)
        # must send a continuation prompt once the round is seen as stuck;
        # mirror each sent prompt with the user message the server would
        # record (the original task send already appended its own).
        sent_besides_task = 0
        reply_added = False
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            QTest.qWait(30)
            while sent_besides_task < len(fake.sent) - 1:
                prompt = fake.sent[sent_besides_task + 1][2]
                fake.add(
                    user_message(
                        f"u_cont{sent_besides_task}",
                        prompt,
                        4000 + sent_besides_task * 10,
                    )
                )
                sent_besides_task += 1
            if sent_besides_task >= 1 and not reply_added:
                # the continuation round completes with a final reply,
                # parent-linked to the newest prompt (the one the server
                # answered)
                fake.add(
                    assistant_message(
                        "a_final",
                        6000,
                        completed=6100,
                        finish="stop",
                        parts=[text_part("recovered final")],
                        parent_id=fake.user_ids[-1],
                    )
                )
                reply_added = True
            record = window._workflow.registry.record("task-miss-001")
            if reply_added and record is not None and record["state"] == "COMPLETED":
                break

        QTest.qWait(300)
        record = window._workflow.registry.record("task-miss-001")
        assert record is not None and record["state"] == "COMPLETED", (
            f"B-side task did not complete: state={record and record['state']!r} "
            f"error={record and record.get('error')!r} sent={fake.sent}"
        )
        # at least one automatic continuation prompt was sent before
        # completion
        assert len(fake.sent) >= 2
        # the reply is the B-side task's answer under the ORIGINAL TASK_ID.
        # Read the persisted reply (the listener is paused in this test, so
        # the wrapped response is not necessarily on the clipboard).
        text = (
            window._workflow.load_reply("task-miss-001")
            or window._last_response
            or (qapp.clipboard().text() or "")
        )
        assert "IN_REPLY_TO: task-miss-001" in text
        assert "recovered final" in text
        # the monitor never wrapped the round as a manual-* task
        assert manual_completed_ids(window) == set()
        assert not any(
            token.split(":")[1].startswith("manual-")
            for token in window._monitor_workers
        )
    finally:
        drain_monitor(window)