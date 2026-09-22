"""Minimal PySide6 control window for AI Relay."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.clipboard import ClipboardListener
from core.error_translator import translate_error
from core.openchamber import (
    MONITOR_CONTINUE_PROMPT,
    MONITOR_MAX_TRANSPORT_RETRIES,
    MONITOR_TRANSPORT_RETRY_DELAYS,
    MonitorRoundTracker,
    MonitorScan,
    OpenChamberClient,
    OpenChamberError,
    OpenChamberUnavailableError,
    _latest_assistant_stat,
    extract_agent_model_sets,
    is_message_wrapped,
    mark_message_wrapped,
    monitor_probe,
    monitor_scan,
    normalize_directory,
)
from core.protocol import MessageType, ProtocolFormat, RelayProtocolError, parse_message, wrap_response
from core.reasonix_uia import ReasonixAutomation
from core.relay import (
    CANCELLED_MARKER,
    MODEL_REJECTION_AGAIN,
    MODEL_REJECTION_PROMPT,
    RelayWorkflow,
    TaskOutcome,
    directory_key,
    resolve_executor_kind,
)
from core.relay_settings import (
    DEFAULT_OPENCHAMBER_URL,
    TARGET_OPENCHAMBER,
    TARGET_REASONIX,
    RelaySettings,
)
from core.runtime_paths import data_dir
from core.session_rotation import SessionRotation

LOG_PATH = data_dir() / "relay.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
LOGGER = logging.getLogger("ai_relay_b")


def _model_info_text(record: dict) -> str:
    """Three-layer model detail restored from a registry record; missing
    values are shown as 未指定 (requested/resolved) or 未知 (actual)."""
    requested = record.get("requested_model")
    resolved = record.get("resolved_model")
    actual = record.get("actual_model")
    note = record.get("model_note")
    if note:
        return note
    return (
        f"请求 {requested or '未指定'}，"
        f"解析 {resolved or '未指定'}，"
        f"实际 {actual or '未知'}"
    )


# Auto-recovery budget for a stalled manually monitored round: at most 3 auto
# continuation attempts, each preceded by a short wait; only a REAL final
# reply (or a stop) resets the budget to zero.
_MONITOR_AUTO_CONTINUE_INTERVAL: float = 10.0
_MONITOR_AUTO_CONTINUE_MAX = 3


def _classify_continue_failure(exc: BaseException) -> str:
    """Route an auto-continuation SEND failure for recovery.

    ``transport`` (``OpenChamberUnavailableError``: SSE timeout / connection
    reset / TLS handshake) means the prompt may never have reached the
    service: before it counts toward the 3-attempt budget the session is
    re-queried for progress.  Everything else (HTTP 400 / Aborted / session
    missing / auth / any unexpected error) is ``non_retryable``: automatic
    recovery stops and the operator is offered the manual buttons.
    """
    if isinstance(exc, OpenChamberUnavailableError):
        return "transport"
    return "non_retryable"


class WorkerSignals(QObject):
    status = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)
    session_started = Signal(object)
    sessions = Signal(object)
    finished = Signal()


class RelayTask(QRunnable):
    def __init__(
        self,
        workflow: RelayWorkflow,
        text: str,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__()
        self.workflow = workflow
        self.text = text
        self.cancel_event = cancel_event
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.workflow.process(
                self.text,
                self.signals.status.emit,
                self.signals.session_started.emit,
                self.cancel_event,
            )
            self.signals.succeeded.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class CompactTaskSignals(QObject):
    done = Signal(bool)
    finished = Signal()


class CompactTask(QRunnable):
    """A post-reply first-class opencode compaction of one OpenChamber
    session, run on a worker so the main thread never blocks.  The
    compaction is dispatched through the dedicated
    ``/api/session/{id}/compact`` API (never through the AI_RELAY protocol
    formatter, never as a text prompt) and finishes before the next queued
    task is allowed to start."""

    def __init__(self, workflow: RelayWorkflow, session_id: str, directory: str):
        super().__init__()
        self.workflow = workflow
        self.session_id = session_id
        self.directory = directory
        self.signals = CompactTaskSignals()

    @Slot()
    def run(self):
        ok = False
        try:
            LOGGER.info(
                "compact worker start session=%s directory=%s",
                self.session_id, self.directory,
            )
            ok = self.workflow.compact_session(
                self.session_id, self.directory, lambda _status: None
            )
            LOGGER.info(
                "compact worker done session=%s ok=%s", self.session_id, ok
            )
        except Exception as exc:
            LOGGER.warning(
                "compact worker error session=%s error=%s", self.session_id, exc
            )
            ok = False
        finally:
            self.signals.done.emit(ok)
            self.signals.finished.emit()


class _StaleRecoverySignals(QObject):
    done = Signal(object)
    failed = Signal(str)
    finished = Signal()


class StaleRecoveryTask(QRunnable):
    """Startup reconciliation of stale RECEIVED/PROCESSING records left by a
    previous run (invariant 5: every task ends in a terminal state).

    The blocking OpenChamber queries run on this worker thread, but the
    registry is only READ here -- every registry WRITE and every clipboard
    write happens on the main thread when the results arrive, so tasks.json
    is never touched from a worker and nothing is ever re-sent.
    """

    def __init__(self, workflow: RelayWorkflow, records: list[dict[str, str]]):
        super().__init__()
        self.workflow = workflow
        self.records = records
        self.signals = _StaleRecoverySignals()

    @Slot()
    def run(self):
        try:
            results = [
                self.workflow.reconcile_stale_task(
                    record["task_id"], record, lambda _status: None
                )
                for record in self.records
            ]
            self.signals.done.emit(results)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class ContinueTask(QRunnable):
    """One manual "继续当前任务" execution against the interrupted task."""

    def __init__(
        self,
        workflow: RelayWorkflow,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__()
        self.workflow = workflow
        self.cancel_event = cancel_event
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.workflow.continue_openchamber_task(
                self.signals.status.emit,
                self.signals.session_started.emit,
                self.cancel_event,
            )
            self.signals.succeeded.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class NewSessionRetryTask(QRunnable):
    """One manual "新会话重试" execution against a model-rejected task."""

    def __init__(
        self,
        workflow: RelayWorkflow,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__()
        self.workflow = workflow
        self.cancel_event = cancel_event
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.workflow.retry_model_rejected_task(
                self.signals.status.emit,
                self.signals.session_started.emit,
                self.cancel_event,
            )
            self.signals.succeeded.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class RefreshMetaTask(QRunnable):
    def __init__(self, url: str, directory: str, auth_token: str | None = None):
        super().__init__()
        self.url = url
        self.directory = directory
        self.auth_token = auth_token
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            client = OpenChamberClient(self.url, auth_token=self.auth_token)
            sessions = client.list_sessions(self.directory)
            if not sessions and self.directory:
                all_sessions = client.list_sessions_with_projects()
                sessions = client.match_project_sessions(
                    self.directory, all_sessions
                )
            agents: set[str] = set()
            models: set[str] = set()
            for session_id, _title in sessions:
                try:
                    messages = client.messages(session_id, self.directory)
                except OpenChamberError:
                    continue
                session_agents, session_models = extract_agent_model_sets(
                    messages
                )
                agents |= session_agents
                models |= session_models
            self.signals.sessions.emit((agents, models))
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class RotateSessionTask(QRunnable):
    """Background automatic session rotation (create + persist + inherit)."""

    def __init__(
        self,
        url: str,
        directory: str,
        previous_session_id: str | None,
        settings: RelaySettings,
        inherit_auto_accept: bool = False,
        base_title_hint: str | None = None,
    ):
        super().__init__()
        self.url = url
        self.directory = directory
        self.previous_session_id = previous_session_id
        self.settings = settings
        self.inherit_auto_accept = inherit_auto_accept
        self.base_title_hint = base_title_hint
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            rotation = SessionRotation(
                settings=self.settings,
                openchamber=OpenChamberClient(
                    self.url,
                    auth_token=self.settings.openchamber_auth_token or None,
                ),
            )
            session_id = rotation.rotate(
                self.directory,
                previous_session_id=self.previous_session_id,
                inherit_auto_accept=self.inherit_auto_accept,
                base_title_hint=self.base_title_hint,
            )
            self.signals.succeeded.emit(session_id)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class SelfCheckTask(QRunnable):
    def __init__(self, reasonix: ReasonixAutomation):
        super().__init__()
        self.reasonix = reasonix
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            checks = self.reasonix.self_check()
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(f"自检未通过: {', '.join(failed)}")
            self.signals.succeeded.emit("")
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class MonitorSignals(QObject):
    reply = Signal(str, str)
    failed = Signal(str)
    status = Signal(str)
    interrupted = Signal(object)
    transport_error = Signal(str)
    transport_recovered = Signal()
    finished = Signal()


class MonitorPollTask(QRunnable):
    """Background poller for the manual OpenChamber monitor.

    Runs in the thread pool so the UI thread is never blocked.  Each probe
    scans the fixed session (captured at start) for new completed assistant
    replies after the baseline AND drives the round tracker that detects an
    abnormal idle (``tool-calls -> idle -> no final reply``).  A
    ``threading.Event`` stops the loop promptly.

    The first successful probe seeds the baseline: everything the session
    already contains is history, and an already-idle history session never
    triggers an interruption.

    While the session is busy, the poller also hard-monitors an unchanged
    signature: if a busy session produces NO progress (same message ids,
    same body lengths, same finish, same tool count) for
    ``busy_without_progress_threshold`` seconds, a single interruption with
    reason ``monitor_busy_without_progress`` is emitted (see
    ``_emit_busy_interruption``); any message/status change resets the
    window.  An optional ``busy_reset_event`` lets a successful manual
    continue zero the in-progress timer.  The poller NEVER auto-continues.
    """

    def __init__(
        self,
        client: OpenChamberClient,
        session_id: str,
        directory: str,
        baseline: set[str],
        interval: float,
        stop_event,
        transport_delays=MONITOR_TRANSPORT_RETRY_DELAYS,
        busy_stale_log_interval: float = 30.0,
        busy_without_progress_threshold: float = 120.0,
        busy_reset_event=None,
        generation: int = 0,
    ):
        super().__init__()
        self.client = client
        self.session_id = session_id
        self.directory = directory
        self.baseline = set(baseline)
        self.interval = interval
        self.stop_event = stop_event
        # Generation token captured when the window starts this monitor: every
        # emitted signal carries it so stale workers never update a newer UI.
        self.generation = int(generation)
        self.busy_stale_log_interval = float(busy_stale_log_interval)
        # A session that keeps reporting "busy" with a COMPLETELY unchanged
        # message signature (id / seen ids / text length / finish / tool
        # count) for this many seconds is considered stuck: the poller emits
        # a listening interruption so the operator can continue or stop.
        # It never auto-continues.
        self.busy_without_progress_threshold = float(busy_without_progress_threshold)
        # Optional event set by the UI after a manual "继续当前任务" send
        # succeeds: resets the busy-without-progress timer on the next poll.
        self.busy_reset_event = busy_reset_event
        self.transport_delays = tuple(transport_delays)
        self.signals = MonitorSignals()

    def _emit_interruption(self, outcome: MonitorStepOutcome) -> None:
        self.signals.interrupted.emit(
            {
                "session_id": self.session_id,
                "directory": self.directory,
                "generation": self.generation,
                "last_message_id": outcome.last_message_id,
                "last_finish": outcome.last_finish,
                "reason": outcome.reason,
            }
        )

    def _emit_busy_interruption(self, latest_stat) -> None:
        self.signals.interrupted.emit(
            {
                "session_id": self.session_id,
                "directory": self.directory,
                "generation": self.generation,
                "last_message_id": latest_stat.message_id if latest_stat else None,
                "last_finish": latest_stat.finish if latest_stat else None,
                "reason": "monitor_busy_without_progress",
            }
        )

    @Slot()
    def run(self):
        try:
            self._pump()
        finally:
            # Fired on EVERY exit path (normal loop end, stop, or an
            # unexpected exception) so the window can decay its worker
            # registry and only then finalise the stop / close sequence.
            self.signals.finished.emit()

    def _pump(self):
        # Round detection starts from the SAME snapshot the UI took when the
        # monitor began (``self.baseline``); the wrap baseline is advanced
        # ONLY with replies that were actually wrapped, so a still-streaming
        # message observed later stays eligible to wrap once it completes.
        tracker = MonitorRoundTracker(self.session_id, self.directory)
        tracker.reset_round(frozenset(self.baseline), reason="monitor_start")
        LOGGER.info(
            "监听轮询开始 baseline=%d interval=%.1f",
            len(self.baseline), self.interval,
        )
        seen = set(self.baseline)
        transport_failures = 0
        last_log = None
        busy_stale = None
        while not self.stop_event.is_set():
            try:
                probe = monitor_probe(
                    self.client, self.session_id, self.directory, frozenset(seen)
                )
            except OpenChamberError as exc:
                if isinstance(exc, OpenChamberUnavailableError) and transport_failures < MONITOR_MAX_TRANSPORT_RETRIES:
                    # Transient transport error: re-query only, never resend.
                    transport_failures += 1
                    self.signals.status.emit(
                        f"OpenChamber 连接暂时中断，正在重连（{transport_failures}/"
                        f"{MONITOR_MAX_TRANSPORT_RETRIES}）……"
                    )
                    delay = self.transport_delays[transport_failures - 1]
                    if self.stop_event.wait(delay):
                        break
                    continue
                if isinstance(exc, OpenChamberUnavailableError):
                    self.signals.transport_error.emit(
                        "OpenChamber 连接多次中断，暂时无法确认会话状态。"
                    )
                    if self.stop_event.wait(self.interval):
                        break
                    continue
                LOGGER.warning(
                    "监听模式轮询失败（非传输）exc=%s", exc,
                )
                self.signals.failed.emit(str(exc))
                if self.stop_event.wait(self.interval):
                    break
                continue
            except Exception as exc:
                # Unexpected poller failure must never kill the monitor
                # silently: report it and keep trying on the next interval.
                LOGGER.warning(
                    "监听模式轮询异常（未捕获）exc=%s", exc,
                )
                self.signals.failed.emit(str(exc))
                if self.stop_event.wait(self.interval):
                    break
                continue
            if transport_failures > 0:
                transport_failures = 0
                self.signals.transport_recovered.emit()

            wrapped = False
            for msg_id, text in probe.new_completed:
                # Only finished and submitted messages enter the wrap
                # baseline: streaming (not yet completed) messages stay out
                # so they are retried by the next poll.  `finish=="stop"`
                # + non-empty text (enforced in ``_message_completed``) is
                # the ONLY shape that may wrap; tool-calls, finish==None and
                # empty assistant messages never reach this loop.
                if is_message_wrapped(msg_id, self.session_id) or msg_id in seen:
                    seen.add(msg_id)
                    LOGGER.debug(
                        "监听模式跳过已包装/已见消息 msg_id=%s", msg_id,
                    )
                    continue
                self.signals.reply.emit(text, msg_id)
                seen.add(msg_id)
                wrapped = True
                LOGGER.info(
                    "监听模式已包装最终回复 msg_id=%s text_len=%d",
                    msg_id, len(text),
                )
            if wrapped:
                # This round produced a real final reply: start the next
                # round cold so a following history-idle is never flagged.
                tracker.reset_round(
                    probe.seen_ids, reason="wrapped_final_reply"
                )
                LOGGER.info(
                    "监听模式重置本轮基线 reason=wrapped_final_reply "
                    "baseline=%d",
                    len(probe.seen_ids),
                )
                continue
            outcome = tracker.step(probe)
            # Evidence-only "no progress while busy" observation: while the
            # session keeps reporting "busy" and the message signature (id /
            # text length / finish / tool count) AND the seen id set stay
            # unchanged for a long time, a periodic line is written to
            # relay.log so a stuck-busy state can be told apart from "idle
            # not recognised".  It NEVER resumes anything: long tool runs are
            # legitimate and must keep working untouched.
            latest_stat = _latest_assistant_stat(probe)
            stale_sig = (
                (
                    latest_stat.message_id,
                    latest_stat.text_length,
                    latest_stat.finish,
                    latest_stat.tool_count,
                )
                if latest_stat is not None
                else None
            )
            if probe.status == "busy":
                if self.busy_reset_event is not None and self.busy_reset_event.is_set():
                    # A manual "继续当前任务" was just sent and succeeded:
                    # zero the busy-without-progress timer and continue
                    # listening for the resumed round.
                    self.busy_reset_event.clear()
                    busy_stale = None
                    LOGGER.info(
                        "监听续接后清零 busy 无进展计时 session_id=%s",
                        self.session_id,
                    )
                mark = (frozenset(probe.seen_ids), stale_sig)
                now = time.monotonic()
                if busy_stale is None or busy_stale["mark"] != mark:
                    # Fresh signature (new message / text growth / finish /
                    # tool-count change) restarts the 120s window and
                    # re-arms the interruption for this new stuck period.
                    busy_stale = {
                        "mark": mark,
                        "start": now,
                        "last_log": 0.0,
                        "emitted": False,
                    }
                else:
                    duration = now - busy_stale["start"]
                    if (
                        duration >= self.busy_stale_log_interval
                        and now - busy_stale["last_log"] >= self.busy_stale_log_interval
                    ):
                        busy_stale["last_log"] = now
                        LOGGER.info(
                            "会话仍报告 busy，但消息没有进展，持续时间=%d 秒",
                            int(duration),
                        )
                    if (
                        not busy_stale["emitted"]
                        and duration >= self.busy_without_progress_threshold
                    ):
                        # Long-lived busy with NO signature change at all:
                        # the session is very likely stuck.  Emit the
                        # listening interruption ONCE per stuck period (the
                        # operator must click; never auto-continue).
                        busy_stale["emitted"] = True
                        busy_stale["last_log"] = now
                        LOGGER.warning(
                            "监听模式 busy 无进展超过阈值，触发中断 "
                            "event=interrupted duration=%d last_message_id=%s "
                            "last_finish=%s reason=monitor_busy_without_progress",
                            int(duration),
                            latest_stat.message_id if latest_stat else None,
                            latest_stat.finish if latest_stat else None,
                        )
                        self._emit_busy_interruption(latest_stat)
            else:
                busy_stale = None
            log_key = (
                outcome.event,
                outcome.idle_confirmations,
                probe.status,
                outcome.last_message_id,
            )
            if outcome.event == "interrupted":
                # Log BEFORE the signal so the relay.log audit line is already
                # written by the time the main thread reacts to the flag.
                LOGGER.info(
                    "监听模式判定异常停顿 event=interrupted idle=%d/%d "
                    "last_message_id=%s last_finish=%s reason=%s",
                    outcome.idle_confirmations,
                    outcome.required_idle,
                    outcome.last_message_id,
                    outcome.last_finish,
                    outcome.reason,
                )
                self._emit_interruption(outcome)
                last_log = log_key
            elif outcome.event == "idle_confirm":
                self.signals.status.emit(
                    f"疑似异常 idle：{outcome.idle_confirmations}/{outcome.required_idle}；"
                    "正在确认会话是否异常停顿……"
                )
                LOGGER.info(
                    "监听模式轮询 status=%s event=idle_confirm idle=%d/%d "
                    "last_message_id=%s last_finish=%s reason=%s",
                    probe.status,
                    outcome.idle_confirmations,
                    outcome.required_idle,
                    outcome.last_message_id,
                    outcome.last_finish,
                    outcome.reason,
                )
                last_log = log_key
            elif outcome.event == "activity":
                # While the unchanged-busy interruption is already displayed,
                # keep the stuck status/buttons visible instead of re-painting
                # the "activity" echo on every busy poll.
                if not (busy_stale and busy_stale["emitted"]):
                    self.signals.status.emit("观察到新的执行活动，正在跟踪本轮……")
            if log_key != last_log:
                last_log = log_key
                LOGGER.info(
                    "监听模式轮询 status=%s event=%s idle=%d/%d "
                    "last_message_id=%s last_finish=%s reason=%s",
                    probe.status,
                    outcome.event,
                    outcome.idle_confirmations,
                    outcome.required_idle,
                    outcome.last_message_id,
                    outcome.last_finish,
                    outcome.reason,
                )
            if self.stop_event.wait(self.interval):
                break


class MonitorContinueTask(QRunnable):
    """One manual "继续当前任务" for a MANUALLY MONITORED round.

    Sends the fixed continue prompt to the original session (original
    project directory, current Agent and Model) WITHOUT creating an AI Relay
    TASK, WITHOUT writing tasks.json and WITHOUT touching auto rotation.
    The monitor poller keeps running and wraps the eventual reply.
    """

    def __init__(
        self,
        url: str,
        session_id: str,
        directory: str,
        agent: str | None = None,
        model=None,
        auth_token: str | None = None,
    ):
        super().__init__()
        self.url = url
        self.session_id = session_id
        self.directory = directory
        self.agent = agent
        self.model = model
        self.auth_token = auth_token
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            client = OpenChamberClient(self.url, auth_token=self.auth_token)
            self.signals.status.emit(
                f"正在向会话 {self.session_id} 发送续接提示……"
            )
            client.send(
                self.session_id,
                MONITOR_CONTINUE_PROMPT,
                self.directory,
                agent=self.agent,
                model=self.model,
            )
            self.signals.succeeded.emit("")
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class MonitorAutoContinueTask(QRunnable):
    """One AUTO "继续当前任务" for a stalled manually monitored round.

    Waits ``interval`` seconds (a stop event aborts the wait and cancels the
    send) then sends the fixed continue prompt to the ORIGINAL session with
    the current Agent/Model -- exactly like the manual MonitorContinueTask,
    never a new TASK, never tasks.json, never a new session, never auto
    rotation.  Failures are emitted CLASSIFIED so the window re-queries
    transport outages before counting the attempt.
    """

    class Signals(QObject):
        status = Signal(str)
        succeeded = Signal(str)
        failed = Signal(str, str)  # (error text, category)
        finished = Signal()

    def __init__(
        self,
        url: str,
        session_id: str,
        directory: str,
        agent: str | None = None,
        model=None,
        interval: float = _MONITOR_AUTO_CONTINUE_INTERVAL,
        stop_event=None,
        auth_token: str | None = None,
    ):
        super().__init__()
        self.url = url
        self.session_id = session_id
        self.directory = directory
        self.agent = agent
        self.model = model
        self.interval = float(interval)
        self.stop_event = stop_event
        self.auth_token = auth_token
        self.signals = self.Signals()

    @Slot()
    def run(self):
        try:
            if self.stop_event is not None and self.stop_event.wait(self.interval):
                return  # operator stopped before the wait elapsed
            if self.stop_event is not None and self.stop_event.is_set():
                return
            client = OpenChamberClient(self.url, auth_token=self.auth_token)
            self.signals.status.emit(f"正在自动续接会话 {self.session_id}……")
            client.send(
                self.session_id,
                MONITOR_CONTINUE_PROMPT,
                self.directory,
                agent=self.agent,
                model=self.model,
            )
            self.signals.succeeded.emit("")
        except Exception as exc:
            self.signals.failed.emit(str(exc), _classify_continue_failure(exc))
        finally:
            self.signals.finished.emit()


class MonitorRecoveryQueryTask(QRunnable):
    """Re-queries a stalled session after a TRANSPORT-failed auto continue.

    A transport send failure (SSE timeout / connection reset / cert
    handshake) means the prompt may never have reached the service.  Before
    the attempt is counted toward the 3-attempt budget the session is
    re-queried up to ``retries`` times: if the latest assistant message id is
    still the stalled one, the result is ``no_progress`` (the attempt counts),
    otherwise ``progress`` (the attempt is rolled back and listening resumes).
    """

    class Signals(QObject):
        result = Signal(str)  # "progress" | "no_progress"
        finished = Signal()

    def __init__(
        self,
        url: str,
        session_id: str,
        directory: str,
        reference_message_id: str | None,
        retries: int = MONITOR_MAX_TRANSPORT_RETRIES,
        delays=MONITOR_TRANSPORT_RETRY_DELAYS,
        stop_event=None,
        auth_token: str | None = None,
    ):
        super().__init__()
        self.url = url
        self.session_id = session_id
        self.directory = directory
        self.reference_message_id = reference_message_id
        self.retries = int(retries)
        self.delays = tuple(delays)
        self.stop_event = stop_event
        self.auth_token = auth_token
        self.signals = self.Signals()

    @Slot()
    def run(self):
        try:
            self._pump()
        finally:
            self.signals.finished.emit()

    def _pump(self):
        client = OpenChamberClient(self.url, auth_token=self.auth_token)
        for attempt in range(self.retries):
            if self.stop_event is not None and self.stop_event.is_set():
                self.signals.result.emit("no_progress")
                return
            try:
                probe = monitor_probe(
                    client, self.session_id, self.directory, frozenset()
                )
                latest = _latest_assistant_stat(probe)
                current_id = latest.message_id if latest else None
                if current_id != self.reference_message_id:
                    self.signals.result.emit("progress")
                    return
            except OpenChamberError as exc:
                # A transport hiccup during the re-query retries under the
                # delay slot below; any other error (or an unreadable session)
                # means NO confirmed progress either.
                if not isinstance(exc, OpenChamberUnavailableError):
                    self.signals.result.emit("no_progress")
                    return
            except Exception:
                self.signals.result.emit("no_progress")
                return
            if attempt < len(self.delays) and self.delays[attempt] > 0:
                if self.stop_event is not None:
                    if self.stop_event.wait(self.delays[attempt]):
                        self.signals.result.emit("no_progress")
                        return
                    continue
                try:
                    time.sleep(self.delays[attempt])
                except Exception:
                    self.signals.result.emit("no_progress")
                    return
        self.signals.result.emit("no_progress")


class RelayWindow(QMainWindow):
    def __init__(self, app: QApplication, pool: QThreadPool | None = None):
        super().__init__()
        self.setWindowTitle("AI Relay")
        self.setMinimumSize(680, 620)
        self.resize(760, 980)

        self._busy = False
        self._compacting = False
        self._app = app
        self._clipboard_paused = False
        self._startup_check_pending = False
        self._task_cancel_event: threading.Event | None = None
        self._refresh_session_after_success = False
        self._current_task_is_auto = False
        self._rotation_pending = False
        self._rotation_directory = ""
        self._pool = pool or QThreadPool.globalInstance()
        self._reasonix = ReasonixAutomation()
        self._settings = RelaySettings.load()
        self._startup_check_pending = (
            self._settings.default_target == TARGET_REASONIX
        )
        self._workflow = RelayWorkflow(
            self._reasonix, settings=self._settings
        )
        # Snapshot of task ids recorded as non-terminal when THIS window was
        # constructed: startup reconciliation must only touch records left by
        # a PREVIOUS run.  A task claimed later in this run (auto-start
        # clipboard pickup, or a direct delivery) can already be RECEIVED by
        # the time the singleShot(0) recovery timer fires; without the
        # snapshot the recovery would misclassify that fresh in-flight task
        # as stale and reconcile it to RECOVERY_REQUIRED under the real
        # worker, aborting it with "task was already processed".
        self._startup_stale_task_ids = {
            r["task_id"] for r in self._workflow.registry.stale_records()
        }
        self._rotation = SessionRotation(self._settings)
        self._last_response: str | None = None
        self._last_outcome: TaskOutcome | None = None
        self._current_outcome: TaskOutcome | None = None
        # The A-side task currently EXECUTING (at most one worker runs at a
        # time).  Its id is the target of a manual 包装内容 wrap -- the operator
        # says "this clipboard text is the final reply of THIS task", so the
        # wrap is bound to the ORIGINAL task id, not a manual-* id.  Its
        # generation is bumped both when a task launches and when the operator
        # manually completes it, so a superseded worker's late success/failure
        # can never update a newer round or overwrite the manual completion.
        self._current_a_task_id: str | None = None
        self._current_a_task_generation: int = 0
        # A-side TASK in-flight ownership (invariants 1/4): window-memory
        # set of project directories / OpenChamber sessions currently
        # occupied by an in-flight A-side task (RECEIVED .. terminal).
        # Window memory ONLY: stale PROCESSING records left in tasks.json by
        # a previous instance must never block a new run's monitor.
        # Populated on claim / _on_session_started, cleared on _finish_task.
        self._a_side_inflight_dirs: set[str] = set()
        self._a_side_inflight_sessions: set[str] = set()
        # session_id -> A-side task_id currently owning that session: lets the
        # monitor's yield/continue decisions be logged with the owning task
        # (invariant 3) without guessing from window state.
        self._a_side_session_owner: dict[str, str] = {}
        # True while an OpenChamber-targeted task has not yet revealed its
        # session/directory (RECEIVED before PROCESSING, no directory known):
        # the monitor yields conservatively until the task is identified or
        # ends.
        self._a_side_oc_pending = False
        # Message ids the monitor yielded (did NOT wrap) while an A-side
        # task owned the session: remembered so the yield is logged once.
        self._oc_monitor_yielded: set[str] = set()
        self._listener = ClipboardListener(app.clipboard())
        self._listener.task_received.connect(self._on_clipboard_text)

        self._oc_monitor_active = False
        self._oc_monitor_directory = ""
        self._oc_monitor_session = ""
        self._oc_monitor_seen: set[str] = set()
        self._oc_monitor_stop = threading.Event()
        self._oc_monitor_task: QRunnable | None = None
        self._oc_monitor_interrupted = False
        self._oc_monitor_continuing = False
        self._oc_monitor_transport_down = False
        self._monitor_transport_delays: tuple[float, ...] = MONITOR_TRANSPORT_RETRY_DELAYS
        # Evidence-only knob (no behavior change): while the session keeps
        # reporting "busy" with NO signature change for this many seconds,
        # the poller records a periodic "busy without progress" line in
        # relay.log so an un-freed busy state can be told apart from a real
        # working round.  It never auto-resumes anything.
        self._monitor_busy_stale_log_interval: float = 30.0
        # A session that stays "busy" with NO message-signature change at all
        # for this many seconds is treated as stuck: the poller emits a
        # listening interruption, which the window answers with an AUTO
        # continuation (up to 3 attempts) or the manual continue/stop offer.
        self._monitor_busy_without_progress_threshold: float = 120.0
        # Auto-recovery: when a stall is detected the window auto-sends the
        # fixed continue prompt (up to 3 attempts, each preceded by a short
        # wait).  auto_count is the current budget position (kept across
        # stalls, reset to 0 only by a wrapped final reply or a stop);
        # auto_disabled stops any further auto send until the round finishes;
        # auto_pending guards against concurrent/duplicate sends.
        self._monitor_auto_continue_interval: float = _MONITOR_AUTO_CONTINUE_INTERVAL
        self._oc_monitor_auto_count = 0
        self._oc_monitor_auto_disabled = False
        self._oc_monitor_auto_pending = False
        self._oc_monitor_last_stall_message_id: str | None = None
        # Monitor thread lifetime control: every monitor worker (poll, manual
        # continue, auto continue, recovery query) is registered in
        # ``_monitor_workers`` at start and removed when its ``finished``
        # signal arrives.  ``_monitor_generation`` is bumped on both start
        # and stop so late signals from an old run can never update a newer
        # UI.  Stripping down is asynchronous: the full state is reset only
        # once the LAST worker emits ``finished``.
        self._monitor_generation = 0
        self._monitor_workers: set[str] = set()
        self._monitor_task_refs: dict[str, QRunnable] = {}
        self._oc_monitor_stopping = False
        self._monitor_close_timeout = 5.0
        self._monitor_close_deadline: float | None = None
        self._close_requested = False
        self._close_after_monitors = False
        # General worker lifetime control (RelayTask, SelfCheckTask, session
        # and rotation tasks): the QRunnable is held strongly, auto-delete is
        # disabled and ``finished`` is awaited before the window closes, so a
        # finished task can never delete its Signals QObject on a worker
        # thread while the window is torn down (the access-violation race).
        self._general_workers: set[str] = set()
        self._general_task_refs: dict[str, QRunnable] = {}
        # A task already on the clipboard when the app auto-starts is picked
        # up once; if the pickup runs while a startup task (e.g. the Reasonix
        # self-check) is busy it is retried right after that task finishes.
        self._startup_pickup_pending = False
        # True while a startup reconciliation of stale non-terminal task
        # records (left PROCESSING/RECEIVED by a previous instance) is
        # running: queued tasks must not start until it settles, and a
        # startup clipboard pickup is held for it.
        self._startup_recovery_pending = False
        # A recovered reply that could not be written to the clipboard yet
        # (a task was still running): flushed to the clipboard as soon as
        # the lane is free, so side A never misses the recovered reply.
        self._pending_recovered_reply: str | None = None

        self.status_label = QLabel("服务状态：未启动")
        self.task_label = QLabel("当前任务：无")
        self.detail_label = QLabel(
            "启动任务接收后，处理 AI_RELAY/1 与旧版剪贴板任务；"
            "TARGET 为 REASONIX / OPENCHAMBER / EXECUTOR（默认执行端）。"
        )
        self.detail_label.setWordWrap(True)
        self._clipboard_status_label = QLabel("任务接收：未启动")
        self._monitor_status_label = QLabel("OpenChamber监控：未启动")
        # FIFO queue position readout (invariant 1): accepted-but-waiting
        # B-side tasks (sourced from A-side) are never lost, they simply run
        # in order.
        self._queue_status_label = QLabel("等待任务：0")
        self.start_button = QPushButton("启动任务接收")
        self.pause_button = QPushButton("暂停任务接收")
        self.check_button = QPushButton("测试 Reasonix 连接")
        self.monitor_button = QPushButton("监控 OpenChamber")
        self.wrap_button = QPushButton("包装内容")
        self.continue_button = QPushButton("继续当前任务")
        self.new_session_button = QPushButton("新会话重试")
        self.stop_button = QPushButton("停止当前任务")
        self.open_session_button = QPushButton("打开当前会话")
        self.recopy_button = QPushButton("重新复制回复")
        self._saved_task_combo = QComboBox()
        self._saved_task_combo.setMinimumWidth(200)
        self.pause_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.new_session_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.open_session_button.setEnabled(False)
        self.recopy_button.setEnabled(False)
        self._refresh_saved_tasks()
        # Reflect any tasks still QUEUED in tasks.json (left by a previous
        # run) in the status card before startup recovery kicks in.
        self._update_queue_label()

        self.start_button.setObjectName("primaryButton")
        self.monitor_button.setObjectName("primaryButton")
        self.stop_button.setObjectName("dangerButton")

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(14)
        layout.addWidget(self._build_status_card())
        layout.addWidget(self._build_actions_group())
        layout.addWidget(self._build_settings_group())
        layout.addWidget(self._build_results_group())

        container = QWidget()
        container.setLayout(layout)
        # Scrollable center: on short screens the groups used to be squeezed
        # and the last settings rows (e.g. 自动压缩会话) vanished; now the
        # full content is always reachable.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
        self._apply_styles()

        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.check_button.clicked.connect(self._self_check)
        self.monitor_button.clicked.connect(self._toggle_monitor)
        self.wrap_button.clicked.connect(self._wrap_clipboard_content)
        self.continue_button.clicked.connect(self._continue_task)
        self.new_session_button.clicked.connect(self._retry_new_session)
        self.stop_button.clicked.connect(self._stop_task)
        self.open_session_button.clicked.connect(self._open_current_session)
        self.recopy_button.clicked.connect(self._recopy_reply)
        self._saved_task_combo.currentIndexChanged.connect(self._saved_task_selected)
        self._save_settings_button.clicked.connect(self._save_settings)

        QTimer.singleShot(0, self._ensure_clipboard_listener_started)
        if self._startup_check_pending:
            QTimer.singleShot(0, self._self_check)
        QTimer.singleShot(0, self._maybe_start_startup_recovery)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_status_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("statusCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        self.status_label.setObjectName("statusTitle")
        self.task_label.setObjectName("mutedText")
        self.detail_label.setObjectName("mutedText")
        self._queue_status_label.setObjectName("mutedText")
        layout.addWidget(self.status_label)
        layout.addWidget(self.task_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self._clipboard_status_label)
        layout.addWidget(self._monitor_status_label)
        layout.addWidget(self._queue_status_label)
        return card

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("快捷操作")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(self.start_button, 0, 0)
        grid.addWidget(self.pause_button, 0, 1)
        grid.addWidget(self.monitor_button, 1, 0)
        grid.addWidget(self.wrap_button, 1, 1)
        grid.addWidget(self.check_button, 2, 0, 1, 2)
        recovery = QHBoxLayout()
        recovery.setSpacing(8)
        recovery.addWidget(self.continue_button)
        recovery.addWidget(self.new_session_button)
        recovery.addWidget(self.stop_button)
        grid.addLayout(recovery, 3, 0, 1, 2)
        return group

    def _build_results_group(self) -> QGroupBox:
        group = QGroupBox("会话与回复")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        layout.addWidget(self.open_session_button, 0, 0, 1, 3)
        layout.addWidget(QLabel("已保存任务"), 1, 0)
        layout.addWidget(self._saved_task_combo, 1, 1)
        layout.addWidget(self.recopy_button, 1, 2)
        layout.setColumnStretch(1, 1)
        return group

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111318; color: #f3f4f6; font-family: "Microsoft YaHei UI"; font-size: 13px; }
            QGroupBox { background: #181b22; border: 1px solid #2a2f3a; border-radius: 10px; margin-top: 10px; padding: 14px 12px 12px 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #d7dbe3; }
            QFrame#statusCard { background: #181b22; border: 1px solid #2a2f3a; border-radius: 12px; }
            QLabel#statusTitle { color: #ffffff; font-size: 16px; font-weight: 700; }
            QLabel#mutedText { color: #9ca3af; }
            QPushButton { min-height: 36px; padding: 0 14px; background: #252a34; border: 1px solid #343b48; border-radius: 7px; font-weight: 600; }
            QPushButton:hover { background: #303643; }
            QPushButton:pressed { background: #20242c; }
            QPushButton:disabled { color: #626975; background: #1a1d23; border-color: #252a32; }
            QPushButton#primaryButton { background: #2563eb; border-color: #3474f5; color: white; }
            QPushButton#primaryButton:hover { background: #3474f5; }
            QPushButton#dangerButton { color: #fecaca; border-color: #7f1d1d; background: #3a1c21; }
            QLineEdit, QComboBox { min-height: 34px; padding: 0 9px; background: #101218; border: 1px solid #343b48; border-radius: 7px; selection-background-color: #2563eb; }
            QLineEdit:focus, QComboBox:focus { border-color: #4f8cff; }
            QComboBox QAbstractItemView { background: #181b22; border: 1px solid #343b48; selection-background-color: #2563eb; }
            """
        )

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("执行端设置")
        form = QFormLayout()

        self._executor_combo = QComboBox()
        self._executor_combo.addItem("Reasonix", TARGET_REASONIX)
        self._executor_combo.addItem("OpenChamber", TARGET_OPENCHAMBER)
        index = self._executor_combo.findData(self._settings.default_target)
        self._executor_combo.setCurrentIndex(max(index, 0))

        self._url_edit = QLineEdit(self._settings.openchamber_url or DEFAULT_OPENCHAMBER_URL)
        self._directory_edit = QLineEdit(self._settings.openchamber_directory)
        self._browse_directory_button = QPushButton("选择文件夹…")
        self._browse_directory_button.clicked.connect(self._browse_directory)
        directory_row = QHBoxLayout()
        directory_row.addWidget(self._directory_edit)
        directory_row.addWidget(self._browse_directory_button)
        self._agent_combo = QComboBox()
        self._agent_combo.setEditable(True)
        self._agent_combo.setInsertPolicy(QComboBox.NoInsert)
        self._agent_combo.addItems(["build", "plan"])
        configured_agent = self._settings.openchamber_agent.strip()
        if configured_agent:
            self._agent_combo.setCurrentText(configured_agent)
        self._model_combo = QComboBox()
        # Model must be picked from the refreshed / known list only; the
        # combo is not editable so an invalid or truncated name can never be
        # typed or saved.
        self._model_combo.setEditable(False)
        self._model_combo.addItems(["4090/qwen3.8-27b", "opencode/big-pickle"])
        configured_model = self._settings.openchamber_model.strip()
        if configured_model:
            self._model_combo.setCurrentText(configured_model)
        self._refresh_meta_button = QPushButton("刷新 Agent/Model")
        self._refresh_meta_button.clicked.connect(self._refresh_meta)
        agent_model_row = QHBoxLayout()
        agent_model_row.addWidget(self._agent_combo)
        agent_model_row.addWidget(self._model_combo)
        agent_model_row.addWidget(self._refresh_meta_button)

        form.addRow("默认执行端（TARGET: EXECUTOR）", self._executor_combo)
        form.addRow("OpenChamber 地址", self._url_edit)
        form.addRow("项目目录", directory_row)
        form.addRow("Agent / Model", agent_model_row)

        self._auto_compact_check = QCheckBox("任务完成后自动压缩会话")
        self._auto_compact_check.setChecked(self._settings.auto_compact_after_response)
        form.addRow(self._auto_compact_check)

        self._save_settings_button = QPushButton("保存设置")
        form.addRow(self._save_settings_button)

        group.setLayout(form)
        return group

    # ------------------------------------------------------------------ #
    # monitoring control
    # ------------------------------------------------------------------ #

    @Slot()
    def _start(self):
        started = self._ensure_clipboard_listener_started("manual")
        if started:
            self._set_status("空闲：正在监听剪贴板")
        else:
            self._update_runtime_status()

    @Slot()
    def _pause(self):
        self._listener.pause()
        self._clipboard_paused = True
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self._update_runtime_status()
        self._set_status("已暂停")
        LOGGER.info("任务接收已暂停")

    def _ensure_clipboard_listener_started(self, source: str = "auto") -> bool:
        """Start the clipboard listener if it is not already enabled.

        ``source`` picks the startup log line: startup pickup, manual button,
        or the auto start triggered by clicking the OpenChamber monitor."""
        if self._listener.enabled or self._close_requested:
            return False
        self._listener.start()
        if source == "auto":
            self._startup_deliver_current_clipboard_task()
        self._update_runtime_status()
        self._set_controls_enabled(not self._busy)
        if source == "auto":
            LOGGER.info("任务接收已自动启动")
        elif source == "manual":
            LOGGER.info("任务接收已手动启动")
        else:
            LOGGER.info("启动OpenChamber监控时自动开启任务接收")
        return True

    def _startup_deliver_current_clipboard_task(self) -> None:
        """Process a task already sitting on the clipboard exactly once after
        the automatic start, so work queued while AI Relay B was closed is
        not lost.  The listener snapshots the same text as seen so the very
        next change is the only other emission."""
        if self._busy:
            self._startup_pickup_pending = True
            return
        if self._startup_recovery_pending:
            # Reconciliation is still settling: the recovered reply must not
            # race a clipboard write, so the pickup waits for it.
            self._startup_pickup_pending = True
            return
        self._startup_pickup_pending = False
        text = self._app.clipboard().text() or ""
        if not self._listener.looks_like_relay_message(text):
            return
        self._on_clipboard_text(text)

    # ------------------------------------------------------------------ #
    # manual OpenChamber monitor (independent of the clipboard listener)
    # ------------------------------------------------------------------ #

    @Slot()
    def _toggle_monitor(self):
        if self._oc_monitor_stopping:
            # A stop was just requested and monitor threads are still
            # draining: ignore the click, the button stays disabled until the
            # finalisation below completes.
            return
        if self._oc_monitor_active:
            self._stop_monitor()
        else:
            started_here = self._ensure_clipboard_listener_started("monitor")
            self._start_monitor()
            if (
                started_here
                and self._clipboard_paused
                and self._oc_monitor_active
            ):
                self._clipboard_paused = False
                self._set_status("任务接收当前已暂停，已同时重新启动。")
                LOGGER.info("监听 OpenChamber 时自动恢复任务接收")

    def _monitor_live(self, generation: int) -> bool:
        """True only for signals coming from the CURRENT monitor run.

        Late signals emitted by an older generation (already stopped/restarted
        session) must never touch the new UI state."""
        return (
            generation == self._monitor_generation
            and self._oc_monitor_active
            and not self._oc_monitor_stopping
        )

    def _guard(self, generation: int, slot, *args):
        if self._monitor_live(generation):
            slot(*args)

    def _track_worker(self, task, role: str, generation: int) -> None:
        """Register a monitor worker and forget it when it emits ``finished``.

        ``finished`` is emitted on EVERY exit path of the run method, so the
        registry always drains and the window never tears itself down while a
        worker may still be emitting a late Qt signal.  The task is held by a
        strong reference and ``autoDelete`` is disabled: with the default
        auto-delete a finished QRunnable is destroyed on the WORKER thread,
        which can drop the queued ``finished`` emission (and any still-queued
        status/reply events) before the main thread processes them -- the
        intermittent drain-hang / teardown crash.  The reference is released
        only after ``finished`` was delivered on the main thread.
        """
        token = f"{generation}:{role}:{uuid4().hex[:8]}"
        self._monitor_workers.add(token)
        self._monitor_task_refs[token] = task
        task.setAutoDelete(False)
        task.signals.finished.connect(
            lambda tok=token: self._on_monitor_worker_finished(tok)
        )

    @Slot()
    def _on_monitor_worker_finished(self, token: str):
        self._monitor_workers.discard(token)
        # Drop the strong reference that kept the finished task (and its
        # Signals QObject) alive until its queued ``finished`` emission was
        # delivered; garbage collection now runs safely on the main thread.
        self._monitor_task_refs.pop(token, None)
        if self._oc_monitor_stopping and not self._monitor_workers:
            self._finish_monitor_stop()

    def _track_general_worker(self, task) -> None:
        """Register a non-monitor worker and forget it when ``finished`` is
        delivered, mirroring ``_track_worker``: auto-delete off, strong
        reference kept until the main thread processed the final signal."""
        token = f"w:{uuid4().hex[:8]}"
        self._general_workers.add(token)
        self._general_task_refs[token] = task
        task.setAutoDelete(False)
        task.signals.finished.connect(
            lambda tok=token: self._on_general_worker_finished(tok)
        )
        self._pool.start(task)

    @Slot()
    def _on_general_worker_finished(self, token: str):
        self._general_workers.discard(token)
        self._general_task_refs.pop(token, None)

    def _start_monitor(self):
        if self._oc_monitor_stopping or self._monitor_workers:
            # The previous monitor's threads have not all finished yet: a new
            # generation must never overlap the old one (avoids late-signal
            # races).  Require a completed stop before the next start.
            self._set_status("正在等待上一次监听完全停止……")
            return
        url = self._settings.openchamber_url.strip()
        directory = self._settings.openchamber_directory.strip()
        session_id = self._settings.openchamber_session_id.strip()
        if not directory:
            self._show_error("请先在设置中填写 OpenChamber 项目目录")
            return
        if not self._directory_is_valid(directory):
            self._show_error(f"OpenChamber 项目目录不存在：{directory}")
            return
        if not session_id:
            self._show_error("请先在设置中选择或填写 OpenChamber 会话 ID")
            return

        client = OpenChamberClient(
            url, auth_token=self._settings.openchamber_auth_token or None
        )
        try:
            if not client.session_exists(session_id, directory):
                self._show_error(
                    f"OpenChamber 会话 {session_id} 不存在，无法监听；"
                    "请在设置中重新选择会话"
                )
                return
        except Exception as exc:
            self._show_error(
                f"验证 OpenChamber 会话失败：{translate_error(str(exc))}；"
                "未启动监听"
            )
            return

        # Baseline: only the assistant messages that are ALREADY fully completed
        # (finish=stop, non-empty) may seed it -- history is never re-packaged.
        # A message that is still streaming when the monitor starts must stay
        # OUT of the baseline so it is wrapped once it completes later (and the
        # round tracker keeps watching it).  Using the full seen-id set here
        # would permanently skip a pre-existing message that completes after.
        try:
            scan = monitor_scan(client, session_id, directory)
        except Exception as exc:
            self._show_error(
                f"读取 OpenChamber 会话消息失败：{translate_error(str(exc))}；"
                "未启动监听"
            )
            return
        baseline = set(scan.completed_history_ids)

        self._oc_monitor_active = True
        self._oc_monitor_stopping = False
        self._monitor_generation += 1
        generation = self._monitor_generation
        self._oc_monitor_directory = directory
        self._oc_monitor_session = session_id
        self._oc_monitor_seen = set(baseline)
        self._oc_monitor_yielded = set()
        self._oc_monitor_stop = threading.Event()
        self._oc_monitor_interrupted = False
        self._oc_monitor_continuing = False
        self._oc_monitor_transport_down = False
        self._oc_monitor_auto_count = 0
        self._oc_monitor_auto_disabled = False
        self._oc_monitor_auto_pending = False
        self._oc_monitor_last_stall_message_id = None
        self._oc_monitor_busy_reset = threading.Event()
        self.monitor_button.setText("停止监控 OpenChamber")

        task = MonitorPollTask(
            client,
            session_id,
            directory,
            baseline,
            self._settings.poll_interval,
            self._oc_monitor_stop,
            transport_delays=self._monitor_transport_delays,
            busy_stale_log_interval=self._monitor_busy_stale_log_interval,
            busy_without_progress_threshold=self._monitor_busy_without_progress_threshold,
            busy_reset_event=self._oc_monitor_busy_reset,
            generation=generation,
        )
        task.signals.reply.connect(
            lambda text, mid, g=generation: self._guard(g, self._on_monitor_reply, text, mid)
        )
        task.signals.failed.connect(
            lambda error, g=generation: self._guard(g, self._on_monitor_failed, error)
        )
        task.signals.status.connect(
            lambda s, g=generation: self._guard(g, self._set_status, s)
        )
        task.signals.interrupted.connect(
            lambda payload, g=generation: self._guard(g, self._on_monitor_interrupted, payload)
        )
        task.signals.transport_error.connect(
            lambda message, g=generation: self._guard(g, self._on_monitor_transport_error, message)
        )
        task.signals.transport_recovered.connect(
            lambda g=generation: self._guard(g, self._on_monitor_transport_recovered)
        )
        self._track_worker(task, "poll", generation)
        self._oc_monitor_task = task
        self._pool.start(task)

        LOGGER.info(
            "监听开始 session_id=%s directory=%s",
            session_id, directory,
        )
        self._set_controls_enabled(True)
        self._set_status(f"正在监听 OpenChamber：会话 {session_id}")
        self._update_runtime_status()

    def _stop_monitor(self):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        # Asynchronous stop: request a graceful exit from every worker, block
        # new work and new UI updates, then finalise ONLY when the last
        # worker emits ``finished`` (see ``_on_monitor_worker_finished``).
        self._oc_monitor_stopping = True
        self._monitor_generation += 1
        self._oc_monitor_stop.set()
        self.monitor_button.setEnabled(False)
        self.monitor_button.setText("正在停止监控…")
        self._set_status("正在停止监听……")
        LOGGER.info(
            "监听停止请求 session_id=%s directory=%s",
            self._oc_monitor_session, self._oc_monitor_directory,
        )
        if not self._monitor_workers:
            self._finish_monitor_stop()

    def _finish_monitor_stop(self):
        """Final teardown, run on the main thread once ALL monitor workers
        have emitted ``finished`` (never while a thread may still emit)."""
        self._oc_monitor_active = False
        self._oc_monitor_stopping = False
        self._oc_monitor_task = None
        self._oc_monitor_seen = set()
        self._oc_monitor_yielded = set()
        self._oc_monitor_interrupted = False
        self._oc_monitor_continuing = False
        self._oc_monitor_transport_down = False
        self._oc_monitor_auto_count = 0
        self._oc_monitor_auto_disabled = False
        self._oc_monitor_auto_pending = False
        self._oc_monitor_last_stall_message_id = None
        if getattr(self, "_oc_monitor_busy_reset", None) is not None:
            self._oc_monitor_busy_reset.clear()
        self._oc_monitor_stop = threading.Event()
        self._oc_monitor_busy_reset = threading.Event()
        self.monitor_button.setText("监控 OpenChamber")
        self.monitor_button.setEnabled(True)
        self._set_controls_enabled(True)
        self._update_runtime_status()
        LOGGER.info(
            "用户停止监听 OpenChamber session_id=%s directory=%s",
            self._oc_monitor_session, self._oc_monitor_directory,
        )
        self._set_status("已停止监听 OpenChamber，会话和历史记录均已保留。")
        if self._close_after_monitors:
            self._close_after_monitors = False
            self.monitor_button.setEnabled(False)
            self.close()

    @Slot(object)
    def _on_monitor_interrupted(self, payload):
        if payload.get("generation") != self._monitor_generation:
            # Late interruption from an older monitor run (stopped / switched
            # session): never update the new state.
            return
        if (
            not self._oc_monitor_active
            or self._oc_monitor_stopping
            or payload.get("session_id") != self._oc_monitor_session
        ):
            # Stale poller (session switched meanwhile): never update the UI.
            return
        if self._monitor_blocked_by_a_side():
            # Session is owned by an in-flight B-side task (sourced from
            # A-side).  Monitor continues stall detection and auto-continue;
            # only _on_monitor_reply yields (final reply wrapping belongs to
            # the B-side task flow).
            LOGGER.info(
                "会话由B端任务流程负责，监听继续监控并自动续接 "
                "session_id=%s owner_task_id=%s reason=%s",
                self._oc_monitor_session,
                self._a_side_owner_task_id() or "unknown",
                payload.get("reason"),
            )
            self._set_status(
                f"当前会话由B端任务负责（{self._a_side_owner_task_id() or '未知'}），"
                "监听继续监控中"
            )
        reason = payload.get("reason")
        last_finish = payload.get("last_finish")
        if last_finish == "length":
            # A truncated-at-length-limit reply is a REAL final reply: never
            # auto-continue it (the operator may still continue or stop).
            self._monitor_stall_manual(payload, reason)
            return
        if self._oc_monitor_auto_pending or self._oc_monitor_continuing:
            # An auto/manual continuation is already waiting or being sent:
            # never stack a concurrent send (each stuck period emits at most
            # once anyway; this guard also rejects duplicate signals).
            LOGGER.info(
                "自动续接进行中，忽略重复中断信号 session_id=%s reason=%s",
                self._oc_monitor_session, reason,
            )
            return
        if (
            not self._oc_monitor_auto_disabled
            and self._oc_monitor_auto_count < _MONITOR_AUTO_CONTINUE_MAX
        ):
            self._begin_auto_monitor_continue(
                reason=reason,
                last_message_id=payload.get("last_message_id"),
            )
            return
        self._monitor_stall_manual(payload, reason)

    def _monitor_stall_manual(self, payload, reason):
        """Plain stall UI: abnormal-interrupted state, buttons and the
        continue/stop offer.  Once the 3 auto attempts are used up the copy
        is the requirement's "请人工继续或停止" message."""
        self._oc_monitor_interrupted = True
        self._set_controls_enabled(True)
        if self._oc_monitor_auto_count >= _MONITOR_AUTO_CONTINUE_MAX:
            self._oc_monitor_auto_disabled = True
            self._set_status("自动恢复3次仍未完成，请人工继续或停止")
            self.detail_label.setText(
                "已自动续接3次仍没有进展，不再自动发送。"
                "可点击“继续当前任务”再尝试一次，"
                "或点击“停止当前任务”结束监听。"
            )
            LOGGER.info(
                "自动恢复次数已达3次，转入人工等待 session_id=%s",
                self._oc_monitor_session,
            )
            return
        if reason == "monitor_busy_without_progress":
            self._set_status("检测到 OpenCode 长时间没有进展，可能已卡住。")
            self.detail_label.setText(
                "会话持续显示运行中，但超过120秒没有新消息或内容变化。"
                "可点击“继续当前任务”尝试续接，或点击“停止当前任务”结束监听。"
            )
        else:
            self._set_status(
                "检测到 OpenCode 已异常停顿：会话已进入空闲状态，但没有最终回复。"
            )
            self.detail_label.setText(
                "可能是工具结果续接失败、SSE读取超时或连接被重置。"
                "可点击“继续当前任务”从中断位置继续，"
                "或点击“停止当前任务”结束监听。"
            )
        LOGGER.info(
            "监听模式检测到工具续接中断 session_id=%s last_message_id=%s "
            "last_finish=%s reason=%s",
            payload.get("session_id"),
            payload.get("last_message_id"),
            payload.get("last_finish"),
            payload.get("reason") or reason,
        )

    def _begin_auto_monitor_continue(self, reason, last_message_id):
        """Attempt N (1-based) of the auto-recovery for a stalled round.

        Waits the short interval, then sends the fixed continue prompt to the
        ORIGINAL session with the current Agent/Model -- a single
        MonitorAutoContinueTask, no TASK, no tasks.json, no new session, no
        rotation.  Its outcome decides the next action.

        When a B-side task (sourced from A-side) owns the session, the
        auto-continue still fires: the monitor is responsible for session
        health, while the B-side task flow owns the final reply."""
        self._oc_monitor_auto_count += 1
        attempt = self._oc_monitor_auto_count
        self._oc_monitor_auto_pending = True
        self._oc_monitor_last_stall_message_id = last_message_id
        self._set_controls_enabled(True)
        self.task_label.setText(
            f"自动续接 {attempt}/3：OpenChamber 会话 {self._oc_monitor_session}"
        )
        self._set_status(f"检测到停滞，正在自动续接 {attempt}/3……")
        self.detail_label.setText(
            f"检测到 OpenCode 停滞，等待 {int(self._monitor_auto_continue_interval)} 秒后自动续接；"
            "可点击“停止当前任务”随时取消。"
        )
        LOGGER.info(
            "自动续接 %d/3 开始 session_id=%s directory=%s reason=%s",
            attempt, self._oc_monitor_session, self._oc_monitor_directory, reason,
        )
        generation = self._monitor_generation
        task = MonitorAutoContinueTask(
            url=self._settings.openchamber_url.strip(),
            session_id=self._oc_monitor_session,
            directory=self._oc_monitor_directory,
            agent=self._settings.openchamber_agent.strip() or None,
            model=self._settings.openchamber_model_ref(),
            interval=self._monitor_auto_continue_interval,
            stop_event=self._oc_monitor_stop,
            auth_token=self._settings.openchamber_auth_token or None,
        )
        task.signals.status.connect(
            lambda s, g=generation: self._guard(g, self._set_status, s)
        )
        task.signals.succeeded.connect(
            lambda label, g=generation: self._guard(g, self._on_auto_continue_succeeded, label)
        )
        task.signals.failed.connect(
            lambda error, cat, g=generation: self._guard(g, self._on_auto_continue_failed, error, cat)
        )
        self._track_worker(task, f"auto-{attempt}", generation)
        self._pool.start(task)

    @Slot(str)
    def _on_auto_continue_succeeded(self, _label: str):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        attempt = self._oc_monitor_auto_count
        self._oc_monitor_auto_pending = False
        self._oc_monitor_interrupted = False
        # The sent prompt creates a new user message: the poller observes it
        # as fresh activity, resets its counters and tracks the resumed round.
        # auto_count is KEPT: the next full stall tries the next attempt, and
        # only a wrapped final reply (or a stop) zeroes the budget.
        if getattr(self, "_oc_monitor_busy_reset", None) is not None:
            self._oc_monitor_busy_reset.set()
        self._set_controls_enabled(True)
        LOGGER.info(
            "自动续接 %d/3 发送成功 session_id=%s",
            attempt, self._oc_monitor_session,
        )
        self._set_status(f"已自动续接 {attempt}/3，等待 OpenChamber 完成……")
        self.detail_label.setText(
            "已发送续接提示。若再次停滞将自动继续续接，直至3次后转入人工。"
        )

    @Slot(str, str)
    def _on_auto_continue_failed(self, error: str, category: str):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        attempt = self._oc_monitor_auto_count
        if category == "transport":
            LOGGER.error(
                "自动续接 %d/3 发送失败（传输）session_id=%s error=%r",
                attempt, self._oc_monitor_session, error,
            )
            self._set_status(f"自动续接 {attempt}/3 发送失败（传输），正在复查会话……")
            self._confirm_auto_continue_progress()
            return
        # Non-retryable (HTTP 400 / Aborted / session missing / auth /
        # unexpected): never auto-continue this session again.
        self._oc_monitor_auto_pending = False
        self._oc_monitor_auto_disabled = True
        self._oc_monitor_interrupted = True
        self._set_controls_enabled(True)
        LOGGER.error(
            "自动续接 %d/3 发送失败（不可重试），停止自动恢复 session_id=%s error=%r",
            attempt, self._oc_monitor_session, error,
        )
        self._set_status(f"自动续接 {attempt}/3 失败，已停止自动恢复")
        self.detail_label.setText(
            f"自动续接失败：{translate_error(str(error))}。"
            "不再自动发送；请人工检查会话并点击“继续当前任务”或“停止当前任务”。"
        )

    def _confirm_auto_continue_progress(self):
        """A transport-failed auto continue (SSE timeout / connection reset /
        cert handshake) is re-queried before it counts toward the 3-attempt
        budget: if the session actually progressed, the attempt is NOT
        counted; otherwise the count stands and the next attempt runs."""
        task = MonitorRecoveryQueryTask(
            url=self._settings.openchamber_url.strip(),
            session_id=self._oc_monitor_session,
            directory=self._oc_monitor_directory,
            reference_message_id=self._oc_monitor_last_stall_message_id,
            delays=self._monitor_transport_delays,
            stop_event=self._oc_monitor_stop,
            auth_token=self._settings.openchamber_auth_token or None,
        )
        generation = self._monitor_generation
        task.signals.result.connect(
            lambda result, g=generation: self._guard(g, self._on_auto_recovery_query_result, result)
        )
        self._track_worker(task, "recovery", generation)
        self._pool.start(task)

    @Slot(str)
    def _on_auto_recovery_query_result(self, result: str):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        attempt = self._oc_monitor_auto_count
        self._oc_monitor_auto_pending = False
        if result == "progress":
            # The session DID move on despite the send failure: roll the
            # attempt back and simply keep monitoring the resumed round.
            self._oc_monitor_auto_count = max(0, attempt - 1)
            self._oc_monitor_interrupted = False
            if getattr(self, "_oc_monitor_busy_reset", None) is not None:
                self._oc_monitor_busy_reset.set()
            self._set_controls_enabled(True)
            LOGGER.info(
                "自动续接 %d/3 发送失败但会话已出现进展，不计入该次 session_id=%s",
                attempt, self._oc_monitor_session,
            )
            self._set_status("自动续接传输失败，但会话已有进展，继续正常监听……")
            return
        # No visible progress after the re-queries: the attempt counts.
        LOGGER.info(
            "自动续接 %d/3 发送失败已确认会话无进展，计入该次 session_id=%s",
            attempt, self._oc_monitor_session,
        )
        if attempt < _MONITOR_AUTO_CONTINUE_MAX:
            self._set_status(f"自动续接 {attempt}/3 失败，准备第 {attempt + 1} 次……")
            self._begin_auto_monitor_continue(
                reason="monitor_auto_transport_retry",
                last_message_id=self._oc_monitor_last_stall_message_id,
            )
            return
        # 3 transport-failed attempts: same exhausted manual state as 3 stalls.
        self._oc_monitor_auto_disabled = True
        self._monitor_stall_manual(
            {
                "session_id": self._oc_monitor_session,
                "directory": self._oc_monitor_directory,
                "last_message_id": self._oc_monitor_last_stall_message_id,
                "last_finish": None,
                "reason": "monitor_auto_transport_exhausted",
            },
            "monitor_auto_transport_exhausted",
        )

    @Slot(str)
    def _on_monitor_transport_error(self, message: str):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        self._oc_monitor_transport_down = True
        self._set_controls_enabled(True)
        self._set_status(message)
        self.detail_label.setText(
            "OpenChamber 连接多次中断，暂时无法确认会话状态。"
            "可点击“停止当前任务”结束监听；恢复连接后将自动继续监听。"
        )
        LOGGER.info(
            "监听模式传输异常 session_id=%s", self._oc_monitor_session,
        )

    @Slot()
    def _on_monitor_transport_recovered(self):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        self._oc_monitor_transport_down = False
        self._set_controls_enabled(True)
        self._set_status("OpenChamber 连接已恢复，继续监听……")

    @Slot(str, str)
    def _on_monitor_reply(self, text: str, msg_id: str):
        # Final reply wrapping belongs to the B-side task flow.  When a B-side
        # task (sourced from A-side) owns this session, the monitor yields and
        # lets the auto relay consume the final reply.
        if self._monitor_blocked_by_a_side():
            if msg_id not in self._oc_monitor_yielded:
                self._oc_monitor_yielded.add(msg_id)
                LOGGER.info(
                    "会话由B端任务流程负责，监听不包装回复 "
                    "session_id=%s owner_task_id=%s msg_id=%s",
                    self._oc_monitor_session,
                    self._a_side_owner_task_id() or "unknown",
                    msg_id,
                )
                self._set_status(
                    f"当前会话回复包装权属于B端任务流程"
                    f"（{self._a_side_owner_task_id() or '未知'}），"
                    "监听不包装回复"
                )
            return
        # Unified dedup: never package a message the auto relay (or an
        # earlier poll of this monitor) already wrapped.
        if is_message_wrapped(msg_id, self._oc_monitor_session) or msg_id in self._oc_monitor_seen:
            return
        self._oc_monitor_seen.add(msg_id)
        mark_message_wrapped(msg_id, self._oc_monitor_session)

        task_id = f"manual-{uuid4()}"
        try:
            response = wrap_response(
                text,
                task_id,
                ProtocolFormat.V1,
                round_number=0,
                max_rounds=1,
            )
        except Exception as exc:
            self._show_error(f"包装 OpenChamber 回复失败：{translate_error(str(exc))}")
            return
        try:
            self._workflow.save_reply(task_id, response)
            self._workflow.registry.mark(
                task_id,
                "COMPLETED",
                executor=TARGET_OPENCHAMBER,
                session_id=self._oc_monitor_session,
                directory=self._oc_monitor_directory,
                reply_file=str(
                    self._workflow.reply_file_for(task_id)
                ),
            )
        except Exception:
            # Registry/reply persistence is best-effort for manual replies;
            # the clipboard copy below is the primary product.
            pass
        self._listener.write_response(response)
        self._refresh_saved_tasks(select_newest=True)
        # A real final reply ends the interrupted round: reset the abnormal
        # state so the user can start a new round (continue disabled again).
        self._oc_monitor_interrupted = False
        self._oc_monitor_continuing = False
        self._oc_monitor_transport_down = False
        # The final reply also resets the auto-recovery budget: the next stall
        # starts a FRESH round from attempt 1/3 (requirement).
        self._oc_monitor_auto_count = 0
        self._oc_monitor_auto_disabled = False
        self._oc_monitor_auto_pending = False
        self._oc_monitor_last_stall_message_id = None
        self._set_controls_enabled(True)
        LOGGER.info(
            "监听模式收到有效最终回复 session_id=%s last_message_id=%s",
            self._oc_monitor_session, msg_id,
        )
        self._set_status("检测到新回复，已包装并复制")
        self.detail_label.setText(
            f"OpenChamber 手动回复已包装（任务 {task_id[:12]}…）并复制到剪贴板。"
        )
        self._maybe_auto_compact_after_monitor_reply()

    def _maybe_auto_compact_after_monitor_reply(self):
        """Compact the monitored OpenChamber session after a manual/monitor
        reply was wrapped and copied, mirroring the auto-compact performed on
        the A-side task lane.  Runs on a worker and only touches the
        ``_compacting`` flag (the monitor path owns no task lane)."""
        if not getattr(
            self._settings, "auto_compact_after_response", False
        ) or self._compacting:
            return
        if not (
            self._oc_monitor_session
            and self._oc_monitor_directory
        ):
            return
        self._compacting = True
        self._set_status("回复已复制；正在压缩当前会话上下文…")
        task = CompactTask(
            self._workflow,
            self._oc_monitor_session,
            self._oc_monitor_directory,
        )
        task.signals.done.connect(self._on_monitor_compact_done)
        self._track_general_worker(task)

    @Slot(bool)
    def _on_monitor_compact_done(self, ok: bool):
        self._compacting = False
        if ok:
            self.detail_label.setText("回复已复制；当前会话上下文压缩完成。")
        else:
            LOGGER.warning("监听模式 auto compact failed after a reply")
            self.detail_label.setText(
                "回复已完成并复制，但上下文压缩失败；不影响已完成的回复。"
            )

    @Slot(str)
    def _on_monitor_failed(self, error: str):
        self._set_status(f"监听失败：{translate_error(error)}")

    def closeEvent(self, event):
        # Never destroy the window while a worker (monitor OR general, e.g.
        # RelayTask/SelfCheckTask) may still emit a Qt signal: request a
        # graceful close, keep checking via timer (never blocking the event
        # loop) and only accept the close once every worker emitted
        # ``finished``.  After a bounded wait a warning is logged but threads
        # are never force-terminated.
        self._listener.pause()
        self._close_requested = True
        if self._monitor_workers or self._general_workers:
            self._close_after_monitors = True
            if not self._oc_monitor_stopping:
                self._stop_monitor()
            self._schedule_close_check()
            event.ignore()
            return
        if self._oc_monitor_stopping:
            self._finish_monitor_stop()
        super().closeEvent(event)

    def _schedule_close_check(self):
        now = time.monotonic()
        pending = len(self._monitor_workers) + len(self._general_workers)
        if self._monitor_close_deadline is None:
            self._monitor_close_deadline = now + self._monitor_close_timeout
        elif now > self._monitor_close_deadline and pending:
            LOGGER.warning(
                "窗口关闭等待超时，仍有 %d 个 worker 未结束；"
                "继续等待，不再强制终止线程。",
                pending,
            )
            self._monitor_close_deadline = now + self._monitor_close_timeout
        QTimer.singleShot(200, self._check_close_ready)

    @Slot()
    def _check_close_ready(self):
        if self._monitor_workers or self._general_workers or self._oc_monitor_stopping:
            self._schedule_close_check()
            return
        if self._close_after_monitors:
            self._close_after_monitors = False
            self.close()

    @Slot(str)
    def _on_clipboard_text(self, text: str):
        if not self._listener.looks_like_relay_message(text):
            return

        try:
            message = parse_message(text)
        except RelayProtocolError as exc:
            self._show_error(
                f"剪贴板 AI_RELAY 消息格式错误：{translate_error(str(exc))}"
            )
            return

        if message.message_type is not MessageType.TASK:
            return

        existing = self._workflow.registry.record(message.message_id)
        if existing is not None:
            # Invariant 8: a repeated TASK_ID is never re-executed; side A
            # gets the existing state, or the saved reply when the task
            # already produced one (success or structured failure).
            LOGGER.info(
                "重复TASK task_id=%s existing_state=%s，返回已有状态/回复",
                message.message_id, existing.get("state", "?"),
            )
            self._answer_duplicate_task(message, existing)
            return

        # Invariant 1/2: a valid A-side TASK is never rejected for the
        # window being busy.  It is claimed synchronously (persisting its
        # full protocol text plus a monotonic sequence) and either starts
        # now or enters the persistent FIFO queue and runs in order.
        self._accept_a_side_task(message, text)

    # ------------------------------------------------------------------ #
    # A-side task priority (invariants 1/4/7/8)
    # ------------------------------------------------------------------ #

    def _register_a_side_claim(self, message):
        """Record which directories / sessions this A-side task may occupy.

        The claim already knows the project directory (task workdir or the
        configured default); the OpenChamber session id is learned later in
        ``_on_session_started``.  Only OpenChamber-targeted tasks can occupy
        a monitored session, so a REASONIX task never blocks the monitor.
        """
        try:
            executor = resolve_executor_kind(message.target, self._settings)
        except Exception:
            return
        if executor != TARGET_OPENCHAMBER:
            return
        directory = (message.workdir or self._settings.openchamber_directory or "").strip()
        if directory:
            self._a_side_inflight_dirs.add(directory_key(directory))
        else:
            # No directory known yet (and none configured): the task will
            # fail fast with a structured response; meanwhile yield to it.
            self._a_side_oc_pending = True

    def _a_side_active(self) -> bool:
        return bool(
            self._a_side_inflight_sessions
            or self._a_side_inflight_dirs
            or self._a_side_oc_pending
        )

    def _monitor_blocked_by_a_side(self) -> bool:
        """True when the monitored session is occupied by an in-flight B-side
        task (sourced from A-side).

        The monitor yields reply wrapping (final replies belong to the B-side
        task flow) but continues stall detection and auto-continue for session
        health."""
        if not self._a_side_active():
            return False
        if self._a_side_oc_pending:
            # An OC task not yet identified may be using ANY session: yield.
            return True
        if self._oc_monitor_session in self._a_side_inflight_sessions:
            return True
        directory = (self._oc_monitor_directory or "").strip()
        return bool(directory) and directory_key(directory) in self._a_side_inflight_dirs

    def _a_side_owner_task_id(self) -> str:
        """Task id of the B-side task (sourced from A-side) that owns the
        currently monitored session ("" when unknown).  Used in the monitor's
        yield/continue log lines for auditability."""
        return self._a_side_session_owner.get(self._oc_monitor_session, "")

    def _answer_duplicate_task(self, message, existing: dict):
        """A side may re-send the same TASK_ID: never re-execute, always
        answer with the existing state or the already-saved reply, and the
        response stays bound to the ORIGINAL TASK_ID (never a manual-* id).
        """
        state = str(existing.get("state", "UNKNOWN"))
        response = None
        if state in ("COMPLETED", "FAILED", "STOPPED_BY_USER"):
            response = self._workflow.load_reply(message.message_id)
        if response is None:
            if state in ("RECEIVED", "PROCESSING"):
                body = (
                    f"任务 {message.message_id} 当前状态为 {state}：正在执行中，"
                    "完成后将返回最终结果，请勿重复发送。"
                )
            else:
                error = str(existing.get("error") or "").strip()
                body = f"任务 {message.message_id} 当前状态为 {state}"
                if error:
                    body += f"，错误：{error}"
                body += "。"
            try:
                response = wrap_response(
                    body,
                    message.message_id,
                    message.protocol_format,
                    message.round_number,
                    message.max_rounds,
                )
            except Exception as exc:
                self._show_error(f"生成重复任务状态响应失败：{translate_error(str(exc))}")
                return
        self._listener.write_response(response)
        self.detail_label.setText(
            f"重复任务 {message.message_id[:12]}…（{state}）：已返回已有状态/回复，未重复执行。"
        )
        self._set_status(f"重复任务 {message.message_id[:12]}…（{state}）已返回状态/回复")

    # ------------------------------------------------------------------ #
    # FIFO queue (invariants 1/2/5): accepted B-side tasks (sourced from
    # A-side) run in order and are never lost, even across a program restart.
    # ------------------------------------------------------------------ #

    def _accept_a_side_task(self, message, raw_text: str):
        """Claim a NEW valid A-side TASK synchronously on the main thread.

        The claim persists the full protocol text plus a monotonic sequence
        into tasks.json, so the task survives a restart.  If a task is
        already running (or the window is closing) the new task is QUEUED
        -- never rejected or dropped -- and runs in FIFO order once the
        current task reaches a terminal state; otherwise it starts now."""
        registry = self._workflow.registry
        sequence = registry.next_sequence()
        received_at = datetime.now().isoformat()
        if self._busy or self._close_requested:
            registry.mark(
                message.message_id,
                "QUEUED",
                raw_message=raw_text,
                sequence=str(sequence),
                received_at=received_at,
            )
            queued = registry.queued_records()
            ahead = max(len(queued) - 1, 0)
            LOGGER.info(
                "收到A端TASK task_id=%s state=QUEUED queue_ahead=%d（当前忙碌，排队执行）",
                message.message_id, ahead,
            )
            self._update_queue_label()
            self._set_status(
                f"已接收任务，当前前方有 {ahead} 个等待任务。"
            )
            self.detail_label.setText(
                f"已接收任务 {message.message_id[:12]}…（排队位置 {ahead + 1}）。"
                "当前任务进入终态后将按顺序自动执行，无需重新复制。"
            )
            return
        # Idle: claim as RECEIVED (process() accepts exactly this state)
        # and start immediately.
        registry.mark(
            message.message_id,
            "RECEIVED",
            raw_message=raw_text,
            sequence=str(sequence),
            received_at=received_at,
        )
        LOGGER.info(
            "收到A端TASK task_id=%s state=RECEIVED round=%s/%s",
            message.message_id, message.round_number, message.max_rounds,
        )
        self._update_queue_label()
        self._launch_a_side_task(message, raw_text)

    def _launch_a_side_task(self, message, raw_text: str):
        """Start one claimed B-side task (sourced from A-side) as a single
        worker (only one RelayTask runs at a time) and register its
        session/directory ownership so the monitor yields reply wrapping."""
        self._busy = True
        self._current_outcome = None
        self._current_task_is_auto = True
        self._task_cancel_event = threading.Event()
        # Record the executing task id and start a fresh generation: a manual
        # 包装内容 of THIS round bumps the generation so this worker's late
        # success/failure is treated as stale once the operator resolves it.
        self._current_a_task_id = message.message_id
        self._current_a_task_generation += 1
        generation = self._current_a_task_generation
        self._register_a_side_claim(message)
        workdir = message.workdir or self._settings.openchamber_directory or "未配置"
        self.task_label.setText(
            f"当前任务：{message.message_id}｜目标 {message.target}｜项目 {workdir}"
        )
        self._set_controls_enabled(False)

        task = RelayTask(self._workflow, raw_text, self._task_cancel_event)
        task.signals.status.connect(self._set_status)
        # The generation guard lives in the lambda (plain Python, so the
        # handlers keep their single-arg signatures): once this round is
        # manually wrapped (or the task is stopped) the generation is bumped,
        # and any success/failure/session signal that a superseded worker
        # emits late is dropped instead of re-writing the clipboard or
        # re-finalising an already-resolved round.
        task.signals.succeeded.connect(
            lambda result, g=generation: self._task_succeeded(result)
            if g == self._current_a_task_generation
            else None
        )
        task.signals.failed.connect(
            lambda error, g=generation: self._task_failed(error)
            if g == self._current_a_task_generation
            else None
        )
        task.signals.session_started.connect(
            lambda outcome, g=generation: self._on_session_started(outcome)
            if g == self._current_a_task_generation
            else None
        )
        self._track_general_worker(task)

    def _start_next_queued_task(self):
        """Dequeue and start the next QUEUED A-side task, if any (FIFO).

        Runs on the Qt main thread, scheduled from every task terminal exit
        (``_finish_task``); it never starts a task while the window is
        closing, a task is already running, or the startup reconciliation
        of stale records is still in progress.
        """
        if self._close_requested or self._busy or self._startup_recovery_pending:
            return
        records = self._workflow.registry.queued_records()
        if not records:
            return
        head = records[0]
        task_id = head["task_id"]
        raw_text = head.get("raw_message") or ""
        if not raw_text:
            # A QUEUED task whose body was never persisted cannot be
            # re-executed: park it as terminal RECOVERY_REQUIRED and move
            # on, so one broken record can never wedge the queue.
            self._workflow.registry.mark(
                task_id,
                "RECOVERY_REQUIRED",
                "队列任务缺少原始协议内容，无法重新执行，需要人工检查",
            )
            LOGGER.error("队列任务缺少 raw_message，标记RECOVERY_REQUIRED task_id=%s", task_id)
            self._update_queue_label()
            QTimer.singleShot(0, self._start_next_queued_task)
            return
        try:
            message = parse_message(raw_text)
        except RelayProtocolError as exc:
            self._workflow.registry.mark(
                task_id, "FAILED", f"队列任务协议无法解析：{exc}"
            )
            LOGGER.error("队列任务协议无法解析，标记FAILED task_id=%s error=%s", task_id, exc)
            self._update_queue_label()
            QTimer.singleShot(0, self._start_next_queued_task)
            return
        # QUEUED -> RECEIVED (raw_message/sequence are preserved); process()
        # then proceeds to PROCESSING.  The task_id is re-executed from the
        # persisted body -- it is never re-sent from the clipboard.
        self._workflow.registry.mark(task_id, "RECEIVED")
        LOGGER.info("开始执行队列中的任务 task_id=%s", task_id)
        self._update_queue_label()
        self._launch_a_side_task(message, raw_text)

    def _update_queue_label(self):
        queued = self._workflow.registry.queued_records()
        n = len(queued)
        self._queue_status_label.setText(f"等待任务：{n}")
        self._queue_status_label.setStyleSheet(
            "color: #fbbf24;" if n else "color: #9ca3af;"
        )

    # ------------------------------------------------------------------ #
    # task results
    # ------------------------------------------------------------------ #

    @Slot()
    def _continue_task(self):
        pending = self._workflow.pending_continue
        if pending is not None and not self._busy:
            # An auto-relay interrupted task awaits its manual continue
            # (the recovery-exhausted FAILED state keeps pending set, so
            # this button is the operator's manual "继续当前任务").
            self._run_continue_from(pending)
            return
        # The button is also offered while a B-side task is still BUSY
        # (recovery exhausted) or while the manual monitor is active: in
        # both cases the manual continuation goes through the MONITOR,
        # which watches the same session and shares the running task's
        # cancel event -- the busy B-side worker consumes the resulting
        # reply and completes the task.  With no pending auto-relay task
        # the monitor offer is the (only) manual continuation.
        if (
            self._oc_monitor_active
            and self._oc_monitor_interrupted
            and not self._oc_monitor_continuing
            and not self._oc_monitor_auto_pending
            and not self._oc_monitor_transport_down
        ) and (pending is None or self._busy):
            self._start_monitor_continue()
            return
        if pending is not None and self._busy:
            self._show_error(
                "任务正在等待中：请改用“停止当前任务”，或等待监控端手动续接可用"
            )
            return
        self._show_error("没有中断中的 OpenChamber 任务可继续")

    def _run_continue_from(self, pending):
        """Run the manual "继续当前任务" for an interrupted task record.

        Allowed both when idle and while a B-side task is still busy: in
        the busy case the operator is finishing a round whose automatic
        recovery was exhausted, and the existing busy worker consumes the
        continuation reply (the task's completion timeout keeps applying).
        """
        self._busy = True
        self._current_task_is_auto = False
        self._task_cancel_event = threading.Event()
        self.task_label.setText(
            f"继续任务：{pending.message.message_id}｜"
            f"OpenChamber 会话 {pending.session_id}｜项目 {pending.directory}"
        )
        self._set_controls_enabled(False)

        task = ContinueTask(self._workflow, self._task_cancel_event)
        task.signals.status.connect(self._set_status)
        task.signals.succeeded.connect(self._task_succeeded)
        task.signals.failed.connect(self._task_failed)
        task.signals.session_started.connect(self._on_session_started)
        self._track_general_worker(task)

    def _start_monitor_continue(self):
        """One manual "继续当前任务" for a manually monitored round: sends
        the fixed continue prompt to the ORIGINAL session with the current
        Agent/Model.  Never creates an AI Relay TASK, never writes
        tasks.json, never triggers auto rotation.

        When a B-side task owns the session, the continue fires: the monitor
        supports session health while the B-side task flow owns the reply."""
        self._oc_monitor_continuing = True
        LOGGER.info(
            "用户发送监听续接 session_id=%s directory=%s continuation_attempts=1",
            self._oc_monitor_session, self._oc_monitor_directory,
        )
        self.task_label.setText(
            f"监听续接：OpenChamber 会话 {self._oc_monitor_session}｜"
            f"项目 {self._oc_monitor_directory}"
        )
        self._set_controls_enabled(False)
        self._set_status("正在发送监听续接提示……")

        task = MonitorContinueTask(
            url=self._settings.openchamber_url.strip(),
            session_id=self._oc_monitor_session,
            directory=self._oc_monitor_directory,
            agent=self._settings.openchamber_agent.strip() or None,
            model=self._settings.openchamber_model_ref(),
            auth_token=self._settings.openchamber_auth_token or None,
        )
        generation = self._monitor_generation
        task.signals.status.connect(
            lambda s, g=generation: self._guard(g, self._set_status, s)
        )
        task.signals.succeeded.connect(
            lambda result, g=generation: self._guard(g, self._on_monitor_continue_succeeded, result)
        )
        task.signals.failed.connect(
            lambda error, g=generation: self._guard(g, self._on_monitor_continue_failed, error)
        )
        self._track_worker(task, "continue", generation)
        self._pool.start(task)

    @Slot(str)
    def _on_monitor_continue_succeeded(self, _result: str):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        self._oc_monitor_continuing = False
        # The sent prompt creates a new user message: the poller observes it
        # as fresh activity, resets its confirmation counter and tracks the
        # resumed round.  Clear the abnormal state so buttons reflect that.
        self._oc_monitor_interrupted = False
        # Zero the busy-without-progress timer in the poller so a fresh 120s
        # window starts for the resumed round (no automatic continue).
        if getattr(self, "_oc_monitor_busy_reset", None) is not None:
            self._oc_monitor_busy_reset.set()
        self._set_controls_enabled(True)
        LOGGER.info("监听续接发送成功 session_id=%s", self._oc_monitor_session)
        self._set_status("已发送续接提示，正在等待 OpenChamber 完成……")

    @Slot(str)
    def _on_monitor_continue_failed(self, error: str):
        if not self._oc_monitor_active or self._oc_monitor_stopping:
            return
        self._oc_monitor_continuing = False
        self._set_controls_enabled(True)
        LOGGER.error(
            "监听续接发送失败 session_id=%s error=%r",
            self._oc_monitor_session, error,
        )
        self._show_error(f"发送监听续接提示失败：{translate_error(str(error))}")

    @Slot()
    def _retry_new_session(self):
        """One fresh-session retry of a model-rejected task (one-shot)."""
        if self._busy:
            return
        pending = self._workflow.pending_rejection
        if pending is None:
            self._show_error("没有可重试的模型拒绝任务")
            return
        self._busy = True
        self._current_task_is_auto = False
        self._task_cancel_event = threading.Event()
        self._refresh_session_after_success = True
        self.task_label.setText(
            f"新会话重试：{pending.message.message_id}｜项目 {pending.directory}"
        )
        self._set_controls_enabled(False)

        task = NewSessionRetryTask(self._workflow, self._task_cancel_event)
        task.signals.status.connect(self._set_status)
        task.signals.succeeded.connect(self._task_succeeded)
        task.signals.failed.connect(self._task_failed)
        task.signals.session_started.connect(self._on_session_started)
        self._track_general_worker(task)

    @Slot()
    def _stop_task(self):
        # Only interrupts the wait loops of a RUNNING clipboard OpenChamber
        # task; the OpenChamber session itself is never deleted/terminated.
        if self._busy and self._task_cancel_event is not None:
            self._set_status("正在停止等待…")
            self.stop_button.setEnabled(False)
            self._task_cancel_event.set()
            return
        # Not running: an interrupted task is awaiting manual "继续当前任务".
        # The auto-relay offer takes priority over the manual monitor, and
        # the operator can stop pursuing EITHER one (the OpenChamber session
        # itself is always kept).
        wf = self._workflow
        pending = wf.pending_continue
        if pending is not None and pending.message is not None:
            wf.pending_continue = None
            try:
                # Invariant 7: stopping an interrupted A-side task is a
                # terminal outcome too -- side A gets a structured stop
                # response bound to the ORIGINAL TASK_ID (never silent).
                # force=True: the task may already be FAILED (recovery
                # exhausted), an explicit operator re-entry that may still
                # write the terminal STOPPED_BY_USER state.
                wf._finalize_failure(
                    pending.message,
                    "STOPPED_BY_USER",
                    "openchamber_stop:user stopped interrupted task",
                    executor=TARGET_OPENCHAMBER,
                    session_id=pending.session_id,
                    directory=pending.directory,
                    force=True,
                )
            except Exception as exc:
                LOGGER.warning("stop interrupted task failed to record: %s", exc)
            failure = wf.failure_response
            wf.failure_response = None
            if failure:
                try:
                    self._listener.write_response(failure)
                    self._last_response = failure
                except Exception as exc:
                    LOGGER.error("写入停止响应到剪贴板失败: %s", exc)
            self._set_controls_enabled(True)
            queued = [] if self._close_requested else self._workflow.registry.queued_records()
            suffix = (
                f"；等待任务 {len(queued)} 个，将按顺序继续执行。" if queued else ""
            )
            if self._pending_recovered_reply is not None and not self._close_requested:
                recovered = self._pending_recovered_reply
                self._pending_recovered_reply = None
                try:
                    self._listener.write_response(recovered)
                    self._last_response = recovered
                except Exception as exc:
                    LOGGER.error("延迟写入恢复的回复到剪贴板失败: %s", exc)
            self._set_status(f"已停止当前任务{suffix}")
            self.detail_label.setText(
                "已停止当前任务；OpenChamber 会话仍保留，"
                "可点击“监控 OpenChamber”捕获迟到回复。" + suffix
            )
            if queued:
                QTimer.singleShot(0, self._start_next_queued_task)
            return
        if self._oc_monitor_active:
            # Stop the manual monitor (interrupted or transport-down or
            # simply observing).  The session + history are all kept.
            self._stop_monitor()
            return
        self._show_error("当前没有正在运行或可停止的 OpenChamber 任务")

    @Slot(str)
    def _task_succeeded(self, response: str):
        LOGGER.info("task completed response_length=%d", len(response))
        self._last_response = response
        self._last_outcome = self._workflow.outcome
        self._refresh_saved_tasks(select_newest=True)
        self._listener.write_response(response)

        detail = "已将包装后的回复写入剪贴板。"
        outcome = self._workflow.outcome
        rotation_trigger = None
        if (
            self._current_task_is_auto
            and self._settings.auto_rotate_enabled
            and outcome is not None
            and outcome.executor == TARGET_OPENCHAMBER
            and outcome.directory
        ):
            directory = outcome.directory
            if self._rotation.note_auto_success(directory):
                rotation_trigger = directory
            detail += (
                f" 自动轮换进度：{self._rotation.count(directory)}/"
                f"{self._settings.auto_rotate_threshold}"
            )
        if outcome is not None:
            if outcome.executor == TARGET_OPENCHAMBER and outcome.session_id:
                detail += f"（执行端 OpenChamber，会话 {outcome.session_id}）"
            if outcome.note:
                detail += f" {outcome.note}"
            elif outcome.model_info:
                detail += f" {outcome.model_info}"
        self.detail_label.setText(detail)
        if self._refresh_session_after_success:
            # A successful fresh-session retry adopted a NEW OpenChamber
            # session: just surface the switch (the session selector is gone).
            self._refresh_session_after_success = False
            if outcome is not None and outcome.session_id:
                self._set_status("已切换到新会话")
            if outcome is not None and outcome.directory:
                self._rotation.reset(outcome.directory)
        if (
            getattr(self._settings, "auto_compact_after_response", False)
            and not self._compacting
            and outcome is not None
            and outcome.executor == TARGET_OPENCHAMBER
            and outcome.session_id
            and outcome.directory
        ):
            # Reply is wrapped and clipped BEFORE this point; the compact lane
            # keeps the task busy so new A-side tasks only queue, and the next
            # queued task starts only after the compaction completes (or fails).
            LOGGER.info(
                "auto compact trigger session=%s directory=%s setting=%s",
                outcome.session_id, outcome.directory,
                getattr(self._settings, "auto_compact_after_response", False),
            )
            self._start_auto_compact(outcome, rotation_trigger)
            return
        if not getattr(self._settings, "auto_compact_after_response", False):
            LOGGER.info("auto compact skipped: setting disabled")
        elif self._compacting:
            LOGGER.info("auto compact skipped: already compacting")
        elif outcome is None:
            LOGGER.info("auto compact skipped: no outcome")
        elif outcome.executor != TARGET_OPENCHAMBER:
            LOGGER.info("auto compact skipped: executor=%s", outcome.executor)
        else:
            LOGGER.info(
                "auto compact skipped: missing session/directory "
                "session=%r directory=%r",
                getattr(outcome, "session_id", None),
                getattr(outcome, "directory", None),
            )
        self._finish_task("完成：等待下一个任务")
        if rotation_trigger is not None:
            self._start_auto_rotation(rotation_trigger)

    def _start_auto_compact(self, outcome, rotation_trigger=None):
        """Trigger a first-class opencode compaction of the OpenChamber
        session that produced the reply just copied, and keep the task lane
        open until that compaction finishes.  Runs on a worker so the Qt main
        thread does not block while the compaction completes."""
        self._compacting = True
        self._set_status("回复已复制；正在压缩当前会话上下文…")
        task = CompactTask(self._workflow, outcome.session_id, outcome.directory)
        task.signals.done.connect(
            lambda ok: self._on_auto_compact_done(ok, rotation_trigger)
        )
        self._track_general_worker(task)

    @Slot()
    def _on_auto_compact_done(self, ok: bool, rotation_trigger):
        self._compacting = False
        LOGGER.info("auto compact finished ok=%s", ok)
        if ok:
            self._finish_task("完成：等待下一个任务")
            self.detail_label.setText("回复已复制；当前会话上下文压缩完成。")
        else:
            LOGGER.warning("auto compact failed after a successful reply")
            self._finish_task("回复完成，但上下文压缩失败")
            self.detail_label.setText(
                "回复已完成并复制，但上下文压缩失败；不影响已完成的回复。"
            )
        if rotation_trigger is not None:
            self._start_auto_rotation(rotation_trigger)

    def _start_auto_rotation(self, directory: str):
        if self._rotation_pending or not self._settings.auto_rotate_enabled:
            return
        self._rotation_pending = True
        self._rotation_directory = directory
        url = self._url_edit.text().strip() or DEFAULT_OPENCHAMBER_URL
        previous_session_id = None
        outcome = self._workflow.outcome
        if outcome is not None and outcome.session_id:
            previous_session_id = outcome.session_id
        # ponytail: the fixed-session UI row is gone; a persisted base title
        # in settings always wins over any UI hint, so no hint is passed.
        base_title_hint = None
        task = RotateSessionTask(
            url,
            directory,
            previous_session_id,
            self._settings,
            inherit_auto_accept=self._settings.auto_rotate_inherit_auto_accept,
            base_title_hint=base_title_hint,
        )
        task.signals.succeeded.connect(self._rotation_succeeded)
        task.signals.failed.connect(self._rotation_failed)
        self._set_status("已达到轮换阈值，正在创建并切换新会话…")
        self._track_general_worker(task)

    @Slot(str)
    def _rotation_succeeded(self, session_id: str):
        self._rotation_pending = False
        directory = self._rotation_directory
        self._rotation.reset(directory)
        self._set_status("自动轮换：已切换到新会话")
        self.detail_label.setText(
            f"自动轮换：达到阈值后已创建并切换到新会话 {session_id}。"
        )

    @Slot(str)
    def _rotation_failed(self, error: str):
        self._rotation_pending = False
        self._show_error(
            f"自动轮换会话失败：{translate_error(error)}；"
            "计数已保留，下次成功任务将重试轮换"
        )

    @Slot(str)
    def _task_failed(self, error: str):
        LOGGER.error("task failed: %s", error)
        if self._startup_check_pending:
            self._startup_check_pending = False
            self._finish_task("Reasonix 自检失败（不影响 OpenChamber）")
            self.detail_label.setText(
                f"Reasonix 自检失败：{translate_error(error)}。"
                "监听仍会启动；REASONIX 任务将失败，OPENCHAMBER 任务不受影响。"
            )
            self._auto_start_after_self_check()
            return
        # Invariant 7: a task that truly failed (or was stopped) must still
        # answer side A with a structured RESPONSE bound to the ORIGINAL
        # TASK_ID -- never a silent stop.  A first model rejection is NOT
        # terminal (the "新会话重试" offer keeps the task resumable), so its
        # failure response is withheld until the retry ends or the task is
        # stopped.
        failure = self._workflow.failure_response
        self._workflow.failure_response = None
        if failure and (
            CANCELLED_MARKER in error
            or MODEL_REJECTION_AGAIN in error
            or MODEL_REJECTION_PROMPT not in error
        ):
            try:
                self._listener.write_response(failure)
                self._last_response = failure
                LOGGER.info(
                    "已向剪贴板返回任务失败/停止响应（关联原TASK_ID）"
                )
            except Exception as exc:
                LOGGER.error("写入失败响应到剪贴板失败: %s", exc)
        if CANCELLED_MARKER in error:
            self._finish_task("已停止等待，OpenChamber 会话仍保留")
            self.detail_label.setText(
                "已停止等待；OpenChamber 会话仍保留。"
                "可点击“监控 OpenChamber”捕获迟到回复，或发送新任务。"
            )
            return
        if MODEL_REJECTION_AGAIN in error:
            # The fresh-session retry itself was rejected again: no third
            # session, buttons restored, operator checks the model service.
            self._finish_task("新会话重试失败")
            self.detail_label.setText(
                f"{MODEL_REJECTION_AGAIN} "
                "不再创建新会话；请检查模型服务日志后另发新任务。"
            )
            return
        if MODEL_REJECTION_PROMPT in error:
            # Non-retryable model rejection: wait stopped, no auto-recovery,
            # the fresh-session retry offer is kept → enable it.
            self._finish_task("任务已暂停，等待“新会话重试”")
            self.detail_label.setText(
                f"{MODEL_REJECTION_PROMPT} "
                "原任务信息已保留；可点击“新会话重试”在全新会话中重试。"
            )
            return
        self._finish_task("错误")
        self._show_error(translate_error(error))

    def _finish_task(self, status: str):
        self._busy = False
        # The B-side task reached a terminal state: release its session /
        # directory ownership so the monitor may resume full control.
        self._a_side_inflight_dirs.clear()
        self._a_side_inflight_sessions.clear()
        self._a_side_session_owner.clear()
        self._a_side_oc_pending = False
        self._current_a_task_id = None
        self._set_controls_enabled(True)
        self._set_status(status)
        if self._startup_pickup_pending:
            self._startup_pickup_pending = False
            QTimer.singleShot(0, self._startup_deliver_current_clipboard_task)
        # Flush a recovered reply that was held back while a task ran, so
        # side A gets the OLDER task's reply before the NEXT task's reply.
        if self._pending_recovered_reply is not None and not self._close_requested:
            recovered = self._pending_recovered_reply
            self._pending_recovered_reply = None
            try:
                self._listener.write_response(recovered)
                self._last_response = recovered
            except Exception as exc:
                LOGGER.error("延迟写入恢复的A端回复到剪贴板失败: %s", exc)
        # Invariant 1/5: EVERY terminal exit funnels through here, so the
        # FIFO queue advances from exactly one place (skipped while the
        # window is closing or startup recovery still holds the lane).
        QTimer.singleShot(0, self._start_next_queued_task)
        if not self._close_requested:
            queued = self._workflow.registry.queued_records()
            if queued:
                self._set_status(
                    f"{status}；等待任务 {len(queued)} 个，将按顺序继续执行。"
                )

    # ------------------------------------------------------------------ #
    # self check (informational only: must never block monitoring)
    # ------------------------------------------------------------------ #

    def _auto_start_after_self_check(self):
        # "auto" (not "manual") so a task already on the clipboard when the
        # app started is delivered exactly once instead of being lost.
        if not self._listener.enabled:
            self._ensure_clipboard_listener_started("auto")

    # ------------------------------------------------------------------ #
    # startup recovery: stale non-terminal records from a previous run
    # ------------------------------------------------------------------ #

    def _maybe_start_startup_recovery(self):
        """Reconcile RECEIVED/PROCESSING records a previous run left behind
        (invariant 5): each is safely re-checked against its OpenChamber
        session (bounded, read-only, never re-sent) and moved to a terminal
        state.  QUEUED tasks need no reconciliation -- they simply run from
        the queue once the lane is free."""
        if self._close_requested or self._startup_recovery_pending:
            return
        stale = [
            record
            for record in self._workflow.registry.stale_records()
            # Only records that predate this window's construction are true
            # leftovers of a previous run; a record this run already claimed
            # (RECEIVED via the auto-start pickup or a direct delivery) is an
            # in-flight task that must NEVER be reconciled.
            if record["task_id"] in self._startup_stale_task_ids
        ]
        if not stale:
            return
        self._startup_recovery_pending = True
        self._set_status("正在核对上次未结束的任务状态……")
        self.detail_label.setText(
            f"检测到 {len(stale)} 个上次未结束的任务，正在安全核对"
            "（不会重复发送任务）……"
        )
        LOGGER.info(
            "startup recovery: %d stale record(s) %s",
            len(stale), [r["task_id"] for r in stale],
        )
        task = StaleRecoveryTask(self._workflow, stale)
        task.signals.done.connect(self._on_recovery_done)
        task.signals.failed.connect(self._on_recovery_failed)
        self._track_general_worker(task)

    def _recovered_response(self, result: dict) -> str | None:
        """Wrap a verified recovered final reply under the ORIGINAL task id
        (invariant 4: never a manual-* id).  The protocol headers come from
        the persisted raw message when available, else the V1 defaults."""
        task_id = result["task_id"]
        record = self._workflow.registry.record(task_id) or {}
        raw = str(record.get("raw_message") or "")
        protocol_format = ProtocolFormat.V1
        round_number = 0
        max_rounds = 3
        if raw:
            try:
                message = parse_message(raw)
                protocol_format = message.protocol_format
                round_number = message.round_number
                max_rounds = message.max_rounds
            except RelayProtocolError:
                pass
        try:
            return wrap_response(
                str(result.get("final_text") or ""),
                task_id,
                protocol_format,
                round_number,
                max_rounds,
            )
        except Exception as exc:
            LOGGER.error("recovered reply wrap failed task=%s: %s", task_id, exc)
            return None

    def _on_recovery_done(self, results: list[dict]):
        if self._close_requested:
            self._startup_recovery_pending = False
            return
        self._startup_recovery_pending = False
        completed: list[str] = []
        failed: list[str] = []
        needs_recovery: list[str] = []
        for result in results:
            task_id = result["task_id"]
            state = result["state"]
            error = str(result.get("error") or "")
            if state == "COMPLETED":
                # Persist exactly like the auto relay would: wrapped reply
                # saved under the original task id, message ids marked so
                # the monitor can never wrap the same reply again.
                response = self._recovered_response(result)
                if response is None:
                    self._workflow.registry.mark(
                        task_id, "RECOVERY_REQUIRED",
                        "已核实最终回复，但回复包装失败，需人工检查",
                    )
                    needs_recovery.append(task_id)
                    continue
                try:
                    reply_file = self._workflow.save_reply(task_id, response)
                except Exception as exc:
                    LOGGER.error("recovered reply save failed task=%s: %s", task_id, exc)
                    reply_file = None
                for message_id in result.get("reply_message_ids") or ():
                    mark_message_wrapped(str(message_id), result.get("session_id") or "")
                extra = {}
                if reply_file is not None:
                    extra["reply_file"] = str(reply_file)
                self._workflow.registry.mark(
                    task_id, "COMPLETED", **extra
                )
                LOGGER.info(
                    "startup recovery: task %s reconciled COMPLETED", task_id
                )
                completed.append(task_id)
                # Clipboard hand-off only when this run produced no newer
                # reply: the recovered reply belongs to an OLDER task, so it
                # must never clobber a newer task's reply on the clipboard.
                # Otherwise it is held and flushed when the lane frees up.
                if not self._busy and self._last_response is None:
                    try:
                        self._listener.write_response(response)
                        self._last_response = response
                    except Exception as exc:
                        LOGGER.error(
                            "写入恢复的回复到剪贴板失败 task=%s: %s", task_id, exc
                        )
                else:
                    self._pending_recovered_reply = response
            elif state == "FAILED":
                self._workflow.registry.mark(task_id, "FAILED", error or None)
                LOGGER.info(
                    "startup recovery: task %s reconciled FAILED error=%s",
                    task_id, error,
                )
                failed.append(task_id)
            else:
                self._workflow.registry.mark(
                    task_id, "RECOVERY_REQUIRED", error or None
                )
                LOGGER.info(
                    "startup recovery: task %s reconciled RECOVERY_REQUIRED error=%s",
                    task_id, error,
                )
                needs_recovery.append(task_id)
        self._refresh_saved_tasks()
        self._update_queue_label()
        parts = []
        if completed:
            parts.append(f"{len(completed)} 个已确认完成并返回原任务ID回复")
        if failed:
            parts.append(f"{len(failed)} 个已确认失败")
        if needs_recovery:
            ids = "、".join(t[:12] + "…" for t in needs_recovery[:5])
            parts.append(f"{len(needs_recovery)} 个无法自动核实（{ids}），需人工检查")
        summary = "；".join(parts) if parts else "无遗留任务"
        self._set_status(f"遗留任务核对完成：{summary}")
        self.detail_label.setText(
            f"遗留任务核对完成：{summary}。"
            "完成的任务可通过“重新复制回复”取回回复。"
        )
        # The lane is free again: advance the FIFO queue (and any pickup
        # that was held for the reconciliation).
        if self._startup_pickup_pending:
            self._startup_pickup_pending = False
            QTimer.singleShot(0, self._startup_deliver_current_clipboard_task)
        QTimer.singleShot(0, self._start_next_queued_task)

    def _on_recovery_failed(self, error: str):
        if self._close_requested:
            self._startup_recovery_pending = False
            return
        self._startup_recovery_pending = False
        # Reconciliation itself broke: park EVERY stale record as
        # RECOVERY_REQUIRED so none is silently left non-terminal, and never
        # guess which ones completed.
        for record in self._workflow.registry.stale_records():
            self._workflow.registry.mark(
                record["task_id"],
                "RECOVERY_REQUIRED",
                f"启动核对失败：{translate_error(error)}",
            )
        LOGGER.error("startup recovery failed: %s", error)
        self._update_queue_label()
        self._set_status("遗留任务核对失败，已标记为需人工检查")
        self.detail_label.setText(
            f"遗留任务核对失败（{translate_error(error)}），相关任务已标记为"
            "需人工检查；队列任务将按顺序继续执行。"
        )
        QTimer.singleShot(0, self._start_next_queued_task)

    @Slot()
    def _self_check(self):
        if self._busy:
            return
        self._busy = True
        self._set_controls_enabled(False)
        self._set_status("正在检查 Reasonix")
        # Snapshot the detail area: the startup check is HOW-TO information
        # only, it must never clobber text a higher-priority UI owner (session
        # result / current task / monitor / error) wrote while the real-UIA
        # probe was running.
        self._detail_snapshot = self.detail_label.text()

        task = SelfCheckTask(self._reasonix)
        task.signals.succeeded.connect(self._self_check_succeeded)
        task.signals.failed.connect(self._task_failed)
        self._track_general_worker(task)

    @Slot(str)
    def _self_check_succeeded(self, _unused: str):
        if self._close_requested:
            LOGGER.info(
                "Reasonix 启动自检迟到完成，窗口已关闭，结果已忽略"
            )
            return
        authoritative = (
            self._startup_check_pending
            and not self._oc_monitor_active
            and not self._oc_monitor_interrupted
            and not self._oc_monitor_auto_pending
            and self.detail_label.text() == self._detail_snapshot
        )
        if authoritative:
            self.detail_label.setText("Reasonix 窗口、输入框和发送按钮均可通过 UIA 识别。")
        else:
            LOGGER.info(
                "Reasonix 启动自检通过；详情分区已被更高优先级内容占用，保留现有提示"
            )
        self._finish_task("Reasonix 连接自检通过")
        if self._startup_check_pending:
            self._startup_check_pending = False
            self._auto_start_after_self_check()

    # ------------------------------------------------------------------ #
    # session / reply actions
    # ------------------------------------------------------------------ #

    @Slot(object)
    def _on_session_started(self, outcome: TaskOutcome):
        self._current_outcome = outcome
        # Invariant 4/6: the session the A-side task now occupies belongs to
        # that task -- rotation may swap sessions but never ownership, so
        # both the old and the new session id stay registered in-flight.
        if outcome.session_id:
            self._a_side_inflight_sessions.add(outcome.session_id)
            if outcome.task_id:
                # Ownership map: the monitor's yield/continue decisions for
                # this session are logged with the owning A-side task id.
                self._a_side_session_owner[outcome.session_id] = outcome.task_id
            self._a_side_oc_pending = False
        if outcome.directory:
            self._a_side_inflight_dirs.add(directory_key(outcome.directory))
            self._a_side_oc_pending = False
        self.task_label.setText(
            f"当前任务：{outcome.task_id}｜OpenChamber 会话 {outcome.session_id}｜"
            f"项目 {outcome.directory}"
        )
        # The open-session button targets THIS task from the moment its
        # session exists and is persisted, even while it is still running.
        self.open_session_button.setEnabled(True)

    @Slot()
    def _open_current_session(self):
        outcome = self._current_outcome
        if outcome is None or not outcome.session_id:
            self._show_error("当前没有可打开的 OpenChamber 会话")
            return
        try:
            OpenChamberClient(
                self._settings.openchamber_url,
                auth_token=self._settings.openchamber_auth_token or None,
            ).open_session(
                outcome.session_id
            )
        except Exception as exc:
            self._show_error(f"打开会话失败：{translate_error(str(exc))}")
            return
        self.detail_label.setText(
            "已请求 OpenChamber 打开会话"
            f"（{outcome.session_id}）；请在桌面窗口确认显示。"
        )
        self._set_status("已请求打开 OpenChamber 会话")

    @Slot(int)
    def _saved_task_selected(self, _index: int):
        if self._busy:
            return
        self._set_controls_enabled(True)
        task_id = self._saved_task_combo.currentData()
        if task_id:
            record = self._workflow.registry.record(task_id)
            if record is not None:
                self.detail_label.setText(
                    f"已保存任务 {task_id[:12]}…；"
                    f"{_model_info_text(record)}"
                )

    @Slot()
    def _recopy_reply(self):
        selected = self._saved_task_combo.currentData()
        if selected:
            # an explicit selection is authoritative: read THAT task's
            # reply, never another task's and never the last response.
            response = self._workflow.load_reply(selected)
            if response is None:
                self._show_error(
                    f"所选任务 {selected[:12]}… 的回复文件缺失或读取失败，"
                    "无法复制；不会回退为其他任务的回复"
                )
                return
            copied = f"（已保存任务 {selected[:12]}…）"
        else:
            # no selection: only the clearly identified last success reply
            response = self._last_response
            if response is None and self._last_outcome is not None:
                response = self._workflow.load_reply(self._last_outcome.task_id)
            if response is None:
                self._show_error("没有可重新复制的回复")
                return
            copied = "（上次成功任务）"
        self._listener.write_response(response)
        self.detail_label.setText(
            f"已重新复制回复到剪贴板{copied}（未重复执行任务）。"
        )
        self._set_status("已重新复制回复")

    @Slot()
    def _wrap_clipboard_content(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text() or ""
        if not text.strip():
            self._show_error("剪切板没有可包装内容")
            return
        if ClipboardListener.looks_like_relay_message(text):
            self._show_error("剪贴板已是完整 AI Relay 协议内容，拒绝重复包装")
            return

        # Binding target: the UNIQUE in-flight A-side task (RECEIVED /
        # PROCESSING), if any.  The operator is declaring this clipboard text
        # to be THAT task's final reply, so it is bound to the task's original
        # TASK_ID and ends the round.  A QUEUED task is never bound (it is not
        # the current round), and with no in-flight task a standalone
        # manual-{uuid} response is produced.
        current_id = self._current_a_task_id
        record = (
            self._workflow.registry.record(current_id) if current_id else None
        )
        if current_id and record is not None and record.get("state") in (
            "RECEIVED",
            "PROCESSING",
        ):
            self._wrap_current_a_task(current_id, record, text)
        else:
            self._wrap_standalone_manual(text)

    def _wrap_standalone_manual(self, text: str) -> None:
        """No in-flight A-side task: wrap the clipboard text as a standalone
        manual RESPONSE (manual-{uuid}, ROUND 0, MAX_ROUNDS 1) and copy it
        back.  The auto relay never sees it."""
        task_id = f"manual-{uuid4()}"
        try:
            response = wrap_response(
                text,
                task_id,
                ProtocolFormat.V1,
                round_number=0,
                max_rounds=1,
            )
        except Exception as exc:
            self._show_error(f"包装剪贴板内容失败：{translate_error(str(exc))}")
            return
        try:
            self._workflow.save_reply(task_id, response)
            self._workflow.registry.mark(
                task_id,
                "COMPLETED",
                executor=TARGET_OPENCHAMBER,
                reply_file=str(self._workflow.reply_file_for(task_id)),
            )
        except Exception as exc:
            LOGGER.warning("manual wrap persist failed task=%s: %s", task_id, exc)
        self._listener.write_response(response)
        self.detail_label.setText(
            f"已将剪贴板内容包装为手动 RESPONSE（任务 {task_id[:12]}…）并复制回剪贴板。"
        )
        self._set_status("已包装剪贴板内容")

    def _wrap_current_a_task(self, task_id: str, record: dict, text: str) -> None:
        """Bind the clipboard text to the running A-side task as ITS final
        reply: invalidate the worker's signals, wrap under the ORIGINAL
        TASK_ID, mark the task COMPLETED (``completion_source=manual_wrap``)
        *before* cancelling the worker, hand it to side A and advance the FIFO
        queue via the unified exit.  A late worker can never clobber this
        completion (generation guard + ``mark_if_state``)."""
        # 1) invalidate this worker's LATE SIGNALS first: bump the generation
        #    so any success/failure/session signal it emits afterwards is
        #    dropped (no clipboard re-write, no re-finalise of the round).
        self._current_a_task_generation += 1
        # 2) wrap under the ORIGINAL task id, reusing the task's protocol
        #    header (round / max_rounds) when the raw message is known.
        protocol_format = ProtocolFormat.V1
        round_number = 0
        max_rounds = 1
        raw = str(record.get("raw_message") or "")
        if raw:
            try:
                header = parse_message(raw)
                protocol_format = header.protocol_format
                round_number = header.round_number
                max_rounds = header.max_rounds
            except RelayProtocolError:
                pass
        try:
            response = wrap_response(
                text,
                task_id,
                protocol_format,
                round_number=round_number,
                max_rounds=max_rounds,
            )
        except Exception as exc:
            self._show_error(f"包装剪贴板内容失败：{translate_error(str(exc))}")
            return
        # 3) persist the reply and mark the task COMPLETED BEFORE cancelling
        #    the worker: this is the authoritative resolution of the round.
        #    The worker's later terminal write is guarded by ``mark_if_state``
        #    (task no longer RECEIVED/PROCESSING) so it cannot flip the task
        #    back to STOPPED/FAILED -- even if it unwinds on another thread
        #    before the cancel is observed.
        try:
            reply_file = self._workflow.save_reply(task_id, response)
            self._workflow.registry.mark(
                task_id,
                "COMPLETED",
                completion_source="manual_wrap",
                reply_file=str(reply_file),
            )
        except Exception as exc:
            LOGGER.error("manual wrap persist failed task=%s: %s", task_id, exc)
        # 4) only now cancel the worker; its terminal update is a no-op.
        if self._task_cancel_event is not None:
            self._task_cancel_event.set()
        # 5) hand the final reply to side A and record it for re-copy.
        self._listener.write_response(response)
        self._last_response = response
        self._refresh_saved_tasks(select_newest=True)
        self.detail_label.setText(
            f"已将剪贴板内容作为任务 {task_id[:12]}… 的最终回复并结束本轮。"
        )
        self._set_status("已人工包装当前回复")
        # 6) unified exit: clears _busy + session ownership and starts the
        #    next QUEUED task (or waits for A-side's next TASK).  The
        #    remaining queue is untouched.
        self._finish_task("已人工包装当前回复")

    def _refresh_saved_tasks(self, select_newest: bool = False):
        self._saved_task_combo.blockSignals(True)
        self._saved_task_combo.clear()
        self._saved_task_combo.addItem("— 选择已保存任务 —", None)
        records = self._workflow.registry.completed_records()
        for record in records:
            self._saved_task_combo.addItem(
                f"{record['task_id'][:12]}（{record.get('executor', '?')}）",
                record["task_id"],
            )
        if select_newest and records:
            self._saved_task_combo.setCurrentIndex(len(records))
        else:
            self._saved_task_combo.setCurrentIndex(0)
        self._saved_task_combo.blockSignals(False)

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #

    @staticmethod
    def _directory_is_valid(directory: str) -> bool:
        """The configured OpenChamber project directory must actually exist
        and be a folder (not a file)."""
        return bool(directory) and os.path.isdir(directory)

    def _clear_session_selection(self, directory: str):
        """Discard any previously chosen fixed session.  Called when the
        project directory changes so a session saved for another project is
        not reused across projects."""
        self._settings.openchamber_session_id = ""
        key = directory_key(directory)
        self._settings.openchamber_sessions.pop(key, None)

    @Slot()
    def _browse_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 OpenChamber 项目目录",
            self._directory_edit.text().strip(),
        )
        if not directory:
            return
        if not self._directory_is_valid(directory):
            self._show_error(
                f"所选目录不存在或不是文件夹，未采用：{directory}"
            )
            return
        old = self._directory_edit.text().strip()
        self._directory_edit.setText(directory)
        if os.path.normcase(os.path.abspath(old)) != os.path.normcase(
            os.path.abspath(directory)
        ):
            self._clear_session_selection(directory)
            self._rotation.reset(directory)
        if not self._save_settings():
            return
        self._refresh_meta()

    @Slot()
    def _refresh_meta(self):
        directory = self._directory_edit.text().strip()
        if not directory:
            self._show_error("请先填写项目目录，再刷新 Agent/Model")
            return
        if not self._directory_is_valid(directory):
            self._show_error(
                f"项目目录不存在或不是文件夹，无法刷新 Agent/Model：{directory}"
            )
            return
        directory = normalize_directory(directory)
        url = self._url_edit.text().strip() or DEFAULT_OPENCHAMBER_URL
        self._set_controls_enabled(False)
        self._set_status("正在刷新 Agent/Model")
        task = RefreshMetaTask(
            url,
            directory,
            auth_token=self._settings.openchamber_auth_token or None,
        )
        task.signals.sessions.connect(self._meta_loaded)
        task.signals.failed.connect(self._meta_failed)
        self._track_general_worker(task)

    @Slot(object)
    def _meta_loaded(self, payload):
        agents, models = payload
        current_agent = self._agent_combo.currentText().strip()
        current_model = self._model_combo.currentText().strip()
        merged_agents = list(dict.fromkeys(["build", "plan", *sorted(agents)]))
        merged_models = list(
            dict.fromkeys(["4090/qwen3.8-27b", "opencode/big-pickle", *sorted(models)])
        )
        self._agent_combo.blockSignals(True)
        self._agent_combo.clear()
        self._agent_combo.addItems(merged_agents)
        if current_agent:
            self._agent_combo.setCurrentText(current_agent)
        self._agent_combo.blockSignals(False)
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(merged_models)
        if current_model:
            self._model_combo.setCurrentText(current_model)
        self._model_combo.blockSignals(False)
        self._set_controls_enabled(not self._busy)
        self.detail_label.setText(
            f"Agent/Model 已刷新：Agent {merged_agents}；Model {merged_models}"
        )
        self._set_status("Agent/Model 已刷新")

    @Slot(str)
    def _meta_failed(self, error: str):
        self._set_controls_enabled(not self._busy)
        self._show_error(f"刷新 Agent/Model 失败：{translate_error(error)}")

    @Slot()
    def _save_settings(self):
        model = self._model_combo.currentText().strip()
        known_models = {
            self._model_combo.itemText(i)
            for i in range(self._model_combo.count())
        } - {""}
        if not model:
            self._show_error(
                "模型不能为空：请在 Agent/Model 行选择有效模型后再保存"
            )
            return False
        if model not in known_models:
            self._show_error(
                f"模型 {model!r} 不在可用模型列表中（{sorted(known_models)}）；"
                "请使用下拉框重新选择后保存"
            )
            return False
        try:
            self._settings.default_target = (
                self._executor_combo.currentData() or TARGET_REASONIX
            )
            self._settings.openchamber_url = self._url_edit.text().strip()
            self._settings.openchamber_directory = self._directory_edit.text().strip()
            self._settings.openchamber_agent = self._agent_combo.currentText().strip()
            self._settings.openchamber_model = model
            self._settings.auto_compact_after_response = (
                self._auto_compact_check.isChecked()
            )
            self._settings.validate()
            self._settings.save()
        except Exception as exc:
            self._show_error(f"保存设置失败：{translate_error(str(exc))}")
            return False
        self.detail_label.setText("设置已保存。")
        self._set_status("设置已保存")
        return True

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _set_controls_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled and not self._listener.enabled)
        self.pause_button.setEnabled(enabled and self._listener.enabled)
        self.check_button.setEnabled(enabled)
        self._save_settings_button.setEnabled(enabled)
        self._browse_directory_button.setEnabled(enabled)
        self._refresh_meta_button.setEnabled(enabled)
        self._auto_compact_check.setEnabled(enabled)
        # "停止当前任务" stays clickable WHILE a task runs (never grayed out
        # by busy); it only stops the wait, never the OpenChamber session.
        # It is ALSO offered while an interrupted auto-relay task awaits a
        # manual "继续当前任务" AND while the manual monitor is active -- in
        # both cases the operator can stop pursuing the task instead.
        interrupted_idle = (
            self._workflow.pending_continue is not None
            and self._workflow.pending_rejection is None
        )
        # A round whose automatic recovery is exhausted (relay one-shot used
        # up, or the monitor's 3 auto continuations) must offer manual
        # continue/stop even WHILE a B-side task is still busy waiting: the
        # operator is the only way out of that state, so the two buttons
        # must never be grayed out by ``_busy``.
        manual_recovery_offered = (
            interrupted_idle
            or (
                self._oc_monitor_active
                and self._oc_monitor_interrupted
                and not self._oc_monitor_continuing
                and not self._oc_monitor_auto_pending
                and not self._oc_monitor_transport_down
            )
        )
        self.stop_button.setEnabled(
            (self._busy and self._task_cancel_event is not None)
            or (enabled and interrupted_idle)
            or (self._busy and manual_recovery_offered)
            or (
                (enabled or self._oc_monitor_continuing)
                and self._oc_monitor_active
                and not self._busy
            )
        )
        # "继续当前任务" is offered only for an interrupted-incomplete task.
        # The AUTO-RELAY offer (pending continue) takes priority; when none
        # exists, the MANUAL MONITOR can claim the same button for its own
        # continuation after an abnormal idle -- but never while a
        # continuation is being sent and never while the transport is down
        # (rechecking gives up).  Both offers stay clickable while a B-side
        # task is busy (the monitor's manual continue is then the action the
        # operator takes; the B-side worker consumes the resulting reply).
        monitor_continue_offered = (
            self._oc_monitor_active
            and self._oc_monitor_interrupted
            and not self._oc_monitor_continuing
            and not self._oc_monitor_auto_pending
            and not self._oc_monitor_transport_down
        )
        self.continue_button.setEnabled(
            enabled
            and (
                self._workflow.pending_continue is not None
                or monitor_continue_offered
            )
            or (self._busy and manual_recovery_offered)
        )
        # "新会话重试" is offered only for a non-retryable model rejection.
        # When such a rejection is pending, "继续当前任务" is naturally off:
        # a rejected session must never be sent a continuation.
        self.new_session_button.setEnabled(
            enabled and self._workflow.pending_rejection is not None
        )
        self.open_session_button.setEnabled(False)
        self.recopy_button.setEnabled(False)
        if enabled:
            self.open_session_button.setEnabled(
                self._current_outcome is not None
                and self._current_outcome.session_id is not None
            )
            self.recopy_button.setEnabled(
                self._last_response is not None
                or self._saved_task_combo.count() > 1
                or self._last_outcome is not None
            )

    def _update_runtime_status(self):
        if self._listener.enabled:
            self._clipboard_status_label.setText("任务接收：运行中")
            self._clipboard_status_label.setStyleSheet("color: #4ade80;")
        elif self._clipboard_paused:
            self._clipboard_status_label.setText("任务接收：已暂停")
            self._clipboard_status_label.setStyleSheet("color: #fbbf24;")
        else:
            self._clipboard_status_label.setText("任务接收：未启动")
            self._clipboard_status_label.setStyleSheet("color: #9ca3af;")
        if self._oc_monitor_active:
            self._monitor_status_label.setText("OpenChamber监控：运行中")
            self._monitor_status_label.setStyleSheet("color: #4ade80;")
        else:
            self._monitor_status_label.setText("OpenChamber监控：未启动")
            self._monitor_status_label.setStyleSheet("color: #9ca3af;")

    def _set_status(self, status: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(f"服务状态：[{now}] {status}")

    def _show_error(self, error: str):
        self.detail_label.setText(error)
        self._set_status("错误")
