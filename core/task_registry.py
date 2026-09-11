"""Small persistent task registry used to prevent repeated execution."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from core.runtime_paths import data_dir


class TaskRegistryError(RuntimeError):
    pass


# States a task may NOT be moved out of by an ordinary (possibly late /
# superseded) worker write: once a task reached any of them, only an
# explicit active re-entry (``mark_reentry``: manual continue, fresh-session
# retry, operator stop of an interrupted task, startup reconciliation) may
# move it on.  Ordinary ``mark_if_active`` writes are a no-op afterwards, so
# a late worker can never flip COMPLETED / STOPPED_BY_USER / FAILED back to
# PROCESSING or another terminal state.
TERMINAL_STATES = frozenset({"COMPLETED", "STOPPED_BY_USER", "FAILED"})


class TaskRegistry:
    EXTRA_FIELDS = frozenset(
        {
            "executor",
            "session_id",
            "directory",
            "reply_file",
            "requested_model",
            "resolved_model",
            "actual_model",
            "model_note",
            # FIFO queue bookkeeping (invariant 1/2/5): the full AI_RELAY
            # protocol text is persisted so a QUEUED task can be re-executed
            # after a restart without re-copying the clipboard; ``sequence``
            # is the monotonic claim order (stored as a string) and
            # ``received_at`` is the ISO receive time.
            "raw_message",
            "sequence",
            "received_at",
            # How a terminal task was resolved: "auto_relay" (the normal path)
            # or "manual_wrap" (the operator pressed 包装内容 to end the round).
            # Preserved across later state writes so a manual completion is
            # auditable and never silently rewritten by a late worker.
            "completion_source",
        }
    )

    def __init__(self, path: Path | None = None):
        self.path = path or (data_dir() / "tasks.json")
        self._states = self._load()
        # Serialises every read-modify-write of the in-memory state so a
        # worker thread and the UI thread can never observe a half-applied
        # update (see ``mark_if_state``: the state check and the write must be
        # one atomic step or a late worker can clobber a manual completion).
        self._lock = threading.Lock()

    def contains(self, task_id: str) -> bool:
        return task_id in self._states

    def record(self, task_id: str) -> dict[str, str] | None:
        return self._states.get(task_id)

    def completed_records(self) -> list[dict[str, str]]:
        """Saved COMPLETED tasks in completion order (newest last), each
        record carrying its ``task_id``, for repeat copy after a restart."""
        return [
            {**record, "task_id": task_id}
            for task_id, record in self._states.items()
            if record.get("state") == "COMPLETED"
        ]

    def queued_records(self) -> list[dict[str, str]]:
        """FIFO queue: every QUEUED task, ordered by claim ``sequence``
        (falling back to receive time), each carrying its ``task_id`` and the
        persisted ``raw_message`` so it can be re-executed after a restart."""
        items = [
            (task_id, record)
            for task_id, record in self._states.items()
            if record.get("state") == "QUEUED"
        ]

        def _seq(record: dict[str, str]) -> int:
            try:
                return int(record.get("sequence", "0") or 0)
            except ValueError:
                return 0

        items.sort(key=lambda kv: (_seq(kv[1]), kv[1].get("received_at", "")))
        return [{**record, "task_id": task_id} for task_id, record in items]

    def stale_records(self) -> list[dict[str, str]]:
        """Non-terminal records (RECEIVED/PROCESSING) left behind by a
        previous run: candidates for startup reconciliation.  QUEUED tasks
        are NOT stale -- they are simply executed from the queue."""
        return [
            {**record, "task_id": task_id}
            for task_id, record in self._states.items()
            if record.get("state") in ("RECEIVED", "PROCESSING")
        ]

    def next_sequence(self) -> int:
        """Monotonic claim order for the next accepted task (max + 1).

        Computed from the persisted records so the order survives a restart
        without a separate counter file; all claims happen on the Qt main
        thread and persist synchronously, so values never collide."""
        highest = 0
        for record in self._states.values():
            try:
                highest = max(highest, int(record.get("sequence", "0") or 0))
            except ValueError:
                continue
        return highest + 1

    def mark(
        self,
        task_id: str,
        state: str,
        error: str | None = None,
        **extra: str,
    ) -> None:
        with self._lock:
            self._apply(task_id, state, error, extra)

    def mark_if_state(
        self,
        task_id: str,
        expected_states,
        state: str,
        error: str | None = None,
        **extra: str,
    ) -> bool:
        """Move ``task_id`` to ``state`` ONLY if it is still in one of
        ``expected_states`` (e.g. ``("RECEIVED", "PROCESSING")``).

        This is the guard that keeps a late worker from clobbering a task
        that the operator already resolved: once the task is COMPLETED (via a
        manual wrap or a normal success) a worker's delayed FAILED /
        STOPPED_BY_USER / COMPLETED write is a no-op.  The state check and the
        write happen under the same lock, so the pair is atomic.  Returns
        ``True`` when the update was applied, ``False`` when the task had
        already moved to a different state."""
        expected = tuple(expected_states)
        with self._lock:
            current = self._states.get(task_id, {}).get("state")
            if current not in expected:
                return False
            self._apply(task_id, state, error, extra)
            return True

    def mark_if_active(
        self,
        task_id: str,
        state: str,
        error: str | None = None,
        **extra: str,
    ) -> bool:
        """Move ``task_id`` to ``state`` UNLESS it already reached ANY
        terminal state (COMPLETED / STOPPED_BY_USER / FAILED).

        This is the guard a long-lived worker uses for EVERY state write
        (PROCESSING as well as terminal COMPLETED/FAILED/STOPPED_BY_USER):
        once the operator has resolved the round (or the task already ended)
        a superseded worker's delayed write is a no-op, so it can never flip
        a finished task back to PROCESSING or overwrite another terminal
        state.  Attention states a re-entry flow starts from are moved
        onward ONLY through :meth:`mark_reentry`.  The check and write are
        atomic under the lock.
        Returns ``True`` when applied, ``False`` when the task was already
        terminal.
        """
        with self._lock:
            current = self._states.get(task_id, {}).get("state")
            if current in TERMINAL_STATES:
                return False
            self._apply(task_id, state, error, extra)
            return True

    def mark_reentry(
        self,
        task_id: str,
        state: str,
        error: str | None = None,
        **extra: str,
    ) -> None:
        """Explicit ACTIVE re-entry write, allowed even from a terminal state.

        Only deliberate operator / relay flows that knowingly re-open a
        resolved task may use it: manual "继续当前任务", "新会话重试" and
        the operator's stop of an interrupted task.  Ordinary late workers
        must keep using :meth:`mark_if_active` so they can never modify a
        terminal state.
        """
        with self._lock:
            self._apply(task_id, state, error, extra)

    def _apply(
        self,
        task_id: str,
        state: str,
        error: str | None,
        extra: dict[str, str],
    ) -> None:
        """In-memory state update + save.  Callers hold ``self._lock``."""
        existing = self._states.get(task_id, {})
        record: dict[str, str] = {
            field: value
            for field, value in existing.items()
            if field in self.EXTRA_FIELDS
        }
        record["state"] = state
        if error is not None:
            record["error"] = error
        for field, value in extra.items():
            if field in self.EXTRA_FIELDS and isinstance(value, str) and value:
                record[field] = value
        self._states[task_id] = record
        self._save()

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskRegistryError(f"failed to load task registry: {self.path}") from exc

        if not isinstance(data, dict):
            raise TaskRegistryError("task registry has an invalid structure")

        states: dict[str, dict[str, str]] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise TaskRegistryError("task registry has an invalid task id")
            if isinstance(value, str):
                states[key] = {"state": value}
            elif isinstance(value, dict) and isinstance(value.get("state"), str):
                states[key] = {
                    field: item
                    for field, item in value.items()
                    if field in {"state", "error", *self.EXTRA_FIELDS}
                    and isinstance(item, str)
                }
            else:
                raise TaskRegistryError("task registry has an invalid task record")
        return states

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        # Write to a temp file then atomically replace, so a crash mid-write
        # never leaves a truncated tasks.json behind.  Windows can transiently
        # lock a freshly written file (AV / search indexer), which makes
        # ``replace`` raise EPERM/EBUSY for a moment: retry briefly before
        # giving up so a real save is never lost to a scanner blip.
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                temporary.write_text(
                    json.dumps(self._states, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(self.path)
                return
            except OSError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
        raise TaskRegistryError(
            f"failed to save task registry: {self.path}"
        ) from last_error
