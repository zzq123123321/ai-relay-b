"""Small persistent task registry used to prevent repeated execution."""

from __future__ import annotations

import json
from pathlib import Path

from core.runtime_paths import data_dir


class TaskRegistryError(RuntimeError):
    pass


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
        }
    )

    def __init__(self, path: Path | None = None):
        self.path = path or (data_dir() / "tasks.json")
        self._states = self._load()

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

    def mark(
        self,
        task_id: str,
        state: str,
        error: str | None = None,
        **extra: str,
    ) -> None:
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
        try:
            temporary.write_text(
                json.dumps(self._states, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError as exc:
            raise TaskRegistryError(f"failed to save task registry: {self.path}") from exc
