"""Persistent relay settings (executor defaults and OpenChamber parameters)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.runtime_paths import data_dir

DEFAULT_OPENCHAMBER_URL = "http://127.0.0.1:57123"
TARGET_REASONIX = "REASONIX"
TARGET_OPENCHAMBER = "OPENCHAMBER"
TARGET_EXECUTOR = "EXECUTOR"
KNOWN_TARGETS = frozenset({TARGET_REASONIX, TARGET_OPENCHAMBER, TARGET_EXECUTOR})


class RelaySettingsError(RuntimeError):
    pass


@dataclass(slots=True)
class RelaySettings:
    default_target: str = TARGET_REASONIX
    openchamber_url: str = DEFAULT_OPENCHAMBER_URL
    # Optional OpenChamber bearer token attached to every API request.
    # Empty means the client stays unauthenticated (the pre-1.23 default).
    # Tokens are operator-supplied secrets; never log them, never write them
    # into the task registry or any report.
    openchamber_auth_token: str = ""
    openchamber_directory: str = ""
    openchamber_session_id: str = ""
    openchamber_sessions: dict[str, str] = field(default_factory=dict)
    openchamber_agent: str = ""
    openchamber_model: str = ""
    completion_timeout: float = 0.0
    poll_interval: float = 2.0
    auto_rotate_enabled: bool = False
    auto_rotate_threshold: int = 5
    auto_rotate_inherit_auto_accept: bool = False
    # Send a bare `/compact` control command to the OpenChamber session that
    # just produced a reply, and wait for the compaction to finish before the
    # next queued task starts.  The reply itself is never affected by a
    # compact failure.  Disable while debugging/measuring.
    auto_compact_after_response: bool = True
    # Per normalized project directory: the last committed auto-rotation
    # sequence (the directory has already been rotated up to "第N个"; the
    # next rotation creates "第N+1个").  Persisted so the sequence survives
    # restarts, is never re-derived from the session list, and is never
    # reused after sessions are deleted or manually switched away.
    auto_rotate_sequences: dict[str, int] = field(default_factory=dict)
    # Per normalized project directory: the persisted rotation base title
    # (e.g. "Task 14H"); once saved it is reused for every later rotation
    # so a suffix-carrying current session is never re-derived.
    auto_rotate_base_titles: dict[str, str] = field(default_factory=dict)
    _path: Path | None = field(default=None, init=False, repr=False)

    def validate(self) -> None:
        if self.default_target not in KNOWN_TARGETS:
            raise RelaySettingsError(
                f"default_target must be one of {sorted(KNOWN_TARGETS)}"
            )
        if not self.openchamber_url.strip():
            raise RelaySettingsError("openchamber_url must not be empty")
        if self.completion_timeout < 0 or self.poll_interval <= 0:
            raise RelaySettingsError(
                "completion_timeout must be >= 0 (0 = no timeout) "
                "and poll_interval must be positive"
            )
        if not 1 <= self.auto_rotate_threshold <= 100:
            raise RelaySettingsError(
                "auto_rotate_threshold must be between 1 and 100"
            )

    @classmethod
    def load(cls, path: Path | None = None) -> "RelaySettings":
        path = path or (data_dir() / "relay_settings.json")
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RelaySettingsError(
                f"failed to load relay settings: {path}"
            ) from exc
        if not isinstance(data, dict):
            raise RelaySettingsError("relay settings has an invalid structure")
        settings = cls()
        known = {field for field in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        for key, value in data.items():
            if key not in known:
                continue
            current = getattr(settings, key)
            if isinstance(current, bool):
                if isinstance(value, bool):
                    setattr(settings, key, value)
            elif isinstance(current, int):
                if isinstance(value, int) and not isinstance(value, bool):
                    setattr(settings, key, int(value))
            elif isinstance(current, float) and not isinstance(current, bool):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    setattr(settings, key, float(value))
            elif isinstance(current, str):
                if isinstance(value, str):
                    setattr(settings, key, value)
            elif isinstance(current, dict) and isinstance(value, dict):
                coerced = {}
                for item_key, item_value in value.items():
                    if not isinstance(item_key, str):
                        continue
                    if isinstance(item_value, bool) or not isinstance(
                        item_value, (str, int)
                    ):
                        continue
                    if key == "openchamber_sessions" or isinstance(
                        item_value, str
                    ):
                        coerced[item_key] = str(item_value)
                    else:
                        # int-valued dicts (auto_rotate_sequences) keep ints
                        coerced[item_key] = int(item_value)
                setattr(settings, key, coerced)
        settings.validate()
        settings._path = path
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or self._path or (data_dir() / "relay_settings.json")
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        key: value
                        for key, value in asdict(self).items()
                        if not key.startswith("_")
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
            self._path = path
        except OSError as exc:
            raise RelaySettingsError(
                f"failed to save relay settings: {path}"
            ) from exc

    def openchamber_model_ref(self):
        from core.openchamber import ModelRef

        return ModelRef.parse(self.openchamber_model or None)
