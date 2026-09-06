"""Persistent relay settings (executor defaults and OpenChamber parameters)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    openchamber_directory: str = ""
    openchamber_agent: str = ""
    openchamber_model: str = ""
    completion_timeout: float = 900.0
    poll_interval: float = 2.0

    def validate(self) -> None:
        if self.default_target not in KNOWN_TARGETS:
            raise RelaySettingsError(
                f"default_target must be one of {sorted(KNOWN_TARGETS)}"
            )
        if not self.openchamber_url.strip():
            raise RelaySettingsError("openchamber_url must not be empty")
        if self.completion_timeout <= 0 or self.poll_interval <= 0:
            raise RelaySettingsError("timeouts must be positive")

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
            if isinstance(current, float) and not isinstance(current, bool):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    setattr(settings, key, float(value))
            elif isinstance(current, str):
                if isinstance(value, str):
                    setattr(settings, key, value)
        settings.validate()
        return settings

    def save(self, path: Path | None = None) -> None:
        path = path or (data_dir() / "relay_settings.json")
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(asdict(self), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            raise RelaySettingsError(
                f"failed to save relay settings: {path}"
            ) from exc

    def openchamber_model_ref(self):
        from core.openchamber import ModelRef

        return ModelRef.parse(self.openchamber_model or None)