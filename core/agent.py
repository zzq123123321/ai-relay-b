"""Agent configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class AgentConfigurationError(ValueError):
    """Raised when an Agent configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Agent:
    """Configuration required to identify and operate an Agent window."""

    agent_id: str
    display_name: str
    window_title: str
    template_dir: Path
    enabled: bool = True

    def __post_init__(self) -> None:
        for field_name in ("agent_id", "display_name", "window_title"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AgentConfigurationError(f"{field_name} must be a non-empty string")
        if not isinstance(self.template_dir, Path):
            raise AgentConfigurationError("template_dir must be a pathlib.Path")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> Agent:
        """Build an Agent from a configuration mapping."""
        required = ("agent_id", "display_name", "window_title", "template_dir")
        missing = [name for name in required if not config.get(name)]
        if missing:
            raise AgentConfigurationError("missing required Agent fields: " + ", ".join(missing))
        return cls(
            agent_id=config["agent_id"],
            display_name=config["display_name"],
            window_title=config["window_title"],
            template_dir=Path(config["template_dir"]),
            enabled=config.get("enabled", True),
        )
