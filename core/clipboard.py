"""Qt clipboard listener with self-write and duplicate protection."""

from __future__ import annotations

import hashlib
from hashlib import sha256

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QClipboard

from core.protocol import LEGACY_BEGIN, PROTOCOL_MARKER


class ClipboardListener(QObject):
    task_received = Signal(str)

    def __init__(self, clipboard: QClipboard):
        super().__init__()
        self._clipboard = clipboard
        self._enabled = False
        self._last_seen_digest: str | None = None
        self._self_write_digest: str | None = None
        clipboard.dataChanged.connect(self._on_changed)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        self._last_seen_digest = self._digest(self._clipboard.text() or "")
        self._enabled = True

    def pause(self) -> None:
        self._enabled = False

    def write_response(self, text: str) -> None:
        digest = self._digest(text)
        self._self_write_digest = digest
        self._last_seen_digest = digest
        self._clipboard.setText(text)

    def _on_changed(self) -> None:
        if not self._enabled:
            return
        text = self._clipboard.text() or ""
        digest = self._digest(text)
        if digest == self._self_write_digest:
            return
        if digest == self._last_seen_digest:
            return
        self._last_seen_digest = digest
        self.task_received.emit(text)

    @staticmethod
    def looks_like_relay_message(text: str) -> bool:
        normalized = text.replace("\r\n", "\n").lstrip()
        return normalized.startswith(PROTOCOL_MARKER) or normalized.startswith(LEGACY_BEGIN)

    @staticmethod
    def _digest(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
