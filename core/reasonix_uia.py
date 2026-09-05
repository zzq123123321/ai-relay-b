"""Reasonix Desktop background automation through Windows UI Automation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable


class ReasonixAutomationError(RuntimeError):
    """Raised when a Reasonix UIA operation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class ReasonixConfig:
    window_name: str = "Reasonix"
    window_class: str = "wailsWindow"
    composer_id: str = "composer-input"
    send_class: str = "composer__btn composer__btn--send"
    stop_class: str = "composer__btn composer__btn--stop"
    reply_copy_class: str = "copybtn"
    poll_interval: float = 0.5
    lookup_timeout: float = 8.0
    generation_start_timeout: float = 20.0
    response_timeout: float = 600.0
    stable_seconds: float = 2.5
    send_attempts: int = 3


class ReasonixAutomation:
    def __init__(self, config: ReasonixConfig | None = None):
        self.config = config or ReasonixConfig()

    def self_check(self) -> dict[str, bool]:
        with self._automation() as auto:
            window = self._window(auto)
            composer = self._find_composer(window)
            send = self._find_send(window, required=False)
            value_pattern = composer.GetValuePattern()
            return {
                "window": window.Exists(0, 0),
                "composer": composer.Exists(0, 0),
                "composer_writable": bool(
                    value_pattern is not None and not value_pattern.IsReadOnly
                ),
                "send_button": send is not None,
            }

    def execute(self, task: str) -> str:
        if not task.strip():
            raise ReasonixAutomationError("Reasonix task must not be empty")
        with self._automation() as auto:
            window = self._window(auto)
            baseline = self._reply_count(window)
            self._set_composer_text(window, task)
            self._invoke_send(auto, baseline)
            self._wait_for_generation_start(auto, baseline)
            return self._wait_for_response(auto, baseline)

    def _set_composer_text(self, window, task: str) -> None:
        try:
            from comtypes import COMError
        except ImportError:
            raise ReasonixAutomationError("comtypes is required by uiautomation")

        normalized = task.replace("\r\n", "\n")

        def operation():
            composer = self._find_composer(window)
            value_pattern = composer.GetValuePattern()
            if value_pattern is None:
                raise COMError(
                    0x80070057,
                    "Reasonix composer value pattern unavailable",
                    (None, None, None, 0, None),
                )
            value_pattern.SetValue(task, waitTime=0)
            current = (value_pattern.Value or "").replace("\r\n", "\n")
            if current != normalized:
                raise COMError(
                    0x80070057,
                    "Reasonix composer value was not applied",
                    (None, None, None, 0, None),
                )
            return None

        self._retry_webview_read(operation, "composer input")

    def _invoke_send(self, auto, baseline: int) -> None:
        try:
            from comtypes import COMError
        except ImportError:
            raise ReasonixAutomationError("comtypes is required by uiautomation")
        last_error = None
        for _attempt in range(self.config.send_attempts):
            window = self._window(auto)
            send = self._find_send(window)
            if not send.IsEnabled:
                raise ReasonixAutomationError("Reasonix send button is disabled")
            try:
                send.GetInvokePattern().Invoke()
                return
            except COMError as exc:
                last_error = exc
                time.sleep(self.config.poll_interval)
                window = self._window(auto)
                if self._find_by_class(window, self.config.stop_class, required=False):
                    # Generation already started, the invocation succeeded.
                    return
                if self._reply_count(window) > baseline:
                    return
        raise ReasonixAutomationError(
            f"Reasonix send invocation failed after {self.config.send_attempts} attempts"
        ) from last_error

    def _wait_for_generation_start(self, auto, baseline: int) -> None:
        deadline = time.monotonic() + self.config.generation_start_timeout
        while time.monotonic() < deadline:
            window = self._window(auto)
            if self._find_by_class(window, self.config.stop_class, required=False):
                return
            if self._reply_count(window) > baseline:
                return
            time.sleep(self.config.poll_interval)
        raise ReasonixAutomationError("Reasonix did not enter generation state")

    def _wait_for_response(self, auto, baseline: int) -> str:
        deadline = time.monotonic() + self.config.response_timeout
        stable_since = None
        previous = ""
        while time.monotonic() < deadline:
            window = self._window(auto)
            reply_count = self._reply_count(window)
            reply = self._last_reply(window) if reply_count > baseline else ""
            if reply_count > baseline and reply.strip():
                if reply == previous:
                    stable_since = stable_since if stable_since is not None else time.monotonic()
                    if time.monotonic() - stable_since >= self.config.stable_seconds:
                        return reply
                else:
                    previous = reply
                    stable_since = time.monotonic()
            else:
                stable_since = None
            time.sleep(self.config.poll_interval)
        raise ReasonixAutomationError("timed out waiting for Reasonix response")

    def _window(self, auto) -> Any:
        deadline = time.monotonic() + self.config.lookup_timeout
        while time.monotonic() < deadline:
            window = auto.WindowControl(
                searchDepth=1,
                Name=self.config.window_name,
                ClassName=self.config.window_class,
            )
            if window.Exists(0.5, 0.1):
                return self._ensure_window_ready(window)
            # Fallback: match by title only; the class can change while the
            # application is updating or showing an overlay.
            window = auto.WindowControl(searchDepth=1, Name=self.config.window_name)
            if window.Exists(0.5, 0.1):
                return self._ensure_window_ready(window)
            time.sleep(0.1)
        raise ReasonixAutomationError("Reasonix window was not found")

    @staticmethod
    def _ensure_window_ready(window) -> Any:
        """Restore/activate the Reasonix window so its WebView2 stays interactive."""
        try:
            if window.IsMinimize():
                window.Restore(0)
            rect = window.BoundingRectangle
            if rect.width() <= 0 or rect.height() <= 0:
                window.SetActive(0)
        except Exception:
            pass
        return window

    def _find_composer(self, window) -> Any:
        composer = window.EditControl(searchDepth=30, AutomationId=self.config.composer_id)
        if not composer.Exists(self.config.lookup_timeout, 0.25):
            raise ReasonixAutomationError("Reasonix composer input was not found")
        return composer

    def _find_send(self, window, required: bool = True) -> Any:
        send = window.ButtonControl(searchDepth=30, ClassName=self.config.send_class)
        if send.Exists(self.config.lookup_timeout if required else 0.5, 0.2):
            return send
        if required:
            raise ReasonixAutomationError("Reasonix send button was not found")
        return None

    def _find_by_class(self, window, class_name: str, required: bool = True) -> Any:
        control = window.Control(searchDepth=30, ClassName=class_name)
        exists = control.Exists(self.config.lookup_timeout if required else 0.1, 0.05)
        if required and not exists:
            raise ReasonixAutomationError(f"Reasonix control was not found: {class_name}")
        return control if exists else None

    def _reply_count(self, window) -> int:
        return self._retry_webview_read(
            lambda: self._reply_count_once(window), "reply count"
        )

    def _reply_count_once(self, window) -> int:
        return sum(
            1
            for control in self._walk(window)
            if control.ClassName == self.config.reply_copy_class
        )

    def _last_reply(self, window) -> str:
        return self._retry_webview_read(
            lambda: self._last_reply_once(window), "reply text"
        )

    def _last_reply_once(self, window) -> str:
        # Walk once and record, for every control, whether it lives inside a
        # reasoning block. GetParentControl() is unreliable for WebView2
        # content (it skips intermediate wrappers), so we track the ancestor
        # class names while walking the child tree instead.
        controls: list[Any] = []
        reasoning_flags: list[bool] = []
        self._walk_with_reasoning(window, False, controls, reasoning_flags)

        copy_indexes = [
            index
            for index, control in enumerate(controls)
            if control.ClassName == self.config.reply_copy_class
        ]
        if not copy_indexes:
            raise ReasonixAutomationError("Reasonix reply boundary was not found")
        end = copy_indexes[-1]
        start = copy_indexes[-2] + 1 if len(copy_indexes) > 1 else 0
        for index in range(end - 1, start - 1, -1):
            if "msg-meta" in controls[index].ClassName:
                start = index + 1
                break
        parts = []
        for index in range(start, end):
            control = controls[index]
            if control.ControlTypeName != "TextControl":
                continue
            if reasoning_flags[index]:
                continue
            value = control.Name.strip()
            if not value:
                continue
            parts.append(value)
        reply = "\n".join(parts).strip()
        # An empty value here usually means WebView2 is still lazily rendering
        # the new turn; return "" so the caller keeps polling instead of
        # failing the whole task.
        return reply

    @classmethod
    def _walk_with_reasoning(
        cls, node, inherited_reasoning: bool, controls: list, reasoning_flags: list
    ) -> None:
        """Pre-order walk (same order as _walk) tracking reasoning-block ancestry."""
        for child in node.GetChildren():
            try:
                class_name = child.ClassName or ""
            except Exception:
                class_name = ""
            lower = class_name.lower()
            child_reasoning = inherited_reasoning or "reasoning" in lower or "turn-collapse__body" in lower
            controls.append(child)
            reasoning_flags.append(child_reasoning)
            cls._walk_with_reasoning(child, child_reasoning, controls, reasoning_flags)

    def _retry_webview_read(self, operation, label: str):
        try:
            from comtypes import COMError
        except ImportError:
            raise ReasonixAutomationError("comtypes is required by uiautomation")
        last_error = None
        for _attempt in range(self.config.send_attempts):
            try:
                return operation()
            except COMError as exc:
                last_error = exc
                time.sleep(self.config.poll_interval)
        raise ReasonixAutomationError(
            f"Reasonix WebView2 {label} remained unstable after "
            f"{self.config.send_attempts} attempts"
        ) from last_error

    @staticmethod
    def _has_reasoning_ancestor(control, root) -> bool:
        parent = control.GetParentControl()
        while parent and parent != root:
            if "reasoning" in parent.ClassName.lower():
                return True
            parent = parent.GetParentControl()
        return False

    @classmethod
    def _walk(cls, root) -> Iterable[Any]:
        for child in root.GetChildren():
            yield child
            yield from cls._walk(child)

    def _automation(self):
        try:
            import uiautomation as auto
        except ImportError as exc:
            raise ReasonixAutomationError(
                "uiautomation is not installed; run: py -m pip install -r requirements.txt"
            ) from exc
        return _initializer_context(auto)


class _AutomationContext:
    def __init__(self, auto: Any):
        self.auto = auto
        self.initializer = None

    def __enter__(self) -> Any:
        self.initializer = self.auto.UIAutomationInitializerInThread()
        return self.auto

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.initializer.Uninitialize()


def _initializer_context(auto):
    return _AutomationContext(auto)
