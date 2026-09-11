"""Shared test fixtures for AI Relay B tests."""

from __future__ import annotations

import os
import time

# Keep every PySide6 test headless.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Do not quit a shared QApplication; later tests may reuse it.


@pytest.fixture(autouse=True)
def _clear_wrapped_registry_between_tests():
    """Reset the process-global wrapped-message registry around every test.

    ``mark_message_wrapped`` lives on a module-global set shared by every
    test in this process; without a reset one test's wrapped state leaks
    into the next and the suite becomes order-dependent.  The registry is
    cleared BEFORE and AFTER each test so test file order can never matter.
    Production flows never call the reset (see
    ``core.openchamber.reset_wrapped_message_ids``); this is strictly a
    test-isolation fixture."""
    from core.openchamber import reset_wrapped_message_ids

    reset_wrapped_message_ids()
    yield
    reset_wrapped_message_ids()


@pytest.fixture(autouse=True)
def _clear_system_clipboard_between_tests():
    """Keep tests hermetic against the system-global clipboard and stale
    windows.

    The clipboard lives on the shared QApplication, so a task left over from
    one window can leak into the next window's automatic clipboard start.
    Every test is also followed by a strict worker teardown (see
    ``_shutdown_all_windows``): no window - whether still referenced in the
    test body or already out of scope - may keep a running worker, an enabled
    clipboard listener, or a live stop/queue behind the current test."""
    _pause_all_listeners()
    yield
    _pause_all_listeners()
    _shutdown_all_windows()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.clipboard().clear()


def _pause_all_listeners() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        listener = getattr(widget, "_listener", None)
        if listener is not None:
            listener.pause()


def _shutdown_all_windows() -> None:
    """Strict worker-lifecycle teardown after every test.

    For every window still alive - including windows the test already closed
    but whose C++ widget lingers hidden on the QApplication - the sequence is:

      1. pause its clipboard listener;
      2. set every stop/cancel event (monitor stop, current task cancel);
      3. wait until ``_monitor_workers``, ``_general_workers`` and
         ``_oc_monitor_stopping`` are all empty (bounded; a leak is a test
         failure, never a silent pass-through);
      4. close the window with WA_DeleteOnClose so the widget is actually
         destroyed and removed from the application, not just hidden.

    This must be mirrored by every individual test's ``finally`` (see
    ``tests.test_ui_startup.shutdown_window``); the fixture enforces the
    invariant even when a test forgets or fails before reaching its own
    teardown."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    if app is None:
        return
    for _ in range(30):
        QTest.qWait(50)
        remaining = []
        for w in app.topLevelWidgets():
            if not isinstance(w, QWidget):
                continue
            try:
                _ = w.objectName()
            except RuntimeError:
                continue
            remaining.append(w)
        if not remaining:
            return
        for widget in remaining:
            try:
                _shutdown_one_window(widget)
            except RuntimeError:
                continue
    remaining = [
        f"{w.__class__.__name__}({hex(id(w))})"
        for w in app.topLevelWidgets()
        if isinstance(w, QWidget)
    ]
    raise AssertionError(f"windows did not shut down within budget: {remaining}")


def _shutdown_one_window(widget) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    listener = getattr(widget, "_listener", None)
    if listener is not None:
        listener.pause()

    # Request a graceful stop from every running worker.
    stop = getattr(widget, "_oc_monitor_stop", None)
    if stop is not None:
        stop.set()
    cancel = getattr(widget, "_task_cancel_event", None)
    if cancel is not None:
        cancel.set()
    if getattr(widget, "_oc_monitor_active", False) and not getattr(
        widget, "_oc_monitor_stopping", False
    ):
        widget._stop_monitor()

    deadline = time.monotonic() + 5.0
    while (
        getattr(widget, "_monitor_workers", set())
        or getattr(widget, "_general_workers", set())
        or getattr(widget, "_oc_monitor_stopping", False)
    ):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{widget.__class__.__name__}({hex(id(widget))}) leaked workers: "
                f"monitor={sorted(widget._monitor_workers)} "
                f"general={sorted(widget._general_workers)} "
                f"stopping={widget._oc_monitor_stopping}"
            )
        QTest.qWait(50)

    widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    widget.close()
    # A window the test already closed (hidden) no longer triggers the
    # WA_DeleteOnClose deletion on a second close; schedule the deletion
    # explicitly (safe: every worker had ``finished`` already) so the widget
    # really leaves the QApplication instead of lingering hidden.
    widget.deleteLater()
    QTest.qWait(50)