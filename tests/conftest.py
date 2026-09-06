"""Shared test fixtures for AI Relay B tests."""

from __future__ import annotations

import os

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