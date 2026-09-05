"""AI Relay desktop application entry point."""

from pathlib import Path

from core.agent import Agent
from core.state_machine import create_default_state_machine


def build_agents():
    return (
        Agent(
            agent_id="chatgpt",
            display_name="ChatGPT",
            window_title="ChatGPT",
            template_dir=Path("templates/chatgpt"),
        ),
        Agent(
            agent_id="reasonix",
            display_name="Reasonix",
            window_title="Reasonix",
            template_dir=Path("templates/reasonix"),
        ),
    )


def main():
    try:
        from PySide6.QtCore import QLockFile, QStandardPaths
        from PySide6.QtWidgets import QApplication
        from ui import RelayWindow
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is not installed; run: py -m pip install -r requirements.txt"
        ) from exc

    build_agents()
    create_default_state_machine()

    app = QApplication([])
    lock_path = Path(QStandardPaths.writableLocation(QStandardPaths.TempLocation)) / "ai-relay-b.lock"
    lock = QLockFile(str(lock_path))
    lock.setStaleLockTime(1000)
    if not lock.tryLock(100):
        lock.removeStaleLockFile()
        if not lock.tryLock(100):
            raise RuntimeError("AI Relay B端已在运行")

    window = RelayWindow(app)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
