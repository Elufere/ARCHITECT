from __future__ import annotations

import os
from pathlib import Path
import tempfile
from uuid import UUID

from .models import InterviewState


def session_directory() -> Path:
    default = Path(__file__).resolve().parents[1] / "adaptive_pm_sessions"
    return Path(os.getenv("ADAPTIVE_PM_SESSION_DIR", str(default)))


def session_path(session_id: str) -> Path:
    return session_directory() / f"{UUID(session_id)}.json"


def save_state(state: InterviewState) -> None:
    path = session_path(state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(state.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_state(session_id: str) -> InterviewState:
    return InterviewState.model_validate_json(session_path(session_id).read_text(encoding="utf-8"))


def list_states() -> list[InterviewState]:
    result: list[InterviewState] = []
    directory = session_directory()
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            result.append(InterviewState.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return result
