from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from uuid import UUID

from .models import DiscoveryState


def session_directory() -> Path:
    default = Path(__file__).resolve().parents[1] / "sessions_v2"
    return Path(os.getenv("ARCHITECT_PM_V2_SESSION_DIR", str(default)))


def session_path(session_id: str) -> Path:
    return session_directory() / f"{UUID(session_id)}.json"


def save_state(state: DiscoveryState) -> None:
    path = session_path(state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump(mode="json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_state(session_id: str) -> DiscoveryState:
    payload = json.loads(session_path(session_id).read_text(encoding="utf-8"))
    return DiscoveryState.model_validate(payload)


def unfinished_states() -> list[DiscoveryState]:
    result = []
    directory = session_directory()
    if not directory.exists():
        return result
    for path in sorted(
        directory.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            state = DiscoveryState.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (ValueError, OSError):
            continue
        if not state.complete:
            result.append(state)
    return result
