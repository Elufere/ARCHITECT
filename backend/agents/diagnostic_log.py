"""Persistent terminal-style diagnostic logs for Architect sessions.

The backend keeps all existing stdout/stderr behavior, while duplicating output
produced inside a session context into one append-only UTF-8 log file. This lets
long interviews be inspected after terminal scrollback has disappeared.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import traceback
from threading import RLock
from uuid import UUID


_session_id: ContextVar[str | None] = ContextVar(
    "architect_diagnostic_session_id",
    default=None,
)
_project_id: ContextVar[str | None] = ContextVar(
    "architect_diagnostic_project_id",
    default=None,
)
_operation: ContextVar[str | None] = ContextVar(
    "architect_diagnostic_operation",
    default=None,
)

_write_lock = RLock()
_install_lock = RLock()


def diagnostic_log_directory() -> Path:
    return Path(
        os.getenv(
            "ARCHITECT_LOG_DIR",
            str(Path(__file__).resolve().parents[1] / "logs" / "sessions"),
        )
    )


def diagnostic_log_path(session_id: str) -> Path:
    canonical = str(UUID(session_id))
    return diagnostic_log_directory() / f"{canonical}.log"


def _append(session_id: str, content: str) -> None:
    if not content:
        return
    path = diagnostic_log_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()


class _SessionTee:
    """Proxy a text stream while duplicating writes into the active session log."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, content):
        written = self._wrapped.write(content)
        session_id = _session_id.get()
        if session_id:
            try:
                _append(session_id, content)
            except Exception:
                # Diagnostics must never break product execution.
                pass
        return written

    def flush(self):
        return self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    def fileno(self):
        return self._wrapped.fileno()

    @property
    def encoding(self):
        return getattr(self._wrapped, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._wrapped, "errors", None)

    def writable(self):
        return True

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def install_diagnostic_streams() -> None:
    """Install stdout/stderr teeing exactly once for this Python process."""

    with _install_lock:
        # Test runners, notebook shells and reloaders can replace sys.stdout or
        # sys.stderr after startup. Re-wrap the current streams when necessary.
        if not isinstance(sys.stdout, _SessionTee):
            sys.stdout = _SessionTee(sys.stdout)
        if not isinstance(sys.stderr, _SessionTee):
            sys.stderr = _SessionTee(sys.stderr)


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def diagnostic_session(
    session_id: str,
    *,
    project_id: str | None = None,
    operation: str = "session",
):
    """Capture terminal diagnostics for one durable Architect operation."""

    canonical = str(UUID(session_id))
    install_diagnostic_streams()

    session_token = _session_id.set(canonical)
    project_token = _project_id.set(project_id)
    operation_token = _operation.set(operation)
    status = "completed"
    try:
        print(
            "\n"
            + "=" * 88
            + f"\nARCHITECT SESSION LOG | {_stamp()}"
            + f"\nproject: {project_id or 'cli'}"
            + f"\nsession: {canonical}"
            + f"\noperation: {operation}"
            + "\n"
            + "=" * 88
        )
        yield
    except BaseException as exc:
        status = f"failed ({type(exc).__name__}: {exc})"
        # FastAPI often translates expected processing failures into an HTTP
        # response, so Uvicorn may never print their traceback. Preserve it in
        # the session log without adding noisy stack traces to normal terminal
        # output.
        try:
            _append(
                canonical,
                "\n===== ARCHITECT CAPTURED TRACEBACK =====\n"
                + traceback.format_exc()
                + "===== END CAPTURED TRACEBACK =====\n",
            )
        except Exception:
            pass
        raise
    finally:
        print(
            "\n"
            + "-" * 88
            + f"\nARCHITECT SESSION OPERATION END | {_stamp()}"
            + f"\noperation: {operation}"
            + f"\nstatus: {status}"
            + "\n"
            + "-" * 88
            + "\n"
        )
        _operation.reset(operation_token)
        _project_id.reset(project_token)
        _session_id.reset(session_token)


def session_log_exists(session_id: str) -> bool:
    return diagnostic_log_path(session_id).is_file()


def session_log_size(session_id: str) -> int:
    path = diagnostic_log_path(session_id)
    return path.stat().st_size if path.exists() else 0
