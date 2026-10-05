"""Persistent terminal-style diagnostic logs for Architect sessions.

The backend keeps all existing stdout/stderr behavior, while duplicating output
produced inside a session context into one append-only UTF-8 log file. This lets
long interviews be inspected after terminal scrollback has disappeared.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import logging
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import traceback
from threading import RLock
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage


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
_log_handler = None


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


class _SessionLogHandler(logging.Handler):
    """Duplicate standard logging records into the active session trace."""

    def emit(self, record: logging.LogRecord) -> None:
        session_id = _session_id.get()
        if not session_id:
            return
        try:
            message = self.format(record)
            _append(session_id, message + "\n")
        except Exception:
            # Diagnostics must never interfere with Architect execution.
            pass


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
    """Install stream teeing and logging capture exactly once per process."""

    global _log_handler
    with _install_lock:
        # Test runners, notebook shells and reloaders can replace sys.stdout or
        # sys.stderr after startup. Re-wrap the current streams when necessary.
        if not isinstance(sys.stdout, _SessionTee):
            sys.stdout = _SessionTee(sys.stdout)
        if not isinstance(sys.stderr, _SessionTee):
            sys.stderr = _SessionTee(sys.stderr)

        if _log_handler is None:
            handler = _SessionLogHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            ))
            logging.getLogger().addHandler(handler)
            _log_handler = handler


@lru_cache(maxsize=1)
def runtime_code_identity() -> tuple[str, str, str]:
    """Return branch, commit and dirty state for the code producing a log."""
    root = Path(__file__).resolve().parents[2]
    branch = os.getenv("ARCHITECT_GIT_BRANCH")
    revision = os.getenv("ARCHITECT_GIT_SHA")
    dirty = os.getenv("ARCHITECT_GIT_DIRTY")

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        except Exception:
            return None

    branch = branch or git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    revision = revision or git("rev-parse", "HEAD") or "unknown"
    if dirty is None:
        status = git("status", "--porcelain")
        dirty = "unknown" if status is None else ("yes" if status else "no")
    return branch, revision, dirty


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
        branch, revision, dirty = runtime_code_identity()
        print(
            "\n"
            + "=" * 88
            + f"\nARCHITECT SESSION LOG | {_stamp()}"
            + f"\nlog_file: {diagnostic_log_path(canonical)}"
            + f"\nproject: {project_id or 'cli'}"
            + f"\nsession: {canonical}"
            + f"\noperation: {operation}"
            + f"\ncode_branch: {branch}"
            + f"\ncode_revision: {revision}"
            + f"\ncode_dirty: {dirty}"
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


def log_state_snapshot(state: dict, *, label: str) -> None:
    """Print a stable, compact checkpoint summary into the active diagnostic log."""

    selected = state.get("selected_inquiry") or {}
    requirement = state.get("selected_requirement_candidate") or {}
    topic = state.get("current_topic")
    scope = state.get("discovery_scope")
    summary = {
        "turn_count": state.get("turn_count"),
        "scope": getattr(scope, "value", scope),
        "checkpoint_cursor": state.get("checkpoint_cursor"),
        "interview_status": state.get("interview_status"),
        "conversation_intent": state.get("conversation_intent"),
        "current_topic": getattr(topic, "value", topic),
        "current_gap": state.get("current_gap"),
        "current_objective": state.get("current_objective"),
        "active_thread": state.get("active_discovery_thread"),
        "decision_key": selected.get("decision_key"),
        "inquiry_id": selected.get("inquiry_id") or selected.get("id"),
        "requirement_id": (
            selected.get("requirement_id")
            or requirement.get("requirement_id")
        ),
        "open_inquiries": len(state.get("open_inquiries", []) or []),
        "active_requirements": sum(
            1
            for item in (state.get("active_requirements", {}) or {}).values()
            if getattr(
                (
                    item.get("status")
                    if isinstance(item, dict)
                    else getattr(item, "status", None)
                ),
                "value",
                (
                    item.get("status")
                    if isinstance(item, dict)
                    else getattr(item, "status", None)
                ),
            )
            == "ACTIVE"
        ),
        "discovery_boundaries": len(state.get("discovery_boundaries", []) or []),
        "founder_requested_completion": state.get("founder_requested_completion", False),
        "completion_arbitration_complete": state.get("completion_arbitration_complete", False),
        "prd_confirmation_pending": state.get("prd_confirmation_pending", False),
        "ready_to_compile": state.get("ready_to_compile", False),
        "prd_ready": state.get("prd_contract") is not None,
        "extraction_status": state.get("extraction_status"),
    }

    print(f"===== ARCHITECT STATE {label.upper()} =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("==========================================")


def log_messages_since(
    state: dict,
    *,
    start_index: int = 0,
    label: str = "NEW CONVERSATION MESSAGES",
) -> None:
    """Print founder/Architect messages added since a known message index."""

    print(f"===== {label} =====")
    found = False
    for index, message in enumerate(state.get("messages", [])[start_index:], start=start_index):
        if isinstance(message, HumanMessage):
            role = "FOUNDER"
        elif isinstance(message, AIMessage):
            role = "ARCHITECT"
        else:
            continue
        found = True
        content = message.content if isinstance(message.content, str) else str(message.content)
        print(f"[{index}] {role}:")
        print(content)
        print()
    if not found:
        print("(none)")
    print("=" * len(f"===== {label} ====="))


def session_log_exists(session_id: str) -> bool:
    return diagnostic_log_path(session_id).is_file()


def session_log_size(session_id: str) -> int:
    path = diagnostic_log_path(session_id)
    return path.stat().st_size if path.exists() else 0
