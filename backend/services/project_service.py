"""Application service for project creation and listing."""
from __future__ import annotations

from datetime import datetime

from agents.diagnostic_log import diagnostic_session
from agents.interview_checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from agents.llm_errors import ExtractionFailed, LLMCallFailed
from api.schemas import ProjectSummary, WorkspaceSnapshot
from services.discovery_session import (
    advance_discovery_to_waiting,
    create_discovery_session,
)
from services.project_repository import (
    create_project_record,
    list_projects,
)
from services.workspace import build_project_summary, build_workspace_snapshot


class ProjectInitializationError(RuntimeError):
    """The Project exists, but Architect could not reach its first user boundary."""

    def __init__(
        self,
        project_id: str,
        *,
        retryable: bool,
        reason: str,
    ):
        self.project_id = project_id
        self.retryable = retryable
        self.reason = reason
        super().__init__(reason)


def _mark_initialization_failure(session_id: str) -> None:
    """Preserve the checkpoint while recording extraction failure when applicable."""

    try:
        state = load_checkpoint(session_id)
    except Exception:
        return
    if state.get("checkpoint_cursor") == "extract":
        state["extraction_status"] = "EXTRACTION_FAILED"
        save_checkpoint(state)


def create_project_workspace(*, name: str, description: str) -> WorkspaceSnapshot:
    """Create Project + durable discovery session, then ask Architect's first question."""

    state = create_discovery_session(description)
    try:
        project = create_project_record(
            name=name,
            description=description,
            discovery_session_id=state["session_id"],
        )
    except Exception:
        # A session created for a Project that never persisted is an orphan.
        # Best-effort cleanup is safe here because no Project owns it yet.
        try:
            checkpoint_path(state["session_id"]).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    try:
        with diagnostic_session(
            project.discovery_session_id,
            project_id=project.id,
            operation="project_initialization",
        ):
            advance_discovery_to_waiting(project.discovery_session_id)
    except LLMCallFailed as exc:
        _mark_initialization_failure(project.discovery_session_id)
        raise ProjectInitializationError(
            project.id,
            retryable=exc.retryable,
            reason=str(exc),
        ) from exc
    except ExtractionFailed as exc:
        _mark_initialization_failure(project.discovery_session_id)
        raise ProjectInitializationError(
            project.id,
            retryable=True,
            reason=str(exc),
        ) from exc

    return build_workspace_snapshot(project.id)


def list_project_summaries() -> list[ProjectSummary]:
    """List persisted web Projects with status/freshness derived from live checkpoints."""

    summaries = [build_project_summary(project.id) for project in list_projects()]
    return sorted(
        summaries,
        key=lambda item: datetime.fromisoformat(item.updatedAt.replace("Z", "+00:00")),
        reverse=True,
    )
