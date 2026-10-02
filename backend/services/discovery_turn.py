"""Project-oriented discovery turn and retry application service."""
from __future__ import annotations

from agents.diagnostic_log import diagnostic_session
from agents.llm_errors import ExtractionFailed, LLMCallFailed
from api.schemas import DiscoveryTurnInput, WorkspaceSnapshot
from services.discovery_session import (
    DiscoverySessionBusyError,
    DiscoverySessionStateError,
    retry_discovery_session,
    submit_discovery_session_turn,
)
from services.project_repository import get_project
from services.workspace import build_workspace_snapshot


class DiscoveryProcessingError(RuntimeError):
    """A durable Project exists, but its graph processing did not finish."""

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


def _processing_failure(project_id: str, exc: Exception) -> DiscoveryProcessingError:
    return DiscoveryProcessingError(
        project_id,
        retryable=(exc.retryable if isinstance(exc, LLMCallFailed) else True),
        reason=str(exc),
    )


def submit_project_discovery_turn(
    project_id: str,
    turn: DiscoveryTurnInput,
) -> WorkspaceSnapshot:
    project = get_project(project_id)
    try:
        with diagnostic_session(
            project.discovery_session_id,
            project_id=project.id,
            operation=f"discovery_turn:{turn.type}",
        ):
            submit_discovery_session_turn(
                project.discovery_session_id,
                turn_type=turn.type,
                message=turn.message,
            )
    except (LLMCallFailed, ExtractionFailed) as exc:
        raise _processing_failure(project.id, exc) from exc

    return build_workspace_snapshot(project.id)


def retry_project_discovery(project_id: str) -> WorkspaceSnapshot:
    project = get_project(project_id)
    try:
        with diagnostic_session(
            project.discovery_session_id,
            project_id=project.id,
            operation="discovery_retry",
        ):
            retry_discovery_session(project.discovery_session_id)
    except (LLMCallFailed, ExtractionFailed) as exc:
        raise _processing_failure(project.id, exc) from exc

    return build_workspace_snapshot(project.id)
