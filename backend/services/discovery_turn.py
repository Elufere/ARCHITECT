"""Project-oriented discovery turn and retry application service."""
from __future__ import annotations

from agents.diagnostic_log import (
    diagnostic_session,
    log_messages_since,
    log_state_snapshot,
)
from agents.llm_errors import ExtractionFailed, LLMCallFailed
from api.schemas import DiscoveryTurnInput, WorkspaceSnapshot
from agents.interview_checkpoint import load_checkpoint
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
            before = load_checkpoint(project.discovery_session_id)
            before_count = len(before.get("messages", []))
            log_state_snapshot(before, label="before discovery turn")
            print("===== DISCOVERY TURN REQUEST =====")
            print(f"type: {turn.type}")
            if turn.message is not None:
                print("message:")
                print(turn.message)
            print("==================================")
            try:
                after = submit_discovery_session_turn(
                    project.discovery_session_id,
                    turn_type=turn.type,
                    message=turn.message,
                )
            except Exception:
                try:
                    failed = load_checkpoint(project.discovery_session_id)
                    log_messages_since(
                        failed,
                        start_index=before_count,
                        label="MESSAGES SAVED BEFORE FAILURE",
                    )
                    log_state_snapshot(failed, label="saved state after discovery failure")
                except Exception as snapshot_exc:
                    print(
                        "DIAGNOSTIC SNAPSHOT ERROR: "
                        f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                    )
                raise
            log_messages_since(
                after,
                start_index=before_count,
                label="NEW CONVERSATION MESSAGES",
            )
            log_state_snapshot(after, label="after discovery turn")
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
            before = load_checkpoint(project.discovery_session_id)
            before_count = len(before.get("messages", []))
            log_state_snapshot(before, label="before discovery retry")
            try:
                after = retry_discovery_session(project.discovery_session_id)
            except Exception:
                try:
                    failed = load_checkpoint(project.discovery_session_id)
                    log_messages_since(
                        failed,
                        start_index=before_count,
                        label="MESSAGES SAVED BEFORE RETRY FAILURE",
                    )
                    log_state_snapshot(failed, label="saved state after retry failure")
                except Exception as snapshot_exc:
                    print(
                        "DIAGNOSTIC SNAPSHOT ERROR: "
                        f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                    )
                raise
            log_messages_since(
                after,
                start_index=before_count,
                label="NEW CONVERSATION MESSAGES",
            )
            log_state_snapshot(after, label="after discovery retry")
    except (LLMCallFailed, ExtractionFailed) as exc:
        raise _processing_failure(project.id, exc) from exc

    return build_workspace_snapshot(project.id)
