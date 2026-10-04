"""Application service for project creation and listing."""
from __future__ import annotations

from datetime import datetime

from agents.diagnostic_log import (
    diagnostic_log_path,
    diagnostic_session,
    log_messages_since,
    log_state_snapshot,
)
from agents.interview_checkpoint import (
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    session_lock,
)
from agents.llm_errors import ExtractionFailed, LLMCallFailed
from api.schemas import ProjectSummary, WorkspaceSnapshot
from services.discovery_session import (
    DiscoverySessionBusyError,
    advance_discovery_to_waiting,
    create_discovery_session,
)
from services.project_repository import (
    create_project_record,
    delete_project_record,
    get_project,
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
            before = load_checkpoint(project.discovery_session_id)
            before_count = len(before.get("messages", []))
            log_state_snapshot(before, label="before project initialization")
            print("===== FOUNDER PROJECT IDEA =====")
            print(project.description)
            print("================================")
            try:
                after = advance_discovery_to_waiting(project.discovery_session_id)
            except Exception:
                try:
                    failed = load_checkpoint(project.discovery_session_id)
                    log_messages_since(
                        failed,
                        start_index=before_count,
                        label="MESSAGES SAVED BEFORE INITIALIZATION FAILURE",
                    )
                    log_state_snapshot(
                        failed,
                        label="saved state after initialization failure",
                    )
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
            log_state_snapshot(after, label="after project initialization")
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



def delete_project_workspace(project_id: str) -> None:
    """Delete a Project and best-effort remove its private session artifacts.

    The session lock prevents deletion while Architect is processing the project.
    The Project record is removed first so a successful delete can never leave a
    broken project visible in the workspace if ancillary cleanup later fails.
    """

    project = get_project(project_id)
    session_id = project.discovery_session_id
    lock_path = checkpoint_path(session_id).with_suffix(".lock")

    try:
        with session_lock(session_id):
            delete_project_record(project.id)

            for artifact in (
                checkpoint_path(session_id),
                diagnostic_log_path(session_id),
            ):
                try:
                    artifact.unlink(missing_ok=True)
                except OSError as exc:
                    # The Project itself is already deleted. Do not resurrect it
                    # because a stale local diagnostic/session artifact could not be
                    # removed; surface the cleanup issue in server logs instead.
                    print(
                        "PROJECT DELETE CLEANUP WARNING: "
                        f"{artifact} | {type(exc).__name__}: {exc}"
                    )
    except RuntimeError as exc:
        if str(exc) == "This interview is already open in another process":
            raise DiscoverySessionBusyError(
                "This project is currently being processed and cannot be deleted yet."
            ) from exc
        raise

    try:
        lock_path.unlink(missing_ok=True)
    except OSError as exc:
        print(
            "PROJECT DELETE CLEANUP WARNING: "
            f"{lock_path} | {type(exc).__name__}: {exc}"
        )
