"""Project-facing PRD generation service."""
from __future__ import annotations

from agents.diagnostic_log import (
    diagnostic_session,
    log_messages_since,
    log_state_snapshot,
)
from agents.graph import build_graph
from agents.interview_checkpoint import load_checkpoint, save_checkpoint, session_lock
from agents.interview_planner import all_discovery_resolved
from agents.llm_errors import LLMCallFailed
from api.schemas import WorkspaceSnapshot
from services.discovery_session import DiscoverySessionBusyError
from services.project_repository import get_project
from services.workspace import build_workspace_snapshot


class PrdGenerationStateError(RuntimeError):
    """The project is not at a state where PRD generation is allowed."""


class PrdGenerationProcessingError(RuntimeError):
    """Founder approval exists, but PRD compilation did not complete."""

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


def _run_with_session_lock(session_id: str, operation):
    try:
        with session_lock(session_id):
            return operation()
    except RuntimeError as exc:
        if str(exc) == "This interview is already open in another process":
            raise DiscoverySessionBusyError(
                "This project is already processing another request."
            ) from exc
        raise


def _compile_approved_session(project_id: str, session_id: str):
    state = load_checkpoint(session_id)

    # Generation is idempotent. Once a verified contract exists, never invoke
    # the compiler again just because the client repeated the request.
    if state.get("prd_contract") is not None:
        return state

    if state.get("prd_confirmation_pending"):
        raise PrdGenerationStateError(
            "Explicit founder confirmation is required before PRD generation."
        )
    if not state.get("ready_to_compile"):
        raise PrdGenerationStateError(
            "PRD generation has not been authorized by the founder."
        )
    if not all_discovery_resolved(state):
        raise PrdGenerationStateError(
            "PRD generation is blocked while material discovery remains unresolved."
        )

    # Confirmation normally routes directly into compile_prd. The explicit API
    # also supports an approved checkpoint that stopped before compilation, and
    # retries a prior failed compilation without replaying founder input.
    if state.get("checkpoint_cursor") != "compile_prd":
        state.update(
            checkpoint_cursor="compile_prd",
            interview_status="COMPILING_PRD",
        )
        save_checkpoint(state)

    graph = build_graph()
    graph.invoke(
        state,
        config={"metadata": {"openai_usage_session": session_id}},
    )
    compiled = load_checkpoint(session_id)

    if compiled.get("prd_contract") is None:
        errors = compiled.get("compilation_errors", [])
        reason = errors[-1] if errors else "PRD compilation did not produce a verified document."
        raise PrdGenerationProcessingError(
            project_id,
            retryable=True,
            reason=reason,
        )

    return compiled


def generate_project_prd(project_id: str) -> WorkspaceSnapshot:
    """Generate or return the verified PRD for one Project."""

    project = get_project(project_id)

    def operation():
        return _compile_approved_session(
            project.id,
            project.discovery_session_id,
        )

    try:
        with diagnostic_session(
            project.discovery_session_id,
            project_id=project.id,
            operation="prd_generation",
        ):
            before = load_checkpoint(project.discovery_session_id)
            before_count = len(before.get("messages", []))
            log_state_snapshot(before, label="before PRD generation")
            _run_with_session_lock(project.discovery_session_id, operation)
            after = load_checkpoint(project.discovery_session_id)
            log_messages_since(
                after,
                start_index=before_count,
                label="NEW CONVERSATION MESSAGES",
            )
            log_state_snapshot(after, label="after PRD generation")
    except LLMCallFailed as exc:
        raise PrdGenerationProcessingError(
            project.id,
            retryable=exc.retryable,
            reason=str(exc),
        ) from exc

    return build_workspace_snapshot(project.id)
