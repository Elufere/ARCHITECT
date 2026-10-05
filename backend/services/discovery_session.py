"""Lifecycle helpers for durable Architect discovery sessions."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from agents.llm_errors import ExtractionFailed, LLMCallFailed

from agents.interview_checkpoint import (
    load_checkpoint,
    save_checkpoint,
    session_lock,
)
from agents.state import AgentState, DiscoveryScope


def create_initial_discovery_state(description: str, *, session_id: str | None = None) -> AgentState:
    """Create the same durable discovery state used by the CLI, without running a model."""

    session_id = session_id or str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    idea_message = HumanMessage(
        content=description,
        id=str(uuid4()),
        additional_kwargs={"created_at": created_at},
    )

    return AgentState(
        session_id=session_id,
        messages=[idea_message],
        raw_idea=description,
        prd_contract=None,
        pm_is_complete=False,
        discovery_scope=DiscoveryScope.USER_APP,
        turn_count=0,
        awaiting_confirmation=False,
        prd_confirmation_pending=False,
        ready_to_compile=False,
        discovered_knowledge=[],
        superseded_knowledge=[],
        product_concepts=[],
        external_systems=[],
        captured_observations=[],
        discovery_boundaries=[],
        founder_gap_guidance=[],
        awaiting_gap_guidance=False,
        model_implications=[],
        thread_planning_enabled=True,
        discovery_threads={},
        active_discovery_thread=None,
        thread_frontier=None,
        thread_relevant_requirement_ids=[],
        active_requirements={},
        requirement_coverage={},
        requirement_dependency_state={},
        eligible_requirement_keys=[],
        open_inquiries=[],
        selected_inquiry=None,
        question_candidates=[],
        eligible_question_candidates=[],
        question_candidate_eligibility={},
        ranked_question_candidates=[],
        question_candidate_priority={},
        requirement_question_history=[],
        planner_source="model",
        selected_requirement_candidate=None,
        selected_requirement_priority=None,
        validation_issues=[],
        validation_pair_cache={},
        validation_blocking=False,
        validation_candidate_blocking=False,
        selected_validation_issue=None,
        product_model={},
        gap_coverage={},
        fact_acquisition={},
        asked_gap=None,
        active_answer_result=None,
        checkpoint_cursor="conversation_manager",
        interview_status="PROCESSING_ANSWER",
        extraction_status="PENDING",
        current_topic=None,
        current_gap=None,
        current_objective=None,
        question_hint=None,
        known_keys=[],
        missing_keys=[],
        inferred_gap_evidence=[],
        known_gap_evidence=[],
        relevant_context=[],
        next_discovery_move=None,
        conversation_intent=None,
        is_correction=False,
        founder_requested_completion=False,
        completion_request_evidence=None,
        completion_arbitration_complete=False,
        current_role=None,
        question_retry_count=0,
        question_retry_exhausted=False,
        answer_followup=None,
        compilation_errors=[],
    )


def create_discovery_session(description: str) -> AgentState:
    """Persist the founder's initial idea before any LLM work happens."""

    state = create_initial_discovery_state(description)
    save_checkpoint(state)
    return state


class DiscoverySessionBusyError(RuntimeError):
    pass


class DiscoverySessionStateError(RuntimeError):
    pass


TURN_TEXT = {
    "request_suggestion": "What do you suggest?",
    "unknown": "I haven't decided yet.",
    "defer_design": "Leave this to design or engineering.",
    "continue_discovery": "There is more I want to cover.",
    "confirm_prd": "Yes, generate the PRD.",
}


def _turn_content(turn_type: str, message: str | None) -> str:
    if turn_type == "answer":
        value = (message or "").strip()
        if not value:
            raise DiscoverySessionStateError("An answer turn requires a message.")
        return value
    try:
        return TURN_TEXT[turn_type]
    except KeyError as exc:
        raise DiscoverySessionStateError(
            f"Unsupported discovery turn type '{turn_type}'."
        ) from exc


def _invoke_graph_locked(state: AgentState) -> AgentState:
    """Invoke from a loaded state while the caller already owns the session lock."""

    from agents.graph import build_graph

    session_id = state["session_id"]
    graph = build_graph()
    graph.invoke(
        state,
        config={"metadata": {"openai_usage_session": session_id}},
    )
    return load_checkpoint(session_id)


def _record_processing_failure(session_id: str) -> None:
    """Keep the durable cursor intact and mark failed extraction when applicable."""

    try:
        state = load_checkpoint(session_id)
    except Exception:
        return
    if state.get("checkpoint_cursor") == "extract":
        state["extraction_status"] = "EXTRACTION_FAILED"
        save_checkpoint(state)


def _is_terminal(state: AgentState) -> bool:
    return bool(
        state.get("prd_contract") is not None
        or state.get("checkpoint_cursor") in {"phase_complete", "completed"}
    )


def _run_with_session_lock(session_id: str, operation):
    try:
        with session_lock(session_id):
            return operation()
    except RuntimeError as exc:
        if str(exc) == "This interview is already open in another process":
            raise DiscoverySessionBusyError(
                "This project is already processing another discovery request."
            ) from exc
        raise


def advance_discovery_to_waiting(session_id: str) -> AgentState:
    """Resume a durable graph until it reaches the next user boundary."""

    def operation():
        state = load_checkpoint(session_id)
        if state.get("checkpoint_cursor") == "waiting":
            return state
        return _invoke_graph_locked(state)

    return _run_with_session_lock(session_id, operation)


def submit_discovery_session_turn(
    session_id: str,
    *,
    turn_type: str,
    message: str | None = None,
) -> AgentState:
    """Persist one founder turn before any model work, then advance the graph."""

    def operation():
        state = load_checkpoint(session_id)
        if _is_terminal(state):
            raise DiscoverySessionStateError(
                "Discovery is complete for this project."
            )

        confirmation_pending = state.get("prd_confirmation_pending", False)
        if confirmation_pending and turn_type not in {"confirm_prd", "continue_discovery"}:
            raise DiscoverySessionStateError(
                "Architect is waiting for the founder's PRD confirmation decision."
            )
        if not confirmation_pending and turn_type in {"confirm_prd", "continue_discovery"}:
            raise DiscoverySessionStateError(
                "PRD confirmation is not currently pending for this project."
            )

        if state.get("checkpoint_cursor") != "waiting":
            raise DiscoverySessionStateError(
                "Architect is not waiting for a new founder turn. Retry the saved work instead."
            )
        if not state.get("messages") or not isinstance(state["messages"][-1], AIMessage):
            raise DiscoverySessionStateError(
                "Architect has saved work that must be retried before another answer can be submitted."
            )

        content = _turn_content(turn_type, message)
        metadata = {"created_at": datetime.now(timezone.utc).isoformat()}
        if turn_type != "answer":
            metadata["architect_turn_type"] = turn_type

        state["messages"].append(
            HumanMessage(
                content=content,
                id=str(uuid4()),
                additional_kwargs=metadata,
            )
        )
        state["turn_count"] = state.get("turn_count", 0) + 1
        state.update(
            checkpoint_cursor="conversation_manager",
            interview_status="PROCESSING_ANSWER",
            active_answer_result=None,
            extraction_status="PENDING",
        )
        save_checkpoint(state)
        return _invoke_graph_locked(state)

    try:
        return _run_with_session_lock(session_id, operation)
    except (LLMCallFailed, ExtractionFailed):
        _record_processing_failure(session_id)
        raise


def retry_discovery_session(session_id: str) -> AgentState:
    """Resume the saved cursor without appending or replaying founder input."""

    def operation():
        state = load_checkpoint(session_id)
        if _is_terminal(state):
            raise DiscoverySessionStateError(
                "Discovery is already complete for this project."
            )

        cursor = state.get("checkpoint_cursor")
        if cursor == "waiting":
            last = state.get("messages", [])[-1:] or [None]
            if not isinstance(last[0], HumanMessage):
                raise DiscoverySessionStateError(
                    "There is no interrupted discovery work to retry."
                )

        return _invoke_graph_locked(state)

    try:
        return _run_with_session_lock(session_id, operation)
    except (LLMCallFailed, ExtractionFailed):
        _record_processing_failure(session_id)
        raise
