"""Lifecycle helpers for durable Architect discovery sessions."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import HumanMessage

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
        discovered_knowledge=[],
        superseded_knowledge=[],
        product_concepts=[],
        captured_observations=[],
        discovery_boundaries=[],
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


def advance_discovery_to_waiting(session_id: str) -> AgentState:
    """Resume the durable graph until it reaches the next user boundary."""

    # Lazy import keeps API startup independent from graph/model construction.
    from agents.graph import build_graph

    with session_lock(session_id):
        state = load_checkpoint(session_id)
        if state.get("checkpoint_cursor") == "waiting":
            return state

        graph = build_graph()
        graph.invoke(
            state,
            config={"metadata": {"openai_usage_session": session_id}},
        )
        return load_checkpoint(session_id)
