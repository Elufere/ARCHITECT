from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.conversation_manager import conversation_manager_node
from agents.graph import route_after_conversation_manager
from agents.interview_checkpoint import load_checkpoint, save_checkpoint
from agents.llm_errors import LLMCallFailed
from api.schemas import DiscoveryTurnInput
from services.discovery_session import (
    DiscoverySessionStateError,
    retry_discovery_session,
    submit_discovery_session_turn,
)
from services.discovery_turn import (
    DiscoveryProcessingError,
    retry_project_discovery,
    submit_project_discovery_turn,
)
from services.project_repository import create_project_record
from services.discovery_session import create_initial_discovery_state


def _project_waiting_for_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    state = create_initial_discovery_state("An escrow product.")
    state["messages"].append(
        AIMessage(
            content="Who are the main users?",
            id="architect-question",
            additional_kwargs={"created_at": "2026-09-30T08:00:00+00:00"},
        )
    )
    state["checkpoint_cursor"] = "waiting"
    state["interview_status"] = "WAITING_FOR_USER"
    save_checkpoint(state)

    project = create_project_record(
        name="Escrow App",
        description="An escrow product.",
        discovery_session_id=state["session_id"],
    )
    return project, state["session_id"]


def test_submit_turn_checkpoints_founder_answer_before_graph_processing(
    tmp_path, monkeypatch
):
    project, session_id = _project_waiting_for_answer(tmp_path, monkeypatch)

    import services.discovery_session as session_service

    def fake_invoke(state):
        durable = load_checkpoint(session_id)
        assert durable["checkpoint_cursor"] == "conversation_manager"
        assert durable["turn_count"] == 1
        assert durable["messages"][-1].content == "Customers."
        assert isinstance(durable["messages"][-1], HumanMessage)

        durable["messages"].append(
            AIMessage(content="What should customers be able to do?", id="next-question")
        )
        durable["checkpoint_cursor"] = "waiting"
        durable["interview_status"] = "WAITING_FOR_USER"
        save_checkpoint(durable)
        return durable

    monkeypatch.setattr(session_service, "_invoke_graph_locked", fake_invoke)

    workspace = submit_project_discovery_turn(
        project.id,
        DiscoveryTurnInput(type="answer", message="Customers."),
    )

    state = load_checkpoint(session_id)
    assert state["turn_count"] == 1
    assert [message.content for message in state["messages"] if isinstance(message, HumanMessage)] == [
        "An escrow product.",
        "Customers.",
    ]
    assert workspace.discovery.activePrompt == "What should customers be able to do?"


def test_failed_turn_retry_resumes_without_duplicate_answer_or_turn_count(
    tmp_path, monkeypatch
):
    project, session_id = _project_waiting_for_answer(tmp_path, monkeypatch)

    import services.discovery_session as session_service

    def fail_once(state):
        raise LLMCallFailed(
            "conversation_manager.classify_question",
            retryable=True,
            failure_kind="connection",
        )

    monkeypatch.setattr(session_service, "_invoke_graph_locked", fail_once)

    with pytest.raises(DiscoveryProcessingError) as exc:
        submit_project_discovery_turn(
            project.id,
            DiscoveryTurnInput(type="answer", message="Customers."),
        )

    assert exc.value.project_id == project.id
    assert exc.value.retryable is True

    failed = load_checkpoint(session_id)
    assert failed["checkpoint_cursor"] == "conversation_manager"
    assert failed["turn_count"] == 1
    assert sum(
        1 for message in failed["messages"]
        if isinstance(message, HumanMessage) and message.content == "Customers."
    ) == 1

    def succeed_retry(state):
        # Retry starts from the exact saved state; it does not append input.
        assert state["turn_count"] == 1
        assert sum(
            1 for message in state["messages"]
            if isinstance(message, HumanMessage) and message.content == "Customers."
        ) == 1
        state["messages"].append(
            AIMessage(content="What should customers be able to do?", id="next-question")
        )
        state["checkpoint_cursor"] = "waiting"
        state["interview_status"] = "WAITING_FOR_USER"
        save_checkpoint(state)
        return state

    monkeypatch.setattr(session_service, "_invoke_graph_locked", succeed_retry)

    workspace = retry_project_discovery(project.id)
    resumed = load_checkpoint(session_id)

    assert resumed["turn_count"] == 1
    assert sum(
        1 for message in resumed["messages"]
        if isinstance(message, HumanMessage) and message.content == "Customers."
    ) == 1
    assert workspace.discovery.activePrompt == "What should customers be able to do?"


def test_retry_rejects_project_with_no_interrupted_work(tmp_path, monkeypatch):
    _, session_id = _project_waiting_for_answer(tmp_path, monkeypatch)

    with pytest.raises(DiscoverySessionStateError):
        retry_discovery_session(session_id)


def test_structured_defer_design_forces_control_intent_without_text_regex():
    state = create_initial_discovery_state("Idea", session_id=str(uuid4()))
    state["messages"].append(AIMessage(content="How should this screen work?"))
    state["messages"].append(
        HumanMessage(
            content="Handle it however makes sense.",
            additional_kwargs={"architect_turn_type": "defer_design"},
        )
    )
    state["turn_count"] = 1
    state["current_objective"] = "Choose the screen interaction."
    state["selected_inquiry"] = {"decision_key": "screen_interaction"}

    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "design_deferral"
    assert update["discovery_boundaries"][-1]["type"] == "design_deferral"
    assert "design/engineering" in update["discovery_boundaries"][-1]["instruction"]


def test_structured_unknown_and_continue_discovery_replan_instead_of_becoming_facts():
    unknown_state = create_initial_discovery_state("Idea", session_id=str(uuid4()))
    unknown_state["messages"].append(AIMessage(content="What should happen?"))
    unknown_state["messages"].append(
        HumanMessage(
            content="Anything",
            additional_kwargs={"architect_turn_type": "unknown"},
        )
    )

    unknown_update = conversation_manager_node(unknown_state)
    assert unknown_update["conversation_intent"] == "uncertainty"
    assert route_after_conversation_manager({**unknown_state, **unknown_update}) == "plan_threads"

    continue_state = create_initial_discovery_state("Idea", session_id=str(uuid4()))
    continue_state["awaiting_confirmation"] = True
    continue_state["messages"].append(AIMessage(content="Anything else?"))
    continue_state["messages"].append(
        HumanMessage(
            content="There is more.",
            additional_kwargs={"architect_turn_type": "continue_discovery"},
        )
    )

    continue_update = conversation_manager_node(continue_state)
    assert continue_update["conversation_intent"] == "continue_discovery"
    assert continue_update["awaiting_confirmation"] is False
    assert route_after_conversation_manager({**continue_state, **continue_update}) == "plan_threads"


def test_discovery_turn_input_validation_and_normalization():
    answer = DiscoveryTurnInput(type="answer", message="  Customers can buy and sell.  ")
    assert answer.message == "Customers can buy and sell."

    continuation = DiscoveryTurnInput(type="continue_discovery")
    assert continuation.message is None

    with pytest.raises(ValueError):
        DiscoveryTurnInput(type="answer", message="   ")

    with pytest.raises(ValueError):
        DiscoveryTurnInput(type="unknown", message="extra")
