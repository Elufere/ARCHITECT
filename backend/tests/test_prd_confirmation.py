import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.conversation_manager import conversation_manager_node
from agents.discovery_completion import should_review_completion_intent
from agents.graph import (
    PRD_CONFIRMATION_PROMPT,
    compile_prd_when_approved,
    request_prd_confirmation_node,
    route_after_conversation_manager,
    route_after_plan,
)
from agents.interview_checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from agents.state import DiscoveryScope
from services.discovery_session import (
    DiscoverySessionStateError,
    create_initial_discovery_state,
    submit_discovery_session_turn,
)


def _resolved_state():
    state = create_initial_discovery_state("A product", session_id=str(uuid4()))
    state.update(
        open_inquiries=[],
        active_requirements={},
        validation_blocking=False,
        prd_confirmation_pending=True,
        ready_to_compile=False,
    )
    return state


def test_planner_completion_routes_to_confirmation_not_compilation():
    state = _resolved_state()

    assert route_after_plan(state) == "request_prd_confirmation"


def test_confirmation_prompt_is_deterministic_and_does_not_approve_compilation():
    state = _resolved_state()

    update = request_prd_confirmation_node(state)

    assert update["messages"][0].content == PRD_CONFIRMATION_PROMPT
    assert update["ready_to_compile"] is False


def test_compiler_requires_explicit_founder_approval(monkeypatch):
    state = _resolved_state()

    with pytest.raises(RuntimeError, match="explicit founder confirmation"):
        compile_prd_when_approved(state)

    import agents.graph as graph

    monkeypatch.setattr(graph, "pm_compile_node", lambda current: {"compiled": True})
    approved = {**state, "prd_confirmation_pending": False, "ready_to_compile": True}

    assert compile_prd_when_approved(approved) == {"compiled": True}


def test_structured_confirmation_sets_compile_authorization_only_at_pending_boundary():
    state = _resolved_state()
    state["messages"].append(AIMessage(content=PRD_CONFIRMATION_PROMPT))
    state["messages"].append(
        HumanMessage(
            content="Yes, generate the PRD.",
            additional_kwargs={"architect_turn_type": "confirm_prd"},
        )
    )

    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "confirm_prd"
    assert update["prd_confirmation_pending"] is False
    assert update["ready_to_compile"] is True
    assert route_after_conversation_manager({**state, **update}) == "compile_prd"


def test_normal_answer_is_blocked_while_prd_confirmation_is_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))

    state = _resolved_state()
    state["messages"].append(AIMessage(content=PRD_CONFIRMATION_PROMPT))
    state["checkpoint_cursor"] = "waiting"
    state["interview_status"] = "AWAITING_PRD_CONFIRMATION"
    save_checkpoint(state)

    with pytest.raises(DiscoverySessionStateError, match="PRD confirmation"):
        submit_discovery_session_turn(
            state["session_id"],
            turn_type="answer",
            message="yes",
        )

    durable = load_checkpoint(state["session_id"])
    assert durable["turn_count"] == 0
    assert not any(
        isinstance(message, HumanMessage) and message.content == "yes"
        for message in durable["messages"]
    )


def test_confirmation_actions_are_rejected_outside_confirmation_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))

    state = create_initial_discovery_state("A product")
    state["messages"].append(AIMessage(content="Who are the users?"))
    state["checkpoint_cursor"] = "waiting"
    state["interview_status"] = "WAITING_FOR_USER"
    save_checkpoint(state)

    for turn_type in ("confirm_prd", "continue_discovery"):
        with pytest.raises(DiscoverySessionStateError, match="not currently pending"):
            submit_discovery_session_turn(
                state["session_id"],
                turn_type=turn_type,
            )


def test_legacy_auto_compile_checkpoint_is_migrated_back_to_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))

    state = create_initial_discovery_state("A product")
    state["awaiting_confirmation"] = True
    state["checkpoint_cursor"] = "compile_prd"
    state["interview_status"] = "COMPILING_PRD"
    save_checkpoint(state)

    path = checkpoint_path(state["session_id"])
    document = json.loads(path.read_text(encoding="utf-8"))
    # Simulate a checkpoint written before explicit PRD confirmation existed.
    document["state"].pop("prd_confirmation_pending", None)
    document["state"].pop("ready_to_compile", None)
    path.write_text(json.dumps(document), encoding="utf-8")

    migrated = load_checkpoint(state["session_id"])

    assert migrated["awaiting_confirmation"] is False
    assert migrated["prd_confirmation_pending"] is True
    assert migrated["ready_to_compile"] is False
    assert migrated["checkpoint_cursor"] == "request_prd_confirmation"

def test_substantive_nothing_answer_is_not_treated_as_discovery_completion():
    state = _resolved_state()
    state.update(
        prd_confirmation_pending=False,
        current_objective="Determine whether completed tasks remain editable.",
        current_gap=None,
        selected_inquiry={
            "id": "question|USER_APP|thread|task-lifecycle|edit-completed-task"
        },
        founder_requested_completion=False,
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Should users be able to edit a completed task, or must they "
                "return it to active first?"
            )
        )
    )

    answer = (
        "nothing can be done on a completed task other than returning "
        "the status to active"
    )

    assert should_review_completion_intent(state, answer) is False

    state["messages"].append(HumanMessage(content=answer))
    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "product_information"
    assert not update.get("founder_requested_completion", False)


def test_explicit_finish_request_can_close_even_with_active_inquiry():
    state = _resolved_state()
    state.update(
        prd_confirmation_pending=False,
        current_objective="Clarify one product decision.",
        selected_inquiry={"id": "question|USER_APP|thread|example|decision"},
    )
    state["messages"].append(AIMessage(content="What should happen next?"))

    assert should_review_completion_intent(
        state,
        "We've covered everything. Generate the PRD.",
    ) is True

