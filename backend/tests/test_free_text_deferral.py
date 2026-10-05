from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.conversation_manager import (
    GapGuidanceReview,
    conversation_manager_node,
)
from agents.discovery_deferrals import (
    DeferralKind,
    FreeTextDeferralReview,
    apply_reopen,
)
from agents.graph import route_after_conversation_manager
from agents.inquiries import InquirySource
from agents.question_candidates import (
    CandidateBlockReason,
    QuestionCandidate,
    filter_question_candidates,
)
from agents.interview_checkpoint import load_checkpoint, save_checkpoint
from agents.requirements import (
    ActiveRequirement,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import DiscoveryScope, DiscoveryTopic
from agents.understanding_projection import build_understanding_projection
from services.discovery_session import create_initial_discovery_state


SCOPE = DiscoveryScope.USER_APP


def _state_with_requirement():
    state = create_initial_discovery_state("An escrow product.", session_id=str(uuid4()))
    requirement = ActiveRequirement(
        id="fee_policy",
        scope=SCOPE,
        topic=DiscoveryTopic.BUSINESS_RULES,
        parent_gap="limits",
        label="Transaction fee policy",
        description="Decide how transaction fees should work.",
    )
    key = requirement_store_key(SCOPE, requirement.id)
    state["active_requirements"] = {key: requirement}
    state["eligible_requirement_keys"] = [key]
    state["selected_inquiry"] = {
        "id": "fee-policy-inquiry",
        "source": "REQUIREMENT",
        "requirement_key": key,
        "requirement_id": requirement.id,
        "decision_key": "transaction_fee_policy",
        "objective": "Decide how transaction fees should work.",
    }
    state["selected_requirement_candidate"] = {
        "requirement_key": key,
        "requirement_id": requirement.id,
        "decision_key": "transaction_fee_policy",
    }
    state["current_objective"] = "Decide how transaction fees should work."
    state["current_gap"] = "limits"
    state["active_discovery_thread"] = "payments"
    state["messages"].append(
        AIMessage(content="How should transaction fees work?", id="fee-question")
    )
    return state, key


def test_free_text_deferral_becomes_durable_control_state(monkeypatch):
    state, key = _state_with_requirement()
    state["turn_count"] = 4
    state["messages"].append(
        HumanMessage(content="Let's decide that later.", id="founder-deferral")
    )

    import agents.conversation_manager as manager

    monkeypatch.setattr(
        manager,
        "review_free_text_deferral",
        lambda current, message: FreeTextDeferralReview(
            action="defer",
            primary_control_intent=True,
            kind=DeferralKind.DECISION,
            decision_summary="Transaction fee policy",
            resolution_stage="later",
            reason="Founder explicitly postponed the decision.",
        ),
    )

    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "decision_deferral"
    assert route_after_conversation_manager({**state, **update}) == "plan_threads"
    assert update["active_requirements"][key].status == RequirementStatus.DEFERRED
    assert key not in update["eligible_requirement_keys"]

    boundary = update["discovery_boundaries"][-1]
    assert boundary["type"] == "decision_deferral"
    assert boundary["evidence"] == "Let's decide that later."
    assert boundary["decision_key"] == "transaction_fee_policy"
    assert boundary["decision_summary"] == "Transaction fee policy"
    assert boundary["resolution_stage"] == "later"
    assert boundary["explicit_authorization"] is True
    assert boundary["requirement_key"] == key
    assert boundary["id"].startswith("deferral:")


def test_mixed_answer_keeps_product_information_route_while_persisting_deferral(monkeypatch):
    state, key = _state_with_requirement()
    state["turn_count"] = 5
    state["messages"].append(
        HumanMessage(
            content="Charge 2% for now, but we can decide the cap later.",
            id="mixed-answer",
        )
    )

    import agents.conversation_manager as manager

    monkeypatch.setattr(
        manager,
        "review_free_text_deferral",
        lambda current, message: FreeTextDeferralReview(
            action="defer",
            primary_control_intent=False,
            kind=DeferralKind.DECISION,
            decision_summary="Maximum transaction fee cap",
            resolution_stage="later",
        ),
    )

    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "product_information"
    assert route_after_conversation_manager({**state, **update}) == "extract"
    assert update["discovery_boundaries"][-1]["decision_summary"] == "Maximum transaction fee cap"
    # The requirement stays durably deferred while the answered product facts
    # continue through extraction on this same turn.
    assert update["active_requirements"][key].status == RequirementStatus.DEFERRED


def test_reopen_restores_deferred_requirement_and_preserves_history():
    state, key = _state_with_requirement()
    requirement = state["active_requirements"][key]
    state["active_requirements"][key] = requirement.model_copy(
        update={"status": RequirementStatus.DEFERRED}
    )
    state["eligible_requirement_keys"] = []
    state["turn_count"] = 9
    state["discovery_boundaries"] = [
        {
            "id": "deferral:fees",
            "type": "decision_deferral",
            "scope": SCOPE.value,
            "source_turn": 4,
            "evidence": "Let's decide that later.",
            "decision_summary": "Transaction fee policy",
            "requirement_key": key,
            "requirement_id": "fee_policy",
        }
    ]

    update = apply_reopen(
        state,
        FreeTextDeferralReview(
            action="reopen",
            primary_control_intent=True,
            reopened_boundary_id="deferral:fees",
        ),
    )

    assert update["active_requirements"][key].status == RequirementStatus.ACTIVE
    assert key in update["eligible_requirement_keys"]
    assert update["discovery_boundaries"][0]["reopened_at_turn"] == 9


def test_deferral_survives_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    state, key = _state_with_requirement()
    state["active_requirements"][key] = state["active_requirements"][key].model_copy(
        update={"status": RequirementStatus.DEFERRED}
    )
    state["discovery_boundaries"] = [
        {
            "id": "deferral:phase2",
            "type": "decision_deferral",
            "kind": "release_scope",
            "scope": SCOPE.value,
            "source_turn": 6,
            "evidence": "Leave reporting for phase 2.",
            "decision_summary": "Reporting scope",
            "resolution_stage": "phase 2",
            "requirement_key": key,
            "requirement_id": "fee_policy",
            "explicit_authorization": True,
        }
    ]

    save_checkpoint(state)
    restored = load_checkpoint(state["session_id"])

    assert restored["discovery_boundaries"][0]["id"] == "deferral:phase2"
    assert restored["discovery_boundaries"][0]["resolution_stage"] == "phase 2"
    assert restored["active_requirements"][key].status == RequirementStatus.DEFERRED


def test_understanding_projection_shows_active_deferral_and_hides_reopened_one():
    state = create_initial_discovery_state("A product.", session_id=str(uuid4()))
    state["discovery_boundaries"] = [
        {
            "id": "deferral:active",
            "type": "decision_deferral",
            "kind": "release_scope",
            "scope": SCOPE.value,
            "source_turn": 3,
            "evidence": "We'll leave reporting for phase 2.",
            "decision_summary": "Reporting scope",
            "resolution_stage": "phase 2",
            "explicit_authorization": True,
        },
        {
            "id": "deferral:reopened",
            "type": "decision_deferral",
            "kind": "decision",
            "scope": SCOPE.value,
            "source_turn": 2,
            "evidence": "Decide fees later.",
            "decision_summary": "Fee policy",
            "reopened_at_turn": 7,
            "explicit_authorization": True,
        },
    ]

    projection = build_understanding_projection(state)
    section = next(item for item in projection.sections if item.id == "deferred_decisions")

    assert [item.label for item in section.items] == ["Reporting scope"]
    assert "phase 2" in (section.items[0].detail or "")
    assert "We'll leave reporting for phase 2." in (section.items[0].detail or "")



def test_question_candidate_for_active_deferral_is_ineligible():
    state = create_initial_discovery_state("A product.", session_id=str(uuid4()))
    state["discovery_boundaries"] = [
        {
            "id": "deferral:fees",
            "type": "decision_deferral",
            "scope": SCOPE.value,
            "source_turn": 2,
            "evidence": "We'll decide fees later.",
            "decision_summary": "Fee policy",
            "decision_key": "fee_policy",
            "explicit_authorization": True,
        }
    ]
    candidate = QuestionCandidate(
        id="question:fees",
        inquiry_id="inquiry:fees",
        source=InquirySource.MODEL,
        scope=SCOPE,
        topic=DiscoveryTopic.BUSINESS_RULES,
        objective="Decide the fee policy.",
        decision_key="fee_policy",
        thread_id="payments",
    )

    eligible, decisions = filter_question_candidates(state, [candidate])

    assert eligible == []
    assert CandidateBlockReason.EXPLICITLY_DEFERRED_DECISION in decisions[
        candidate.id
    ].reasons



def test_open_question_list_after_continue_discovery_is_gap_guidance_not_deferral(
    monkeypatch,
):
    state = create_initial_discovery_state(
        "A personal todo app.",
        session_id=str(uuid4()),
    )
    state["awaiting_gap_guidance"] = True
    state["prd_confirmation_pending"] = False
    state["messages"].append(
        AIMessage(
            content="Sure. What product decision or area do you want to add or revisit?"
        )
    )
    founder_text = (
        "What information a task contains — for example, just a title, or title + description.\n"
        "Whether active and completed tasks are shown together or in separate views/filters.\n"
        "Whether completed tasks can still be edited or deleted.\n"
        "What exactly happens on delete — immediate deletion or confirmation first.\n"
        "How tasks are ordered, if ordering matters at all.\n"
        "Whether tasks should persist after closing/reopening the app."
    )
    state["messages"].append(HumanMessage(content=founder_text))
    state["turn_count"] = 4

    import agents.conversation_manager as manager

    monkeypatch.setattr(
        manager,
        "review_gap_guidance",
        lambda current, message: GapGuidanceReview(
            is_gap_guidance=True,
            contains_product_decisions=False,
            unresolved_items=[
                "What information a task should contain",
                "Whether active and completed tasks need separate views or filters",
                "Whether completed tasks can still be edited or deleted",
                "What should happen when a task is deleted",
                "How tasks should be ordered, if ordering matters",
                "Whether tasks should persist after closing and reopening the app",
            ],
            reason="Founder named unresolved product questions.",
        ),
    )
    monkeypatch.setattr(
        manager,
        "review_free_text_deferral",
        lambda *_: pytest.fail(
            "Pure gap guidance must not be sent through deferral classification"
        ),
    )

    update = conversation_manager_node(state)
    merged = {**state, **update}

    assert update["conversation_intent"] == "gap_guidance"
    assert update["awaiting_gap_guidance"] is False
    assert route_after_conversation_manager(merged) == "plan_threads"
    assert update.get("discovery_boundaries", state["discovery_boundaries"]) == []
    assert len(update["founder_gap_guidance"]) == 1
    guidance = update["founder_gap_guidance"][0]
    assert guidance["evidence"] == founder_text
    assert "completed tasks can still be edited or deleted" in " ".join(
        guidance["items"]
    ).lower()
    assert "not deferred decisions" in guidance["instruction"].lower()


def test_deferral_model_cannot_create_boundary_without_explicit_postponement(
    monkeypatch,
):
    state, key = _state_with_requirement()
    state["messages"].append(
        HumanMessage(
            content=(
                "Whether completed tasks can still be edited or deleted. "
                "What exactly happens on delete."
            )
        )
    )

    import agents.conversation_manager as manager

    monkeypatch.setattr(
        manager,
        "review_free_text_deferral",
        lambda current, message: FreeTextDeferralReview(
            action="defer",
            primary_control_intent=True,
            kind=DeferralKind.DECISION,
            decision_summary="Completed task behavior",
            reason="Incorrect model verdict.",
        ),
    )
    monkeypatch.setattr(
        manager,
        "question_is_clarification",
        lambda *_: False,
    )

    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "product_information"
    assert "discovery_boundaries" not in update
    assert state["active_requirements"][key].status == RequirementStatus.ACTIVE


def test_continue_discovery_sets_gap_guidance_boundary_for_next_founder_turn():
    state = create_initial_discovery_state(
        "A personal todo app.",
        session_id=str(uuid4()),
    )
    state["prd_confirmation_pending"] = True
    state["messages"].append(
        AIMessage(
            content=(
                "I think we've covered the important product decisions.\n\n"
                "Do you think everything important has been covered before I generate the PRD?"
            )
        )
    )
    state["messages"].append(
        HumanMessage(
            content="There is more I want to cover.",
            additional_kwargs={"architect_turn_type": "continue_discovery"},
        )
    )

    update = conversation_manager_node(state)

    assert update["conversation_intent"] == "continue_discovery"
    assert update["awaiting_gap_guidance"] is True
    assert update["prd_confirmation_pending"] is False
    assert update["messages"][0].content == (
        "Sure. What product decision or area do you want to add or revisit?"
    )
