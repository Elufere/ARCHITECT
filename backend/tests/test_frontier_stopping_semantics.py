from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agents.conversation_manager as conversation
import agents.interview_planner as planner
import agents.question_candidates as candidates
from agents.discovery_completion import CompletionIntentReview
from agents.discovery_threads import discovery_thread_node
from agents.graph import route_after_conversation_manager, route_after_plan
from agents.question_candidates import (
    CandidateBlockReason,
    CandidateEligibilityDecision,
    CompletionRequirementArbitration,
    QuestionCandidate,
    arbitrate_completion_candidates,
    filter_question_candidates,
)
from agents.inquiries import InquirySource
from agents.requirements import (
    ActiveRequirement,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import DiscoveryScope, DiscoveryTopic
from services.discovery_session import create_initial_discovery_state


SCOPE = DiscoveryScope.USER_APP


def requirement(requirement_id="payment_recovery"):
    return ActiveRequirement(
        id=requirement_id,
        scope=SCOPE,
        topic=DiscoveryTopic.CORE_WORKFLOW,
        parent_gap="workflow_steps",
        label="Payment recovery",
        description="Decide what users should do when payment fails.",
    )


def base_state():
    return create_initial_discovery_state(
        "An escrow app.",
        session_id=str(uuid4()),
    )


def test_structured_design_deferral_uses_durable_decision_boundary():
    state = base_state()
    req = requirement()
    key = requirement_store_key(SCOPE, req.id)
    state["active_requirements"] = {key: req}
    state["eligible_requirement_keys"] = [key]
    state["selected_inquiry"] = {
        "id": "question|payment-recovery",
        "inquiry_id": "payment-recovery",
        "source": InquirySource.REQUIREMENT.value,
        "requirement_key": key,
        "requirement_id": req.id,
        "decision_key": "payment_failure_recovery",
        "objective": req.description,
    }
    state["current_objective"] = req.description
    state["active_discovery_thread"] = "payments"
    state["turn_count"] = 7
    state["messages"].extend([
        AIMessage(content="What should users do if a Flutterwave payment fails?"),
        HumanMessage(
            content="Leave this to design or engineering.",
            additional_kwargs={"architect_turn_type": "defer_design"},
        ),
    ])

    update = conversation.conversation_manager_node(state)

    assert update["conversation_intent"] == "design_deferral"
    assert route_after_conversation_manager({**state, **update}) == "plan_threads"
    assert update["active_requirements"][key].status == RequirementStatus.DEFERRED
    assert key not in update["eligible_requirement_keys"]

    boundary = update["discovery_boundaries"][-1]
    assert boundary["type"] == "decision_deferral"
    assert boundary["kind"] == "design_implementation"
    assert boundary["decision_key"] == "payment_failure_recovery"
    assert boundary["requirement_key"] == key


@pytest.mark.parametrize(
    "text",
    [
        "I answered this question already.",
        "I answered that question already.",
        "Answered this question already.",
        "You asked me already.",
    ],
)
def test_repetition_feedback_variants_are_objections(text):
    assert conversation.classify_turn(text) == "objection"


def test_already_answered_feedback_is_objection_not_generic_correction():
    text = "I answered this question already, and I said the admin should decide."

    assert conversation.classify_turn(text) == "objection"


def test_rejected_inquiry_hard_blocks_same_decision_after_objection():
    state = base_state()
    state["selected_inquiry"] = {
        "id": "question|dispute-resolution",
        "inquiry_id": "dispute-resolution",
        "source": InquirySource.MODEL.value,
        "decision_key": "dispute_resolution_authority",
    }
    state["current_objective"] = "Decide who resolves a dispute."
    state["active_discovery_thread"] = "disputes"
    state["turn_count"] = 9
    state["messages"].extend([
        AIMessage(content="Who should decide the final dispute outcome?"),
        HumanMessage(
            content="I answered this question already, and I said the admin should decide."
        ),
    ])

    update = conversation.conversation_manager_node(state)
    assert update["conversation_intent"] == "objection"

    candidate = QuestionCandidate(
        id="question|dispute-resolution-again",
        inquiry_id="dispute-resolution",
        source=InquirySource.MODEL,
        scope=SCOPE,
        topic=DiscoveryTopic.BUSINESS_RULES,
        objective="Decide who resolves a dispute.",
        decision_key="dispute_resolution_authority",
        thread_id="disputes",
    )
    filter_state = {
        **state,
        **update,
        "open_inquiries": [],
        "requirement_question_history": [],
    }

    eligible, decisions = filter_question_candidates(filter_state, [candidate])

    assert eligible == []
    assert CandidateBlockReason.EXPLICITLY_REJECTED_DECISION in decisions[
        candidate.id
    ].reasons


def test_founder_wrap_up_becomes_completion_control_state(monkeypatch):
    state = base_state()
    state["turn_count"] = 12
    state["current_objective"] = "Check whether anything material remains."
    state["messages"].extend([
        AIMessage(content="Is there any other part of the app you want to cover?"),
        HumanMessage(content="No, we have covered everything."),
    ])

    monkeypatch.setattr(
        conversation,
        "should_review_free_text_deferral",
        lambda current: False,
    )
    monkeypatch.setattr(
        conversation,
        "review_completion_intent",
        lambda current, message: CompletionIntentReview(
            wants_to_finish_discovery=True,
            reason="Founder explicitly closed discovery.",
        ),
    )

    update = conversation.conversation_manager_node(state)

    assert update["conversation_intent"] == "close_discovery"
    assert update["founder_requested_completion"] is True
    assert update["completion_request_evidence"] == "No, we have covered everything."
    assert update["completion_arbitration_complete"] is False
    assert route_after_conversation_manager({**state, **update}) == "plan_threads"


def test_plain_no_to_product_question_is_not_forced_to_close(monkeypatch):
    state = base_state()
    state["turn_count"] = 4
    state["messages"].extend([
        AIMessage(content="Should the buyer be allowed to cancel after acceptance?"),
        HumanMessage(content="No"),
    ])

    monkeypatch.setattr(
        conversation,
        "should_review_free_text_deferral",
        lambda current: False,
    )
    monkeypatch.setattr(
        conversation,
        "review_completion_intent",
        lambda current, message: CompletionIntentReview(
            wants_to_finish_discovery=False,
            reason="No answers the product question.",
        ),
    )

    update = conversation.conversation_manager_node(state)

    assert update["conversation_intent"] == "product_information"
    assert not update.get("founder_requested_completion", False)


def test_founder_closure_bypasses_new_thread_frontier(monkeypatch):
    state = base_state()
    req = requirement("late_confirmation")
    key = requirement_store_key(SCOPE, req.id)
    state["active_requirements"] = {key: req}
    state["founder_requested_completion"] = True

    import agents.discovery_threads as threads

    monkeypatch.setattr(
        threads,
        "plan_discovery_thread",
        lambda current: (_ for _ in ()).throw(
            AssertionError("closure must not generate a fresh exploratory frontier")
        ),
    )

    update = discovery_thread_node(state)

    assert update["thread_frontier"] is None
    assert update["thread_relevant_requirement_ids"] == [req.id]
    assert update["completion_arbitration_complete"] is False


def _requirement_candidate(req: ActiveRequirement) -> QuestionCandidate:
    key = requirement_store_key(SCOPE, req.id)
    return QuestionCandidate(
        id=f"question|{req.id}",
        inquiry_id=f"inquiry|{req.id}",
        source=InquirySource.REQUIREMENT,
        requirement_key=key,
        requirement_id=req.id,
        scope=SCOPE,
        topic=req.topic,
        objective=req.description or req.label,
        reason="Activated requirement still has an unresolved decision.",
        target_facets=["behavior"],
        architecture_impact=0.4,
        business_risk=0.3,
    )


def test_completion_arbitration_drops_nonmaterial_requirement(monkeypatch):
    req = requirement("notification_content")
    candidate = _requirement_candidate(req)
    state = {
        "founder_requested_completion": True,
        "completion_request_evidence": "I think we have covered everything.",
        "product_model": {"CORE_WORKFLOW": ["Core transaction flow is known."]},
        "discovery_boundaries": [],
    }
    decisions = {
        candidate.id: CandidateEligibilityDecision(
            candidate_id=candidate.id,
            eligible=True,
        )
    }

    class FakeModel:
        def invoke(self, messages):
            return CompletionRequirementArbitration(
                blocking_candidate_ids=[],
                nonblocking_candidate_ids=[candidate.id],
                reasons={candidate.id: "Optional depth after founder closure."},
            )

    monkeypatch.setattr(candidates, "completion_requirement_model", lambda: FakeModel())

    eligible, updated, complete = arbitrate_completion_candidates(
        state,
        [candidate],
        decisions,
    )

    assert eligible == []
    assert complete is True
    assert CandidateBlockReason.FOUNDER_CLOSURE_NONBLOCKING in updated[
        candidate.id
    ].reasons


def test_foundational_candidate_still_blocks_founder_closure(monkeypatch):
    candidate = QuestionCandidate(
        id="question|core-actors",
        inquiry_id="model.core_actors",
        source=InquirySource.MODEL,
        scope=SCOPE,
        topic=DiscoveryTopic.USER_ROLES,
        objective="Identify the primary users.",
    )
    state = {
        "founder_requested_completion": True,
        "completion_request_evidence": "We are done.",
    }
    decisions = {
        candidate.id: CandidateEligibilityDecision(
            candidate_id=candidate.id,
            eligible=True,
        )
    }

    monkeypatch.setattr(
        candidates,
        "completion_requirement_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("foundational candidates do not need requirement arbitration")
        ),
    )

    eligible, _, complete = arbitrate_completion_candidates(
        state,
        [candidate],
        decisions,
    )

    assert eligible == [candidate]
    assert complete is False


def _stub_empty_frontier_with_block_reasons(monkeypatch, state, reasons):
    inquiry_id = "USER_APP|thread|task-due-dates|stale-decision"
    candidate_id = f"question|{inquiry_id}"
    state["open_inquiries"] = [{
        "id": inquiry_id,
        "source": InquirySource.MODEL.value,
        "scope": SCOPE.value,
        "topic": DiscoveryTopic.BUSINESS_RULES.value,
        "objective": "A stale decision",
        "question_hint": "Ask about a decision already closed.",
        "reason": "Regression fixture.",
        "thread_id": "task-due-dates",
        "decision_key": "stale-decision",
    }]
    decisions = {
        candidate_id: {
            "candidate_id": candidate_id,
            "eligible": False,
            "reasons": reasons,
        }
    }

    def refresh(current):
        refreshed = {
            **current,
            "ranked_question_candidates": [],
            "eligible_question_candidates": [],
            "question_candidate_eligibility": decisions,
        }
        updates = {
            "open_inquiries": current["open_inquiries"],
            "question_candidates": [{"id": candidate_id}],
            "eligible_question_candidates": [],
            "question_candidate_eligibility": decisions,
            "ranked_question_candidates": [],
            "question_candidate_priority": {},
            "completion_arbitration_complete": False,
        }
        return refreshed, updates

    monkeypatch.setattr(planner, "_refresh_inquiry_frontier", refresh)


def test_planner_can_recover_when_only_terminally_blocked_inquiries_remain(monkeypatch):
    state = base_state()
    _stub_empty_frontier_with_block_reasons(
        monkeypatch,
        state,
        ["REPEATED_THREAD_DECISION", "EXPLICITLY_DEFERRED_DECISION"],
    )

    update = planner.interview_planner_node(state)

    assert update["prd_confirmation_pending"] is True
    assert update["ready_to_compile"] is False


def test_planner_still_fails_for_unexpected_candidate_blockers(monkeypatch):
    state = base_state()
    _stub_empty_frontier_with_block_reasons(
        monkeypatch,
        state,
        ["WRONG_SCOPE"],
    )

    with pytest.raises(
        RuntimeError,
        match="Open product inquiries exist but none survived candidate eligibility/prioritization",
    ):
        planner.interview_planner_node(state)


def test_completion_ready_can_reach_prd_confirmation_with_nonblocking_backlog(
    monkeypatch,
):
    state = base_state()
    req = requirement("optional_followup")
    state["active_requirements"] = {
        requirement_store_key(SCOPE, req.id): req,
    }
    state["open_inquiries"] = [
        {
            "id": "optional-open",
            "source": InquirySource.REQUIREMENT.value,
            "scope": SCOPE.value,
            "topic": DiscoveryTopic.CORE_WORKFLOW.value,
            "objective": "Optional follow-up detail",
            "question_hint": "Ask about optional detail.",
            "reason": "More detail is possible.",
        }
    ]
    state["ranked_question_candidates"] = []
    state["founder_requested_completion"] = True
    state["completion_request_evidence"] = "I think we have covered everything."
    state["completion_arbitration_complete"] = True
    state["validation_issues"] = []

    monkeypatch.setattr(
        planner,
        "_refresh_inquiry_frontier",
        lambda current: (
            {
                **current,
                "ranked_question_candidates": [],
                "completion_arbitration_complete": True,
            },
            {
                "open_inquiries": current["open_inquiries"],
                "question_candidates": [],
                "eligible_question_candidates": [],
                "question_candidate_eligibility": {},
                "ranked_question_candidates": [],
                "question_candidate_priority": {},
                "completion_arbitration_complete": True,
            },
        ),
    )

    update = planner.interview_planner_node(state)
    routed_state = {**state, **update}

    assert update["prd_confirmation_pending"] is True
    assert planner.all_discovery_resolved(routed_state) is True
    assert route_after_plan(routed_state) == "request_prd_confirmation"



def test_generated_question_cannot_jump_to_another_founder_named_gap():
    from agents.guardrails import founder_gap_objective_drift

    state = {
        "planner_source": "model",
        "discovery_scope": SCOPE,
        "current_objective": (
            "Clarify whether tasks should persist after closing and reopening the app."
        ),
        "question_hint": (
            "Should tasks remain saved and visible after the user closes and reopens "
            "the app, or should the task list reset each time?"
        ),
        "founder_gap_guidance": [
            {
                "scope": SCOPE.value,
                "items": [
                    "What exactly happens on delete",
                    "How tasks are ordered",
                    "Whether tasks should persist after closing/reopening the app",
                ],
            }
        ],
    }

    reason = founder_gap_objective_drift(
        state,
        "When a user deletes a task, should it delete immediately or ask for confirmation?",
    )

    assert reason is not None
    assert "another founder-named open gap" in reason


def test_generated_question_allows_selected_founder_gap():
    from agents.guardrails import founder_gap_objective_drift

    state = {
        "planner_source": "model",
        "discovery_scope": SCOPE,
        "current_objective": (
            "Clarify whether tasks should persist after closing and reopening the app."
        ),
        "question_hint": (
            "Should tasks remain saved and visible after the user closes and reopens "
            "the app, or should the task list reset each time?"
        ),
        "founder_gap_guidance": [
            {
                "scope": SCOPE.value,
                "items": [
                    "What exactly happens on delete",
                    "Whether tasks should persist after closing/reopening the app",
                ],
            }
        ],
    }

    assert founder_gap_objective_drift(
        state,
        "Should tasks remain saved after the user closes and reopens the app?",
    ) is None
