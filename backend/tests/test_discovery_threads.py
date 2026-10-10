"""Discovery-thread planning keeps the interview causal and prevents semantic loops."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import discovery_threads as threads
from agents.inquiries import identify_open_inquiries
from agents.question_candidates import (
    CandidateBlockReason,
    QuestionCandidate,
    filter_question_candidates,
)
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem
from agents.requirements import ActiveRequirement, RequirementFacet, requirement_store_key
from agents.requirement_coverage import RequirementCoverageRecord, RequirementCoverageStatus
from agents.product_concepts import ProductConcept, ProductConceptKind


def fact(topic, key, value, *, turn=1):
    return KnowledgeItem(
        topic=topic,
        scope=S.USER_APP,
        key=key,
        value=value,
        evidence=value,
        confidence=1,
        source_turn=turn,
    )


def test_thread_planner_persists_active_thread_and_frontier(monkeypatch):
    response = {
        "thread_id": "core_transaction",
        "thread_label": "Core transaction",
        "thread_objective": "Understand how a deal moves from creation to completion.",
        "parent_thread_id": None,
        "frontier": {
            "decision_key": "transaction_initiation",
            "topic": "CORE_WORKFLOW",
            "anchor_gap": "workflow_steps",
            "objective": "Understand how one customer starts a transaction with another.",
            "question_hint": "Ask who creates the deal and how the other party joins.",
            "reason": "The actors are known but the transaction entry point is not.",
            "related_fact_ids": [],
            "information_gain": 0.95,
            "causal_relevance": 1,
            "conversation_continuity": 1,
            "architecture_impact": 0.8,
            "business_risk": 0.5,
            "question_cost": 0,
        },
        "relevant_requirement_ids": [],
        "rationale": "Continue the core transaction before exceptions.",
    }
    monkeypatch.setattr(
        threads,
        "thread_planner_model",
        lambda: SimpleNamespace(invoke=lambda _: response),
    )
    state = {
        "messages": [
            AIMessage(content="Who uses the app?"),
            HumanMessage(content="Customers can be buyers or sellers."),
        ],
        "raw_idea": "An escrow app.",
        "discovery_scope": S.USER_APP,
        "thread_planning_enabled": True,
        "discovered_knowledge": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "requirement_question_history": [],
        "discovery_threads": {},
        "active_discovery_thread": None,
        "turn_count": 1,
    }

    update = threads.discovery_thread_node(state)

    assert update["active_discovery_thread"] == "core_transaction"
    assert update["thread_frontier"]["decision_key"] == "transaction_initiation"
    assert update["thread_frontier"]["anchor_gap"] == "workflow_steps"
    assert update["discovery_threads"]["core_transaction"]["status"] == "ACTIVE"


def test_thread_frontier_replaces_legacy_actor_goal_sequence():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="primary_users",
                value="customer",
                evidence="Customers use the app.",
                roles=["customer"],
                confidence=1,
            )
        ],
        "fact_acquisition": {},
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "active_discovery_thread": "core_transaction",
        "thread_relevant_requirement_ids": [],
        "thread_frontier": {
            "thread_id": "core_transaction",
            "thread_label": "Core transaction",
            "thread_objective": "Understand the normal transaction.",
            "decision_key": "transaction_initiation",
            "topic": "CORE_WORKFLOW",
            "anchor_gap": "workflow_steps",
            "objective": "Understand how the transaction starts between customers.",
            "question_hint": "Ask who creates the deal and how the other joins.",
            "reason": "This is the next causal link.",
            "related_fact_ids": [],
            "information_gain": 1,
            "causal_relevance": 1,
            "conversation_continuity": 1,
            "architecture_impact": 0.8,
            "business_risk": 0.5,
            "question_cost": 0,
        },
    }

    inquiries = identify_open_inquiries(state)

    assert len(inquiries) == 1
    assert inquiries[0].thread_id == "core_transaction"
    assert inquiries[0].decision_key == "transaction_initiation"
    assert inquiries[0].anchor_gap == "workflow_steps"
    assert "actor_actions" not in inquiries[0].id
    assert "actor_goal" not in inquiries[0].id


def test_bounded_planner_fallback_releases_active_thread_to_foundational_inquiries(monkeypatch):
    state = {
        "messages": [
            AIMessage(content="Should reminders be rescheduled?"),
            HumanMessage(content="That decision has already been answered."),
        ],
        "raw_idea": "A personal to-do app.",
        "discovery_scope": S.USER_APP,
        "thread_planning_enabled": True,
        "discovered_knowledge": [],
        "product_concepts": [],
        "external_systems": [],
        "captured_observations": [],
        "founder_gap_guidance": [],
        "discovery_boundaries": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "requirement_question_history": [],
        "discovery_threads": {
            "task-due-dates": {
                "id": "task-due-dates",
                "label": "Due dates and reminders",
                "objective": "Define task due-date behavior.",
                "scope": S.USER_APP.value,
                "status": "ACTIVE",
                "trigger_fact_ids": [],
                "last_active_turn": 17,
            }
        },
        "active_discovery_thread": "task-due-dates",
        "thread_frontier": None,
        "thread_relevant_requirement_ids": [],
        "deferred_discovery_frontiers": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "founder_requested_completion": False,
        "turn_count": 18,
        "extraction_status": "SUCCESS",
    }

    def proposal(decision_key):
        return threads.DiscoveryThreadPlan(
            thread_id="task-due-dates",
            thread_label="Due dates and reminders",
            thread_objective="Define task due-date behavior.",
            frontier=threads.ThreadFrontierInquiry(
                decision_key=decision_key,
                topic=T.BUSINESS_RULES,
                objective="Determine a decision that has already been answered.",
                question_hint="Should the Architect repeat an already confirmed decision?",
                reason="This fixture simulates a stale frontier.",
            ),
        )

    proposals = iter([
        proposal("completed-task-reminders"),
        proposal("due-date-edit-reminder-rescheduling"),
        proposal("overdue-task-visibility"),
    ])
    monkeypatch.setattr(
        threads,
        "_invoke_thread_plan",
        lambda *_args, **_kwargs: next(proposals),
    )
    monkeypatch.setattr(
        threads,
        "_plan_problem",
        lambda *_args, **_kwargs: "Decision already received a usable answer.",
    )

    fallback = threads.plan_discovery_thread(state)

    assert fallback.frontier is None
    assert fallback._used_safe_fallback is True

    monkeypatch.setattr(threads, "plan_discovery_thread", lambda _: fallback)
    update = threads.discovery_thread_node(state)

    assert update["active_discovery_thread"] is None
    assert update["thread_frontier"] is None
    assert update["discovery_threads"]["task-due-dates"]["status"] == "PAUSED"

    inquiries = identify_open_inquiries({**state, **update})
    assert [item.id for item in inquiries] == ["USER_APP|model.core_actors"]


def test_pre_thread_state_can_still_use_minimal_foundational_actor_inquiry():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "active_discovery_thread": None,
        "thread_relevant_requirement_ids": [],
        "thread_frontier": None,
    }

    inquiries = identify_open_inquiries(state)

    assert len(inquiries) == 1
    assert inquiries[0].id == "USER_APP|model.core_actors"
    assert inquiries[0].requirement_id is None


def test_no_thread_frontier_does_not_restore_legacy_checklist_after_thread_planning():
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "active_discovery_thread": "core_transaction",
        "thread_relevant_requirement_ids": [],
        "thread_frontier": None,
    }

    assert identify_open_inquiries(state) == []


def test_same_thread_decision_is_hard_blocked_after_clear_answer():
    candidate = QuestionCandidate(
        id="question|core",
        inquiry_id="USER_APP|thread|core_transaction|transaction_initiation",
        source="MODEL",
        scope=S.USER_APP,
        topic=T.CORE_WORKFLOW,
        anchor_gap="workflow_steps",
        objective="Understand how a transaction starts.",
        thread_id="core_transaction",
        decision_key="transaction_initiation",
    )
    state = {
        "discovery_scope": S.USER_APP,
        "question_candidates": [candidate.model_dump(mode="json")],
        "open_inquiries": [{
            "id": candidate.inquiry_id,
            "source": "MODEL",
            "scope": S.USER_APP.value,
            "topic": T.CORE_WORKFLOW.value,
            "anchor_gap": "workflow_steps",
            "objective": candidate.objective,
            "question_hint": "Ask how it starts.",
            "reason": "Next causal link.",
            "thread_id": "core_transaction",
            "decision_key": "transaction_initiation",
        }],
        "requirement_question_history": [{
            "inquiry_id": candidate.inquiry_id,
            "thread_id": "core_transaction",
            "decision_key": "transaction_initiation",
            "question": "How does a transaction start?",
            "turn": 2,
        }],
        "extraction_status": "SUCCESS",
    }

    eligible, decisions = filter_question_candidates(state, [candidate])

    assert eligible == []
    assert CandidateBlockReason.REPEATED_THREAD_DECISION in decisions[candidate.id].reasons


def test_one_rephrase_is_allowed_only_when_previous_answer_produced_no_facts():
    candidate = QuestionCandidate(
        id="question|core",
        inquiry_id="USER_APP|thread|core_transaction|transaction_initiation",
        source="MODEL",
        scope=S.USER_APP,
        topic=T.CORE_WORKFLOW,
        anchor_gap="workflow_steps",
        objective="Understand how a transaction starts.",
        thread_id="core_transaction",
        decision_key="transaction_initiation",
    )
    state = {
        "discovery_scope": S.USER_APP,
        "requirement_question_history": [{
            "inquiry_id": candidate.inquiry_id,
            "thread_id": "core_transaction",
            "decision_key": "transaction_initiation",
            "question": "How does a transaction start?",
            "turn": 2,
        }],
        "open_inquiries": [{
            "id": candidate.inquiry_id,
            "source": "MODEL",
            "scope": S.USER_APP.value,
            "topic": T.CORE_WORKFLOW.value,
            "anchor_gap": "workflow_steps",
            "objective": candidate.objective,
            "question_hint": "Ask how it starts.",
            "reason": "Next causal link.",
            "thread_id": "core_transaction",
            "decision_key": "transaction_initiation",
        }],
        "extraction_status": "NO_FACTS_FOUND",
    }

    eligible, _ = filter_question_candidates(state, [candidate])

    assert eligible == [candidate]


def test_unrelated_active_requirement_is_deferred_while_thread_frontier_exists():
    requirement = ActiveRequirement(
        id="dispute.evidence_collection",
        scope=S.USER_APP,
        topic=T.EXCEPTIONS,
        parent_gap="invalid_actions",
        label="Dispute evidence collection",
        facets=[
            RequirementFacet(
                id="evidence_submission",
                label="Evidence submission",
                description="How dispute evidence is submitted.",
            ),
        ],
    )
    key = requirement_store_key(S.USER_APP, requirement.id)
    coverage = RequirementCoverageRecord(
        requirement_id=requirement.id,
        scope=S.USER_APP,
        status=RequirementCoverageStatus.UNSEEN,
        facets={},
    )
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "active_requirements": {key: requirement},
        "requirement_coverage": {key: coverage.model_dump(mode="json")},
        "eligible_requirement_keys": [key],
        "active_discovery_thread": "core_transaction",
        "thread_relevant_requirement_ids": [],
        "thread_frontier": {
            "thread_id": "core_transaction",
            "thread_label": "Core transaction",
            "thread_objective": "Understand the normal transaction path.",
            "decision_key": "transaction_initiation",
            "topic": T.CORE_WORKFLOW.value,
            "anchor_gap": "workflow_steps",
            "objective": "Understand how one customer starts a transaction with another.",
            "question_hint": "Ask who creates the deal and how the other party joins.",
            "reason": "The normal path is not yet coherent.",
            "related_fact_ids": [],
            "information_gain": 1,
            "causal_relevance": 1,
            "conversation_continuity": 1,
            "architecture_impact": 0.8,
            "business_risk": 0.5,
            "question_cost": 0,
        },
    }

    inquiries = identify_open_inquiries(state)

    assert len(inquiries) == 1
    assert inquiries[0].thread_id == "core_transaction"
    assert inquiries[0].decision_key == "transaction_initiation"
    assert all(inquiry.requirement_id != requirement.id for inquiry in inquiries)


def test_only_requirements_marked_relevant_by_thread_are_askable():
    relevant = ActiveRequirement(
        id="transaction.term_agreement",
        scope=S.USER_APP,
        topic=T.BUSINESS_RULES,
        parent_gap="approval_rules",
        label="Term agreement",
        facets=[
            RequirementFacet(
                id="agreement_condition",
                label="Agreement condition",
                description="What must be agreed before the transaction proceeds.",
            ),
        ],
    )
    deferred = ActiveRequirement(
        id="dispute.evidence_collection",
        scope=S.USER_APP,
        topic=T.EXCEPTIONS,
        parent_gap="invalid_actions",
        label="Dispute evidence collection",
        facets=[
            RequirementFacet(
                id="evidence_submission",
                label="Evidence submission",
                description="How dispute evidence is submitted.",
            ),
        ],
    )
    relevant_key = requirement_store_key(S.USER_APP, relevant.id)
    deferred_key = requirement_store_key(S.USER_APP, deferred.id)
    coverage = lambda req: RequirementCoverageRecord(
        requirement_id=req.id,
        scope=S.USER_APP,
        status=RequirementCoverageStatus.UNSEEN,
        facets={},
    ).model_dump(mode="json")
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
        "answer_followup": None,
        "active_requirements": {
            relevant_key: relevant,
            deferred_key: deferred,
        },
        "requirement_coverage": {
            relevant_key: coverage(relevant),
            deferred_key: coverage(deferred),
        },
        "eligible_requirement_keys": [relevant_key, deferred_key],
        "active_discovery_thread": "core_transaction",
        "thread_relevant_requirement_ids": [relevant.id],
        "thread_frontier": None,
    }

    inquiries = identify_open_inquiries(state)

    assert [inquiry.requirement_id for inquiry in inquiries] == [relevant.id]


def test_thread_planner_receives_latest_model_delta(monkeypatch):
    captured = {}
    response = {
        "thread_id": "event_structure",
        "thread_label": "Event structure",
        "thread_objective": "Understand how an event organizes what guests can buy.",
        "parent_thread_id": None,
        "frontier": {
            "decision_key": "group_meaning",
            "topic": "CORE_WORKFLOW",
            "anchor_gap": "workflow_steps",
            "objective": "Understand what groups represent inside an event.",
            "question_hint": "Ask what the groups represent.",
            "reason": "The founder just introduced groups as a new structural concept.",
            "related_fact_ids": [],
            "information_gain": 1,
            "causal_relevance": 1,
            "conversation_continuity": 1,
            "architecture_impact": 0.8,
            "business_risk": 0.4,
            "question_cost": 0,
        },
        "relevant_requirement_ids": [],
        "rationale": "Follow the newly introduced group structure.",
    }

    class Planner:
        def invoke(self, messages):
            captured["payload"] = json.loads(messages[-1].content)
            return response

    monkeypatch.setattr(
        threads,
        "thread_planner_model",
        lambda: Planner(),
    )
    workflow = fact(
        T.CORE_WORKFLOW,
        "workflow_steps",
        "Guests enter an event and see groups.",
        turn=4,
    )
    old_fact = fact(T.CORE_WORKFLOW, "workflow_steps", "Guests are invited.", turn=3)
    group = ProductConcept(
        kind=ProductConceptKind.ENTITY,
        scope=S.USER_APP,
        subject="group",
        value="There are groups.",
        evidence="There are groups.",
        confidence=1,
        source_turn=4,
    )
    event = ProductConcept(
        kind=ProductConceptKind.ENTITY,
        scope=S.USER_APP,
        subject="event",
        value="Purchasing is tied to an event.",
        evidence="Purchasing is tied to an event.",
        confidence=1,
        source_turn=2,
    )
    state = {
        "messages": [
            AIMessage(content="Does an event contain another level of organization?"),
            HumanMessage(content="There are groups."),
        ],
        "raw_idea": "An event commerce app.",
        "discovery_scope": S.USER_APP,
        "thread_planning_enabled": True,
        "discovered_knowledge": [old_fact, workflow],
        "product_concepts": [
            event.model_dump(mode="json"),
            group.model_dump(mode="json"),
        ],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "requirement_question_history": [],
        "discovery_threads": {},
        "active_discovery_thread": "event_structure",
        "turn_count": 4,
        "extraction_status": "SUCCESS",
    }

    threads.discovery_thread_node(state)

    assert [item["value"] for item in captured["payload"]["new_confirmed_facts_this_turn"]] == [
        "Guests enter an event and see groups."
    ]
    assert [item["subject"] for item in captured["payload"]["new_product_concepts_this_turn"]] == [
        "group"
    ]


def test_invalid_frontier_anchor_degrades_to_none_instead_of_failing():
    frontier = threads.ThreadFrontierInquiry.model_validate({
        "decision_key": "Group Access",
        "topic": "BUSINESS_RULES",
        "anchor_gap": "workflow_steps",
        "objective": "Understand how group access works.",
        "question_hint": "Ask who can see a group's packages.",
        "reason": "Access was introduced by the latest answer.",
    })

    assert frontier.decision_key == "group_access"
    assert frontier.anchor_gap is None


def test_thread_planner_repairs_malformed_first_structured_response(monkeypatch):
    calls = []
    valid = {
        "thread_id": "access",
        "thread_label": "Access",
        "thread_objective": "Understand how invited users gain access to packages.",
        "parent_thread_id": None,
        "frontier": {
            "decision_key": "group_access",
            "topic": "BUSINESS_RULES",
            "anchor_gap": "visibility_rules",
            "objective": "Understand how group membership controls package visibility.",
            "question_hint": "Ask what determines which groups and packages an invited user can see.",
            "reason": "The latest answer introduced invitation-gated visibility.",
            "related_fact_ids": [],
            "information_gain": 0.9,
            "causal_relevance": 1,
            "conversation_continuity": 1,
            "architecture_impact": 0.7,
            "business_risk": 0.5,
            "question_cost": 0,
        },
        "relevant_requirement_ids": [],
        "rationale": "Follow the newly introduced access rule.",
    }

    class Planner:
        def invoke(self, _messages):
            calls.append(1)
            if len(calls) == 1:
                return {
                    "thread_id": "Access Thread",
                    "thread_label": "Access",
                    # Missing thread_objective/frontier/rationale makes the
                    # first structured response invalid.
                }
            return valid

    monkeypatch.setattr(threads, "thread_planner_model", lambda: Planner())
    state = {
        "messages": [
            AIMessage(content="How does someone get access to packages?"),
            HumanMessage(content="Users are invited to a group before they can see packages."),
        ],
        "raw_idea": "An ecommerce app for packages.",
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "product_concepts": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "requirement_question_history": [],
        "discovery_threads": {},
        "active_discovery_thread": "core_workflow",
        "turn_count": 1,
        "extraction_status": "SUCCESS",
    }

    plan = threads.plan_discovery_thread(state)

    assert len(calls) == 2
    assert plan.thread_id == "access"
    assert plan.frontier.decision_key == "group_access"



def _simple_todo_depth_state(*, with_creation_shape=True, material=False):
    knowledge = [
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="primary_users",
            value="individual user",
            evidence="individual user",
            roles=["user"],
            confidence=1,
            source_turn=0,
        ),
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="secondary_users",
            value="none",
            evidence="There are no other user roles",
            roles=[],
            confidence=1,
            source_turn=0,
            absence="none",
        ),
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="responsibilities",
            value="A user can create tasks",
            evidence="A user can create tasks",
            role="user",
            confidence=1,
            source_turn=0,
        ),
    ]
    if with_creation_shape:
        knowledge.append(
            KnowledgeItem(
                topic=T.BUSINESS_RULES,
                scope=S.USER_APP,
                key="validation_rules",
                value="The task title must not be empty",
                evidence="The title cannot be empty",
                confidence=1,
                source_turn=1,
            )
        )
    if material:
        knowledge.append(
            KnowledgeItem(
                topic=T.CORE_WORKFLOW,
                scope=S.USER_APP,
                key="workflow_steps",
                value="The buyer funds payment before completion",
                evidence="The buyer funds payment before completion",
                confidence=1,
                source_turn=1,
            )
        )
    return {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": knowledge,
        "external_systems": [],
        "captured_observations": [],
        "discovery_boundaries": [],
    }


def _frontier(decision_key, objective, hint, reason="Unspecified product behavior."):
    return threads.DiscoveryThreadPlan(
        thread_id="core_task_management",
        thread_label="Core task management",
        thread_objective="Understand the task product.",
        frontier=threads.ThreadFrontierInquiry(
            decision_key=decision_key,
            topic=T.CORE_WORKFLOW,
            objective=objective,
            question_hint=hint,
            reason=reason,
        ),
    )


def test_one_creation_shape_question_is_allowed_before_shape_is_known():
    plan = _frontier(
        "task_creation_details",
        "Clarify how a user creates a task.",
        "What information is required when creating a task, and what validation applies?",
    )

    assert threads._low_signal_crud_depth_frontier(
        plan,
        _simple_todo_depth_state(with_creation_shape=False),
        S.USER_APP,
    ) is False


@pytest.mark.parametrize(
    "decision_key,objective,hint",
    [
        (
            "task_editing_behavior",
            "Clarify how users edit tasks.",
            "What fields can users edit and what validations or restrictions apply?",
        ),
        (
            "historical_effect_of_modifications",
            "Clarify historical effects of editing.",
            "Should edits keep history or simply update the task without past versions?",
        ),
        (
            "removal_lifecycle",
            "Clarify task deletion behavior.",
            "Should deleted tasks be permanent, archived, or restorable after a grace period?",
        ),
        (
            "task_completion_reversibility",
            "Clarify completion behavior.",
            "Can a completed task be reopened or undone after completion?",
        ),
        (
            "task_list_view_behavior",
            "Clarify how tasks are viewed.",
            "How should active and completed tasks be displayed and what interactions are available?",
        ),
    ],
)
def test_low_risk_single_actor_product_rejects_crud_policy_drilling(
    decision_key,
    objective,
    hint,
):
    plan = _frontier(decision_key, objective, hint)

    assert threads._low_signal_crud_depth_frontier(
        plan,
        _simple_todo_depth_state(),
        S.USER_APP,
    ) is True


def test_material_payment_context_does_not_trigger_low_risk_crud_suppression():
    plan = _frontier(
        "transaction_editing",
        "Clarify editing after payment.",
        "Can users edit transaction terms after funding, and what restrictions apply?",
    )

    assert threads._low_signal_crud_depth_frontier(
        plan,
        _simple_todo_depth_state(material=True),
        S.USER_APP,
    ) is False



def test_known_create_action_blocks_redundant_entry_point_question():
    plan = _frontier(
        "task_entry_point",
        "Clarify the first meaningful action in the task workflow.",
        "What is the first meaningful action the user takes to begin managing tasks?",
    )

    assert threads._low_signal_crud_depth_frontier(
        plan,
        _simple_todo_depth_state(),
        S.USER_APP,
    ) is True


def test_low_risk_product_rejects_completion_timing_microdecision():
    plan = _frontier(
        "task_creation_completion_interaction",
        "Clarify completion timing during task creation.",
        "Can the user mark a task completed immediately during creation?",
    )

    assert threads._low_signal_crud_depth_frontier(
        plan,
        _simple_todo_depth_state(),
        S.USER_APP,
    ) is True



def test_founder_gap_guidance_payload_preserves_open_questions_as_control_only():
    state = {
        "founder_gap_guidance": [
            {
                "scope": S.USER_APP.value,
                "source_turn": 4,
                "evidence": (
                    "Whether completed tasks can still be edited or deleted. "
                    "How tasks are ordered."
                ),
                "items": [
                    "Whether completed tasks can still be edited or deleted",
                    "How tasks should be ordered",
                ],
                "instruction": (
                    "Founder identified these as unresolved areas to consider."
                ),
            }
        ]
    }

    payload = threads._gap_guidance_payload(state, S.USER_APP)

    assert payload == state["founder_gap_guidance"]
    assert "edited or deleted" in payload[0]["items"][0]



def _todo_completion_state(*, include_goal=False, include_title=False):
    facts = [
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="primary_users",
            value="user",
            evidence="A user can manage tasks.",
            roles=["user"],
            confidence=1,
        ),
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="secondary_users",
            value="none",
            evidence="There are no other user roles.",
            roles=[],
            absence="none",
            confidence=1,
        ),
        KnowledgeItem(
            topic=T.USER_ROLES,
            scope=S.USER_APP,
            key="responsibilities",
            value="create, edit, delete, and complete tasks",
            evidence="A user can create tasks, edit or delete them, and mark them as completed.",
            role="user",
            confidence=1,
        ),
    ]
    if include_goal:
        facts.append(
            KnowledgeItem(
                topic=T.USER_GOALS,
                scope=S.USER_APP,
                key="primary_user_goals",
                value="keep track of things they need to do",
                evidence="keep track of things they need to do",
                role="user",
                confidence=1,
            )
        )

    concepts = [
        ProductConcept(
            kind=ProductConceptKind.ENTITY,
            scope=S.USER_APP,
            subject="task",
            value="Task",
            evidence="tasks",
            confidence=1,
        ),
        ProductConcept(
            kind=ProductConceptKind.ATTRIBUTE,
            scope=S.USER_APP,
            subject="task",
            relation="status",
            object="active or completed",
            value="A task can be active or completed.",
            evidence="Tasks can be either active or completed.",
            confidence=1,
        ),
    ]
    if include_title:
        concepts.append(
            ProductConcept(
                kind=ProductConceptKind.ATTRIBUTE,
                scope=S.USER_APP,
                subject="task",
                relation="title",
                object="title",
                value="A task has a title.",
                evidence="Each task has a title.",
                confidence=1,
            )
        )

    return {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": facts,
        "product_concepts": [item.model_dump(mode="json") for item in concepts],
        "requirement_question_history": [],
        "discovery_boundaries": [],
        "captured_observations": [],
        "founder_gap_guidance": [],
        "external_systems": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "discovery_threads": {},
        "messages": [],
        "turn_count": 0,
    }


def _null_plan():
    return threads.DiscoveryThreadPlan(
        thread_id="mvp_complete",
        thread_label="MVP complete",
        thread_objective="Determine whether discovery can finish.",
        frontier=None,
        rationale="No material inquiry remains.",
    )


def test_null_frontier_rejected_when_product_has_no_founder_outcome():
    problem = threads._plan_problem(
        _null_plan(),
        _todo_completion_state(include_goal=False),
        [],
    )

    assert problem is not None
    assert "user outcome" in problem.lower() or "product goal" in problem.lower()


def test_null_frontier_rejected_when_created_entity_shape_is_undefined():
    problem = threads._plan_problem(
        _null_plan(),
        _todo_completion_state(include_goal=True, include_title=False),
        [],
    )

    assert problem is not None
    assert "entity" in problem.lower()
    assert "information" in problem.lower() or "shape" in problem.lower()


def test_foundational_completion_guard_releases_once_outcome_and_entity_shape_exist():
    problem = threads._plan_problem(
        _null_plan(),
        _todo_completion_state(include_goal=True, include_title=True),
        [],
    )

    assert problem is None


def test_completed_task_editability_is_not_hard_rejected_as_crud_depth(monkeypatch):
    state = _todo_completion_state(include_goal=True, include_title=True)
    frontier = threads.ThreadFrontierInquiry(
        decision_key="completed_task_editability",
        topic=T.CORE_WORKFLOW,
        anchor_gap=None,
        objective="Clarify whether completed tasks can still be edited.",
        question_hint="After a task is completed, can the user still edit it?",
        reason="This changes the task lifecycle and availability of the edit action.",
    )
    plan = threads.DiscoveryThreadPlan(
        thread_id="task_lifecycle",
        thread_label="Task lifecycle",
        thread_objective="Understand task state behavior.",
        frontier=frontier,
        rationale="Completed is a confirmed state with unresolved behavior.",
    )
    assessment = {
        "supporting_observation_ids": [],
        "recent_answer_supports": False,
        "information_need_resolved": False,
        "missing_information": ["whether completed tasks remain editable"],
        "too_broad": False,
        "recap_of_known_information": False,
        "should_move_on": False,
        "repeats_rejected_frontier": False,
        "repeats_prior_decision": False,
        "matching_prior_question": "",
        "abstraction_level": "PRODUCT_BEHAVIOR",
        "material_product_consequence": True,
        "current_frontier_value": 0.8,
        "best_alternative_value": 0.0,
        "higher_value_elsewhere": False,
        "best_alternative_focus": "",
        "depth_reason": "Editability changes the completed-state product contract.",
        "reason": "The founder has not defined completed-task editability.",
    }
    monkeypatch.setattr(
        threads,
        "inquiry_assessment_model",
        lambda: SimpleNamespace(invoke=lambda _: assessment),
    )

    assert threads._semantic_frontier_problem(
        plan,
        state,
        S.USER_APP,
    ) is None



def test_compact_planner_snapshot_preserves_semantics_without_evidence_duplication():
    state = {
        "discovered_knowledge": [
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="primary_users",
                value="user",
                evidence="A user can create tasks.",
                roles=["user"],
                confidence=1,
            ),
            KnowledgeItem(
                topic=T.USER_ROLES,
                scope=S.USER_APP,
                key="secondary_users",
                value="none",
                evidence="There are no other user roles.",
                roles=[],
                absence="none",
                confidence=1,
            ),
            KnowledgeItem(
                topic=T.USER_GOALS,
                scope=S.USER_APP,
                key="primary_user_goals",
                value="keep track of things they need to do",
                evidence="keep track of things they need to do",
                role="user",
                confidence=1,
            ),
        ],
        "product_concepts": [],
        "external_systems": [],
    }

    snapshot = threads._compact_product_snapshot(state, S.USER_APP)
    encoded = json.dumps(snapshot)

    assert "keep track of things they need to do" in encoded
    assert "explicit absence: none" in encoded
    assert "There are no other user roles." not in encoded
    assert "fact_id" not in encoded
    assert "source_turn" not in encoded
