"""Discovery-thread planning keeps the interview causal and prevents semantic loops."""

import json
from types import SimpleNamespace

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
