"""Discovery-thread planning keeps the interview causal and prevents semantic loops."""

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
