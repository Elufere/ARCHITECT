"""End-to-end integration scenarios across Architect's discovery reasoning pipeline."""
from types import SimpleNamespace
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from agents import consistency_validation as cv
from agents import validation_resolution as vr
from agents.consistency_validation import FactConflictBatch, FactConflictVerdict
from agents.discovery_coverage import fact_id
from agents.interview_checkpoint import load_checkpoint, save_checkpoint
from agents.implications import product_implication_node
from agents.inquiries import inquiry_identification_node
from agents.interview_planner import interview_planner_node
from agents.question_candidates import question_candidate_builder_node, question_candidate_filter_node
from agents.question_priority import question_candidate_priority_node
from agents.requirement_activation import requirement_activation_node
from agents.requirement_coverage import requirement_coverage_node
from agents.requirement_dependencies import requirement_dependency_node
from agents.requirements import RequirementStatus, requirement_store_key
from agents.state import (
    DiscoveryScope as S,
    DiscoveryTopic as T,
    KnowledgeItem,
    TopicMaturity,
)
from agents.validation_resolution import ConflictResolutionDecision, validation_resolution_node
from coverage_test_utils import coverage_for_facts
from discovery_invariant_utils import assert_discovery_invariants


def fact(topic, key, value, *, turn=1, role=None, roles=None, absence=None):
    return KnowledgeItem(
        topic=topic,
        scope=S.USER_APP,
        key=key,
        value=value,
        evidence=value,
        role=role,
        roles=roles,
        confidence=1,
        source_turn=turn,
        absence=absence,
    )


def state_with(*facts):
    return {
        "session_id": str(uuid4()),
        "messages": [],
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": list(facts),
        "superseded_knowledge": [],
        "model_implications": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "requirement_dependency_state": {},
        "eligible_requirement_keys": [],
        "question_candidates": [],
        "eligible_question_candidates": [],
        "question_candidate_eligibility": {},
        "ranked_question_candidates": [],
        "question_candidate_priority": {},
        "requirement_question_history": [],
        "open_inquiries": [],
        "selected_inquiry": None,
        "planner_source": "model",
        "selected_requirement_candidate": None,
        "selected_requirement_priority": None,
        "validation_issues": [],
        "validation_pair_cache": {},
        "validation_blocking": False,
        "validation_candidate_blocking": False,
        "selected_validation_issue": None,
        "gap_coverage": {},
        "fact_acquisition": {},
        "active_answer_result": None,
        "asked_gap": None,
        "topic_status": {},
        "topic_maturity": {},
        "current_topic": None,
        "current_gap": None,
        "turn_count": max([item.source_turn for item in facts] or [0]),
        "awaiting_confirmation": False,
        "pm_is_complete": False,
        "checkpoint_cursor": "infer_implications",
        "interview_status": "PROCESSING_REQUIREMENTS",
        "extraction_status": "COMMITTED",
    }


def run_reasoning_frontier(state):
    """Run the real post-extraction reasoning pipeline through planning."""
    for node in (
        product_implication_node,
        requirement_activation_node,
        requirement_coverage_node,
        requirement_dependency_node,
        cv.consistency_validation_node,
        inquiry_identification_node,
        question_candidate_builder_node,
        question_candidate_filter_node,
        question_candidate_priority_node,
    ):
        state.update(node(state))
    state.update(interview_planner_node(state))
    assert_discovery_invariants(state)
    return state


def mature(state, *topics):
    for topic in topics:
        state["topic_maturity"][topic] = TopicMaturity.COHERENT
    return state


def coherent_foundation():
    return [
        fact(T.USER_ROLES, "primary_users", "customers", roles=["customer"]),
        fact(T.USER_ROLES, "responsibilities", "create and manage transactions", role="customer"),
        fact(T.USER_GOALS, "primary_user_goals", "complete transactions safely", role="customer"),
        fact(T.CORE_WORKFLOW, "workflow_steps", "create, process, and complete a transaction"),
        fact(T.CORE_WORKFLOW, "completion_condition", "the transaction reaches its agreed completed state"),
    ]


def test_simple_task_product_uses_model_frontier_without_dynamic_requirements():
    actor = fact(T.USER_ROLES, "primary_users", "individual task owners", roles=["task owner"])
    state = state_with(actor)
    state["gap_coverage"] = coverage_for_facts(state, T.USER_ROLES)

    result = run_reasoning_frontier(state)

    assert result["active_requirements"] == {}
    assert len(result["ranked_question_candidates"]) == 1
    assert result["ranked_question_candidates"][0]["source"] == "MODEL"
    assert result["planner_source"] == "model"
    assert result["current_topic"] == T.USER_ROLES
    assert result["current_gap"] == "responsibilities::task owner"


def test_external_dependency_flows_from_fact_to_ranked_requirement_to_planner():
    dependency = fact(
        T.CORE_WORKFLOW,
        "downstream_dependency",
        "A bank approval is required before the workflow can finish.",
    )
    state = state_with(*coherent_foundation(), dependency)
    mature(state, T.CORE_WORKFLOW, T.BUSINESS_RULES)

    result = run_reasoning_frontier(state)

    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    requirement = result["active_requirements"][key]
    assert requirement.status == RequirementStatus.ACTIVE
    assert requirement.activation_sources[0].evidence_ref == fact_id(dependency)
    assert result["eligible_requirement_keys"] == [key]
    assert result["ranked_question_candidates"][0]["requirement_key"] == key
    assert result["planner_source"] == "requirement"
    assert result["selected_requirement_candidate"]["requirement_key"] == key
    assert result["current_topic"] == T.EXCEPTIONS
    assert result["current_gap"] == "recovery"


def test_appointment_time_constraint_activates_boundary_lifecycle_after_foundation():
    deadline = fact(
        T.CONSTRAINTS,
        "time_constraints",
        "An appointment invitation expires after 48 hours.",
    )
    state = state_with(*coherent_foundation(), deadline)

    result = run_reasoning_frontier(state)
    key = requirement_store_key(S.USER_APP, "lifecycle.time_boundary_behavior")
    assert key in result["active_requirements"]
    assert result["ranked_question_candidates"][0]["requirement_key"] == key
    assert result["planner_source"] == "requirement"
    assert result["current_topic"] == T.EDGE_CASES
    assert result["current_gap"] == "boundary_conditions"


def test_marketplace_mutability_activates_both_and_prioritizes_lower_question_cost():
    ownership = fact(
        T.BUSINESS_RULES,
        "ownership_rules",
        "Hosts can update active listings and archive listings they own.",
    )
    state = state_with(*coherent_foundation(), ownership)
    mature(state, T.CORE_WORKFLOW)

    result = run_reasoning_frontier(state)

    modification = requirement_store_key(S.USER_APP, "lifecycle.modification_behavior")
    removal = requirement_store_key(S.USER_APP, "lifecycle.removal_behavior")
    assert {modification, removal}.issubset(result["active_requirements"])
    ranked = [item["requirement_key"] for item in result["ranked_question_candidates"]]
    assert removal in ranked and modification in ranked
    assert ranked.index(modification) < ranked.index(removal)

    scores = result["question_candidate_priority"]
    modification_candidate = next(
        item for item in result["ranked_question_candidates"]
        if item["requirement_key"] == modification
    )
    removal_candidate = next(
        item for item in result["ranked_question_candidates"]
        if item["requirement_key"] == removal
    )
    modification_score = scores[modification_candidate["id"]]
    removal_score = scores[removal_candidate["id"]]

    # Removal has higher architecture/risk hints, but it asks about four
    # unresolved facets rather than three. Fix 7 intentionally applies a larger
    # breadth/question-cost penalty, which makes modification the better next
    # question in this otherwise-equal context.
    assert removal_score["components"]["business_risk"] > modification_score["components"]["business_risk"]
    assert removal_score["components"]["architecture_impact"] > modification_score["components"]["architecture_impact"]
    assert removal_score["components"]["question_cost_penalty"] > modification_score["components"]["question_cost_penalty"]
    assert modification_score["score"] > removal_score["score"]
    assert result["selected_requirement_candidate"]["requirement_key"] == modification
    assert result["planner_source"] == "requirement"


def test_contradiction_preempts_candidates_then_user_clarification_heals_graph(monkeypatch):
    five = fact(T.BUSINESS_RULES, "limits", "Maximum five active requests.", turn=1)
    ten = fact(T.BUSINESS_RULES, "limits", "Maximum ten active requests.", turn=2)
    state = state_with(five, ten)
    mature(state, T.CORE_WORKFLOW)

    class ConflictModel:
        def invoke(self, _):
            return FactConflictBatch(verdicts=[
                FactConflictVerdict(
                    pair_id=cv._pair_id(five, ten),
                    contradiction=True,
                    confidence=1,
                    explanation="The same active-request limit is both five and ten.",
                )
            ])

    monkeypatch.setattr(cv, "conflict_model", lambda: ConflictModel())
    conflicted = run_reasoning_frontier(state)

    assert conflicted["validation_blocking"] is True
    assert conflicted["validation_candidate_blocking"] is True
    assert len(conflicted["ranked_question_candidates"]) == 1
    assert conflicted["ranked_question_candidates"][0]["source"] == "VALIDATION"
    assert conflicted["planner_source"] == "validation"
    issue = conflicted["selected_validation_issue"]

    question = "I have limits of five and ten recorded. Which limit should apply now?"
    conflicted["messages"] = [
        AIMessage(content=question),
        HumanMessage(content="Use ten active requests."),
    ]
    conflicted["turn_count"] = 3

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=True,
            superseded_fact_ids=[fact_id(five)],
            retained_fact_ids=[fact_id(ten)],
            evidence="Use ten active requests.",
            confidence=1,
        )),
    )
    update = validation_resolution_node(conflicted)
    conflicted.update(update)

    # The stale pair is gone; the retained limit can still justify its lifecycle
    # requirement and the reasoning graph becomes usable again.
    healed = run_reasoning_frontier(conflicted)
    assert fact_id(five) not in {fact_id(item) for item in healed["discovered_knowledge"]}
    assert fact_id(ten) in {fact_id(item) for item in healed["discovered_knowledge"]}
    assert healed["validation_blocking"] is False
    assert healed["validation_candidate_blocking"] is False
    assert healed["planner_source"] != "validation"
    assert issue["fact_ids"]
    assert healed["superseded_knowledge"]


def test_checkpoint_roundtrip_preserves_reasoning_frontier(monkeypatch, tmp_path):
    ownership = fact(
        T.BUSINESS_RULES,
        "ownership_rules",
        "Hosts can update and archive listings they own.",
    )
    state = state_with(ownership)
    mature(state, T.CORE_WORKFLOW)
    state = run_reasoning_frontier(state)
    state["checkpoint_cursor"] = "waiting"
    state["interview_status"] = "WAITING_FOR_USER"

    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path))
    save_checkpoint(state)
    loaded = load_checkpoint(state["session_id"])

    assert loaded["checkpoint_cursor"] == "waiting"
    assert loaded["planner_source"] == state["planner_source"]
    assert loaded["selected_requirement_candidate"] == state["selected_requirement_candidate"]
    assert loaded["ranked_question_candidates"] == state["ranked_question_candidates"]
    assert loaded["question_candidate_priority"] == state["question_candidate_priority"]
    assert_discovery_invariants(loaded)
