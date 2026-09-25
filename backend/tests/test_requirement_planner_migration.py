"""Focused tests for requirement planning inside the model-driven inquiry frontier."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agents import guardrails, interview_planner, question_generator, requirement_coverage
from agents.discovery_coverage import fact_id
from agents.question_candidates import QuestionCandidate
from agents.requirement_coverage import (
    RequirementCoverageAssessment,
    RequirementCoverageStatus,
)
from agents.requirements import (
    ActiveRequirement,
    RequirementFacet,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import (
    DiscoveryScope as S,
    DiscoveryTopic as T,
    KnowledgeItem,
    KnowledgeState,
    TopicMaturity,
)


def active_requirement(*, status=RequirementStatus.ACTIVE):
    return ActiveRequirement(
        id="workflow.external_dependency_failure",
        scope=S.USER_APP,
        topic=T.EXCEPTIONS,
        parent_gap="recovery",
        label="External dependency failure",
        description="Determine what happens when an external dependency fails.",
        status=status,
        facets=[
            RequirementFacet(
                id="failure_condition",
                label="Failure condition",
                description="What failure condition matters.",
            ),
            RequirementFacet(
                id="expected_behavior",
                label="Expected behavior",
                description="What the product should do next.",
            ),
        ],
    )


def candidate(requirement=None):
    requirement = requirement or active_requirement()
    key = requirement_store_key(requirement.scope, requirement.id)
    return QuestionCandidate(
        id=f"{key}|failure_condition,expected_behavior",
        requirement_key=key,
        requirement_id=requirement.id,
        scope=requirement.scope,
        topic=requirement.topic,
        objective=requirement.description,
        target_facets=["failure_condition", "expected_behavior"],
        known_fact_ids=[],
        activation_rule_ids=["test.rule"],
        coverage_status=RequirementCoverageStatus.UNSEEN,
    )


def planner_state(requirement=None, ranked=True):
    requirement = requirement or active_requirement()
    cand = candidate(requirement)
    key = cand.requirement_key
    return {
        "discovery_scope": S.USER_APP,
        "active_requirements": {key: requirement},
        "ranked_question_candidates": [cand.model_dump(mode="json")] if ranked else [],
        "question_candidate_priority": {
            cand.id: {"candidate_id": cand.id, "score": 0.8, "components": {}, "rationale": []}
        },
        "discovered_knowledge": [],
        "gap_coverage": {},
        "topic_status": {},
        "topic_maturity": {
            T.CORE_WORKFLOW: TopicMaturity.COHERENT,
            T.BUSINESS_RULES: TopicMaturity.COHERENT,
        },
        "active_answer_result": None,
        "current_topic": None,
        "planner_source": "model",
    }


def test_ranked_requirement_candidate_is_selected_from_inquiry_frontier():
    state = planner_state()
    result = interview_planner.interview_planner_node(state)
    assert result["planner_source"] == "requirement"
    assert result["selected_requirement_candidate"]["requirement_id"] == "workflow.external_dependency_failure"
    assert result["current_topic"] == T.EXCEPTIONS
    assert result["current_gap"] == "recovery"
    assert result["next_discovery_move"] == "requirement_discovery"


def test_model_frontier_replaces_schema_fallback_when_no_requirement_candidate_exists():
    state = planner_state(ranked=False)
    state["active_requirements"] = {}
    result = interview_planner.interview_planner_node(state)
    assert result["planner_source"] == "model"
    assert result["selected_requirement_candidate"] is None
    assert result["current_topic"] == T.USER_ROLES
    assert result["current_gap"] == "primary_users"


def test_active_requirement_blocks_combined_completion(monkeypatch):
    state = planner_state()
    monkeypatch.setattr(interview_planner, "all_required_gaps_resolved", lambda _: True)
    assert interview_planner.all_discovery_resolved(state) is False

    req = active_requirement(status=RequirementStatus.RESOLVED)
    state["active_requirements"] = {requirement_store_key(S.USER_APP, req.id): req}
    assert interview_planner.all_discovery_resolved(state) is True


def test_requirement_answer_resolves_facets_from_grounded_fact(monkeypatch):
    req = active_requirement()
    cand = candidate(req)
    key = cand.requirement_key
    question = "If the external approval fails, what should the app do next?"
    answer = "If approval fails, retry once and then stop the workflow."
    item = KnowledgeItem(
        topic=T.EXCEPTIONS,
        scope=S.USER_APP,
        key="recovery",
        value="If approval fails, retry once and then stop the workflow.",
        evidence=answer,
        source_question=question,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=1,
    )
    identity = fact_id(item)
    decision = RequirementCoverageAssessment(
        covered_facets={
            "failure_condition": [identity],
            "expected_behavior": [identity],
        }
    )
    monkeypatch.setattr(
        requirement_coverage,
        "requirement_coverage_assessor",
        lambda: SimpleNamespace(invoke=lambda _: decision),
    )
    state = {
        "planner_source": "requirement",
        "selected_requirement_candidate": cand.model_dump(mode="json"),
        "discovery_scope": S.USER_APP,
        "active_requirements": {key: req},
        "requirement_coverage": {},
        "discovered_knowledge": [item],
        "turn_count": 1,
        "messages": [AIMessage(content=question), HumanMessage(content=answer)],
        "gap_coverage": {},
    }
    result = requirement_coverage.requirement_coverage_node(state)
    assert result["active_requirements"][key].status == RequirementStatus.RESOLVED
    assert result["requirement_coverage"][key]["status"] == RequirementCoverageStatus.RESOLVED.value
    assert result["requirement_coverage"][key]["facets"]["failure_condition"]["fact_ids"] == [identity]
    assert result["requirement_coverage"][key]["facets"]["expected_behavior"]["fact_ids"] == [identity]


def test_requirement_question_receipt_does_not_resolve_parent_schema_gap():
    # Requirement questions never consume legacy schema coverage receipts.
    # Mark the requirement resolved here so this focused planner call does not
    # require a fabricated coverage/dependency frontier.
    req = active_requirement(status=RequirementStatus.RESOLVED)
    key = requirement_store_key(S.USER_APP, req.id)
    item = KnowledgeItem(
        topic=T.EXCEPTIONS,
        scope=S.USER_APP,
        key="recovery",
        value="Retry once.",
        evidence="Retry once.",
        confidence=1,
        source_turn=1,
    )
    identity = fact_id(item)
    state = {
        "planner_source": "requirement",
        "selected_requirement_candidate": candidate(req).model_dump(mode="json"),
        "discovery_scope": S.USER_APP,
        "active_requirements": {key: req},
        "ranked_question_candidates": [],
        "question_candidate_priority": {},
        "discovered_knowledge": [item],
        "gap_coverage": {},
        "topic_status": {},
        "topic_maturity": {},
        "current_topic": T.EXCEPTIONS,
        "current_gap": "recovery",
        "turn_count": 1,
        "asked_gap": {
            "scope": S.USER_APP.value,
            "topic": T.EXCEPTIONS.value,
            "gap": "recovery",
            "question": "What happens after failure?",
        },
        "active_answer_result": {
            "scope": S.USER_APP.value,
            "topic": T.EXCEPTIONS.value,
            "gap": "recovery",
            "source_turn": 1,
            "fact_ids": [identity],
            "resolution": "DIRECT_ANSWER",
        },
    }
    result = interview_planner.interview_planner_node(state)
    assert result["gap_coverage"] == {}


def test_requirement_generator_receives_requirement_and_facet_context(monkeypatch):
    req = active_requirement()
    cand = candidate(req)
    prompts = []

    def invoke(messages):
        prompts.extend(messages)
        return AIMessage(content="If the external approval fails, what should the app do next?")

    monkeypatch.setattr(
        question_generator,
        "get_chat_model",
        lambda **_: SimpleNamespace(invoke=invoke),
    )
    state = {
        **planner_state(req),
        "planner_source": "requirement",
        "selected_requirement_candidate": cand.model_dump(mode="json"),
        "current_topic": T.EXCEPTIONS,
        "current_gap": "recovery",
        "current_objective": req.description,
        "question_hint": "Ask about the failure behavior.",
        "next_discovery_move": "requirement_discovery",
        "current_role": None,
        "relevant_context": [],
        "messages": [HumanMessage(content="It depends on an external approval.")],
        "product_model": {},
        "question_retry_count": 0,
    }
    result = question_generator.question_generator_node(state)
    assert result["messages"][0].content.endswith("?")
    system = prompts[0].content
    assert "REQUIREMENT-DRIVEN DISCOVERY" in system
    assert "External dependency failure" in system
    assert "Failure condition" in system
    assert "Expected behavior" in system


def test_approved_requirement_question_is_recorded_for_priority_history(monkeypatch):
    req = active_requirement()
    cand = candidate(req)
    monkeypatch.setattr(
        guardrails,
        "evaluator_llm",
        SimpleNamespace(invoke=lambda _: SimpleNamespace(
            passed=True,
            stage=guardrails.Stage.PRODUCT_DISCOVERY,
            explanation="",
            guidance="",
        )),
    )
    question = "If the external approval fails, what should the app do next?"
    state = {
        "messages": [HumanMessage(content="It depends on approval."), AIMessage(content=question)],
        "planner_source": "requirement",
        "selected_requirement_candidate": cand.model_dump(mode="json"),
        "current_topic": T.EXCEPTIONS,
        "current_gap": "recovery",
        "current_objective": req.description,
        "question_hint": "Ask about failure behavior.",
        "current_role": None,
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "requirement_question_history": [],
        "turn_count": 3,
        "question_retry_count": 0,
    }
    result = guardrails.guardrail_node(state)
    assert result["requirement_question_history"][0]["candidate_id"] == cand.id
    assert result["requirement_question_history"][0]["question"] == question


def test_requirement_candidate_does_not_wait_for_schema_topic_prerequisites():
    state = planner_state()
    state["topic_maturity"] = {}
    result = interview_planner.interview_planner_node(state)
    assert result["planner_source"] == "requirement"
    assert result["current_topic"] == T.EXCEPTIONS


def test_model_frontier_resumes_after_requirement_move():
    state = planner_state(ranked=False)
    state["planner_source"] = "requirement"
    state["current_topic"] = T.EXCEPTIONS
    state["topic_maturity"] = {}
    state["active_requirements"] = {}
    result = interview_planner.interview_planner_node(state)
    assert result["planner_source"] == "model"
    assert result["current_topic"] == T.USER_ROLES
    assert result["current_gap"] == "primary_users"
