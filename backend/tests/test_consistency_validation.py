"""Focused tests for formal discovery consistency validation."""
from types import SimpleNamespace

import pytest

from agents import consistency_validation as cv
from agents import interview_planner
from agents.consistency_validation import (
    DiscoveryValidationIssue,
    FactConflictBatch,
    FactConflictVerdict,
    ValidationIssueKind,
    ValidationIssueSeverity,
    ValidationResolution,
    consistency_validation_node,
    validate_discovery_consistency,
)
from agents.discovery_coverage import fact_id
from agents.question_candidates import build_question_candidates
from agents.requirement_dependencies import requirement_dependency_node
from agents.requirements import (
    ActiveRequirement,
    RequirementFacet,
    RequirementStatus,
    register_requirement,
    requirement_store_key,
)
from agents.state import (
    DiscoveryScope as S,
    DiscoveryTopic as T,
    KnowledgeItem,
    KnowledgeState,
    TopicMaturity,
)


def fact(topic, key, value, *, scope=S.USER_APP, role=None, turn=1, absence=None):
    return KnowledgeItem(
        topic=topic,
        scope=scope,
        key=key,
        value=value,
        evidence=value,
        role=role,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
        absence=absence,
    )


def requirement(requirement_id, *, status=RequirementStatus.ACTIVE, dependencies=()):
    return ActiveRequirement(
        id=requirement_id,
        scope=S.USER_APP,
        topic=T.BUSINESS_RULES,
        parent_gap="approval_rules",
        label=requirement_id,
        description=requirement_id,
        status=status,
        dependencies=list(dependencies),
        facets=[
            RequirementFacet(id="rule", label="Rule", description="Resolve the rule.")
        ],
    )


def test_positive_fact_and_whole_field_absence_are_blocking_without_llm(monkeypatch):
    monkeypatch.setattr(cv, "conflict_model", lambda: pytest.fail("semantic model should not be needed"))
    absent = fact(T.CONSTRAINTS, "time_constraints", "none", absence="none")
    positive = fact(T.CONSTRAINTS, "time_constraints", "Invites expire after 48 hours")

    # Keep only the deterministic detector under test by pre-caching the positive
    # pair space as empty: absence facts are excluded from semantic pair review.
    issues, cache = validate_discovery_consistency({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [absent, positive],
        "active_requirements": {},
        "requirement_dependency_state": {},
        "requirement_coverage": {},
        "validation_pair_cache": {},
    })
    assert cache == {}
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == ValidationIssueKind.FACT_CONTRADICTION
    assert issue.resolution == ValidationResolution.USER_CLARIFICATION
    assert set(issue.fact_ids) == {fact_id(absent), fact_id(positive)}


def test_semantic_same_bucket_conflict_is_cached(monkeypatch):
    first = fact(T.BUSINESS_RULES, "limits", "Maximum five active requests")
    second = fact(T.BUSINESS_RULES, "limits", "Maximum ten active requests", turn=2)
    calls = []

    class Model:
        def invoke(self, messages):
            calls.append(messages)
            pair_id = cv._pair_id(first, second)
            return FactConflictBatch(verdicts=[
                FactConflictVerdict(
                    pair_id=pair_id,
                    contradiction=True,
                    confidence=0.99,
                    explanation="The same active-request limit is both five and ten.",
                )
            ])

    monkeypatch.setattr(cv, "conflict_model", lambda: Model())
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [first, second],
        "active_requirements": {},
        "requirement_dependency_state": {},
        "requirement_coverage": {},
        "validation_pair_cache": {},
    }
    issues, cache = validate_discovery_consistency(state)
    assert len(calls) == 1
    assert any(issue.kind == ValidationIssueKind.FACT_CONTRADICTION for issue in issues)

    issues_again, cache_again = validate_discovery_consistency({
        **state,
        "validation_pair_cache": cache,
    })
    assert len(calls) == 1
    assert issues_again
    assert cache_again == cache


def test_semantically_compatible_conditional_rules_do_not_conflict(monkeypatch):
    before = fact(
        T.EXCEPTIONS,
        "user_cancellations",
        "Buyers may cancel before approval",
    )
    after = fact(
        T.EXCEPTIONS,
        "user_cancellations",
        "Buyers cannot cancel after approval",
        turn=2,
    )

    class Model:
        def invoke(self, messages):
            return FactConflictBatch(verdicts=[
                FactConflictVerdict(
                    pair_id=cv._pair_id(before, after),
                    contradiction=False,
                    confidence=0.99,
                    explanation="The rules apply in different lifecycle states.",
                )
            ])

    monkeypatch.setattr(cv, "conflict_model", lambda: Model())
    issues, _ = validate_discovery_consistency({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [before, after],
        "active_requirements": {},
        "requirement_dependency_state": {},
        "requirement_coverage": {},
        "validation_pair_cache": {},
    })
    assert not any(issue.kind == ValidationIssueKind.FACT_CONTRADICTION for issue in issues)


def test_cross_field_mvp_conflict_is_semantically_reviewed(monkeypatch):
    must = fact(T.MVP_SCOPE, "must_have_features", "MVP must include card payments")
    excluded = fact(T.MVP_SCOPE, "out_of_scope", "Card payments are excluded from MVP", turn=2)

    class Model:
        def invoke(self, messages):
            return FactConflictBatch(verdicts=[
                FactConflictVerdict(
                    pair_id=cv._pair_id(must, excluded),
                    contradiction=True,
                    confidence=1,
                    explanation="Card payments are simultaneously required and excluded from MVP.",
                )
            ])

    monkeypatch.setattr(cv, "conflict_model", lambda: Model())
    issues, _ = validate_discovery_consistency({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [must, excluded],
        "active_requirements": {},
        "requirement_dependency_state": {},
        "requirement_coverage": {},
        "validation_pair_cache": {},
    })
    issue = next(issue for issue in issues if issue.kind == ValidationIssueKind.FACT_CONTRADICTION)
    assert issue.topic is None
    assert issue.metadata["first_field"].startswith("MVP_SCOPE.")
    assert issue.metadata["second_field"].startswith("MVP_SCOPE.")


def test_semantic_review_rejects_omitted_pair_verdict(monkeypatch):
    first = fact(T.BUSINESS_RULES, "limits", "Maximum five requests")
    second = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", turn=2)
    monkeypatch.setattr(
        cv,
        "conflict_model",
        lambda: SimpleNamespace(invoke=lambda _: FactConflictBatch(verdicts=[])),
    )
    with pytest.raises(Exception):
        validate_discovery_consistency({
            "discovery_scope": S.USER_APP,
            "discovered_knowledge": [first, second],
            "active_requirements": {},
            "requirement_dependency_state": {},
            "requirement_coverage": {},
            "validation_pair_cache": {},
        })


def test_missing_dependency_is_blocking_configuration_issue():
    item = requirement("downstream", dependencies=("missing.upstream",))
    store = register_requirement({}, item)
    dependency = requirement_dependency_node({
        "discovery_scope": S.USER_APP,
        "active_requirements": store,
    })
    result = consistency_validation_node({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "active_requirements": store,
        "requirement_coverage": {},
        "validation_pair_cache": {},
        **dependency,
    })
    kinds = {issue["kind"] for issue in result["validation_issues"]}
    assert ValidationIssueKind.DEPENDENCY_MISSING.value in kinds
    assert result["validation_blocking"] is True
    assert result["validation_candidate_blocking"] is True


def test_dependency_cycle_is_blocking_configuration_issue():
    a = requirement("a", dependencies=("b",))
    b = requirement("b", dependencies=("a",))
    store = register_requirement(register_requirement({}, a), b)
    dependency = requirement_dependency_node({
        "discovery_scope": S.USER_APP,
        "active_requirements": store,
    })
    result = consistency_validation_node({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "active_requirements": store,
        "requirement_coverage": {},
        "validation_pair_cache": {},
        **dependency,
    })
    kinds = {issue["kind"] for issue in result["validation_issues"]}
    assert ValidationIssueKind.DEPENDENCY_CYCLE.value in kinds
    assert result["validation_candidate_blocking"] is True


def test_resolved_downstream_with_reopened_dependency_blocks_compile_not_candidates():
    upstream = requirement("upstream", status=RequirementStatus.ACTIVE)
    downstream = requirement(
        "downstream",
        status=RequirementStatus.RESOLVED,
        dependencies=("upstream",),
    )
    store = register_requirement(register_requirement({}, upstream), downstream)
    dependency = requirement_dependency_node({
        "discovery_scope": S.USER_APP,
        "active_requirements": store,
    })
    result = consistency_validation_node({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "active_requirements": store,
        "requirement_coverage": {},
        "validation_pair_cache": {},
        **dependency,
    })
    issue = next(
        issue for issue in result["validation_issues"]
        if issue["kind"] == ValidationIssueKind.RESOLVED_REQUIREMENT_BLOCKED.value
    )
    assert issue["resolution"] == ValidationResolution.RECOMPUTE_STATE.value
    assert result["validation_blocking"] is True
    assert result["validation_candidate_blocking"] is False


def test_requirement_coverage_status_drift_is_blocking():
    item = requirement("r", status=RequirementStatus.RESOLVED)
    key = requirement_store_key(S.USER_APP, item.id)
    store = register_requirement({}, item)
    result = consistency_validation_node({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "active_requirements": store,
        "requirement_dependency_state": {
            key: {
                "requirement_id": item.id,
                "scope": S.USER_APP.value,
                "requirement_status": RequirementStatus.RESOLVED.value,
                "eligible": False,
                "dependencies": [],
                "blocking_dependencies": {},
            }
        },
        "requirement_coverage": {
            key: {
                "requirement_id": item.id,
                "scope": S.USER_APP.value,
                "status": "NEEDS_EXPANSION",
                "facets": {},
                "candidate_fact_ids": [],
                "expansion_needed": True,
            }
        },
        "validation_pair_cache": {},
    })
    assert any(
        issue["kind"] == ValidationIssueKind.REQUIREMENT_COVERAGE_MISMATCH.value
        for issue in result["validation_issues"]
    )
    assert result["validation_candidate_blocking"] is True


def test_fact_contradiction_routes_planner_to_validation_clarification(monkeypatch):
    old = fact(T.BUSINESS_RULES, "limits", "Maximum five active requests", turn=1)
    new = fact(T.BUSINESS_RULES, "limits", "Maximum ten active requests", turn=2)
    issue = DiscoveryValidationIssue(
        id="conflict",
        kind=ValidationIssueKind.FACT_CONTRADICTION,
        severity=ValidationIssueSeverity.BLOCKING,
        resolution=ValidationResolution.USER_CLARIFICATION,
        scope=S.USER_APP,
        topic=T.BUSINESS_RULES,
        key="limits",
        fact_ids=[fact_id(old), fact_id(new)],
        fact_values=[old.value, new.value],
        message="The active-request limit conflicts.",
    )
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [old, new],
        "validation_issues": [issue.model_dump(mode="json")],
        "validation_blocking": True,
        "validation_candidate_blocking": True,
        "active_requirements": {},
        "ranked_question_candidates": [],
        "question_candidate_priority": {},
        "gap_coverage": {},
        "active_answer_result": None,
        "topic_status": {},
        "topic_maturity": {},
        "current_topic": None,
        "planner_source": "schema",
    }
    result = interview_planner.interview_planner_node(state)
    assert result["planner_source"] == "validation"
    assert result["selected_validation_issue"]["id"] == "conflict"
    assert result["current_topic"] == T.BUSINESS_RULES
    assert result["current_gap"] == "limits"
    assert result["next_discovery_move"] == "resolve_contradiction"
    assert result["known_gap_evidence"] == [old.value, new.value]


def test_validation_question_does_not_resolve_schema_gap(monkeypatch):
    item = fact(T.BUSINESS_RULES, "limits", "Maximum ten active requests", turn=3)
    issue = DiscoveryValidationIssue(
        id="conflict",
        kind=ValidationIssueKind.FACT_CONTRADICTION,
        severity=ValidationIssueSeverity.BLOCKING,
        resolution=ValidationResolution.USER_CLARIFICATION,
        scope=S.USER_APP,
        topic=T.BUSINESS_RULES,
        key="limits",
        fact_ids=[fact_id(item)],
        fact_values=[item.value],
        message="Conflict.",
    )
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [item],
        "validation_issues": [issue.model_dump(mode="json")],
        "validation_blocking": True,
        "validation_candidate_blocking": True,
        "active_requirements": {},
        "ranked_question_candidates": [],
        "question_candidate_priority": {},
        "gap_coverage": {},
        "active_answer_result": {
            "scope": S.USER_APP.value,
            "topic": T.BUSINESS_RULES.value,
            "gap": "limits",
            "source_turn": 3,
            "fact_ids": [fact_id(item)],
            "resolution": "DIRECT_ANSWER",
        },
        "asked_gap": {
            "scope": S.USER_APP.value,
            "topic": T.BUSINESS_RULES.value,
            "gap": "limits",
            "question": "Which limit should apply now?",
        },
        "topic_status": {},
        "topic_maturity": {},
        "current_topic": T.BUSINESS_RULES,
        "current_gap": "limits",
        "turn_count": 3,
        "planner_source": "validation",
    }
    result = interview_planner.interview_planner_node(state)
    assert result["gap_coverage"] == {}


def test_candidate_builder_suppresses_candidates_for_fact_conflict():
    req = requirement("r")
    key = requirement_store_key(S.USER_APP, req.id)
    state = {
        "discovery_scope": S.USER_APP,
        "validation_candidate_blocking": True,
        "active_requirements": {key: req},
        "requirement_coverage": {},
        "eligible_requirement_keys": [key],
    }
    assert build_question_candidates(state) == []


def test_cross_scope_contradictions_do_not_mix(monkeypatch):
    user = fact(T.BUSINESS_RULES, "limits", "Maximum five requests", scope=S.USER_APP)
    admin = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", scope=S.ADMIN_DASHBOARD)
    monkeypatch.setattr(cv, "conflict_model", lambda: pytest.fail("cross-scope pair should not be reviewed"))
    issues, _ = validate_discovery_consistency({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [user, admin],
        "active_requirements": {},
        "requirement_dependency_state": {},
        "requirement_coverage": {},
        "validation_pair_cache": {},
    })
    assert issues == []
