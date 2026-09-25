"""Negative tests proving the end-to-end invariant checker catches impossible states."""
import pytest

from agents.discovery_coverage import fact_id
from agents.requirements import ActiveRequirement, RequirementStatus, requirement_store_key
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem
from discovery_invariant_utils import assert_discovery_invariants


def fact(value="Known fact"):
    return KnowledgeItem(
        topic=T.BUSINESS_RULES,
        scope=S.USER_APP,
        key="limits",
        value=value,
        evidence=value,
        confidence=1,
    )


def requirement(status=RequirementStatus.ACTIVE):
    return ActiveRequirement(
        id="test.requirement",
        scope=S.USER_APP,
        topic=T.BUSINESS_RULES,
        parent_gap="limits",
        label="Test requirement",
        description="Test requirement",
        status=status,
    )


def base():
    item = fact()
    req = requirement()
    key = requirement_store_key(S.USER_APP, req.id)
    return {
        "discovered_knowledge": [item],
        "superseded_knowledge": [],
        "active_requirements": {key: req},
        "requirement_coverage": {},
        "requirement_dependency_state": {
            key: {
                "requirement_id": req.id,
                "scope": S.USER_APP.value,
                "requirement_status": req.status.value,
                "eligible": True,
                "dependencies": [],
                "blocking_dependencies": {},
            }
        },
        "eligible_question_candidates": [],
        "ranked_question_candidates": [],
        "validation_candidate_blocking": False,
    }, item, req, key


def test_invariant_rejects_candidate_for_missing_requirement():
    state, _, _, _ = base()
    state["ranked_question_candidates"] = [{
        "id": "bad",
        "requirement_key": "USER_APP::missing",
    }]
    with pytest.raises(AssertionError, match="missing requirement"):
        assert_discovery_invariants(state)


def test_invariant_rejects_candidate_that_bypasses_dependency_block():
    state, _, req, key = base()
    state["requirement_dependency_state"][key]["eligible"] = False
    state["ranked_question_candidates"] = [{
        "id": "blocked",
        "requirement_key": key,
    }]
    with pytest.raises(AssertionError, match="dependency block"):
        assert_discovery_invariants(state)


def test_invariant_rejects_resolved_requirement_with_nonresolved_coverage():
    state, item, _, key = base()
    state["active_requirements"][key] = requirement(RequirementStatus.RESOLVED)
    state["requirement_coverage"][key] = {
        "requirement_id": "test.requirement",
        "scope": S.USER_APP.value,
        "status": "NEEDS_EXPANSION",
        "candidate_fact_ids": [fact_id(item)],
        "facets": {},
    }
    with pytest.raises(AssertionError):
        assert_discovery_invariants(state)


def test_invariant_rejects_coverage_reference_to_missing_fact():
    state, _, _, key = base()
    state["requirement_coverage"][key] = {
        "requirement_id": "test.requirement",
        "scope": S.USER_APP.value,
        "status": "KNOWN_SHALLOW",
        "candidate_fact_ids": ["missing-fact-id"],
        "facets": {},
    }
    with pytest.raises(AssertionError, match="missing fact"):
        assert_discovery_invariants(state)


def test_invariant_rejects_ranked_candidates_during_consistency_block():
    state, _, _, key = base()
    state["validation_candidate_blocking"] = True
    state["question_candidates"] = []
    state["eligible_question_candidates"] = []
    state["ranked_question_candidates"] = [{
        "id": "should-not-exist",
        "requirement_key": key,
    }]
    with pytest.raises(AssertionError):
        assert_discovery_invariants(state)
