"""Focused tests for deterministic question candidate prioritization."""
from agents.question_candidates import QuestionCandidate
from agents.question_priority import (
    prioritize_question_candidates,
    score_question_candidate,
)
from agents.requirement_coverage import RequirementCoverageStatus
from agents.requirements import (
    ActiveRequirement,
    RequirementFacet,
    RequirementPriorityHints,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


def req(
    requirement_id,
    *,
    topic=T.BUSINESS_RULES,
    dependencies=(),
    unlocks=(),
    architecture_impact=0.5,
    business_risk=0.5,
    status=RequirementStatus.ACTIVE,
):
    return ActiveRequirement(
        id=requirement_id,
        scope=S.USER_APP,
        topic=topic,
        label=requirement_id,
        description=f"Investigate {requirement_id}",
        status=status,
        dependencies=list(dependencies),
        unlocks=list(unlocks),
        facets=[
            RequirementFacet(id="a", label="A", description="A"),
            RequirementFacet(id="b", label="B", description="B"),
            RequirementFacet(id="c", label="C", description="C"),
        ],
        priority_hints=RequirementPriorityHints(
            architecture_impact=architecture_impact,
            business_risk=business_risk,
        ),
    )


def candidate(
    requirement,
    *,
    candidate_id=None,
    target_facets=("a",),
    coverage_status=RequirementCoverageStatus.UNSEEN,
):
    key = requirement_store_key(requirement.scope, requirement.id)
    return QuestionCandidate(
        id=candidate_id or f"{key}|{','.join(target_facets)}",
        requirement_key=key,
        requirement_id=requirement.id,
        scope=requirement.scope,
        topic=requirement.topic,
        objective=requirement.description,
        target_facets=list(target_facets),
        known_fact_ids=[],
        activation_rule_ids=[],
        coverage_status=coverage_status,
    )


def state_for(requirements, candidates, *, current_topic=None, history=None):
    return {
        "discovery_scope": S.USER_APP,
        "active_requirements": {
            requirement_store_key(item.scope, item.id): item
            for item in requirements
        },
        "eligible_question_candidates": [
            item.model_dump(mode="json") for item in candidates
        ],
        "current_topic": current_topic,
        "requirement_question_history": history or [],
    }


def test_more_active_downstream_unlocks_rank_higher():
    root = req("root")
    peer = req("peer")
    child_one = req("child.one", dependencies=("root",))
    child_two = req("child.two", dependencies=("root",))
    root_candidate = candidate(root)
    peer_candidate = candidate(peer)

    state = state_for(
        [root, peer, child_one, child_two],
        [peer_candidate, root_candidate],
    )
    ranked, scores = prioritize_question_candidates(state)
    assert ranked[0].requirement_id == "root"
    assert (
        scores[root_candidate.id].components.dependency_unlock_value
        > scores[peer_candidate.id].components.dependency_unlock_value
    )


def test_resolved_or_inactive_downstream_does_not_inflate_unlock_value():
    root = req("root")
    resolved = req(
        "resolved.child",
        dependencies=("root",),
        status=RequirementStatus.RESOLVED,
    )
    inactive = req(
        "inactive.child",
        dependencies=("root",),
        status=RequirementStatus.INACTIVE,
    )
    root_candidate = candidate(root)
    state = state_for([root, resolved, inactive], [root_candidate])

    score = score_question_candidate(state, root_candidate)
    assert score.components.dependency_unlock_value == 0


def test_more_unresolved_information_scores_as_more_uncertain():
    broad = req("broad")
    narrow = req("narrow")
    broad_candidate = candidate(
        broad,
        target_facets=("a", "b", "c"),
        coverage_status=RequirementCoverageStatus.UNSEEN,
    )
    narrow_candidate = candidate(
        narrow,
        target_facets=("c",),
        coverage_status=RequirementCoverageStatus.NEEDS_EXPANSION,
    )
    state = state_for([broad, narrow], [broad_candidate, narrow_candidate])

    broad_score = score_question_candidate(state, broad_candidate)
    narrow_score = score_question_candidate(state, narrow_candidate)
    assert broad_score.components.uncertainty > narrow_score.components.uncertainty


def test_explicit_priority_hints_affect_score_without_topic_ranking():
    high = req("high", architecture_impact=1, business_risk=1)
    low = req("low", architecture_impact=0, business_risk=0)
    high_candidate = candidate(high)
    low_candidate = candidate(low)
    state = state_for([high, low], [low_candidate, high_candidate])

    ranked, scores = prioritize_question_candidates(state)
    assert ranked[0].requirement_id == "high"
    assert scores[high_candidate.id].score > scores[low_candidate.id].score


def test_current_topic_is_a_small_context_signal():
    same = req("same", topic=T.EXCEPTIONS)
    other = req("other", topic=T.BUSINESS_RULES)
    same_candidate = candidate(same)
    other_candidate = candidate(other)
    state = state_for(
        [same, other],
        [other_candidate, same_candidate],
        current_topic=T.EXCEPTIONS,
    )

    same_score = score_question_candidate(state, same_candidate)
    other_score = score_question_candidate(state, other_candidate)
    assert same_score.components.context_relevance > other_score.components.context_relevance
    assert same_score.score > other_score.score


def test_recent_same_requirement_applies_repetition_penalty():
    item = req("repeat")
    cand = candidate(item)
    clean = state_for([item], [cand])
    repeated = state_for(
        [item],
        [cand],
        history=[
            {
                "requirement_key": cand.requirement_key,
                "target_facets": ["b"],
                "topic": cand.topic.value,
            }
        ],
    )

    clean_score = score_question_candidate(clean, cand)
    repeated_score = score_question_candidate(repeated, cand)
    assert repeated_score.components.repetition_penalty > 0
    assert repeated_score.components.fatigue_penalty > 0
    assert repeated_score.score < clean_score.score


def test_broader_question_target_has_cost_penalty():
    item = req("cost")
    narrow = candidate(item, candidate_id="narrow", target_facets=("a",))
    broad = candidate(item, candidate_id="broad", target_facets=("a", "b", "c"))
    state = state_for([item], [narrow, broad])

    narrow_score = score_question_candidate(state, narrow)
    broad_score = score_question_candidate(state, broad)
    assert broad_score.components.question_cost_penalty > narrow_score.components.question_cost_penalty


def test_ties_are_deterministic_by_candidate_id():
    first = req("a")
    second = req("b")
    candidate_z = candidate(first, candidate_id="z-candidate")
    candidate_a = candidate(second, candidate_id="a-candidate")
    state = state_for([first, second], [candidate_z, candidate_a])

    ranked, scores = prioritize_question_candidates(state)
    assert scores[candidate_z.id].score == scores[candidate_a.id].score
    assert [item.id for item in ranked] == ["a-candidate", "z-candidate"]


def test_only_eligible_candidate_payloads_are_ranked():
    first = req("first")
    second = req("second")
    first_candidate = candidate(first)
    second_candidate = candidate(second)
    state = state_for([first, second], [first_candidate])
    # second exists in requirement state but is intentionally absent from the
    # post-filter candidate pool.
    ranked, scores = prioritize_question_candidates(state)
    assert [item.requirement_id for item in ranked] == ["first"]
    assert list(scores) == [first_candidate.id]


def test_score_components_are_bounded_and_explainable():
    item = req("bounded", architecture_impact=1, business_risk=1)
    cand = candidate(item, target_facets=("a", "b", "c"))
    state = state_for([item], [cand])
    score = score_question_candidate(state, cand)

    assert 0 <= score.score <= 1
    for value in score.components.model_dump().values():
        assert 0 <= value <= 1
    assert score.rationale
