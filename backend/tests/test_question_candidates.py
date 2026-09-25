"""Focused tests for requirement question candidate construction and eligibility."""
from agents.question_candidates import (
    CandidateBlockReason,
    QuestionCandidate,
    build_question_candidates,
    filter_question_candidates,
)
from agents.requirement_coverage import (
    FacetCoverage,
    RequirementCoverageRecord,
    RequirementCoverageStatus,
    RequirementFacetState,
)
from agents.requirement_dependencies import RequirementDependencyDecision
from agents.requirements import (
    ActiveRequirement,
    RequirementFacet,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


def requirement(*, status=RequirementStatus.ACTIVE, scope=S.USER_APP):
    return ActiveRequirement(
        id="workflow.external_dependency_failure",
        scope=scope,
        topic=T.EXCEPTIONS,
        label="External dependency failure",
        description="Determine what happens when the external dependency fails.",
        status=status,
        facets=[
            RequirementFacet(id="failure_condition", label="Failure condition", description="What fails."),
            RequirementFacet(id="expected_behavior", label="Expected behavior", description="What happens."),
            RequirementFacet(id="optional_note", label="Optional note", description="Optional.", required=False),
        ],
    )


def coverage(req, *, status=RequirementCoverageStatus.UNSEEN, covered=()):
    facets = {}
    for facet in req.facets:
        if facet.id in covered:
            facets[facet.id] = FacetCoverage(
                facet_id=facet.id,
                state=RequirementFacetState.COVERED,
                fact_ids=[f"fact-{facet.id}"],
            )
        else:
            facets[facet.id] = FacetCoverage(facet_id=facet.id)
    return RequirementCoverageRecord(
        requirement_id=req.id,
        scope=req.scope,
        status=status,
        facets=facets,
        candidate_fact_ids=["fact-known"] if status != RequirementCoverageStatus.UNSEEN else [],
        expansion_needed=status == RequirementCoverageStatus.NEEDS_EXPANSION,
    )


def base_state(req=None, cov=None, *, dependency_eligible=True):
    req = req or requirement()
    cov = cov or coverage(req)
    key = requirement_store_key(req.scope, req.id)
    dependency = RequirementDependencyDecision(
        requirement_id=req.id,
        scope=req.scope,
        requirement_status=req.status,
        eligible=dependency_eligible,
    )
    return {
        "discovery_scope": req.scope,
        "active_requirements": {key: req},
        "requirement_coverage": {key: cov.model_dump(mode="json")},
        "requirement_dependency_state": {key: dependency.model_dump(mode="json")},
        "eligible_requirement_keys": [key] if dependency_eligible else [],
        "requirement_question_history": [],
    }


def test_builder_targets_only_unresolved_required_facets():
    req = requirement()
    cov = coverage(
        req,
        status=RequirementCoverageStatus.NEEDS_EXPANSION,
        covered=("failure_condition",),
    )
    candidates = build_question_candidates(base_state(req, cov))
    assert len(candidates) == 1
    assert candidates[0].target_facets == ["expected_behavior"]
    assert "optional_note" not in candidates[0].target_facets


def test_builder_carries_known_fact_ids_without_inventing_evidence():
    req = requirement()
    cov = coverage(
        req,
        status=RequirementCoverageStatus.NEEDS_EXPANSION,
        covered=("failure_condition",),
    )
    candidate = build_question_candidates(base_state(req, cov))[0]
    assert candidate.known_fact_ids == ["fact-known", "fact-failure_condition"]


def test_builder_only_uses_dependency_frontier():
    state = base_state(dependency_eligible=False)
    assert build_question_candidates(state) == []


def test_valid_candidate_passes_filter():
    state = base_state()
    candidates = build_question_candidates(state)
    accepted, decisions = filter_question_candidates(state, candidates)
    assert accepted == candidates
    assert decisions[candidates[0].id].eligible is True
    assert decisions[candidates[0].id].reasons == []


def test_requirement_must_still_exist_and_be_active():
    state = base_state()
    candidate = build_question_candidates(state)[0]

    missing_state = {**state, "active_requirements": {}}
    accepted, decisions = filter_question_candidates(missing_state, [candidate])
    assert accepted == []
    assert CandidateBlockReason.REQUIREMENT_MISSING in decisions[candidate.id].reasons

    req = requirement(status=RequirementStatus.RESOLVED)
    key = requirement_store_key(req.scope, req.id)
    resolved_state = {
        **state,
        "active_requirements": {key: req},
    }
    accepted, decisions = filter_question_candidates(resolved_state, [candidate])
    assert accepted == []
    assert CandidateBlockReason.REQUIREMENT_NOT_ACTIVE in decisions[candidate.id].reasons


def test_dependency_must_still_be_eligible():
    state = base_state()
    candidate = build_question_candidates(state)[0]
    blocked = {
        **state,
        "requirement_dependency_state": {
            candidate.requirement_key: {
                "requirement_id": candidate.requirement_id,
                "scope": S.USER_APP.value,
                "requirement_status": RequirementStatus.ACTIVE.value,
                "eligible": False,
                "dependencies": ["upstream"],
                "blocking_dependencies": {"upstream": "UNRESOLVED"},
            }
        },
    }
    accepted, decisions = filter_question_candidates(blocked, [candidate])
    assert accepted == []
    assert CandidateBlockReason.DEPENDENCY_BLOCKED in decisions[candidate.id].reasons


def test_terminal_coverage_is_filtered():
    state = base_state()
    candidate = build_question_candidates(state)[0]
    for status in (
        RequirementCoverageStatus.RESOLVED,
        RequirementCoverageStatus.NOT_APPLICABLE,
        RequirementCoverageStatus.DEFERRED,
        RequirementCoverageStatus.INACTIVE,
    ):
        payload = dict(state["requirement_coverage"][candidate.requirement_key])
        payload["status"] = status.value
        updated = {
            **state,
            "requirement_coverage": {candidate.requirement_key: payload},
        }
        accepted, decisions = filter_question_candidates(updated, [candidate])
        assert accepted == []
        assert CandidateBlockReason.COVERAGE_TERMINAL in decisions[candidate.id].reasons


def test_candidate_becomes_stale_when_coverage_changes():
    req = requirement()
    initial_cov = coverage(req, status=RequirementCoverageStatus.UNSEEN)
    state = base_state(req, initial_cov)
    candidate = build_question_candidates(state)[0]
    assert candidate.target_facets == ["failure_condition", "expected_behavior"]

    later_cov = coverage(
        req,
        status=RequirementCoverageStatus.NEEDS_EXPANSION,
        covered=("failure_condition",),
    )
    stale_state = {
        **state,
        "requirement_coverage": {
            candidate.requirement_key: later_cov.model_dump(mode="json")
        },
    }
    accepted, decisions = filter_question_candidates(stale_state, [candidate])
    assert accepted == []
    assert CandidateBlockReason.NO_UNRESOLVED_FACETS in decisions[candidate.id].reasons


def test_recent_exact_target_is_allowed_when_it_is_the_only_frontier_candidate():
    state = base_state()
    candidate = build_question_candidates(state)[0]
    repeated = {
        **state,
        "requirement_question_history": [{
            "requirement_key": candidate.requirement_key,
            "target_facets": candidate.target_facets,
            "question": "What should happen if the external dependency fails?",
        }],
    }
    accepted, decisions = filter_question_candidates(repeated, [candidate])
    assert accepted == [candidate]
    assert decisions[candidate.id].eligible is True


def test_recent_exact_target_stays_suppressed_when_an_alternative_exists():
    first_state = base_state()
    first = build_question_candidates(first_state)[0]
    other_req = requirement()
    other_req = other_req.model_copy(update={"id": "workflow.other_requirement", "label": "Other"})
    other_key = requirement_store_key(other_req.scope, other_req.id)
    other_cov = coverage(other_req)
    state = {
        **first_state,
        "active_requirements": {
            **first_state["active_requirements"],
            other_key: other_req,
        },
        "requirement_coverage": {
            **first_state["requirement_coverage"],
            other_key: other_cov.model_dump(mode="json"),
        },
        "requirement_dependency_state": {
            **first_state["requirement_dependency_state"],
            other_key: {
                "requirement_id": other_req.id,
                "scope": other_req.scope.value,
                "requirement_status": RequirementStatus.ACTIVE.value,
                "eligible": True,
                "dependencies": [],
                "blocking_dependencies": {},
            },
        },
        "eligible_requirement_keys": [first.requirement_key, other_key],
        "requirement_question_history": [{
            "requirement_key": first.requirement_key,
            "target_facets": first.target_facets,
        }],
    }
    candidates = build_question_candidates(state)
    accepted, decisions = filter_question_candidates(state, candidates)
    assert first.id not in {item.id for item in accepted}
    assert CandidateBlockReason.RECENTLY_ASKED_SAME_TARGET in decisions[first.id].reasons


def test_recent_different_target_does_not_trigger_exact_repetition_filter():
    req = requirement()
    cov = coverage(
        req,
        status=RequirementCoverageStatus.NEEDS_EXPANSION,
        covered=("failure_condition",),
    )
    state = base_state(req, cov)
    candidate = build_question_candidates(state)[0]
    state["requirement_question_history"] = [{
        "requirement_key": candidate.requirement_key,
        "target_facets": ["failure_condition"],
    }]
    accepted, decisions = filter_question_candidates(state, [candidate])
    assert accepted == [candidate]
    assert CandidateBlockReason.RECENTLY_ASKED_SAME_TARGET not in decisions[candidate.id].reasons


def test_wrong_scope_is_filtered_even_if_candidate_is_injected():
    state = base_state()
    original = build_question_candidates(state)[0]
    injected = original.model_copy(update={"scope": S.ADMIN_DASHBOARD})
    accepted, decisions = filter_question_candidates(state, [injected])
    assert accepted == []
    assert CandidateBlockReason.WRONG_SCOPE in decisions[injected.id].reasons
