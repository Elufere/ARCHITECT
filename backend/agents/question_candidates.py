"""Build and deterministically filter requirement question candidates.

This is the bridge between requirement reasoning and the future prioritized
planner. It does not rank candidates and does not change current planner output.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

from agents.requirement_coverage import (
    RequirementCoverageRecord,
    RequirementCoverageStatus,
    RequirementFacetState,
)
from agents.requirements import (
    ActiveRequirement,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic


class QuestionCandidate(BaseModel):
    id: str
    requirement_key: str
    requirement_id: str
    scope: DiscoveryScope
    topic: DiscoveryTopic
    objective: str
    target_facets: List[str] = Field(default_factory=list)
    known_fact_ids: List[str] = Field(default_factory=list)
    activation_rule_ids: List[str] = Field(default_factory=list)
    coverage_status: RequirementCoverageStatus
    dependency_eligible: bool = True


class CandidateBlockReason(str, Enum):
    REQUIREMENT_MISSING = "REQUIREMENT_MISSING"
    WRONG_SCOPE = "WRONG_SCOPE"
    REQUIREMENT_NOT_ACTIVE = "REQUIREMENT_NOT_ACTIVE"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    COVERAGE_MISSING = "COVERAGE_MISSING"
    COVERAGE_TERMINAL = "COVERAGE_TERMINAL"
    NO_UNRESOLVED_FACETS = "NO_UNRESOLVED_FACETS"
    RECENTLY_ASKED_SAME_TARGET = "RECENTLY_ASKED_SAME_TARGET"


class CandidateEligibilityDecision(BaseModel):
    candidate_id: str
    eligible: bool
    reasons: List[CandidateBlockReason] = Field(default_factory=list)


ELIGIBLE_COVERAGE_STATUSES = {
    RequirementCoverageStatus.UNSEEN,
    RequirementCoverageStatus.KNOWN_SHALLOW,
    RequirementCoverageStatus.NEEDS_EXPANSION,
}


def _candidate_id(requirement: ActiveRequirement, target_facets: List[str]) -> str:
    target = ",".join(target_facets) if target_facets else "__requirement__"
    return f"{requirement.scope.value}|{requirement.id}|{target}"


def _target_facets(
    requirement: ActiveRequirement,
    coverage: RequirementCoverageRecord,
) -> List[str]:
    """Target only unresolved required facets, preserving requirement order."""
    result = []
    for facet in requirement.facets:
        if not facet.required:
            continue
        state = coverage.facets.get(facet.id)
        if state is None or state.state == RequirementFacetState.UNKNOWN:
            result.append(facet.id)
    return result


def _known_fact_ids(coverage: RequirementCoverageRecord) -> List[str]:
    result = list(coverage.candidate_fact_ids)
    seen = set(result)
    for facet in coverage.facets.values():
        for identity in facet.fact_ids:
            if identity not in seen:
                seen.add(identity)
                result.append(identity)
    return result


def build_question_candidates(state: AgentState) -> List[QuestionCandidate]:
    """Build candidates only from the current dependency frontier.

    This is intentionally small and deterministic. It does not decide whether a
    candidate should ultimately be asked; the eligibility filter owns that gate.
    """
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    coverage_state = state.get("requirement_coverage", {})
    eligible_keys = state.get("eligible_requirement_keys", [])

    candidates: List[QuestionCandidate] = []
    for key in eligible_keys:
        requirement = store.get(key)
        payload = coverage_state.get(key)
        if requirement is None or payload is None:
            continue
        if requirement.scope != scope:
            continue

        coverage = RequirementCoverageRecord.model_validate(payload)
        target_facets = _target_facets(requirement, coverage)
        activation_rule_ids = list(dict.fromkeys(
            source.activation_rule_id
            for source in requirement.activation_sources
            if source.activation_rule_id
        ))

        candidates.append(QuestionCandidate(
            id=_candidate_id(requirement, target_facets),
            requirement_key=key,
            requirement_id=requirement.id,
            scope=requirement.scope,
            topic=requirement.topic,
            objective=requirement.description or requirement.label,
            target_facets=target_facets,
            known_fact_ids=_known_fact_ids(coverage),
            activation_rule_ids=activation_rule_ids,
            coverage_status=coverage.status,
            dependency_eligible=True,
        ))

    return candidates


def question_candidate_builder_node(state: AgentState) -> dict:
    return {
        "question_candidates": [
            candidate.model_dump(mode="json")
            for candidate in build_question_candidates(state)
        ]
    }


def _history_signature(entry: dict) -> tuple[str | None, tuple[str, ...]]:
    return (
        entry.get("requirement_key"),
        tuple(entry.get("target_facets") or []),
    )


def filter_question_candidates(
    state: AgentState,
    candidates: List[QuestionCandidate] | None = None,
) -> tuple[List[QuestionCandidate], Dict[str, CandidateEligibilityDecision]]:
    """Fail closed on lifecycle/dependency/coverage problems before ranking.

    Recent repetition is intentionally exact at this stage: the same requirement
    and same facet target cannot immediately re-enter the candidate pool. Semantic
    similarity belongs to a later layer and must not be faked here.
    """
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    coverage_state = state.get("requirement_coverage", {})
    dependency_state = state.get("requirement_dependency_state", {})
    history = state.get("requirement_question_history", [])
    recent_signatures = {
        _history_signature(entry)
        for entry in history[-3:]
    }

    if candidates is None:
        candidates = [
            QuestionCandidate.model_validate(item)
            for item in state.get("question_candidates", [])
        ]

    accepted: List[QuestionCandidate] = []
    decisions: Dict[str, CandidateEligibilityDecision] = {}

    for candidate in candidates:
        reasons: List[CandidateBlockReason] = []
        requirement = store.get(candidate.requirement_key)

        if requirement is None:
            reasons.append(CandidateBlockReason.REQUIREMENT_MISSING)
        else:
            if requirement.scope != scope or candidate.scope != scope:
                reasons.append(CandidateBlockReason.WRONG_SCOPE)
            if requirement.status != RequirementStatus.ACTIVE:
                reasons.append(CandidateBlockReason.REQUIREMENT_NOT_ACTIVE)

        dependency = dependency_state.get(candidate.requirement_key)
        if not dependency or not dependency.get("eligible"):
            reasons.append(CandidateBlockReason.DEPENDENCY_BLOCKED)

        coverage_payload = coverage_state.get(candidate.requirement_key)
        coverage = None
        if coverage_payload is None:
            reasons.append(CandidateBlockReason.COVERAGE_MISSING)
        else:
            coverage = RequirementCoverageRecord.model_validate(coverage_payload)
            if coverage.status not in ELIGIBLE_COVERAGE_STATUSES:
                reasons.append(CandidateBlockReason.COVERAGE_TERMINAL)

        if requirement is not None and coverage is not None and requirement.facets:
            unresolved = _target_facets(requirement, coverage)
            if not unresolved:
                reasons.append(CandidateBlockReason.NO_UNRESOLVED_FACETS)
            elif set(candidate.target_facets) != set(unresolved):
                # A stale candidate cannot target facets that no longer match
                # current coverage. Treat it as not eligible rather than mutating
                # it in place.
                reasons.append(CandidateBlockReason.NO_UNRESOLVED_FACETS)

        signature = (
            candidate.requirement_key,
            tuple(candidate.target_facets),
        )
        if signature in recent_signatures:
            reasons.append(CandidateBlockReason.RECENTLY_ASKED_SAME_TARGET)

        decision = CandidateEligibilityDecision(
            candidate_id=candidate.id,
            eligible=not reasons,
            reasons=list(dict.fromkeys(reasons)),
        )
        decisions[candidate.id] = decision
        if decision.eligible:
            accepted.append(candidate)

    return accepted, decisions


def question_candidate_filter_node(state: AgentState) -> dict:
    eligible, decisions = filter_question_candidates(state)
    return {
        "eligible_question_candidates": [
            candidate.model_dump(mode="json")
            for candidate in eligible
        ],
        "question_candidate_eligibility": {
            candidate_id: decision.model_dump(mode="json")
            for candidate_id, decision in decisions.items()
        },
    }
