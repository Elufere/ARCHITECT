"""Deterministic, explainable prioritization of eligible requirement questions."""
from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from agents.question_candidates import QuestionCandidate
from agents.requirement_coverage import RequirementCoverageStatus
from agents.requirements import ActiveRequirement, RequirementStatus, RequirementStore
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic


class CandidatePriorityComponents(BaseModel):
    dependency_unlock_value: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    architecture_impact: float = Field(ge=0, le=1)
    business_risk: float = Field(ge=0, le=1)
    context_relevance: float = Field(ge=0, le=1)
    question_cost_penalty: float = Field(ge=0, le=1)
    repetition_penalty: float = Field(ge=0, le=1)
    fatigue_penalty: float = Field(ge=0, le=1)


class CandidatePriorityScore(BaseModel):
    candidate_id: str
    score: float = Field(ge=0, le=1)
    components: CandidatePriorityComponents
    rationale: List[str] = Field(default_factory=list)


POSITIVE_WEIGHTS = {
    "dependency_unlock_value": 0.30,
    "uncertainty": 0.30,
    "architecture_impact": 0.15,
    "business_risk": 0.15,
    "context_relevance": 0.10,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scope_requirements(
    store: RequirementStore,
    scope: DiscoveryScope,
) -> Dict[str, ActiveRequirement]:
    return {
        requirement.id: requirement
        for requirement in store.values()
        if requirement.scope == scope
    }


def _unlock_count(
    requirement: ActiveRequirement,
    requirements: Dict[str, ActiveRequirement],
) -> int:
    """Count structurally downstream active requirements this requirement can unlock."""
    inferred = {
        downstream.id
        for downstream in requirements.values()
        if downstream.status == RequirementStatus.ACTIVE
        and requirement.id in downstream.dependencies
    }
    declared = {
        requirement_id
        for requirement_id in requirement.unlocks
        if requirement_id in requirements
        and requirements[requirement_id].status == RequirementStatus.ACTIVE
    }
    return len(inferred.union(declared))


def _unlock_value(
    requirement: ActiveRequirement,
    requirements: Dict[str, ActiveRequirement],
) -> float:
    # Three or more downstream nodes is already a strong structural signal.
    return _clamp(_unlock_count(requirement, requirements) / 3.0)


def _uncertainty(candidate: QuestionCandidate, requirement: ActiveRequirement) -> float:
    required = [facet for facet in requirement.facets if facet.required]
    if required:
        unresolved_ratio = len(candidate.target_facets) / len(required)
    else:
        unresolved_ratio = 1.0

    coverage_multiplier = {
        RequirementCoverageStatus.UNSEEN: 1.0,
        RequirementCoverageStatus.KNOWN_SHALLOW: 0.85,
        RequirementCoverageStatus.NEEDS_EXPANSION: 0.65,
    }.get(candidate.coverage_status, 0.0)

    return _clamp(unresolved_ratio * coverage_multiplier)


def _context_relevance(state: AgentState, candidate: QuestionCandidate) -> float:
    current_topic = state.get("current_topic")
    if current_topic is None:
        return 0.5
    return 1.0 if current_topic == candidate.topic else 0.35


def _question_cost_penalty(candidate: QuestionCandidate) -> float:
    # Asking about many unresolved facets at once is harder for the user and more
    # likely to produce a broad answer. One target has no breadth penalty.
    extra_targets = max(0, len(candidate.target_facets) - 1)
    return min(0.15, extra_targets * 0.05)


def _recent_history(state: AgentState) -> Sequence[dict]:
    return state.get("requirement_question_history", [])[-6:]


def _repetition_penalty(state: AgentState, candidate: QuestionCandidate) -> float:
    count = sum(
        1 for entry in _recent_history(state)
        if entry.get("requirement_key") == candidate.requirement_key
    )
    return min(0.21, count * 0.07)


def _fatigue_penalty(state: AgentState, candidate: QuestionCandidate) -> float:
    def topic_value(entry: dict):
        value = entry.get("topic")
        return value.value if isinstance(value, DiscoveryTopic) else value

    count = sum(
        1 for entry in _recent_history(state)
        if topic_value(entry) == candidate.topic.value
    )
    return min(0.12, count * 0.03)


def score_question_candidate(
    state: AgentState,
    candidate: QuestionCandidate,
) -> CandidatePriorityScore:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    requirement = store.get(candidate.requirement_key)
    if requirement is None:
        raise ValueError(f"Cannot score missing requirement '{candidate.requirement_key}'")
    if requirement.scope != scope or candidate.scope != scope:
        raise ValueError("Cannot score candidate outside the current discovery scope")

    requirements = _scope_requirements(store, scope)

    components = CandidatePriorityComponents(
        dependency_unlock_value=_unlock_value(requirement, requirements),
        uncertainty=_uncertainty(candidate, requirement),
        architecture_impact=requirement.priority_hints.architecture_impact,
        business_risk=requirement.priority_hints.business_risk,
        context_relevance=_context_relevance(state, candidate),
        question_cost_penalty=_question_cost_penalty(candidate),
        repetition_penalty=_repetition_penalty(state, candidate),
        fatigue_penalty=_fatigue_penalty(state, candidate),
    )

    positive = (
        POSITIVE_WEIGHTS["dependency_unlock_value"] * components.dependency_unlock_value
        + POSITIVE_WEIGHTS["uncertainty"] * components.uncertainty
        + POSITIVE_WEIGHTS["architecture_impact"] * components.architecture_impact
        + POSITIVE_WEIGHTS["business_risk"] * components.business_risk
        + POSITIVE_WEIGHTS["context_relevance"] * components.context_relevance
    )
    penalties = (
        components.question_cost_penalty
        + components.repetition_penalty
        + components.fatigue_penalty
    )
    score = _clamp(positive - penalties)

    rationale = [
        f"dependency_unlock_value={components.dependency_unlock_value:.2f}",
        f"uncertainty={components.uncertainty:.2f}",
        f"architecture_impact={components.architecture_impact:.2f}",
        f"business_risk={components.business_risk:.2f}",
        f"context_relevance={components.context_relevance:.2f}",
    ]
    if penalties:
        rationale.append(f"penalties={penalties:.2f}")

    return CandidatePriorityScore(
        candidate_id=candidate.id,
        score=round(score, 6),
        components=components,
        rationale=rationale,
    )


def prioritize_question_candidates(
    state: AgentState,
    candidates: List[QuestionCandidate] | None = None,
) -> tuple[List[QuestionCandidate], Dict[str, CandidatePriorityScore]]:
    """Rank only candidates that already passed deterministic eligibility."""
    if candidates is None:
        candidates = [
            QuestionCandidate.model_validate(item)
            for item in state.get("eligible_question_candidates", [])
        ]

    scores = {
        candidate.id: score_question_candidate(state, candidate)
        for candidate in candidates
    }

    # Stable, deterministic tie-breaks: score first, then structural unlock value,
    # then uncertainty, then candidate id. No randomization or LLM tie-breaking.
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -scores[candidate.id].score,
            -scores[candidate.id].components.dependency_unlock_value,
            -scores[candidate.id].components.uncertainty,
            candidate.id,
        ),
    )
    return ranked, scores


def question_candidate_priority_node(state: AgentState) -> dict:
    ranked, scores = prioritize_question_candidates(state)
    return {
        "ranked_question_candidates": [
            candidate.model_dump(mode="json")
            for candidate in ranked
        ],
        "question_candidate_priority": {
            candidate_id: score.model_dump(mode="json")
            for candidate_id, score in scores.items()
        },
    }
