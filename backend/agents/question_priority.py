"""Deterministic, explainable prioritization of inquiry-driven questions."""
from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from agents.inquiries import InquirySource, ProductInquiry
from agents.question_candidates import QuestionCandidate
from agents.requirements import ActiveRequirement, RequirementStatus, RequirementStore
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic


class CandidatePriorityComponents(BaseModel):
    contradiction_pressure: float = Field(default=0, ge=0, le=1)
    decision_impact: float = Field(default=0.5, ge=0, le=1)
    dependency_unlock_value: float = Field(default=0, ge=0, le=1)
    uncertainty: float = Field(default=1, ge=0, le=1)
    architecture_impact: float = Field(default=0.5, ge=0, le=1)
    business_risk: float = Field(default=0.5, ge=0, le=1)
    context_relevance: float = Field(default=0.5, ge=0, le=1)
    question_cost_penalty: float = Field(default=0, ge=0, le=1)
    repetition_penalty: float = Field(default=0, ge=0, le=1)
    fatigue_penalty: float = Field(default=0, ge=0, le=1)
    premature_depth_penalty: float = Field(default=0, ge=0, le=1)


class CandidatePriorityScore(BaseModel):
    candidate_id: str
    score: float = Field(ge=0, le=1)
    components: CandidatePriorityComponents
    rationale: List[str] = Field(default_factory=list)


POSITIVE_WEIGHTS = {
    "contradiction_pressure": 0.20,
    "decision_impact": 0.20,
    "dependency_unlock_value": 0.15,
    "uncertainty": 0.20,
    "architecture_impact": 0.10,
    "business_risk": 0.10,
    "context_relevance": 0.05,
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
    requirement: ActiveRequirement | None,
    requirements: Dict[str, ActiveRequirement],
) -> float:
    if requirement is None:
        return 0.0
    return _clamp(_unlock_count(requirement, requirements) / 3.0)


def _decision_impact(candidate: QuestionCandidate) -> float:
    if candidate.source == InquirySource.VALIDATION:
        return 1.0
    if candidate.source == InquirySource.MODEL:
        anchor = candidate.anchor_gap or ""
        if anchor == "primary_users":
            return 1.0
        if anchor.startswith("responsibilities"):
            return 0.95
        if anchor.startswith("primary_user_goals"):
            return 0.85
        if anchor == "workflow_steps":
            return 1.0
        if anchor in {"completion_condition", "end_state"}:
            return 0.9
        return 0.8
    return 0.75


def _context_relevance(state: AgentState, candidate: QuestionCandidate) -> float:
    current_topic = state.get("current_topic")
    if current_topic is None:
        return 0.6
    return 1.0 if current_topic == candidate.topic else 0.5


def _recent_history(state: AgentState) -> Sequence[dict]:
    return state.get("requirement_question_history", [])[-6:]


def _repetition_penalty(state: AgentState, candidate: QuestionCandidate) -> float:
    identity = candidate.inquiry_id or candidate.requirement_key
    count = sum(
        1 for entry in _recent_history(state)
        if (entry.get("inquiry_id") or entry.get("requirement_key")) == identity
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


def _premature_depth_penalty(state: AgentState, candidate: QuestionCandidate) -> float:
    if candidate.source != InquirySource.REQUIREMENT:
        return 0.0
    for payload in state.get("open_inquiries", []):
        inquiry = ProductInquiry.model_validate(payload)
        if inquiry.source == InquirySource.MODEL:
            return 0.25
    return 0.0


def score_question_candidate(
    state: AgentState,
    candidate: QuestionCandidate,
) -> CandidatePriorityScore:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    requirement = (
        store.get(candidate.requirement_key)
        if candidate.requirement_key
        else None
    )
    if candidate.source == InquirySource.REQUIREMENT and requirement is None:
        raise ValueError(f"Cannot score missing requirement '{candidate.requirement_key}'")
    if candidate.scope != scope or (requirement is not None and requirement.scope != scope):
        raise ValueError("Cannot score candidate outside the current discovery scope")

    requirements = _scope_requirements(store, scope)
    components = CandidatePriorityComponents(
        contradiction_pressure=1.0 if candidate.source == InquirySource.VALIDATION else 0.0,
        decision_impact=_decision_impact(candidate),
        dependency_unlock_value=_unlock_value(requirement, requirements),
        uncertainty=candidate.uncertainty,
        architecture_impact=candidate.architecture_impact,
        business_risk=candidate.business_risk,
        context_relevance=_context_relevance(state, candidate),
        question_cost_penalty=candidate.question_cost,
        repetition_penalty=_repetition_penalty(state, candidate),
        fatigue_penalty=_fatigue_penalty(state, candidate),
        premature_depth_penalty=_premature_depth_penalty(state, candidate),
    )

    positive = sum(
        POSITIVE_WEIGHTS[name] * getattr(components, name)
        for name in POSITIVE_WEIGHTS
    )
    penalties = (
        components.question_cost_penalty
        + components.repetition_penalty
        + components.fatigue_penalty
        + components.premature_depth_penalty
    )
    score = _clamp(positive - penalties)

    rationale = [
        f"source={candidate.source.value}",
        f"decision_impact={components.decision_impact:.2f}",
        f"uncertainty={components.uncertainty:.2f}",
        f"dependency_unlock_value={components.dependency_unlock_value:.2f}",
        f"architecture_impact={components.architecture_impact:.2f}",
        f"business_risk={components.business_risk:.2f}",
    ]
    if components.contradiction_pressure:
        rationale.append("contradiction_pressure=1.00")
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
    if candidates is None:
        candidates = [
            QuestionCandidate.model_validate(item)
            for item in state.get("eligible_question_candidates", [])
        ]

    scores = {
        candidate.id: score_question_candidate(state, candidate)
        for candidate in candidates
    }

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -scores[candidate.id].score,
            -scores[candidate.id].components.contradiction_pressure,
            -scores[candidate.id].components.decision_impact,
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
