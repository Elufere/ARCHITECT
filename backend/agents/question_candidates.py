"""Build and deterministically filter question candidates from open inquiries.

The planner no longer traverses schema gaps. Inquiries can come from the
foundational product model, active product-specific requirements, or blocking
consistency validation.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from agents.inquiries import InquirySource, ProductInquiry
from agents.requirement_coverage import (
    RequirementCoverageRecord,
    RequirementCoverageStatus,
    RequirementFacetState,
)
from agents.requirements import ActiveRequirement, RequirementStatus
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic


class QuestionCandidate(BaseModel):
    id: str
    inquiry_id: Optional[str] = None
    source: InquirySource = InquirySource.REQUIREMENT
    requirement_key: Optional[str] = None
    requirement_id: Optional[str] = None
    scope: DiscoveryScope
    topic: DiscoveryTopic
    anchor_gap: Optional[str] = None
    objective: str
    question_hint: str = ""
    reason: str = ""
    role: Optional[str] = None
    target_facets: List[str] = Field(default_factory=list)
    known_fact_ids: List[str] = Field(default_factory=list)
    activation_rule_ids: List[str] = Field(default_factory=list)
    coverage_status: Optional[RequirementCoverageStatus] = None
    dependency_eligible: bool = True
    uncertainty: float = Field(default=1.0, ge=0, le=1)
    architecture_impact: float = Field(default=0.5, ge=0, le=1)
    business_risk: float = Field(default=0.5, ge=0, le=1)
    question_cost: float = Field(default=0.0, ge=0, le=1)
    thread_id: Optional[str] = None
    decision_key: Optional[str] = None
    information_gain: float = Field(default=0.5, ge=0, le=1)
    causal_relevance: float = Field(default=0.5, ge=0, le=1)
    conversation_continuity: float = Field(default=0.5, ge=0, le=1)


class CandidateBlockReason(str, Enum):
    INQUIRY_MISSING = "INQUIRY_MISSING"
    REQUIREMENT_MISSING = "REQUIREMENT_MISSING"
    WRONG_SCOPE = "WRONG_SCOPE"
    REQUIREMENT_NOT_ACTIVE = "REQUIREMENT_NOT_ACTIVE"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    COVERAGE_MISSING = "COVERAGE_MISSING"
    COVERAGE_TERMINAL = "COVERAGE_TERMINAL"
    NO_UNRESOLVED_FACETS = "NO_UNRESOLVED_FACETS"
    RECENTLY_ASKED_SAME_TARGET = "RECENTLY_ASKED_SAME_TARGET"
    REPEATED_THREAD_DECISION = "REPEATED_THREAD_DECISION"


class CandidateEligibilityDecision(BaseModel):
    candidate_id: str
    eligible: bool
    reasons: List[CandidateBlockReason] = Field(default_factory=list)


ELIGIBLE_COVERAGE_STATUSES = {
    RequirementCoverageStatus.UNSEEN,
    RequirementCoverageStatus.KNOWN_SHALLOW,
    RequirementCoverageStatus.NEEDS_EXPANSION,
}


def _candidate_from_inquiry(inquiry: ProductInquiry) -> QuestionCandidate:
    return QuestionCandidate(
        id=f"question|{inquiry.id}",
        inquiry_id=inquiry.id,
        source=inquiry.source,
        requirement_key=inquiry.requirement_key,
        requirement_id=inquiry.requirement_id,
        scope=inquiry.scope,
        topic=inquiry.topic,
        anchor_gap=inquiry.anchor_gap,
        objective=inquiry.objective,
        question_hint=inquiry.question_hint,
        reason=inquiry.reason,
        role=inquiry.role,
        target_facets=list(inquiry.target_facets),
        known_fact_ids=list(inquiry.known_fact_ids),
        activation_rule_ids=list(inquiry.activation_rule_ids),
        coverage_status=inquiry.coverage_status,
        dependency_eligible=inquiry.dependency_eligible,
        uncertainty=inquiry.uncertainty,
        architecture_impact=inquiry.architecture_impact,
        business_risk=inquiry.business_risk,
        question_cost=inquiry.question_cost,
        thread_id=inquiry.thread_id,
        decision_key=inquiry.decision_key,
        information_gain=inquiry.information_gain,
        causal_relevance=inquiry.causal_relevance,
        conversation_continuity=inquiry.conversation_continuity,
    )


def _legacy_requirement_candidates(state: AgentState) -> List[QuestionCandidate]:
    """Compatibility path for focused unit tests that call this layer directly."""
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    coverage_state = state.get("requirement_coverage", {})
    result: List[QuestionCandidate] = []
    for key in state.get("eligible_requirement_keys", []):
        requirement = store.get(key)
        payload = coverage_state.get(key)
        if requirement is None or payload is None or requirement.scope != scope:
            continue
        coverage = RequirementCoverageRecord.model_validate(payload)
        target_facets = []
        for facet in requirement.facets:
            if not facet.required:
                continue
            facet_state = coverage.facets.get(facet.id)
            if facet_state is None or facet_state.state == RequirementFacetState.UNKNOWN:
                target_facets.append(facet.id)
        known = list(coverage.candidate_fact_ids)
        for facet in coverage.facets.values():
            for identity in facet.fact_ids:
                if identity not in known:
                    known.append(identity)
        rule_ids = list(dict.fromkeys(
            source.activation_rule_id
            for source in requirement.activation_sources
            if source.activation_rule_id
        ))
        inquiry = ProductInquiry(
            id=f"{scope.value}|requirement|{requirement.id}|{','.join(target_facets) or '__requirement__'}",
            source=InquirySource.REQUIREMENT,
            scope=scope,
            topic=requirement.topic,
            anchor_gap=requirement.parent_gap,
            objective=requirement.description or requirement.label,
            question_hint="Ask one natural product question about the unresolved requirement.",
            reason=f"Active requirement '{requirement.label}' still needs a product decision.",
            requirement_key=key,
            requirement_id=requirement.id,
            target_facets=target_facets,
            known_fact_ids=known,
            activation_rule_ids=rule_ids,
            coverage_status=coverage.status,
            architecture_impact=requirement.priority_hints.architecture_impact,
            business_risk=requirement.priority_hints.business_risk,
        )
        result.append(_candidate_from_inquiry(inquiry))
    return result


def build_question_candidates(state: AgentState) -> List[QuestionCandidate]:
    payloads = state.get("open_inquiries")
    if payloads is None:
        return _legacy_requirement_candidates(state)
    return [
        _candidate_from_inquiry(ProductInquiry.model_validate(payload))
        for payload in payloads
    ]


def question_candidate_builder_node(state: AgentState) -> dict:
    return {
        "question_candidates": [
            candidate.model_dump(mode="json")
            for candidate in build_question_candidates(state)
        ]
    }


def _history_signature(entry: dict) -> tuple[str | None, tuple[str, ...]]:
    # Requirement history predates inquiry IDs. Preserve requirement identity
    # when present so migrated checkpoints still suppress the exact target.
    identity = entry.get("requirement_key") or entry.get("inquiry_id")
    return identity, tuple(entry.get("target_facets") or [])


def _candidate_signature(candidate: QuestionCandidate) -> tuple[str | None, tuple[str, ...]]:
    identity = (
        candidate.requirement_key
        if candidate.source == InquirySource.REQUIREMENT
        else candidate.inquiry_id
    )
    return identity, tuple(candidate.target_facets)


def _thread_decision_repeat_count(state: AgentState, candidate: QuestionCandidate) -> int:
    if not candidate.thread_id or not candidate.decision_key:
        return 0
    return sum(
        1
        for entry in state.get("requirement_question_history", [])[-12:]
        if entry.get("thread_id") == candidate.thread_id
        and entry.get("decision_key") == candidate.decision_key
    )


def _requirement_reasons(
    state: AgentState,
    candidate: QuestionCandidate,
) -> List[CandidateBlockReason]:
    reasons: List[CandidateBlockReason] = []
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    coverage_state = state.get("requirement_coverage", {})
    dependency_state = state.get("requirement_dependency_state", {})

    if not candidate.requirement_key:
        return [CandidateBlockReason.REQUIREMENT_MISSING]

    requirement: ActiveRequirement | None = store.get(candidate.requirement_key)
    if requirement is None:
        return [CandidateBlockReason.REQUIREMENT_MISSING]

    if requirement.scope != scope or candidate.scope != scope:
        reasons.append(CandidateBlockReason.WRONG_SCOPE)
    if requirement.status != RequirementStatus.ACTIVE:
        reasons.append(CandidateBlockReason.REQUIREMENT_NOT_ACTIVE)

    dependency = dependency_state.get(candidate.requirement_key)
    if not dependency or not dependency.get("eligible"):
        reasons.append(CandidateBlockReason.DEPENDENCY_BLOCKED)

    payload = coverage_state.get(candidate.requirement_key)
    coverage = None
    if payload is None:
        reasons.append(CandidateBlockReason.COVERAGE_MISSING)
    else:
        coverage = RequirementCoverageRecord.model_validate(payload)
        if coverage.status not in ELIGIBLE_COVERAGE_STATUSES:
            reasons.append(CandidateBlockReason.COVERAGE_TERMINAL)

    if coverage is not None and requirement.facets:
        unresolved = []
        for facet in requirement.facets:
            if not facet.required:
                continue
            facet_state = coverage.facets.get(facet.id)
            if facet_state is None or facet_state.state == RequirementFacetState.UNKNOWN:
                unresolved.append(facet.id)
        if not unresolved or set(unresolved) != set(candidate.target_facets):
            reasons.append(CandidateBlockReason.NO_UNRESOLVED_FACETS)

    return reasons


def filter_question_candidates(
    state: AgentState,
    candidates: List[QuestionCandidate] | None = None,
) -> tuple[List[QuestionCandidate], Dict[str, CandidateEligibilityDecision]]:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    history = state.get("requirement_question_history", [])
    recent_signatures = {_history_signature(entry) for entry in history[-3:]}

    if candidates is None:
        candidates = [
            QuestionCandidate.model_validate(item)
            for item in state.get("question_candidates", [])
        ]

    current_inquiry_ids = {
        ProductInquiry.model_validate(item).id
        for item in state.get("open_inquiries", [])
    } if state.get("open_inquiries") is not None else set()

    accepted: List[QuestionCandidate] = []
    decisions: Dict[str, CandidateEligibilityDecision] = {}

    for candidate in candidates:
        reasons: List[CandidateBlockReason] = []
        if candidate.scope != scope:
            reasons.append(CandidateBlockReason.WRONG_SCOPE)

        if current_inquiry_ids and candidate.inquiry_id not in current_inquiry_ids:
            reasons.append(CandidateBlockReason.INQUIRY_MISSING)

        if candidate.source == InquirySource.REQUIREMENT:
            reasons.extend(_requirement_reasons(state, candidate))

        signature = _candidate_signature(candidate)
        repeat_count = _thread_decision_repeat_count(state, candidate)
        ambiguous_thread_retry = (
            candidate.thread_id
            and repeat_count == 1
            and state.get("extraction_status") == "NO_FACTS_FOUND"
        )
        if signature in recent_signatures and not ambiguous_thread_retry:
            reasons.append(CandidateBlockReason.RECENTLY_ASKED_SAME_TARGET)

        if repeat_count >= 2:
            reasons.append(CandidateBlockReason.REPEATED_THREAD_DECISION)
        elif repeat_count == 1 and state.get("extraction_status") != "NO_FACTS_FOUND":
            reasons.append(CandidateBlockReason.REPEATED_THREAD_DECISION)

        decision = CandidateEligibilityDecision(
            candidate_id=candidate.id,
            eligible=not reasons,
            reasons=list(dict.fromkeys(reasons)),
        )
        decisions[candidate.id] = decision
        if decision.eligible:
            accepted.append(candidate)

    # One rephrased follow-up is allowed only for legacy/non-thread candidates.
    # Thread decisions have a hard circuit breaker so a semantic uncertainty
    # cannot be paraphrased indefinitely.
    if not accepted:
        for candidate in candidates:
            decision = decisions[candidate.id]
            if (
                not candidate.thread_id
                and decision.reasons == [CandidateBlockReason.RECENTLY_ASKED_SAME_TARGET]
            ):
                decisions[candidate.id] = decision.model_copy(
                    update={"eligible": True, "reasons": []}
                )
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
