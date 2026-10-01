"""Build and deterministically filter question candidates from open inquiries.

The planner no longer traverses schema gaps. Inquiries can come from the
foundational product model, active product-specific requirements, or blocking
consistency validation.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.inquiries import InquirySource, ProductInquiry
from agents.llm import get_structured_model
from agents.llm_errors import raise_if_llm_failure
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
    EXPLICITLY_DEFERRED_DECISION = "EXPLICITLY_DEFERRED_DECISION"
    EXPLICITLY_REJECTED_DECISION = "EXPLICITLY_REJECTED_DECISION"
    FOUNDER_CLOSURE_NONBLOCKING = "FOUNDER_CLOSURE_NONBLOCKING"


class CandidateEligibilityDecision(BaseModel):
    candidate_id: str
    eligible: bool
    reasons: List[CandidateBlockReason] = Field(default_factory=list)


class CompletionRequirementArbitration(BaseModel):
    blocking_candidate_ids: List[str] = Field(default_factory=list)
    nonblocking_candidate_ids: List[str] = Field(default_factory=list)
    reasons: Dict[str, str] = Field(default_factory=dict)


_completion_requirement_model = None


def completion_requirement_model():
    global _completion_requirement_model
    if _completion_requirement_model is None:
        _completion_requirement_model = get_structured_model(
            call_name="question_candidates.completion_arbitration",
            schema=CompletionRequirementArbitration,
            max_tokens=700,
        )
    return _completion_requirement_model


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


def _control_boundary_block_reason(
    state: AgentState,
    candidate: QuestionCandidate,
) -> CandidateBlockReason | None:
    """Hard-block decisions the founder already deferred, rejected, or closed."""
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    scope_value = getattr(scope, "value", scope)
    for boundary in state.get("discovery_boundaries", []) or []:
        if not isinstance(boundary, dict) or boundary.get("reopened_at_turn"):
            continue
        boundary_type = boundary.get("type")
        if boundary_type not in {
            "decision_deferral",
            "design_deferral",
            "implementation_deferred",
            "rejected_inquiry",
            "product_scope_closed",
        }:
            continue
        if boundary.get("scope") and boundary.get("scope") != scope_value:
            continue

        matches = any((
            bool(
                boundary.get("requirement_key")
                and candidate.requirement_key
                and boundary.get("requirement_key") == candidate.requirement_key
            ),
            bool(
                boundary.get("requirement_id")
                and candidate.requirement_id
                and boundary.get("requirement_id") == candidate.requirement_id
            ),
            bool(
                boundary.get("inquiry_id")
                and candidate.inquiry_id
                and boundary.get("inquiry_id") == candidate.inquiry_id
            ),
            bool(
                boundary.get("decision_key")
                and candidate.decision_key
                and boundary.get("decision_key") == candidate.decision_key
            ),
        ))
        if not matches:
            continue

        if boundary_type in {
            "decision_deferral",
            "design_deferral",
            "implementation_deferred",
        }:
            return CandidateBlockReason.EXPLICITLY_DEFERRED_DECISION
        return CandidateBlockReason.EXPLICITLY_REJECTED_DECISION
    return None


COMPLETION_ARBITRATION_INSTRUCTION = """The founder has explicitly said product
discovery is sufficiently covered and wants to finish. Review ONLY the supplied
already-activated REQUIREMENT candidates and decide which, if any, are materially
blocking before a PRD confirmation can be requested.

Keep a requirement BLOCKING only when leaving it unresolved would make the
currently confirmed product materially incoherent or ambiguous about a core
product rule already implicated by the founder's model. Examples can include
money movement/finality, irreversible state transitions, authorization, a
material compliance constraint, or a high-risk exception that is already part of
the confirmed workflow.

Mark NONBLOCKING when it is optional depth, hypothetical completeness, UI/content
detail, implementation mechanics, speculative limits, polish, or another detail
that can safely remain unresolved in this PRD. Founder closure is a strong
stopping preference: do not keep asking merely because more detail is possible.

Do not invent new requirements, facts, risks, or scenarios. Do not reinterpret a
deferred/rejected item as blocking. Classify every supplied candidate exactly
once, using its exact candidate id. reasons may briefly explain each choice."""


def arbitrate_completion_candidates(
    state: AgentState,
    candidates: List[QuestionCandidate],
    decisions: Dict[str, CandidateEligibilityDecision],
) -> tuple[
    List[QuestionCandidate],
    Dict[str, CandidateEligibilityDecision],
    bool,
]:
    if not state.get("founder_requested_completion"):
        return candidates, decisions, False

    requirements = [
        candidate
        for candidate in candidates
        if candidate.source == InquirySource.REQUIREMENT
    ]
    always_blocking = [
        candidate
        for candidate in candidates
        if candidate.source != InquirySource.REQUIREMENT
    ]

    blocking_requirement_ids: set[str] = set()
    if requirements:
        payload = {
            "founder_completion_evidence": state.get("completion_request_evidence"),
            "confirmed_product_model": state.get("product_model", {}),
            "active_control_boundaries": state.get("discovery_boundaries", [])[-30:],
            "requirement_candidates": [
                {
                    "candidate_id": candidate.id,
                    "requirement_id": candidate.requirement_id,
                    "objective": candidate.objective,
                    "reason": candidate.reason,
                    "target_facets": candidate.target_facets,
                    "architecture_impact": candidate.architecture_impact,
                    "business_risk": candidate.business_risk,
                }
                for candidate in requirements
            ],
        }
        try:
            result = completion_requirement_model().invoke([
                SystemMessage(content=COMPLETION_ARBITRATION_INSTRUCTION),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ])
            review = (
                result
                if isinstance(result, CompletionRequirementArbitration)
                else CompletionRequirementArbitration.model_validate(result)
            )
        except Exception as exc:
            raise_if_llm_failure(exc)
            raise ValueError("Completion requirement arbitration failed") from exc

        expected = {candidate.id for candidate in requirements}
        blocking = set(review.blocking_candidate_ids)
        nonblocking = set(review.nonblocking_candidate_ids)
        if blocking & nonblocking or blocking | nonblocking != expected:
            raise ValueError(
                "Completion arbitration must classify every requirement candidate exactly once"
            )
        blocking_requirement_ids = blocking

        for candidate in requirements:
            if candidate.id in blocking_requirement_ids:
                continue
            existing = decisions[candidate.id]
            decisions[candidate.id] = existing.model_copy(update={
                "eligible": False,
                "reasons": [
                    *existing.reasons,
                    CandidateBlockReason.FOUNDER_CLOSURE_NONBLOCKING,
                ],
            })

    eligible = [
        *always_blocking,
        *[
            candidate
            for candidate in requirements
            if candidate.id in blocking_requirement_ids
        ],
    ]
    return eligible, decisions, not eligible


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

        boundary_reason = _control_boundary_block_reason(state, candidate)
        if boundary_reason is not None:
            reasons.append(boundary_reason)

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
    eligible, decisions, completion_complete = arbitrate_completion_candidates(
        state,
        eligible,
        decisions,
    )
    return {
        "eligible_question_candidates": [
            candidate.model_dump(mode="json")
            for candidate in eligible
        ],
        "question_candidate_eligibility": {
            candidate_id: decision.model_dump(mode="json")
            for candidate_id, decision in decisions.items()
        },
        "completion_arbitration_complete": completion_complete,
    }
