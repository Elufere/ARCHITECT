"""Coverage state for product-specific active requirements.

This layer is intentionally separate from schema-gap coverage. A broad schema
gap may provide candidate evidence for a requirement, but cannot by itself mark
that more specific requirement resolved.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Sequence

from pydantic import BaseModel, Field, model_validator

from agents.discovery_coverage import fact_id
from agents.requirements import (
    ActiveRequirement,
    RequirementFacet,
    RequirementStatus,
    RequirementStore,
    requirement_store_key,
)
from agents.role_utils import roles_match
from agents.state import AgentState, DiscoveryScope, KnowledgeItem, KnowledgeState


class RequirementCoverageStatus(str, Enum):
    UNSEEN = "UNSEEN"
    KNOWN_SHALLOW = "KNOWN_SHALLOW"
    NEEDS_EXPANSION = "NEEDS_EXPANSION"
    RESOLVED = "RESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DEFERRED = "DEFERRED"
    INACTIVE = "INACTIVE"


class RequirementFacetState(str, Enum):
    UNKNOWN = "UNKNOWN"
    COVERED = "COVERED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FacetCoverage(BaseModel):
    facet_id: str
    state: RequirementFacetState = RequirementFacetState.UNKNOWN
    fact_ids: List[str] = Field(default_factory=list)


class RequirementCoverageRecord(BaseModel):
    requirement_id: str
    scope: DiscoveryScope
    status: RequirementCoverageStatus
    facets: Dict[str, FacetCoverage] = Field(default_factory=dict)
    candidate_fact_ids: List[str] = Field(default_factory=list)
    expansion_needed: bool = False


class RequirementCoverageAssessment(BaseModel):
    """Validated semantic assessment supplied by a future coverage evaluator.

    The assessment may only reference facets declared by the requirement and
    confirmed fact IDs supplied to that evaluator.
    """

    covered_facets: Dict[str, List[str]] = Field(default_factory=dict)
    not_applicable_facets: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_overlap(self):
        overlap = set(self.covered_facets).intersection(self.not_applicable_facets)
        if overlap:
            raise ValueError(f"Facet cannot be covered and not applicable: {sorted(overlap)}")
        return self


def _confirmed_fact_index(knowledge: Sequence[KnowledgeItem], scope: DiscoveryScope) -> Dict[str, KnowledgeItem]:
    return {
        fact_id(item): item
        for item in knowledge
        if item.scope == scope and item.knowledge_state == KnowledgeState.CONFIRMED
    }


def _parent_gap_fact_ids(
    requirement: ActiveRequirement,
    knowledge: Sequence[KnowledgeItem],
) -> List[str]:
    if not requirement.parent_gap:
        return []
    key, _, role = requirement.parent_gap.partition("::")
    result = []
    for item in knowledge:
        if (
            item.scope == requirement.scope
            and item.knowledge_state == KnowledgeState.CONFIRMED
            and item.topic == requirement.topic
            and item.key == key
            and (not role or roles_match(item.role, role))
        ):
            result.append(fact_id(item))
    return result


def candidate_fact_ids(
    requirement: ActiveRequirement,
    knowledge: Sequence[KnowledgeItem],
) -> List[str]:
    """Facts allowed to support requirement coverage.

    Parent-gap facts are candidates because they share the requirement's schema
    area. Explicit requirement evidence references may add cross-topic facts.
    Activation provenance is intentionally excluded: why a requirement exists is
    not evidence that its answer is known.
    """
    index = _confirmed_fact_index(knowledge, requirement.scope)
    ids = list(_parent_gap_fact_ids(requirement, knowledge))
    for evidence in requirement.evidence_refs:
        if evidence.fact_id in index and evidence.fact_id not in ids:
            ids.append(evidence.fact_id)
    return ids


def validate_coverage_assessment(
    requirement: ActiveRequirement,
    assessment: RequirementCoverageAssessment,
    allowed_fact_ids: Iterable[str],
) -> None:
    facet_ids = {facet.id for facet in requirement.facets}
    allowed = set(allowed_fact_ids)

    unknown = set(assessment.covered_facets).union(assessment.not_applicable_facets) - facet_ids
    if unknown:
        raise ValueError(f"Unknown requirement facets: {sorted(unknown)}")

    for facet_id, ids in assessment.covered_facets.items():
        if not ids:
            raise ValueError(f"Covered facet '{facet_id}' requires supporting fact IDs")
        invalid = set(ids) - allowed
        if invalid:
            raise ValueError(
                f"Facet '{facet_id}' references facts outside requirement coverage evidence: {sorted(invalid)}"
            )


def _initial_facets(requirement: ActiveRequirement) -> Dict[str, FacetCoverage]:
    return {
        facet.id: FacetCoverage(facet_id=facet.id)
        for facet in requirement.facets
    }


def _status_from_facets(
    requirement: ActiveRequirement,
    facets: Dict[str, FacetCoverage],
    candidates: List[str],
) -> RequirementCoverageStatus:
    required = [facet.id for facet in requirement.facets if facet.required]

    if not required:
        # Requirements without an explicit facet contract stay shallow when facts
        # exist; they cannot become resolved merely because a parent gap is known.
        return (
            RequirementCoverageStatus.KNOWN_SHALLOW
            if candidates else RequirementCoverageStatus.UNSEEN
        )

    resolved = all(
        facets[facet_id].state in {
            RequirementFacetState.COVERED,
            RequirementFacetState.NOT_APPLICABLE,
        }
        for facet_id in required
    )
    if resolved:
        return RequirementCoverageStatus.RESOLVED

    any_assessed = any(
        facet.state != RequirementFacetState.UNKNOWN
        for facet in facets.values()
    )
    if any_assessed:
        return RequirementCoverageStatus.NEEDS_EXPANSION
    if candidates:
        return RequirementCoverageStatus.KNOWN_SHALLOW
    return RequirementCoverageStatus.UNSEEN


def reconcile_requirement_coverage_record(
    requirement: ActiveRequirement,
    knowledge: Sequence[KnowledgeItem],
    existing: RequirementCoverageRecord | None = None,
) -> RequirementCoverageRecord:
    """Reconcile coverage against current confirmed evidence.

    Stale fact references are removed after corrections. New parent-gap facts are
    candidates only; they are never auto-assigned to facets.
    """
    if requirement.status == RequirementStatus.INACTIVE:
        return RequirementCoverageRecord(
            requirement_id=requirement.id,
            scope=requirement.scope,
            status=RequirementCoverageStatus.INACTIVE,
            facets=_initial_facets(requirement),
        )
    if requirement.status == RequirementStatus.NOT_APPLICABLE:
        return RequirementCoverageRecord(
            requirement_id=requirement.id,
            scope=requirement.scope,
            status=RequirementCoverageStatus.NOT_APPLICABLE,
            facets=_initial_facets(requirement),
        )
    if requirement.status == RequirementStatus.DEFERRED:
        return RequirementCoverageRecord(
            requirement_id=requirement.id,
            scope=requirement.scope,
            status=RequirementCoverageStatus.DEFERRED,
            facets=_initial_facets(requirement),
        )

    candidates = candidate_fact_ids(requirement, knowledge)
    valid = set(candidates)
    facets = _initial_facets(requirement)

    if existing is not None:
        for facet_id, old in existing.facets.items():
            if facet_id not in facets:
                continue
            if old.state == RequirementFacetState.NOT_APPLICABLE:
                facets[facet_id] = old
                continue
            kept = [identity for identity in old.fact_ids if identity in valid]
            facets[facet_id] = FacetCoverage(
                facet_id=facet_id,
                state=RequirementFacetState.COVERED if kept else RequirementFacetState.UNKNOWN,
                fact_ids=kept,
            )

    status = _status_from_facets(requirement, facets, candidates)
    return RequirementCoverageRecord(
        requirement_id=requirement.id,
        scope=requirement.scope,
        status=status,
        facets=facets,
        candidate_fact_ids=candidates,
        expansion_needed=status == RequirementCoverageStatus.NEEDS_EXPANSION,
    )


def apply_requirement_coverage_assessment(
    requirement: ActiveRequirement,
    knowledge: Sequence[KnowledgeItem],
    assessment: RequirementCoverageAssessment,
    existing: RequirementCoverageRecord | None = None,
) -> RequirementCoverageRecord:
    candidates = candidate_fact_ids(requirement, knowledge)
    validate_coverage_assessment(requirement, assessment, candidates)

    base = reconcile_requirement_coverage_record(requirement, knowledge, existing)
    facets = dict(base.facets)

    for facet_id, ids in assessment.covered_facets.items():
        facets[facet_id] = FacetCoverage(
            facet_id=facet_id,
            state=RequirementFacetState.COVERED,
            fact_ids=list(dict.fromkeys(ids)),
        )
    for facet_id in assessment.not_applicable_facets:
        facets[facet_id] = FacetCoverage(
            facet_id=facet_id,
            state=RequirementFacetState.NOT_APPLICABLE,
        )

    status = _status_from_facets(requirement, facets, candidates)
    return base.model_copy(update={
        "facets": facets,
        "status": status,
        "candidate_fact_ids": candidates,
        "expansion_needed": status == RequirementCoverageStatus.NEEDS_EXPANSION,
    })


def reconcile_requirement_coverage(
    store: RequirementStore,
    knowledge: Sequence[KnowledgeItem],
    scope: DiscoveryScope,
    current: Dict[str, dict] | None = None,
) -> tuple[RequirementStore, Dict[str, dict]]:
    """Recompute coverage and synchronize RESOLVED <-> ACTIVE status.

    Applicability statuses remain owned by the activation layer. Coverage owns
    only whether an applicable requirement has enough grounded detail.
    """
    current = current or {}
    updated_store = dict(store)
    coverage: Dict[str, dict] = {}

    for key, requirement in store.items():
        if requirement.scope != scope:
            continue
        existing_payload = current.get(key)
        existing = (
            RequirementCoverageRecord.model_validate(existing_payload)
            if existing_payload else None
        )
        record = reconcile_requirement_coverage_record(requirement, knowledge, existing)
        coverage[key] = record.model_dump(mode="json")

        if requirement.status in {RequirementStatus.ACTIVE, RequirementStatus.RESOLVED}:
            new_status = (
                RequirementStatus.RESOLVED
                if record.status == RequirementCoverageStatus.RESOLVED
                else RequirementStatus.ACTIVE
            )
            if new_status != requirement.status:
                updated_store[key] = requirement.model_copy(update={"status": new_status})

    # Preserve other-scope coverage untouched.
    for key, payload in current.items():
        if key not in coverage:
            coverage[key] = payload

    return updated_store, coverage


def requirement_coverage_node(state: AgentState) -> dict:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store, coverage = reconcile_requirement_coverage(
        state.get("active_requirements", {}),
        state.get("discovered_knowledge", []),
        scope,
        state.get("requirement_coverage", {}),
    )
    return {
        "active_requirements": store,
        "requirement_coverage": coverage,
    }
