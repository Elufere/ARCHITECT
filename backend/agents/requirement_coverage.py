"""Coverage state for product-specific active requirements.

This layer is intentionally separate from schema-gap coverage. A broad schema
gap may provide candidate evidence for a requirement, but cannot by itself mark
that more specific requirement resolved.
"""
from __future__ import annotations

from enum import Enum
import json
from typing import Dict, Iterable, List, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from agents.discovery_coverage import fact_id
from agents.llm import get_structured_model
from agents.llm_errors import ExtractionFailed, raise_if_llm_failure
from agents.requirements import (
    ActiveRequirement,
    RequirementEvidenceRef,
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
    not_applicable_facets: Dict[str, List[str]] = Field(default_factory=dict)

    @field_validator("covered_facets", "not_applicable_facets", mode="before")
    @classmethod
    def normalize_single_fact_ids(cls, value):
        """Structured models sometimes emit one fact ID as a scalar string.

        The protocol is semantically unambiguous in that case, so normalize the
        wire shape instead of spending a repair call or aborting the interview.
        """
        if not isinstance(value, dict):
            return value
        normalized = {}
        for facet_id, fact_ids in value.items():
            normalized[facet_id] = [fact_ids] if isinstance(fact_ids, str) else fact_ids
        return normalized

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

    for state_name, mapping in (
        ("Covered", assessment.covered_facets),
        ("Not-applicable", assessment.not_applicable_facets),
    ):
        for facet_id, ids in mapping.items():
            if not ids:
                raise ValueError(f"{state_name} facet '{facet_id}' requires supporting fact IDs")
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
            kept = [identity for identity in old.fact_ids if identity in valid]
            if old.state == RequirementFacetState.NOT_APPLICABLE:
                facets[facet_id] = FacetCoverage(
                    facet_id=facet_id,
                    state=RequirementFacetState.NOT_APPLICABLE if kept else RequirementFacetState.UNKNOWN,
                    fact_ids=kept,
                )
                continue
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
    for facet_id, ids in assessment.not_applicable_facets.items():
        facets[facet_id] = FacetCoverage(
            facet_id=facet_id,
            state=RequirementFacetState.NOT_APPLICABLE,
            fact_ids=list(dict.fromkeys(ids)),
        )

    status = _status_from_facets(requirement, facets, candidates)
    return base.model_copy(update={
        "facets": facets,
        "status": status,
        "candidate_fact_ids": candidates,
        "expansion_needed": status == RequirementCoverageStatus.NEEDS_EXPANSION,
    })



_requirement_coverage_assessor = None


def requirement_coverage_assessor():
    global _requirement_coverage_assessor
    if _requirement_coverage_assessor is None:
        _requirement_coverage_assessor = get_structured_model(
            call_name="requirement_coverage.assess",
            schema=RequirementCoverageAssessment,
        )
    return _requirement_coverage_assessor


REQUIREMENT_COVERAGE_INSTRUCTION = """Assess ONLY the selected product requirement facets.

You are given:
- the exact PM question,
- the user's latest response,
- a requirement description,
- the unresolved target facets,
- grounded confirmed facts with stable fact IDs.

A facet is COVERED only when one or more supplied facts explicitly establish the
information described by that facet and the latest answer supports using those
facts for this requirement. Do not infer missing behavior from the fact that the
requirement was activated. Do not treat a broad parent schema fact as complete
unless its actual content establishes the facet.

A facet is NOT_APPLICABLE only when the supplied facts explicitly establish that
the facet does not apply. Silence, uncertainty, "I don't know", or lack of detail
is not NOT_APPLICABLE.

Return fact IDs exactly as supplied. Do not invent IDs. EVERY mapping value must
be a JSON ARRAY of fact-ID strings, even when exactly one fact supports the facet.
Leave unresolved facets out of both mappings. Assess only the target facet IDs.
"""


REQUIREMENT_COVERAGE_REPAIR_INSTRUCTION = """
The previous requirement-facet assessment was invalid. Repair only the response
format/protocol. Do not add new product knowledge. Use ONLY the supplied target
facet IDs and ONLY the supplied confirmed fact IDs. A facet may be omitted when
the evidence does not establish it. Return no explanation outside the structured
assessment.
"""


def _normalize_requirement_coverage_assessment(result) -> RequirementCoverageAssessment:
    return (
        result
        if isinstance(result, RequirementCoverageAssessment)
        else RequirementCoverageAssessment.model_validate(result)
    )


def _validate_requirement_coverage_protocol(
    requirement: ActiveRequirement,
    assessment: RequirementCoverageAssessment,
    target_facets: Sequence[str],
    allowed_fact_ids: Sequence[str],
) -> None:
    outside = (
        set(assessment.covered_facets)
        | set(assessment.not_applicable_facets)
    ) - set(target_facets)
    if outside:
        raise ValueError(f"Coverage assessment returned non-target facets: {sorted(outside)}")
    validate_coverage_assessment(requirement, assessment, allowed_fact_ids)


def _assess_requirement_facets_with_repair(
    payload: dict,
    requirement: ActiveRequirement,
    target_facets: Sequence[str],
    allowed_fact_ids: Sequence[str],
) -> RequirementCoverageAssessment:
    messages = [
        SystemMessage(content=REQUIREMENT_COVERAGE_INSTRUCTION),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        assessment = _normalize_requirement_coverage_assessment(
            requirement_coverage_assessor().invoke(messages)
        )
        _validate_requirement_coverage_protocol(
            requirement, assessment, target_facets, allowed_fact_ids
        )
        return assessment
    except Exception as first_exc:
        raise_if_llm_failure(first_exc)
        print(f"REQUIREMENT COVERAGE REPAIR: {first_exc}")

    repair_payload = {
        **payload,
        "repair": {
            "allowed_target_facets": list(target_facets),
            "allowed_fact_ids": list(allowed_fact_ids),
            "instruction": (
                "Return only supported mappings using these exact IDs. "
                "Omit any unresolved facet."
            ),
        },
    }
    try:
        repaired = _normalize_requirement_coverage_assessment(
            requirement_coverage_assessor().invoke([
                SystemMessage(content=(
                    REQUIREMENT_COVERAGE_INSTRUCTION
                    + REQUIREMENT_COVERAGE_REPAIR_INSTRUCTION
                )),
                HumanMessage(content=json.dumps(repair_payload, ensure_ascii=False)),
            ])
        )
        _validate_requirement_coverage_protocol(
            requirement, repaired, target_facets, allowed_fact_ids
        )
        return repaired
    except Exception as second_exc:
        raise_if_llm_failure(second_exc)
        raise ExtractionFailed(
            "Requirement facet coverage assessment failed after one repair attempt"
        ) from second_exc


def _latest_question_and_answer(state: AgentState) -> tuple[str, str] | None:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    answer = messages[-1].content
    question = next(
        (message.content for message in reversed(messages[:-1]) if isinstance(message, AIMessage)),
        "",
    )
    return question, answer


def _attach_current_turn_requirement_evidence(
    requirement: ActiveRequirement,
    knowledge: Sequence[KnowledgeItem],
    *,
    turn: int,
    question: str,
) -> ActiveRequirement:
    refs = list(requirement.evidence_refs)
    seen = {ref.fact_id for ref in refs}
    for item in knowledge:
        if (
            item.scope == requirement.scope
            and item.knowledge_state == KnowledgeState.CONFIRMED
            and item.source_turn == turn
            and (not item.source_question or item.source_question == question)
        ):
            identity = fact_id(item)
            if identity not in seen:
                seen.add(identity)
                refs.append(RequirementEvidenceRef(
                    fact_id=identity,
                    source_turn=item.source_turn,
                    note="direct answer to selected requirement question",
                ))
    return requirement.model_copy(update={"evidence_refs": refs})


def assess_selected_requirement_answer(
    state: AgentState,
    store: RequirementStore,
    coverage: Dict[str, dict],
) -> tuple[RequirementStore, Dict[str, dict]]:
    if state.get("planner_source") != "requirement":
        return store, coverage

    selected = state.get("selected_requirement_candidate")
    exchange = _latest_question_and_answer(state)
    if not selected or exchange is None:
        return store, coverage

    candidate = dict(selected)
    requirement_key = candidate.get("requirement_key")
    target_facets = list(candidate.get("target_facets") or [])
    requirement = store.get(requirement_key)
    if requirement is None or requirement.status != RequirementStatus.ACTIVE:
        return store, coverage

    question, answer = exchange
    knowledge = state.get("discovered_knowledge", [])
    requirement = _attach_current_turn_requirement_evidence(
        requirement,
        knowledge,
        turn=state.get("turn_count", 0),
        question=question,
    )
    updated_store = dict(store)
    updated_store[requirement_key] = requirement

    if not target_facets:
        return updated_store, coverage

    allowed_ids = candidate_fact_ids(requirement, knowledge)
    if not allowed_ids:
        return updated_store, coverage

    fact_index = _confirmed_fact_index(knowledge, requirement.scope)
    supplied_facts = [
        {
            "fact_id": identity,
            "topic": fact_index[identity].topic.value,
            "key": fact_index[identity].key,
            "value": fact_index[identity].value,
            "evidence": fact_index[identity].evidence,
            "source_turn": fact_index[identity].source_turn,
        }
        for identity in allowed_ids
        if identity in fact_index
    ]
    target = {
        facet.id: {
            "label": facet.label,
            "description": facet.description,
        }
        for facet in requirement.facets
        if facet.id in target_facets
    }

    payload = {
        "question": question,
        "latest_response": answer,
        "requirement_id": requirement.id,
        "requirement": requirement.description or requirement.label,
        "target_facets": target,
        "confirmed_facts": supplied_facts,
    }
    assessment = _assess_requirement_facets_with_repair(
        payload,
        requirement,
        target_facets,
        allowed_ids,
    )

    existing_payload = coverage.get(requirement_key)
    existing = (
        RequirementCoverageRecord.model_validate(existing_payload)
        if existing_payload else None
    )
    record = apply_requirement_coverage_assessment(
        requirement,
        knowledge,
        assessment,
        existing,
    )

    updated_coverage = dict(coverage)
    updated_coverage[requirement_key] = record.model_dump(mode="json")
    updated_store[requirement_key] = requirement.model_copy(update={
        "status": (
            RequirementStatus.RESOLVED
            if record.status == RequirementCoverageStatus.RESOLVED
            else RequirementStatus.ACTIVE
        )
    })
    return updated_store, updated_coverage

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
    store = dict(state.get("active_requirements", {}))
    coverage = dict(state.get("requirement_coverage", {}))

    # When the previous planner move was requirement-driven, first map the
    # grounded answer to the selected requirement facets. This never marks the
    # broad parent schema gap resolved.
    store, coverage = assess_selected_requirement_answer(state, store, coverage)

    store, coverage = reconcile_requirement_coverage(
        store,
        state.get("discovered_knowledge", []),
        scope,
        coverage,
    )
    return {
        "active_requirements": store,
        "requirement_coverage": coverage,
    }
