"""Deterministic founder-facing projection of Architect's internal discovery state.

This module is intentionally presentation-facing but backend-owned. It converts
confirmed product knowledge, emergent product structure, unresolved requirements,
current inquiries, and explicit founder deferrals into a stable structure that an
API/UI can render without understanding LangGraph internals.

No LLM calls occur here. The projection must never upgrade inferred knowledge,
invent product facts, or classify a generic product entity as an integration.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from agents.discovery_coverage import fact_id
from agents.product_concepts import ProductConcept, ProductConceptKind, concept_id
from agents.requirement_coverage import (
    RequirementCoverageRecord,
    RequirementCoverageStatus,
    RequirementFacetState,
)
from agents.requirements import ActiveRequirement, RequirementStatus
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeItem, KnowledgeState


class UnderstandingItemState(str, Enum):
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    OPEN = "open"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class UnderstandingItem(BaseModel):
    id: str
    label: str
    detail: Optional[str] = None
    state: UnderstandingItemState
    source: str
    source_refs: list[str] = Field(default_factory=list)
    topic: Optional[DiscoveryTopic] = None
    key: Optional[str] = None
    role: Optional[str] = None


class UnderstandingSection(BaseModel):
    id: str
    title: str
    items: list[UnderstandingItem]


class UnderstandingProjection(BaseModel):
    scope: DiscoveryScope
    sections: list[UnderstandingSection]


TOPIC_SECTIONS: dict[DiscoveryTopic, tuple[str, str]] = {
    DiscoveryTopic.USER_ROLES: ("users", "Users & roles"),
    DiscoveryTopic.USER_GOALS: ("goals", "Goals & outcomes"),
    DiscoveryTopic.CORE_WORKFLOW: ("workflow", "Core workflow"),
    DiscoveryTopic.BUSINESS_RULES: ("business_rules", "Business rules"),
    DiscoveryTopic.CONSTRAINTS: ("constraints", "Constraints"),
    DiscoveryTopic.MVP_SCOPE: ("mvp_scope", "MVP scope"),
    DiscoveryTopic.EXCEPTIONS: ("exceptions", "Exceptions"),
    DiscoveryTopic.EDGE_CASES: ("edge_cases", "Edge cases"),
}

SECTION_ORDER = [
    "users",
    "goals",
    "workflow",
    "business_rules",
    "constraints",
    "mvp_scope",
    "exceptions",
    "edge_cases",
    "product_structure",
    "open_decisions",
    "deferred_decisions",
]

FIELD_LABELS = {
    "primary_users": "Primary users",
    "secondary_users": "Secondary users",
    "responsibilities": "Responsibilities",
    "permissions": "Permissions",
    "multiple_roles": "Multiple roles",
    "role_transitions": "Role transitions",
    "primary_user_goals": "Primary user goals",
    "secondary_user_goals": "Secondary user goals",
    "success_criteria": "Success criteria",
    "motivations": "Motivations",
    "trigger": "Workflow trigger",
    "workflow_steps": "Workflow",
    "completion_condition": "Completion condition",
    "downstream_dependency": "External dependency",
    "end_state": "End state",
    "validation_rules": "Validation rules",
    "approval_rules": "Approval rules",
    "eligibility_rules": "Eligibility rules",
    "limits": "Limits",
    "ownership_rules": "Ownership rules",
    "visibility_rules": "Visibility rules",
    "legal_constraints": "Legal constraints",
    "business_constraints": "Business constraints",
    "operational_constraints": "Operational constraints",
    "geographic_constraints": "Geographic constraints",
    "time_constraints": "Time constraints",
    "must_have_features": "Must-have features",
    "nice_to_have_features": "Nice-to-have features",
    "out_of_scope": "Out of scope",
    "success_metrics": "Success metrics",
    "user_cancellations": "Cancellation behavior",
    "timeouts": "Timeout behavior",
    "invalid_actions": "Invalid actions",
    "recovery": "Recovery",
    "duplicate_actions": "Duplicate actions",
    "boundary_conditions": "Boundary conditions",
    "simultaneous_actions": "Simultaneous actions",
    "rare_scenarios": "Rare scenarios",
}


def _humanize(value: str) -> str:
    return " ".join(part for part in value.replace("_", " ").split()).strip().capitalize()


def _role_label(role: str) -> str:
    return " ".join(part.capitalize() for part in role.replace("_", " ").split())


def _field_label(key: str) -> str:
    return FIELD_LABELS.get(key, _humanize(key))


def _scope_matches(raw_scope: Any, scope: DiscoveryScope) -> bool:
    value = getattr(raw_scope, "value", raw_scope)
    return value == scope.value


def _knowledge_items(state: AgentState, scope: DiscoveryScope) -> list[KnowledgeItem]:
    result: list[KnowledgeItem] = []
    for raw in state.get("discovered_knowledge", []):
        item = raw if isinstance(raw, KnowledgeItem) else KnowledgeItem.model_validate(raw)
        if item.scope == scope and item.knowledge_state == KnowledgeState.CONFIRMED:
            result.append(item)
    return result


def _confirmed_fact_item(item: KnowledgeItem) -> list[UnderstandingItem]:
    identity = fact_id(item)
    field = _field_label(str(item.key))

    if item.absence:
        label = f"No {_humanize(str(item.key))}"
        return [UnderstandingItem(
            id=f"fact:{identity}",
            label=label,
            detail="The founder explicitly confirmed this does not apply.",
            state=UnderstandingItemState.CONFIRMED,
            source="fact",
            source_refs=[identity],
            topic=item.topic,
            key=str(item.key),
            role=item.role,
        )]

    if item.topic == DiscoveryTopic.USER_ROLES and item.key in {"primary_users", "secondary_users"}:
        roles = list(item.roles or [])
        if roles:
            return [
                UnderstandingItem(
                    id=f"fact:{identity}:role:{role}",
                    label=_role_label(role),
                    detail=item.value,
                    state=UnderstandingItemState.CONFIRMED,
                    source="fact",
                    source_refs=[identity],
                    topic=item.topic,
                    key=str(item.key),
                    role=role,
                )
                for role in roles
            ]

    detail_parts = [field]
    if item.role:
        detail_parts.insert(0, _role_label(item.role))
    return [UnderstandingItem(
        id=f"fact:{identity}",
        label=item.value,
        detail=" · ".join(detail_parts),
        state=UnderstandingItemState.CONFIRMED,
        source="fact",
        source_refs=[identity],
        topic=item.topic,
        key=str(item.key),
        role=item.role,
    )]


def _concept_items(state: AgentState, scope: DiscoveryScope) -> list[UnderstandingItem]:
    result: list[UnderstandingItem] = []
    for raw in state.get("product_concepts", []):
        concept = raw if isinstance(raw, ProductConcept) else ProductConcept.model_validate(raw)
        if concept.scope != scope:
            continue
        identity = concept_id(concept)
        if concept.kind == ProductConceptKind.ENTITY:
            label = _humanize(concept.subject)
            detail = concept.value
        else:
            label = concept.value
            detail = (
                f"{_humanize(concept.kind.value)} · "
                f"{_humanize(concept.subject)} / {_humanize(concept.relation or '')} / "
                f"{_humanize(concept.object or '')}"
            )
        result.append(UnderstandingItem(
            id=f"concept:{identity}",
            label=label,
            detail=detail,
            state=UnderstandingItemState.CONFIRMED,
            source="concept",
            source_refs=[identity],
        ))
    return result


def _as_requirement(raw: Any) -> ActiveRequirement:
    return raw if isinstance(raw, ActiveRequirement) else ActiveRequirement.model_validate(raw)


def _as_coverage(raw: Any) -> RequirementCoverageRecord:
    return (
        raw
        if isinstance(raw, RequirementCoverageRecord)
        else RequirementCoverageRecord.model_validate(raw)
    )


def _inquiry_dict(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "model_dump"):
        return raw.model_dump(mode="json")
    return {}


def _selected_inquiry(state: AgentState) -> dict:
    return _inquiry_dict(state.get("selected_inquiry"))


def _selected_matches(inquiry: dict, selected: dict) -> bool:
    if not selected:
        return False
    inquiry_id = inquiry.get("id") or inquiry.get("inquiry_id")
    selected_id = selected.get("id") or selected.get("inquiry_id")
    if inquiry_id and selected_id and inquiry_id == selected_id:
        return True
    requirement_id = inquiry.get("requirement_id")
    return bool(
        requirement_id
        and requirement_id == selected.get("requirement_id")
        and set(inquiry.get("target_facets") or []) == set(selected.get("target_facets") or [])
    )


def _unresolved_facets(
    requirement: ActiveRequirement,
    coverage: RequirementCoverageRecord | None,
) -> list[str]:
    if not requirement.facets:
        return []
    if coverage is None:
        return [facet.label for facet in requirement.facets if facet.required]
    result: list[str] = []
    for facet in requirement.facets:
        if not facet.required:
            continue
        facet_state = coverage.facets.get(facet.id)
        if facet_state is None or facet_state.state == RequirementFacetState.UNKNOWN:
            result.append(facet.label)
    return result


def _dependency_block_detail(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        else:
            return None
    if raw.get("eligible", True):
        return None
    blocking = raw.get("blocking_dependencies") or {}
    if not blocking:
        return "Waiting on another product decision."
    labels = [
        f"{_humanize(str(requirement_id))} ({_humanize(str(reason))})"
        for requirement_id, reason in blocking.items()
    ]
    return "Waiting on: " + ", ".join(labels)


def _requirement_decisions(
    state: AgentState,
    scope: DiscoveryScope,
    selected: dict,
) -> tuple[list[UnderstandingItem], list[UnderstandingItem]]:
    open_items: list[UnderstandingItem] = []
    deferred_items: list[UnderstandingItem] = []
    store = state.get("active_requirements", {})
    coverage_state = state.get("requirement_coverage", {})
    dependency_state = state.get("requirement_dependency_state", {})

    for store_key, raw in store.items():
        requirement = _as_requirement(raw)
        if requirement.scope != scope:
            continue

        raw_coverage = coverage_state.get(store_key)
        coverage = _as_coverage(raw_coverage) if raw_coverage is not None else None

        if requirement.status == RequirementStatus.DEFERRED or (
            coverage is not None and coverage.status == RequirementCoverageStatus.DEFERRED
        ):
            deferred_items.append(UnderstandingItem(
                id=f"requirement:{store_key}",
                label=requirement.label,
                detail=requirement.description,
                state=UnderstandingItemState.DEFERRED,
                source="requirement",
                source_refs=[store_key],
                topic=requirement.topic,
                key=requirement.parent_gap,
            ))
            continue

        if requirement.status != RequirementStatus.ACTIVE:
            continue
        if coverage is not None and coverage.status in {
            RequirementCoverageStatus.RESOLVED,
            RequirementCoverageStatus.NOT_APPLICABLE,
            RequirementCoverageStatus.INACTIVE,
        }:
            continue

        inquiry_shape = {
            "requirement_id": requirement.id,
            "target_facets": [
                facet.id
                for facet in requirement.facets
                if facet.label in _unresolved_facets(requirement, coverage)
            ],
        }
        is_active = (
            selected.get("requirement_id") == requirement.id
            or _selected_matches(inquiry_shape, selected)
        )
        dependency_detail = _dependency_block_detail(dependency_state.get(store_key))
        state_value = (
            UnderstandingItemState.ACTIVE
            if is_active
            else UnderstandingItemState.BLOCKED
            if dependency_detail
            else UnderstandingItemState.OPEN
        )

        details: list[str] = []
        if requirement.description:
            details.append(requirement.description)
        unresolved = _unresolved_facets(requirement, coverage)
        if unresolved:
            details.append("Still unresolved: " + ", ".join(unresolved))
        if dependency_detail and not is_active:
            details.append(dependency_detail)

        open_items.append(UnderstandingItem(
            id=f"requirement:{store_key}",
            label=requirement.label,
            detail=" ".join(details) or None,
            state=state_value,
            source="requirement",
            source_refs=[store_key],
            topic=requirement.topic,
            key=requirement.parent_gap,
        ))

    return open_items, deferred_items


def _inquiry_decisions(
    state: AgentState,
    scope: DiscoveryScope,
    selected: dict,
    existing_requirement_ids: set[str],
) -> list[UnderstandingItem]:
    result: list[UnderstandingItem] = []
    inquiries = list(state.get("open_inquiries", []) or [])
    if selected:
        selected_id = selected.get("id") or selected.get("inquiry_id")
        known_ids = {
            (_inquiry_dict(item).get("id") or _inquiry_dict(item).get("inquiry_id"))
            for item in inquiries
        }
        if selected_id and selected_id not in known_ids:
            inquiries.append(selected)

    for raw in inquiries:
        inquiry = _inquiry_dict(raw)
        if not inquiry or not _scope_matches(inquiry.get("scope"), scope):
            continue
        if inquiry.get("requirement_id") in existing_requirement_ids:
            continue
        identity = inquiry.get("id") or inquiry.get("inquiry_id")
        if not identity:
            continue
        state_value = (
            UnderstandingItemState.ACTIVE
            if _selected_matches(inquiry, selected)
            else UnderstandingItemState.OPEN
        )
        detail = inquiry.get("reason") or inquiry.get("question_hint")
        result.append(UnderstandingItem(
            id=f"inquiry:{identity}",
            label=inquiry.get("objective") or "Open product decision",
            detail=detail,
            state=state_value,
            source="inquiry",
            source_refs=[str(identity)],
            topic=DiscoveryTopic(inquiry["topic"]) if inquiry.get("topic") else None,
            key=inquiry.get("anchor_gap"),
            role=inquiry.get("role"),
        ))
    return result


def _validation_decisions(
    state: AgentState,
    scope: DiscoveryScope,
    existing_ids: set[str],
) -> list[UnderstandingItem]:
    result: list[UnderstandingItem] = []
    selected = state.get("selected_validation_issue") or {}
    selected_id = selected.get("id") if isinstance(selected, dict) else None

    for raw in state.get("validation_issues", []) or []:
        issue = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
        issue_id = issue.get("id")
        if not issue_id or f"validation:{issue_id}" in existing_ids:
            continue
        if not _scope_matches(issue.get("scope"), scope):
            continue
        if issue.get("severity") != "BLOCKING":
            continue
        result.append(UnderstandingItem(
            id=f"validation:{issue_id}",
            label="Resolve conflicting product decisions",
            detail=issue.get("message"),
            state=(
                UnderstandingItemState.ACTIVE
                if issue_id == selected_id
                else UnderstandingItemState.OPEN
            ),
            source="validation",
            source_refs=[str(issue_id)],
            topic=DiscoveryTopic(issue["topic"]) if issue.get("topic") else None,
            key=issue.get("key"),
        ))
    return result


def _boundary_deferrals(
    state: AgentState,
    scope: DiscoveryScope,
) -> list[UnderstandingItem]:
    result: list[UnderstandingItem] = []
    for index, boundary in enumerate(state.get("discovery_boundaries", []) or []):
        if not isinstance(boundary, dict) or not _scope_matches(boundary.get("scope"), scope):
            continue
        if boundary.get("type") not in {"design_deferral", "implementation_deferred"}:
            continue

        label = (
            boundary.get("objective")
            or boundary.get("question")
            or "Design / implementation detail"
        )
        evidence = boundary.get("evidence")
        detail = "Deferred by the founder."
        if evidence:
            detail += f' Founder feedback: "{evidence}"'

        signature = (
            boundary.get("type"),
            boundary.get("source_turn"),
            boundary.get("decision_key"),
            boundary.get("evidence"),
        )
        result.append(UnderstandingItem(
            id=f"boundary:{abs(hash(signature))}:{index}",
            label=str(label),
            detail=detail,
            state=UnderstandingItemState.DEFERRED,
            source="boundary",
            source_refs=[str(boundary.get("source_turn", ""))],
        ))
    return result


def _dedupe(items: Iterable[UnderstandingItem]) -> list[UnderstandingItem]:
    result: list[UnderstandingItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        signature = (
            item.state.value,
            item.label.strip().lower(),
            (item.detail or "").strip().lower(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        result.append(item)
    return result


def build_understanding_projection(state: AgentState) -> UnderstandingProjection:
    """Return the complete founder-facing understanding snapshot for one scope.

    Confirmed sections are derived only from CONFIRMED knowledge and explicit
    product concepts. Open/deferred sections are decision state, not facts.
    """
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    section_items: dict[str, list[UnderstandingItem]] = {}

    for item in _knowledge_items(state, scope):
        section_id, _ = TOPIC_SECTIONS[item.topic]
        section_items.setdefault(section_id, []).extend(_confirmed_fact_item(item))

    concepts = _concept_items(state, scope)
    if concepts:
        section_items["product_structure"] = concepts

    selected = _selected_inquiry(state)
    requirement_open, requirement_deferred = _requirement_decisions(
        state, scope, selected
    )
    active_requirement_ids = {
        _as_requirement(raw).id
        for raw in state.get("active_requirements", {}).values()
        if _as_requirement(raw).scope == scope
    }
    inquiry_open = _inquiry_decisions(
        state,
        scope,
        selected,
        active_requirement_ids,
    )
    validation_open = _validation_decisions(
        state,
        scope,
        {item.id for item in inquiry_open},
    )
    open_items = _dedupe([*requirement_open, *inquiry_open, *validation_open])
    deferred_items = _dedupe([
        *requirement_deferred,
        *_boundary_deferrals(state, scope),
    ])

    if open_items:
        section_items["open_decisions"] = open_items
    if deferred_items:
        section_items["deferred_decisions"] = deferred_items

    titles = {section_id: title for _, (section_id, title) in TOPIC_SECTIONS.items()}
    titles.update({
        "product_structure": "Product structure",
        "open_decisions": "Open decisions",
        "deferred_decisions": "Deferred decisions",
    })

    sections = [
        UnderstandingSection(
            id=section_id,
            title=titles[section_id],
            items=_dedupe(section_items[section_id]),
        )
        for section_id in SECTION_ORDER
        if section_items.get(section_id)
    ]
    return UnderstandingProjection(scope=scope, sections=sections)
