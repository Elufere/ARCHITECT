"""Identify open product inquiries from the current model.

The discovery schema is intentionally not traversed here. Model inquiries are
created only when a small amount of foundational product structure is missing.
Product-specific depth comes from active requirements, and contradictions become
explicit validation inquiries.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from agents.consistency_validation import (
    DiscoveryValidationIssue,
    ValidationIssueKind,
    ValidationResolution,
)
from agents.discovery_coverage import fact_id
from agents.requirement_coverage import (
    RequirementCoverageRecord,
    RequirementCoverageStatus,
    RequirementFacetState,
)
from agents.requirements import RequirementStatus
from agents.state import (
    AgentState,
    DiscoveryScope,
    DiscoveryTopic,
    KnowledgeItem,
    KnowledgeState,
)


class InquirySource(str, Enum):
    MODEL = "MODEL"
    REQUIREMENT = "REQUIREMENT"
    VALIDATION = "VALIDATION"


class ProductInquiry(BaseModel):
    id: str
    source: InquirySource
    scope: DiscoveryScope
    topic: DiscoveryTopic
    anchor_gap: Optional[str] = None
    objective: str
    question_hint: str
    reason: str
    role: Optional[str] = None
    known_fact_ids: List[str] = Field(default_factory=list)
    requirement_key: Optional[str] = None
    requirement_id: Optional[str] = None
    target_facets: List[str] = Field(default_factory=list)
    activation_rule_ids: List[str] = Field(default_factory=list)
    coverage_status: Optional[RequirementCoverageStatus] = None
    dependency_eligible: bool = True
    uncertainty: float = Field(default=1.0, ge=0, le=1)
    architecture_impact: float = Field(default=0.5, ge=0, le=1)
    business_risk: float = Field(default=0.5, ge=0, le=1)
    question_cost: float = Field(default=0.0, ge=0, le=1)


def _confirmed(state: AgentState, *, topic=None, key=None, role=None) -> list[KnowledgeItem]:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    result = []
    for item in state.get("discovered_knowledge", []):
        if item.scope != scope or item.knowledge_state != KnowledgeState.CONFIRMED:
            continue
        if topic is not None and item.topic != topic:
            continue
        if key is not None and item.key != key:
            continue
        if role is not None and (item.role or "").strip().lower().rstrip("s") != role.strip().lower().rstrip("s"):
            continue
        result.append(item)
    return result


def _foundation_facts(state: AgentState, *, topic, key, role=None) -> list[KnowledgeItem]:
    """Facts trusted to satisfy a foundational model decision.

    Direct answers count. Facts volunteered before an active inquiry (for
    example in the raw idea) count. Incidental cross-category extraction during
    another inquiry remains useful context but does not silently skip the next
    foundational question.
    """
    result = []
    acquisition = state.get("fact_acquisition", {})
    for item in _confirmed(state, topic=topic, key=key, role=role):
        record = acquisition.get(fact_id(item))
        if record is None:
            result.append(item)  # compatibility with imported/tests lacking acquisition metadata
            continue
        if record.get("acquisition") == "DIRECT" or record.get("active_gap") is None:
            result.append(item)
    return result


def _primary_roles(state: AgentState) -> list[str]:
    roles: list[str] = []
    seen = set()
    for item in _foundation_facts(
        state, topic=DiscoveryTopic.USER_ROLES, key="primary_users"
    ):
        if item.absence:
            continue
        for role in item.roles or []:
            identity = role.strip().lower().rstrip("s")
            if identity and identity not in seen:
                seen.add(identity)
                roles.append(role)
    return roles


def _model_inquiries(state: AgentState) -> list[ProductInquiry]:
    """Return only the current foundational frontier, never a fixed schema checklist."""
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    roles = _primary_roles(state)

    if not roles:
        return [ProductInquiry(
            id=f"{scope.value}|model.core_actors",
            source=InquirySource.MODEL,
            scope=scope,
            topic=DiscoveryTopic.USER_ROLES,
            anchor_gap="primary_users",
            objective="Understand who directly uses or participates in the product's core value flow.",
            question_hint=(
                "Ask who the product is for and who directly participates in the core interaction. "
                "Keep it broad enough for the founder to mention other important participants naturally."
            ),
            reason="The product model has no confirmed primary actor yet.",
            uncertainty=1.0,
            architecture_impact=0.9,
            business_risk=0.8,
        )]

    missing_responsibilities = [
        role for role in roles
        if not _foundation_facts(
            state,
            topic=DiscoveryTopic.USER_ROLES,
            key="responsibilities",
            role=role,
        )
    ]
    if missing_responsibilities:
        return [
            ProductInquiry(
                id=f"{scope.value}|model.actor_actions|{role.strip().lower()}",
                source=InquirySource.MODEL,
                scope=scope,
                topic=DiscoveryTopic.USER_ROLES,
                anchor_gap=f"responsibilities::{role}",
                objective=f"Understand what the {role} actually does in the product.",
                question_hint=(
                    f"Ask what the {role} needs to do or manage in the product. "
                    "Do not treat contextual role labels as responsibilities."
                ),
                reason=f"The actor '{role}' exists in the product model but has no confirmed core actions.",
                role=role,
                uncertainty=1.0,
                architecture_impact=0.9,
                business_risk=0.7,
            )
            for role in missing_responsibilities
        ]

    missing_goals = [
        role for role in roles
        if not _foundation_facts(
            state,
            topic=DiscoveryTopic.USER_GOALS,
            key="primary_user_goals",
            role=role,
        )
    ]
    if missing_goals:
        return [
            ProductInquiry(
                id=f"{scope.value}|model.actor_goal|{role.strip().lower()}",
                source=InquirySource.MODEL,
                scope=scope,
                topic=DiscoveryTopic.USER_GOALS,
                anchor_gap=f"primary_user_goals::{role}",
                objective=f"Understand the outcome the {role} is trying to achieve by using the product.",
                question_hint=(
                    f"Ask about the desired outcome for the {role}, not the steps, permissions, "
                    "or business rules used to achieve it."
                ),
                reason=f"The model knows what '{role}' does but not the outcome they are trying to achieve.",
                role=role,
                uncertainty=1.0,
                architecture_impact=0.8,
                business_risk=0.6,
            )
            for role in missing_goals
        ]

    workflow = _foundation_facts(
        state,
        topic=DiscoveryTopic.CORE_WORKFLOW,
        key="workflow_steps",
    )
    if not workflow:
        return [ProductInquiry(
            id=f"{scope.value}|model.core_workflow",
            source=InquirySource.MODEL,
            scope=scope,
            topic=DiscoveryTopic.CORE_WORKFLOW,
            anchor_gap="workflow_steps",
            objective="Understand the normal end-to-end product interaction that produces the user's desired outcome.",
            question_hint=(
                "Ask for the normal flow from the point the user starts the core interaction "
                "until the intended outcome is reached. Do not ask for exceptions yet."
            ),
            reason="Actors and outcomes are known, but the product model has no coherent core workflow.",
            uncertainty=1.0,
            architecture_impact=1.0,
            business_risk=0.8,
        )]

    completion = _foundation_facts(
        state,
        topic=DiscoveryTopic.CORE_WORKFLOW,
        key="completion_condition",
    )
    end_state = _foundation_facts(
        state,
        topic=DiscoveryTopic.CORE_WORKFLOW,
        key="end_state",
    )
    if not completion and not end_state:
        return [ProductInquiry(
            id=f"{scope.value}|model.completion",
            source=InquirySource.MODEL,
            scope=scope,
            topic=DiscoveryTopic.CORE_WORKFLOW,
            anchor_gap="completion_condition",
            objective="Understand what makes the core interaction successfully complete.",
            question_hint=(
                "Ask what must be true for the core interaction to count as successfully completed. "
                "Focus on the product outcome, not implementation details."
            ),
            reason="The normal workflow is known, but its successful completion condition is not.",
            uncertainty=0.9,
            architecture_impact=0.9,
            business_risk=0.8,
        )]

    return []


def _requirement_inquiries(state: AgentState) -> list[ProductInquiry]:
    if state.get("validation_candidate_blocking", False):
        return []

    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    coverage_state = state.get("requirement_coverage", {})
    result: list[ProductInquiry] = []

    for key in state.get("eligible_requirement_keys", []):
        requirement = store.get(key)
        payload = coverage_state.get(key)
        if requirement is None or payload is None:
            continue
        if requirement.scope != scope or requirement.status != RequirementStatus.ACTIVE:
            continue

        coverage = RequirementCoverageRecord.model_validate(payload)
        target_facets = []
        for facet in requirement.facets:
            if not facet.required:
                continue
            facet_state = coverage.facets.get(facet.id)
            if facet_state is None or facet_state.state == RequirementFacetState.UNKNOWN:
                target_facets.append(facet.id)
        if requirement.facets and not target_facets:
            continue

        activation_rule_ids = list(dict.fromkeys(
            source.activation_rule_id
            for source in requirement.activation_sources
            if source.activation_rule_id
        ))
        known_fact_ids = list(coverage.candidate_fact_ids)
        for facet in coverage.facets.values():
            for identity in facet.fact_ids:
                if identity not in known_fact_ids:
                    known_fact_ids.append(identity)

        result.append(ProductInquiry(
            id=f"{scope.value}|requirement|{requirement.id}|{','.join(target_facets) or '__requirement__'}",
            source=InquirySource.REQUIREMENT,
            scope=scope,
            topic=requirement.topic,
            anchor_gap=requirement.parent_gap,
            objective=requirement.description or requirement.label,
            question_hint=(
                "Ask one natural product question about the unresolved requirement. "
                "Cover closely related unresolved facets together only when that makes the question easier to answer."
            ),
            reason=f"Active requirement '{requirement.label}' still has unresolved decision facets.",
            known_fact_ids=known_fact_ids,
            requirement_key=key,
            requirement_id=requirement.id,
            target_facets=target_facets,
            activation_rule_ids=activation_rule_ids,
            coverage_status=coverage.status,
            dependency_eligible=True,
            uncertainty=1.0,
            architecture_impact=requirement.priority_hints.architecture_impact,
            business_risk=requirement.priority_hints.business_risk,
            question_cost=min(0.15, max(0, len(target_facets) - 1) * 0.05),
        ))

    return result


def _validation_inquiry(state: AgentState) -> Optional[ProductInquiry]:
    issues = [
        DiscoveryValidationIssue.model_validate(item)
        for item in state.get("validation_issues", [])
    ]
    issue = next(
        (
            item for item in issues
            if item.severity.value == "BLOCKING"
            and item.resolution == ValidationResolution.USER_CLARIFICATION
        ),
        None,
    )
    if issue is None:
        return None

    facts = {
        fact_id(item): item
        for item in state.get("discovered_knowledge", [])
    }
    conflicting = [facts[identity] for identity in issue.fact_ids if identity in facts]
    if not conflicting:
        return None
    anchor = conflicting[0]
    return ProductInquiry(
        id=f"{anchor.scope.value}|validation|{issue.id}",
        source=InquirySource.VALIDATION,
        scope=anchor.scope,
        topic=anchor.topic,
        anchor_gap=anchor.key + (f"::{anchor.role}" if anchor.role else ""),
        objective=(
            "Resolve incompatible confirmed product statements by establishing "
            "which rule or decision currently applies."
        ),
        question_hint=(
            "Briefly present the incompatible statements and ask the founder to establish "
            "the current rule. Do not choose a side."
        ),
        reason=issue.message,
        role=anchor.role,
        known_fact_ids=list(issue.fact_ids),
        uncertainty=1.0,
        architecture_impact=1.0,
        business_risk=1.0,
    )


def identify_open_inquiries(state: AgentState) -> list[ProductInquiry]:
    validation = _validation_inquiry(state)
    if validation is not None:
        return [validation]

    inquiries = [
        *_model_inquiries(state),
        *_requirement_inquiries(state),
    ]
    seen = set()
    result = []
    for inquiry in inquiries:
        if inquiry.id in seen:
            continue
        seen.add(inquiry.id)
        result.append(inquiry)
    return result


def inquiry_identification_node(state: AgentState) -> dict:
    inquiries = identify_open_inquiries(state)
    return {
        "open_inquiries": [
            inquiry.model_dump(mode="json")
            for inquiry in inquiries
        ]
    }
