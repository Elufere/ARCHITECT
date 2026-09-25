"""Deterministic activation of product-specific requirements from confirmed facts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from agents.discovery_coverage import fact_id
from agents.requirements import (
    ActiveRequirement,
    RequirementActivationSource,
    RequirementFacet,
    RequirementStatus,
    RequirementStore,
    requirement_store_key,
)
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeItem, KnowledgeState


@dataclass(frozen=True)
class FactCondition:
    topic: DiscoveryTopic
    key: str
    require_substantive: bool = True
    value_equals: Optional[str] = None

    def matches(self, item: KnowledgeItem, scope: DiscoveryScope) -> bool:
        if (
            item.scope != scope
            or item.knowledge_state != KnowledgeState.CONFIRMED
            or item.topic != self.topic
            or item.key != self.key
        ):
            return False
        if self.require_substantive and item.absence:
            return False
        if self.value_equals is not None and item.value.strip().lower() != self.value_equals.strip().lower():
            return False
        return True


@dataclass(frozen=True)
class RequirementTemplate:
    id: str
    topic: DiscoveryTopic
    parent_gap: Optional[str]
    label: str
    description: str
    facets: tuple[RequirementFacet, ...] = ()
    dependencies: tuple[str, ...] = ()
    unlocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequirementActivationRule:
    id: str
    description: str
    when: tuple[FactCondition, ...]
    activates: tuple[RequirementTemplate, ...]
    match_all: bool = True

    def matching_facts(
        self,
        knowledge: Sequence[KnowledgeItem],
        scope: DiscoveryScope,
    ) -> list[KnowledgeItem]:
        groups = [
            [item for item in knowledge if condition.matches(item, scope)]
            for condition in self.when
        ]
        if self.match_all:
            if any(not group for group in groups):
                return []
            # One rule firing may be justified by several confirmed facts.
            seen = set()
            result = []
            for group in groups:
                for item in group:
                    identity = fact_id(item)
                    if identity not in seen:
                        seen.add(identity)
                        result.append(item)
            return result

        result = []
        seen = set()
        for group in groups:
            for item in group:
                identity = fact_id(item)
                if identity not in seen:
                    seen.add(identity)
                    result.append(item)
        return result


# Keep the initial registry intentionally small and generic. These rules activate
# implementation-relevant discovery only when an already-confirmed generic fact
# makes that requirement applicable. They never supply the requirement's answer.
ACTIVATION_RULES: tuple[RequirementActivationRule, ...] = (
    RequirementActivationRule(
        id="workflow.external_dependency_followup.v1",
        description="A confirmed downstream dependency makes external-dependency behavior relevant.",
        when=(
            FactCondition(
                topic=DiscoveryTopic.CORE_WORKFLOW,
                key="downstream_dependency",
                require_substantive=True,
            ),
        ),
        activates=(
            RequirementTemplate(
                id="workflow.external_dependency_failure",
                topic=DiscoveryTopic.EXCEPTIONS,
                parent_gap="recovery",
                label="External dependency failure behavior",
                description=(
                    "Determine what the product should do when a required external review, "
                    "approval, service, or action does not complete successfully."
                ),
                facets=(
                    RequirementFacet(id="failure_condition", label="Failure condition", description="What failure, delay, or non-completion condition matters."),
                    RequirementFacet(id="expected_behavior", label="Expected behavior", description="What the product should do when that condition occurs."),
                    RequirementFacet(id="recovery_or_escalation", label="Recovery or escalation", description="How the workflow recovers, retries, escalates, or terminates afterward."),
                ),
            ),
        ),
    ),
    RequirementActivationRule(
        id="constraints.time_boundary_followup.v1",
        description="A confirmed time constraint makes time-boundary behavior relevant.",
        when=(
            FactCondition(
                topic=DiscoveryTopic.CONSTRAINTS,
                key="time_constraints",
                require_substantive=True,
            ),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.time_boundary_behavior",
                topic=DiscoveryTopic.EDGE_CASES,
                parent_gap="boundary_conditions",
                label="Time-boundary behavior",
                description=(
                    "Determine product behavior immediately before, at, and after a confirmed "
                    "deadline, expiry period, or other time boundary."
                ),
                facets=(
                    RequirementFacet(id="at_boundary", label="At boundary", description="What happens when the deadline, expiry, or limit is reached."),
                    RequirementFacet(id="after_boundary", label="After boundary", description="What state or behavior applies after the boundary has passed."),
                ),
            ),
        ),
    ),
)


def _source_for(rule: RequirementActivationRule, item: KnowledgeItem) -> RequirementActivationSource:
    return RequirementActivationSource(
        source_type="confirmed_fact",
        source_key=f"{item.topic.value}.{item.key}",
        source_value=item.value,
        source_turn=item.source_turn,
        evidence_ref=fact_id(item),
        activation_rule_id=rule.id,
    )


def _source_identity(source: RequirementActivationSource) -> tuple:
    return (
        source.activation_rule_id,
        source.evidence_ref,
        source.source_key,
        source.source_turn,
    )


def reconcile_active_requirements(
    store: RequirementStore,
    knowledge: Sequence[KnowledgeItem],
    scope: DiscoveryScope,
    rules: Iterable[RequirementActivationRule] = ACTIVATION_RULES,
) -> RequirementStore:
    """Recompute rule-backed requirement applicability for one discovery scope.

    Existing evidence/history is preserved. A requirement is marked INACTIVE only
    when none of its rule-backed activation sources are currently valid. Manual or
    future non-rule activation sources are left untouched.
    """
    updated = dict(store)
    valid_sources_by_key: dict[str, list[RequirementActivationSource]] = {}
    templates_by_key: dict[str, RequirementTemplate] = {}

    for rule in rules:
        matched = rule.matching_facts(knowledge, scope)
        if not matched:
            continue
        sources = [_source_for(rule, item) for item in matched]
        for template in rule.activates:
            key = requirement_store_key(scope, template.id)
            templates_by_key[key] = template
            valid_sources_by_key.setdefault(key, []).extend(sources)

    # Reconcile all requirements in this scope that have rule-backed provenance,
    # plus any newly activated requirements.
    keys_to_reconcile = set(valid_sources_by_key)
    keys_to_reconcile.update(
        key for key, requirement in updated.items()
        if requirement.scope == scope
        and any(source.activation_rule_id for source in requirement.activation_sources)
    )

    for key in keys_to_reconcile:
        current = updated.get(key)
        valid = valid_sources_by_key.get(key, [])
        valid_identities = {_source_identity(source) for source in valid}

        if current is None:
            template = templates_by_key[key]
            updated[key] = ActiveRequirement(
                id=template.id,
                scope=scope,
                topic=template.topic,
                parent_gap=template.parent_gap,
                label=template.label,
                description=template.description,
                status=RequirementStatus.ACTIVE,
                activation_sources=valid,
                facets=list(template.facets),
                dependencies=list(template.dependencies),
                unlocks=list(template.unlocks),
            )
            continue

        # Preserve non-rule provenance, and keep only rule provenance that still
        # has a currently valid triggering fact.
        preserved = [
            source for source in current.activation_sources
            if not source.activation_rule_id or _source_identity(source) in valid_identities
        ]
        existing_identities = {_source_identity(source) for source in preserved}
        preserved.extend(source for source in valid if _source_identity(source) not in existing_identities)

        has_valid_activation = bool(preserved)
        status = current.status
        if has_valid_activation and status == RequirementStatus.INACTIVE:
            status = RequirementStatus.ACTIVE
        elif not has_valid_activation:
            status = RequirementStatus.INACTIVE

        structural = {}
        template = templates_by_key.get(key)
        if template is not None:
            structural = {
                "topic": template.topic,
                "parent_gap": template.parent_gap,
                "label": template.label,
                "description": template.description,
                "facets": list(template.facets),
                "dependencies": list(template.dependencies),
                "unlocks": list(template.unlocks),
            }

        updated[key] = current.model_copy(
            update={"activation_sources": preserved, "status": status, **structural}
        )

    return updated


def requirement_activation_node(state: AgentState) -> dict:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    reconciled = reconcile_active_requirements(
        state.get("active_requirements", {}),
        state.get("discovered_knowledge", []),
        scope,
    )
    return {"active_requirements": reconciled}
