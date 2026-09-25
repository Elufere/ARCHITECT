"""Deterministic activation of product-specific requirements from confirmed facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from agents.discovery_coverage import fact_id
from agents.requirements import (
    ActiveRequirement,
    RequirementActivationSource,
    RequirementFacet,
    RequirementPriorityHints,
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
    value_contains_any: tuple[str, ...] = ()

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
        normalized = item.value.strip().lower()
        if self.value_equals is not None and normalized != self.value_equals.strip().lower():
            return False
        if self.value_contains_any and not any(
            token.lower() in normalized for token in self.value_contains_any
        ):
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
    priority_hints: RequirementPriorityHints = field(default_factory=RequirementPriorityHints)
    metadata: dict[str, Any] = field(default_factory=dict)
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


# Base implication rules remain generic and separate from lifecycle patterns.\nBASE_ACTIVATION_RULES: tuple[RequirementActivationRule, ...] = (\n    RequirementActivationRule(
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
)


# Lifecycle rules are activated only by grounded generic product facts. They
# create questions worth investigating; they never assert the lifecycle answer.
LIFECYCLE_ACTIVATION_RULES: tuple[RequirementActivationRule, ...] = (
    RequirementActivationRule(
        id="lifecycle.start_state.v1",
        description="A known workflow trigger plus workflow steps makes the initial lifecycle state relevant.",
        when=(
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="trigger"),
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="workflow_steps"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.start_state_behavior",
                topic=DiscoveryTopic.CORE_WORKFLOW,
                parent_gap="trigger",
                label="Lifecycle start-state behavior",
                description="Determine the state and conditions immediately after the primary workflow begins.",
                facets=(
                    RequirementFacet(id="start_preconditions", label="Start preconditions", description="What must already be true before the lifecycle can begin."),
                    RequirementFacet(id="initial_state", label="Initial state", description="What state or status exists immediately after the lifecycle starts."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.7, business_risk=0.5),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["creation", "active"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.state_transitions.v1",
        description="A workflow with an explicit end state implies meaningful state transitions worth clarifying.",
        when=(
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="workflow_steps"),
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="end_state"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.state_transition_behavior",
                topic=DiscoveryTopic.CORE_WORKFLOW,
                parent_gap="workflow_steps",
                label="Lifecycle state-transition behavior",
                description="Determine the meaningful state changes in the workflow and what controls movement between them.",
                facets=(
                    RequirementFacet(id="transition_conditions", label="Transition conditions", description="What conditions move the lifecycle from one meaningful state to another."),
                    RequirementFacet(id="transition_result", label="Transition result", description="What state results after each important transition."),
                    RequirementFacet(id="transition_reversibility", label="Transition reversibility", description="Which important transitions can be reversed or reopened, if any."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.85, business_risk=0.65),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["active", "transition", "completion"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.post_completion.v1",
        description="An explicit completion condition and end state make post-completion lifecycle behavior relevant.",
        when=(
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="completion_condition"),
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="end_state"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.post_completion_behavior",
                topic=DiscoveryTopic.CORE_WORKFLOW,
                parent_gap="end_state",
                label="Post-completion lifecycle behavior",
                description="Determine what may still change after completion and whether the completed lifecycle can be reopened or reversed.",
                facets=(
                    RequirementFacet(id="terminality", label="Terminality", description="Whether completion is final or another state can follow."),
                    RequirementFacet(id="post_completion_changes", label="Post-completion changes", description="What may still be changed after completion, if anything."),
                    RequirementFacet(id="reopen_or_reverse", label="Reopen or reverse", description="Whether and under what conditions the completed lifecycle can be reopened or reversed."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.75, business_risk=0.7),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["completion", "post_completion"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.cancellation_transition.v1",
        description="Explicit cancellation behavior implies a lifecycle transition that needs its states and effects defined.",
        when=(
            FactCondition(topic=DiscoveryTopic.EXCEPTIONS, key="user_cancellations"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.cancellation_transition",
                topic=DiscoveryTopic.EXCEPTIONS,
                parent_gap="user_cancellations",
                label="Cancellation lifecycle transition",
                description="Determine when cancellation is allowed, the resulting state, and important effects of cancellation.",
                facets=(
                    RequirementFacet(id="cancellable_states", label="Cancellable states", description="At which lifecycle states cancellation is permitted or blocked."),
                    RequirementFacet(id="cancellation_result", label="Cancellation result", description="What state the lifecycle enters after cancellation."),
                    RequirementFacet(id="cancellation_effects", label="Cancellation effects", description="What important downstream effects, reversals, or retained records follow cancellation."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.7, business_risk=0.8),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["active", "cancellation", "terminal"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.role_transition.v1",
        description="Explicit role transitions imply a user-role lifecycle worth clarifying.",
        when=(
            FactCondition(topic=DiscoveryTopic.USER_ROLES, key="role_transitions"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.role_transition_behavior",
                topic=DiscoveryTopic.USER_ROLES,
                parent_gap="role_transitions",
                label="Role-transition lifecycle behavior",
                description="Determine the conditions, authority, and effects when a user changes product roles.",
                facets=(
                    RequirementFacet(id="transition_conditions", label="Transition conditions", description="What conditions permit or require the role change."),
                    RequirementFacet(id="transition_authority", label="Transition authority", description="Who or what can initiate or approve the role change."),
                    RequirementFacet(id="transition_effects", label="Transition effects", description="How access, responsibilities, or retained data change after the transition."),
                    RequirementFacet(id="transition_reversibility", label="Transition reversibility", description="Whether the role change can be reversed."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.75, business_risk=0.65),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["role_transition"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.time_boundary.v1",
        description="A confirmed time constraint implies lifecycle behavior at and after the time boundary.",
        when=(
            FactCondition(topic=DiscoveryTopic.CONSTRAINTS, key="time_constraints"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.time_boundary_behavior",
                topic=DiscoveryTopic.EDGE_CASES,
                parent_gap="boundary_conditions",
                label="Time-boundary behavior",
                description="Determine product behavior immediately before, at, and after a confirmed deadline, expiry period, or other time boundary.",
                facets=(
                    RequirementFacet(id="at_boundary", label="At boundary", description="What happens when the deadline, expiry, or limit is reached."),
                    RequirementFacet(id="after_boundary", label="After boundary", description="What state or behavior applies after the boundary has passed."),
                    RequirementFacet(id="reopen_or_extend", label="Reopen or extend", description="Whether the time-bound lifecycle can be extended, reopened, or reactivated."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.7, business_risk=0.7),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["active", "boundary", "expired"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.limit_boundary.v1",
        description="A confirmed operational limit implies behavior at and beyond the lifecycle boundary.",
        when=(
            FactCondition(topic=DiscoveryTopic.BUSINESS_RULES, key="limits"),
        ),
        activates=(
            RequirementTemplate(
                id="lifecycle.limit_boundary_behavior",
                topic=DiscoveryTopic.EDGE_CASES,
                parent_gap="boundary_conditions",
                label="Limit-boundary lifecycle behavior",
                description="Determine what happens as a configured product limit is reached, exceeded, or reset.",
                facets=(
                    RequirementFacet(id="at_limit", label="At limit", description="What behavior applies exactly when the limit is reached."),
                    RequirementFacet(id="beyond_limit", label="Beyond limit", description="What happens when an action would exceed the limit."),
                    RequirementFacet(id="reset_or_recovery", label="Reset or recovery", description="How capacity or eligibility becomes available again, if applicable."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.65, business_risk=0.75),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["active", "boundary", "recovery"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.modification.v1",
        description="Explicit edit/update/modify behavior implies lifecycle rules around changes to existing state.",
        when=(
            FactCondition(topic=DiscoveryTopic.USER_ROLES, key="responsibilities", value_contains_any=("edit", "update", "modify", "change")),
            FactCondition(topic=DiscoveryTopic.USER_ROLES, key="permissions", value_contains_any=("edit", "update", "modify", "change")),
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="workflow_steps", value_contains_any=("edit", "update", "modify", "change")),
            FactCondition(topic=DiscoveryTopic.BUSINESS_RULES, key="ownership_rules", value_contains_any=("edit", "update", "modify", "change")),
        ),
        match_all=False,
        activates=(
            RequirementTemplate(
                id="lifecycle.modification_behavior",
                topic=DiscoveryTopic.BUSINESS_RULES,
                parent_gap="ownership_rules",
                label="Modification lifecycle behavior",
                description="Determine when existing product state may be modified and what those changes affect.",
                facets=(
                    RequirementFacet(id="modifiable_states", label="Modifiable states", description="At which lifecycle states modification is allowed or blocked."),
                    RequirementFacet(id="modification_authority", label="Modification authority", description="Who or what may make the modification."),
                    RequirementFacet(id="historical_effect", label="Historical effect", description="Whether changes rewrite existing history or only affect future state."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.8, business_risk=0.7),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["active", "modification"]},
            ),
        ),
    ),
    RequirementActivationRule(
        id="lifecycle.removal.v1",
        description="Explicit delete/disable/archive behavior implies lifecycle rules for removal and historical effects.",
        when=(
            FactCondition(topic=DiscoveryTopic.USER_ROLES, key="responsibilities", value_contains_any=("delete", "disable", "archive", "deactivate", "remove")),
            FactCondition(topic=DiscoveryTopic.USER_ROLES, key="permissions", value_contains_any=("delete", "disable", "archive", "deactivate", "remove")),
            FactCondition(topic=DiscoveryTopic.CORE_WORKFLOW, key="workflow_steps", value_contains_any=("delete", "disable", "archive", "deactivate", "remove")),
            FactCondition(topic=DiscoveryTopic.BUSINESS_RULES, key="ownership_rules", value_contains_any=("delete", "disable", "archive", "deactivate", "remove")),
        ),
        match_all=False,
        activates=(
            RequirementTemplate(
                id="lifecycle.removal_behavior",
                topic=DiscoveryTopic.BUSINESS_RULES,
                parent_gap="ownership_rules",
                label="Removal/disable lifecycle behavior",
                description="Determine when product state can be deleted, disabled, archived, or otherwise removed from active use.",
                facets=(
                    RequirementFacet(id="removal_conditions", label="Removal conditions", description="At which states deletion, disablement, or archival is allowed."),
                    RequirementFacet(id="removal_authority", label="Removal authority", description="Who or what may perform the removal action."),
                    RequirementFacet(id="historical_retention", label="Historical retention", description="What historical records or references remain after removal."),
                    RequirementFacet(id="restoration", label="Restoration", description="Whether disabled or archived state can be restored."),
                ),
                priority_hints=RequirementPriorityHints(architecture_impact=0.85, business_risk=0.85),
                metadata={"requirement_family": "lifecycle", "lifecycle_stages": ["active", "disabled", "archived", "deleted"]},
            ),
        ),
    ),
)

ACTIVATION_RULES: tuple[RequirementActivationRule, ...] = (
    *BASE_ACTIVATION_RULES,
    *LIFECYCLE_ACTIVATION_RULES,
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
                priority_hints=template.priority_hints,
                metadata=dict(template.metadata),
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
                "priority_hints": template.priority_hints,
                "metadata": {**current.metadata, **template.metadata},
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
