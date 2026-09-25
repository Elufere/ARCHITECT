"""Resolve which active product requirements are currently eligible for investigation."""
from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Set

from pydantic import BaseModel, Field

from agents.requirements import (
    ActiveRequirement,
    RequirementStatus,
    RequirementStore,
    requirement_store_key,
)
from agents.state import AgentState, DiscoveryScope


class DependencyBlockReason(str, Enum):
    MISSING = "MISSING"
    UNRESOLVED = "UNRESOLVED"
    DEFERRED = "DEFERRED"
    INACTIVE = "INACTIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CYCLE = "CYCLE"


class RequirementDependencyDecision(BaseModel):
    requirement_id: str
    scope: DiscoveryScope
    eligible: bool
    dependencies: List[str] = Field(default_factory=list)
    blocking_dependencies: Dict[str, DependencyBlockReason] = Field(default_factory=dict)


def _active_scope_requirements(
    store: RequirementStore,
    scope: DiscoveryScope,
) -> Dict[str, ActiveRequirement]:
    return {
        requirement.id: requirement
        for requirement in store.values()
        if requirement.scope == scope
    }


def _cycle_nodes(requirements: Dict[str, ActiveRequirement]) -> Set[str]:
    """Return requirement ids that participate in a dependency cycle.

    Dependencies are same-scope requirement ids. Missing dependencies are not
    graph nodes and therefore cannot create cycles.
    """
    graph = {
        requirement_id: [
            dependency_id
            for dependency_id in requirement.dependencies
            if dependency_id in requirements
        ]
        for requirement_id, requirement in requirements.items()
    }

    visiting: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = []
    cycles: Set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            try:
                start = stack.index(node)
            except ValueError:
                start = 0
            cycles.update(stack[start:])
            cycles.add(node)
            return

        visiting.add(node)
        stack.append(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)

    return cycles


def resolve_requirement_dependencies(
    store: RequirementStore,
    scope: DiscoveryScope,
) -> Dict[str, RequirementDependencyDecision]:
    """Resolve current eligibility without mutating requirement state.

    Only ACTIVE requirements can be eligible for investigation. Every declared
    dependency must exist in the same scope and be RESOLVED. Other dependency
    states block the requirement and are preserved as explicit diagnostics.
    """
    requirements = _active_scope_requirements(store, scope)
    cycles = _cycle_nodes(requirements)
    decisions: Dict[str, RequirementDependencyDecision] = {}

    for requirement_id, requirement in requirements.items():
        blocking: Dict[str, DependencyBlockReason] = {}

        if requirement.status != RequirementStatus.ACTIVE:
            decisions[requirement_id] = RequirementDependencyDecision(
                requirement_id=requirement_id,
                scope=scope,
                eligible=False,
                dependencies=list(requirement.dependencies),
                blocking_dependencies=blocking,
            )
            continue

        if requirement_id in cycles:
            blocking[requirement_id] = DependencyBlockReason.CYCLE

        for dependency_id in requirement.dependencies:
            dependency = requirements.get(dependency_id)

            if dependency_id in cycles:
                blocking[dependency_id] = DependencyBlockReason.CYCLE
                continue

            if dependency is None:
                blocking[dependency_id] = DependencyBlockReason.MISSING
                continue

            if dependency.status == RequirementStatus.RESOLVED:
                continue
            if dependency.status == RequirementStatus.DEFERRED:
                blocking[dependency_id] = DependencyBlockReason.DEFERRED
            elif dependency.status == RequirementStatus.INACTIVE:
                blocking[dependency_id] = DependencyBlockReason.INACTIVE
            elif dependency.status == RequirementStatus.NOT_APPLICABLE:
                blocking[dependency_id] = DependencyBlockReason.NOT_APPLICABLE
            else:
                blocking[dependency_id] = DependencyBlockReason.UNRESOLVED

        decisions[requirement_id] = RequirementDependencyDecision(
            requirement_id=requirement_id,
            scope=scope,
            eligible=not blocking,
            dependencies=list(requirement.dependencies),
            blocking_dependencies=blocking,
        )

    return decisions


def eligible_requirement_keys(
    store: RequirementStore,
    scope: DiscoveryScope,
    decisions: Dict[str, RequirementDependencyDecision] | None = None,
) -> List[str]:
    decisions = decisions or resolve_requirement_dependencies(store, scope)
    return [
        requirement_store_key(scope, requirement_id)
        for requirement_id, decision in decisions.items()
        if decision.eligible
    ]


def requirement_dependency_node(state: AgentState) -> dict:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    store = state.get("active_requirements", {})
    decisions = resolve_requirement_dependencies(store, scope)
    return {
        "requirement_dependency_state": {
            requirement_store_key(scope, requirement_id): decision.model_dump(mode="json")
            for requirement_id, decision in decisions.items()
        },
        "eligible_requirement_keys": eligible_requirement_keys(store, scope, decisions),
    }
