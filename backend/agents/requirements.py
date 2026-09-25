"""Product-specific requirements that are separate from schema gaps and confirmed facts."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator

from agents.state import DiscoveryScope, DiscoveryTopic


class RequirementStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DEFERRED = "DEFERRED"
    INACTIVE = "INACTIVE"


class RequirementActivationSource(BaseModel):
    """Why a requirement exists.

    Activation provenance references the confirmed fact/evidence that made a
    requirement relevant. It never turns the implication into product knowledge.
    """

    source_type: str
    source_key: Optional[str] = None
    source_value: Optional[str] = None
    source_turn: Optional[int] = None
    evidence_ref: Optional[str] = None
    activation_rule_id: Optional[str] = None


class RequirementEvidenceRef(BaseModel):
    """Reference to existing knowledge/evidence rather than a copied fact."""

    fact_id: str
    source_turn: Optional[int] = None
    note: Optional[str] = None


class ActiveRequirement(BaseModel):
    """A product-specific requirement currently relevant to discovery."""

    id: str
    scope: DiscoveryScope = DiscoveryScope.USER_APP
    topic: DiscoveryTopic
    parent_gap: Optional[str] = None
    label: str
    description: Optional[str] = None
    status: RequirementStatus = RequirementStatus.ACTIVE
    activation_sources: List[RequirementActivationSource] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    unlocks: List[str] = Field(default_factory=list)
    evidence_refs: List[RequirementEvidenceRef] = Field(default_factory=list)
    coverage_ref: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "label")
    @classmethod
    def non_empty_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Requirement id and label must be non-empty")
        return value


RequirementStore = Dict[str, ActiveRequirement]


def requirement_store_key(scope: DiscoveryScope, requirement_id: str) -> str:
    return f"{scope.value}|{requirement_id}"


def _resolve_store_key(
    store: RequirementStore,
    requirement_id: str,
    scope: Optional[DiscoveryScope],
) -> Optional[str]:
    if scope is not None:
        key = requirement_store_key(scope, requirement_id)
        return key if key in store else None

    matches = [key for key, item in store.items() if item.id == requirement_id]
    if len(matches) > 1:
        raise ValueError(
            f"Requirement '{requirement_id}' exists in multiple scopes; provide scope"
        )
    return matches[0] if matches else None


def register_requirement(
    store: RequirementStore,
    requirement: ActiveRequirement,
) -> RequirementStore:
    """Return a new store containing the requirement.

    Registration is idempotent for an identical scoped requirement and rejects
    accidental replacement. Runtime reconciliation uses explicit model copies
    rather than treating changed provenance as a new registration.
    """
    updated = dict(store)
    key = requirement_store_key(requirement.scope, requirement.id)
    existing = updated.get(key)
    if existing is not None and existing != requirement:
        raise ValueError(
            f"Requirement '{requirement.id}' already exists in scope '{requirement.scope.value}'"
        )
    updated[key] = requirement
    return updated


def get_requirement(
    store: RequirementStore,
    requirement_id: str,
    *,
    scope: Optional[DiscoveryScope] = None,
) -> Optional[ActiveRequirement]:
    key = _resolve_store_key(store, requirement_id, scope)
    return store.get(key) if key is not None else None


def list_requirements(
    store: RequirementStore,
    *,
    scope: Optional[DiscoveryScope] = None,
    topic: Optional[DiscoveryTopic] = None,
    status: Optional[RequirementStatus] = None,
) -> List[ActiveRequirement]:
    requirements = list(store.values())
    if scope is not None:
        requirements = [item for item in requirements if item.scope == scope]
    if topic is not None:
        requirements = [item for item in requirements if item.topic == topic]
    if status is not None:
        requirements = [item for item in requirements if item.status == status]
    return requirements


def update_requirement_status(
    store: RequirementStore,
    requirement_id: str,
    status: RequirementStatus,
    *,
    scope: Optional[DiscoveryScope] = None,
) -> RequirementStore:
    key = _resolve_store_key(store, requirement_id, scope)
    if key is None:
        raise KeyError(requirement_id)
    updated = dict(store)
    updated[key] = store[key].model_copy(update={"status": status})
    return updated


def attach_requirement_evidence(
    store: RequirementStore,
    requirement_id: str,
    evidence_refs: Iterable[RequirementEvidenceRef],
    *,
    scope: Optional[DiscoveryScope] = None,
) -> RequirementStore:
    key = _resolve_store_key(store, requirement_id, scope)
    if key is None:
        raise KeyError(requirement_id)

    requirement = store[key]
    merged = list(requirement.evidence_refs)
    seen = {item.fact_id for item in merged}
    for evidence in evidence_refs:
        if evidence.fact_id not in seen:
            seen.add(evidence.fact_id)
            merged.append(evidence)

    updated = dict(store)
    updated[key] = requirement.model_copy(update={"evidence_refs": merged})
    return updated
