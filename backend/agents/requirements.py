"""Product-specific requirements that are separate from schema gaps and confirmed facts."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator

from agents.state import DiscoveryTopic


class RequirementStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DEFERRED = "DEFERRED"
    INACTIVE = "INACTIVE"


class RequirementActivationSource(BaseModel):
    """Why a requirement exists.

    This is descriptive provenance only. Activation rules are introduced in a
    later architecture step; the model exists now so requirements can preserve
    their origin without duplicating product knowledge.
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


def register_requirement(
    store: RequirementStore,
    requirement: ActiveRequirement,
) -> RequirementStore:
    """Return a new store containing the requirement.

    Registration is idempotent for an identical requirement id/value and
    rejects accidental replacement of an existing requirement.
    """
    updated = dict(store)
    existing = updated.get(requirement.id)
    if existing is not None and existing != requirement:
        raise ValueError(f"Requirement '{requirement.id}' already exists")
    updated[requirement.id] = requirement
    return updated


def get_requirement(store: RequirementStore, requirement_id: str) -> Optional[ActiveRequirement]:
    return store.get(requirement_id)


def list_requirements(
    store: RequirementStore,
    *,
    topic: Optional[DiscoveryTopic] = None,
    status: Optional[RequirementStatus] = None,
) -> List[ActiveRequirement]:
    requirements = list(store.values())
    if topic is not None:
        requirements = [item for item in requirements if item.topic == topic]
    if status is not None:
        requirements = [item for item in requirements if item.status == status]
    return requirements


def update_requirement_status(
    store: RequirementStore,
    requirement_id: str,
    status: RequirementStatus,
) -> RequirementStore:
    if requirement_id not in store:
        raise KeyError(requirement_id)
    updated = dict(store)
    updated[requirement_id] = store[requirement_id].model_copy(update={"status": status})
    return updated


def attach_requirement_evidence(
    store: RequirementStore,
    requirement_id: str,
    evidence_refs: Iterable[RequirementEvidenceRef],
) -> RequirementStore:
    if requirement_id not in store:
        raise KeyError(requirement_id)

    requirement = store[requirement_id]
    merged = list(requirement.evidence_refs)
    seen = {item.fact_id for item in merged}
    for evidence in evidence_refs:
        if evidence.fact_id not in seen:
            seen.add(evidence.fact_id)
            merged.append(evidence)

    updated = dict(store)
    updated[requirement_id] = requirement.model_copy(update={"evidence_refs": merged})
    return updated
