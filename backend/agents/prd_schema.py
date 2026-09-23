"""Source-linked compiler drafts and the application-verified PRD artifact."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    source_fact_ids: list[str] = Field(min_length=1, description="IDs from the supplied confirmed fact snapshot. Never invent IDs.")
    category: str = Field(description="Exact canonical TOPIC.key, e.g. BUSINESS_RULES.approval_rules. Preserve source meaning.")
    actor_ids: list[str] = Field(description="Actors involved, preserving ownership and capacity-specific restrictions; [] for product-wide claims.")
    conditions: list[str] = Field(description="All relevant source conditions, thresholds, exceptions and negations; [] when unconditional.")


class SourcedClaim(SourceReference):
    text: str = Field(min_length=1)


class ScopeBoundary(StrictModel):
    in_scope: list[SourcedClaim]
    out_of_scope: list[SourcedClaim]


class UserPersona(SourceReference):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    key_behaviors: list[SourcedClaim]


class FunctionalRequirement(SourceReference):
    id: str = Field(min_length=1, description="Unique requirement ID such as FR-01.")
    description: str = Field(min_length=1)
    validation: str = Field(min_length=1, description="An acceptance criterion entailed by the cited sources, or TBD. Do not add new behavior.")


class PRDDraft(StrictModel):
    product_name: SourcedClaim | None = None
    elevator_pitch: list[SourcedClaim]
    scope: ScopeBoundary
    personas: list[UserPersona]
    functional_requirements: list[FunctionalRequirement]
    non_functional_constraints: list[SourcedClaim]
    deferred_items: list[SourcedClaim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list, description="Unresolved questions only, never requirements or promises.")


class SourceFact(StrictModel):
    fact_id: str
    topic: str
    scope: Literal["USER_APP", "ADMIN_DASHBOARD"]
    key: str
    value: str
    evidence: str
    source_question: str | None = None
    roles: list[str] | None = None
    aliases: dict[str, list[str]] | None = None
    role: str | None = None
    confidence: float
    knowledge_state: Literal["CONFIRMED"]
    source_turn: int
    absence: Literal["none", "not_applicable"] | None = None


class ClaimVerdict(StrictModel):
    claim_id: str
    source_evidence_supports_facts: StrictBool
    claim_supported: StrictBool
    category_preserved: StrictBool
    actors_preserved: StrictBool
    conditions_preserved: StrictBool
    validation_supported: StrictBool
    no_conflict_with_confirmed_facts: StrictBool
    explanation: str = Field(min_length=1)


class SemanticCategories(StrictModel):
    categories: list[str] = Field(description="Canonical TOPIC.key categories explicitly supported by the text; [] for a fragment without an assertion.")
    explanation: str = Field(min_length=1)


class PRDContract(PRDDraft):
    schema_version: Literal["2.0"] = "2.0"
    discovery_scope: Literal["USER_APP", "ADMIN_DASHBOARD"]
    source_facts: list[SourceFact]
    validation_report: list[ClaimVerdict]
