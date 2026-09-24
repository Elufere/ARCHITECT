"""Conservative, bucket-local comparison before committing grounded facts."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.extraction_passes import canonical_role


class FactComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: Literal["new", "semantic_duplicate", "refinement", "already_refined", "correction", "contradiction"]
    existing_id: int | None = None
    confidence: float = Field(ge=0, le=1)


COMPARISON_INSTRUCTION = """Compare a grounded candidate with existing facts.
These facts share a scope, topic, field, canonical owner/actor, and confirmation
state. Classify meaning, not word similarity. Return new unless a relationship
to a specific existing_id is clear. Do not invent or rewrite any fact.
semantic_duplicate: exactly the same atomic assertion, including polarity,
conditions, quantities, authority, timing and exceptions; neither adds information.
refinement: candidate preserves ALL information in one existing fact and adds
compatible specificity. already_refined: the existing fact preserves ALL candidate
information plus compatible specificity. Mere overlap or related actions is new.
Different limits, actors, stages, conditions, permissions, or exceptions are NOT
duplicates. Two independently applicable rules must coexist even if similarly worded.
correction: source explicitly corrects, replaces, narrows, or supersedes the prior
assertion. Require evidence about that specific assertion; an intent flag or
correction keyword alone does not supersede unrelated facts. Different conditions
may coexist (permission before a stage and prohibition after another stage).
contradiction:
incompatible assertions without explicit correction. Never label either a duplicate
or refinement. Evidence and source questions are provenance, not extra assertions.
Use only supplied facts and evidence, not domain assumptions. When uncertain use
new with low confidence. Supply existing_id for every non-new relationship.
"""


def same_bucket(first, second):
    return (first.scope == second.scope and first.topic == second.topic
            and first.key == second.key
            and canonical_role(first.role or "") == canonical_role(second.role or "")
            and {canonical_role(role) for role in first.roles or []}
                == {canonical_role(role) for role in second.roles or []}
            and first.knowledge_state == second.knowledge_state
            and first.absence == second.absence
            # Alias additions carry information; never discard them by comparing
            # only the actor declaration's value.
            and first.aliases == second.aliases)


def compare_candidate(candidate, knowledge, decide):
    existing = [item for item in knowledge if same_bucket(candidate, item)]
    for item in existing:
        if candidate.value.lower() == item.value.lower():
            return "exact_duplicate", item
    if not existing or candidate.absence:
        return "new", None
    try:
        result = decide("FACT_COMPARISON", FactComparison, COMPARISON_INSTRUCTION,
                        dict(candidate=candidate.model_dump(mode="json"),
                             existing=[dict(id=index, fact=item.model_dump(mode="json"))
                                       for index, item in enumerate(existing)]))
        if (result.confidence >= 0.95 and result.relation != "new"
                and result.existing_id is not None
                and 0 <= result.existing_id < len(existing)):
            return result.relation, existing[result.existing_id]
    except Exception as exc:
        # Comparison failure cannot justify deleting a grounded, distinct fact.
        print(f"FACT COMPARISON UNRESOLVED: {exc}")
    return "new", None
