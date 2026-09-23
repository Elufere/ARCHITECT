"""Legacy monolithic contract (not the active pass pipeline).

The active shared field meanings live in discovery_fields.py; extraction_passes
and semantic_validation apply them. valid_role_id remains a shared syntax helper.
"""
from typing import Literal
from pydantic import ConfigDict, Field, model_validator
from agents.state import KnowledgeItem, KnowledgeState

ACTOR_KEYS = {"primary_users", "secondary_users"}
ROLE_KEYS = {"responsibilities", "permissions", "primary_user_goals", "secondary_user_goals"}


def valid_role_id(label: str) -> bool:
    # Syntax only: punctuation and sentence fragments must not become gap IDs.
    return bool(label and label == label.strip() and len(label) <= 80
                and all(c.isalnum() or c in "_ -" for c in label))


class ExtractedItem(KnowledgeItem):
    model_config = ConfigDict(extra="forbid")
    knowledge_state: KnowledgeState = Field(description="CONFIRMED for explicitly asserted facts; INFERRED for deductions. Independent of current_gap.")
    assertion_type: Literal["positive", "negative", "hypothetical", "uncertain", "conditional"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1)
    value: str = Field(min_length=1)
    roles: list[str] | None = Field(max_length=1, description=(
        "Actor declarations MUST contain one canonical actor ID, e.g. ['service_provider']. "
        "An empty list is allowed ONLY for an explicit negative statement that no actors exist. "
        "For non-actor facts use null. Never leave positive actor roles empty."))
    role: str | None = Field(default=None, description=(
        "Canonical owner of a responsibility, permission or goal. Null for actor declarations."))
    aliases: dict[str, list[str]] = Field(default_factory=dict, description=(
        "Only for actor declarations: map the exact ID in roles to explicitly supported aliases. "
        "An alias key without the same canonical ID in roles is invalid."))

    @model_validator(mode="after")
    def check_semantic_shape(self):
        if not self.value.strip() or not self.evidence.strip():
            raise ValueError("Value and evidence must not be blank")
        if self.key in ACTOR_KEYS:
            if self.role is not None or self.roles is None or len(self.roles) > 1:
                raise ValueError("Actor records require roles=[] or one canonical actor; role must be null")
            if not self.roles and self.assertion_type != "negative":
                raise ValueError("Only an explicit absence statement may have no actors")
        elif self.roles or self.aliases:
            raise ValueError("Only actor declarations may define roles or aliases")
        if self.key in ROLE_KEYS and not self.role:
            raise ValueError("A role-scoped fact requires an explicit owner")
        if any(not valid_role_id(r) for r in [*(self.roles or []), *([self.role] if self.role else [])]):
            raise ValueError("Malformed canonical role reference")
        if set(self.aliases) - set(self.roles or []):
            raise ValueError("Aliases must reference this record's canonical actor")
        if any(not valid_role_id(a) for aliases in self.aliases.values() for a in aliases):
            raise ValueError("Malformed alias")
        return self


SEMANTIC_CONTRACT = """
Determine meaning before choosing a schema key. current_gap is conversational
context, never a rule for CONFIRMED versus INFERRED or for assigning ownership.
Return knowledge_state explicitly: CONFIRMED for facts explicitly asserted by
the user (including incidental facts); INFERRED only for evidence-based deductions.
Never infer actors merely from domain conventions. A named admin is explicit;
an admin deduced from 'someone from our team approves providers' is INFERRED.
Return assertion_type: positive, negative, hypothetical, uncertain, or conditional.
Preserve negation and all conditions in value. 'No admins' is a CONFIRMED negative
actor fact, not a positive actor. 'We may add drivers next year' is hypothetical,
not an existing actor. A stated cancellation rule with a condition is an explicit
conditional fact; do not mark it inferred merely because it has a condition.
Evidence must be an exact source substring supporting the WHOLE fact, including
ownership, polarity, modality and any alias relationship. Do not add unsupported
claims to a real quote. If support is insufficient, return no item.

Actor declarations (primary_users/secondary_users) have role=null and exactly one
canonical ID in roles per item. An explicit absence of any additional actors uses
roles=[] and assertion_type=negative. A denial of a specific actor names that actor
in roles instead. Do not infer absence of all other actors from a named denial.
Choose primary/secondary from the user's meaning, never a rule that admins must
be secondary. Alias relationships are semantic, not punctuation rules:
'customers, service providers/artisans, and admins' -> separate actor items with
roles=['customers'], ['service_provider'], ['admin']; the provider item carries
aliases={'service_provider': ['service providers', 'artisan', 'artisans']}.
When buyers/sellers means two distinct actors, return two actor items, no alias.
Use stable canonical IDs, reusing the supplied actor registry. Declare alternate
labels only when the source establishes they are the same actor. Do not globally
equate artisans with providers. Never place slash-separated labels in roles.

Responsibilities, permissions and role-scoped goals have one canonical role owner,
roles=null, aliases={}. Resolve pronouns using the actual question and conversation.
Responsibilities include explicit role actions, activities, capabilities, duties,
and processes; they need not be obligations. Shared evidence is permitted across
responsibilities and workflow when both semantic views are supported. Goals still
require outcome meaning and permissions require explicit authorization boundaries.
Return separate facts for different owners. A shared evidence sentence does not
make every fact apply to every actor it mentions. Do not copy other actors' clauses
into an owner's value. If ownership is unresolved, omit that fact rather than guess.

multiple_roles means ONE person/account may hold more than one role. Several actors
participating in a marketplace does NOT establish multiple_roles. Do not output
multiple_roles=true from a list of activities or from customers booking providers.
Omit this key unless the source actually supports that account/role relationship.
"""
