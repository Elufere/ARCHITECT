"""Small, independent structured extraction contracts for one user response."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.state import DiscoveryTopic, KnowledgeState, KnowledgeItem
from agents.extraction_contract import valid_role_id
from agents.discovery_fields import FIELD_DEFINITIONS

GENERIC_ROLES = {"user", "users", "people", "person", "demand_side", "supply_side"}


def absence_label(value: str) -> str | None:
    """Recognize serialized absence, never infer absence from source language."""
    return {"none": "none", "not applicable": "not_applicable",
            "not_applicable": "not_applicable", "n/a": "not_applicable"}.get(value.strip().lower())


def canonical_role(role: str) -> str:
    return "_".join(role.strip().lower().split())


class RawPass(BaseModel):
    # Keep parsing at the item boundary: one malformed sibling is rejected later.
    items: list[dict]


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    knowledge_state: KnowledgeState = KnowledgeState.CONFIRMED
    absence: Literal["none", "not_applicable"] | None = None

    @model_validator(mode="after")
    def mark_serialized_absence(self):
        # A model must not bypass absence auditing by emitting value="none"
        # with absence=null. This normalizes representation, not user meaning.
        label = absence_label(self.value)
        if label:
            if self.absence and self.absence != label:
                raise ValueError("Conflicting absence representations")
            self.absence = label
            self.value = "none" if label == "none" else "not applicable"
        return self


class ActorFact(Fact):
    # Role relationships belong to the existing ACTOR pass, not a new model call.
    key: Literal["primary_users", "secondary_users", "multiple_roles", "role_transitions"]
    roles: list[str] = Field(default_factory=list, max_length=1)
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_actor(self):
        if self.key in ("multiple_roles", "role_transitions"):
            if self.roles or self.aliases:
                raise ValueError("Role policies cannot declare actors or aliases")
            return self
        if self.absence:
            if self.roles or self.aliases:
                raise ValueError("Actor absence cannot declare roles or aliases")
            return self
        if not self.roles:
            raise ValueError("A positive actor requires one canonical role")
        if canonical_role(self.roles[0]) in ("primary_users", "secondary_users", "multiple_roles", "role_transitions"):
            raise ValueError("An actor ID cannot be an actor field name")
        if not valid_role_id(self.roles[0]):
            raise ValueError("Malformed canonical actor role")
        if canonical_role(self.roles[0]) in GENERIC_ROLES:
            raise ValueError("Generic actor role")
        return self


class OwnedFact(Fact):
    key: Literal["responsibilities", "permissions"]
    role: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_owner(self):
        if not valid_role_id(self.role):
            raise ValueError("Malformed owner role")
        if canonical_role(self.role) in GENERIC_ROLES:
            raise ValueError("Generic owner role")
        return self


class ResponsibilityFact(OwnedFact):
    key: Literal["responsibilities"]
    value: str = Field(min_length=1, description=FIELD_DEFINITIONS[DiscoveryTopic.USER_ROLES]["responsibilities"])


class PermissionFact(OwnedFact):
    key: Literal["permissions"]


class GoalFact(Fact):
    key: Literal["primary_user_goals", "secondary_user_goals", "success_criteria", "motivations"]
    role: str | None = Field(default=None, description="Required for primary/secondary goals; optional for product-wide success criteria or motivations.")

    @model_validator(mode="after")
    def owned_goal(self):
        if self.role is None:
            if self.key in ("primary_user_goals", "secondary_user_goals"):
                raise ValueError("A role-specific goal requires an owner")
            return self
        if not valid_role_id(self.role):
            raise ValueError("Malformed goal owner role")
        if canonical_role(self.role) in GENERIC_ROLES:
            raise ValueError("Generic goal owner role")
        return self


class WorkflowFact(Fact):
    key: Literal["trigger", "workflow_steps", "completion_condition", "downstream_dependency", "end_state"]


class RemainingFact(Fact):
    topic: Literal["BUSINESS_RULES", "CONSTRAINTS", "MVP_SCOPE", "EXCEPTIONS", "EDGE_CASES"]

    @model_validator(mode="after")
    def valid_key(self):
        from agents.state import TOPIC_KEY_MAP
        if self.key not in TOPIC_KEY_MAP[DiscoveryTopic(self.topic)]:
            raise ValueError(f"{self.key} is not a key for {self.topic}")
        return self


PASSES = (
    ("ACTOR", ActorFact, DiscoveryTopic.USER_ROLES, """Identify functional product actors only.
Primary actors directly participate in the core value or workflow, including both demand and supply sides.
The service provider is a primary actor when providing the service is part of the product's core value exchange. Secondary does NOT mean supply side.
Secondary actors mainly administer, moderate, support, supervise or audit, BUT emit a secondary actor only when the response explicitly describes one.
Never infer an admin just because providers manage their own profiles or schedules.
Professions such as electricians or dermatologists are categories of a provider actor, not separate product roles.
Keep named functions: patients -> patient; healthcare providers -> healthcare_provider.
Normalize vague people who book artisans to customer.
For a patient booking healthcare professionals, output primary patient and healthcare_provider, with no secondary actors.
For the pattern "patients find and book appointments with healthcare professionals", return TWO primary_users items: roles=["patient"] and roles=["healthcare_provider"]. Do not classify healthcare_provider as secondary_users because it manages its profile or schedule.
For people booking artisans, output primary customer and artisan, with no secondary actors.
Never use demand_side, supply_side, user, users, people or person as canonical role IDs.
"Platform" does not imply an administrator. Use exactly one canonical role per item in roles.
Do not extract responsibilities here. Also extract multiple_roles and role_transitions
when explicitly stated, with roles=[] and aliases=[]. These are policies, not new actors.
Examples: "Users can have both roles" -> multiple_roles with the stated positive policy;
"Each account has only one role" -> multiple_roles with that negative policy;
"A provider can also be a patient" -> multiple_roles;
"Patients become providers after verification" -> role_transitions preserving verification;
"Roles never change" -> role_transitions with that negative policy.
Do not infer either policy merely from a list of actors. A denial is a confirmed
policy, not missing information. Explicit absence of ALL primary/secondary actors
may use absence="none", value="none", roles=[], aliases=[]; never use this for a
specific denied actor ("no admins") or an uncertain/future actor.
If the response says only what separate actors do, emit NO multiple_roles or
role_transitions item. Separate actors do not imply separate accounts. Omit
unmentioned policies entirely: never output none/not_applicable for silence.
Evidence must be copied verbatim as one exact contiguous, case-sensitive substring of the latest user response.
Never shorten a sentence by deleting words from its middle or end and adding punctuation that was not there.
If a concise quote is unavailable, copy the full supporting sentence exactly as written."""),
    (
    "RESPONSIBILITY",
    ResponsibilityFact,
    DiscoveryTopic.USER_ROLES,
    """
Extract the significant actions, activities, duties, capabilities, or processes
that EACH known role explicitly performs or manages within the product.
Ask: "What does this role do in the product?"
In this single pass, review every known actor and every relevant clause; emit
one or more owned items for every actor with explicit role-action evidence.
Do not require the user to call an action a duty or responsibility.
"Can", "should be able to", and "will be able to" establish supported product
capabilities only when the latest response explicitly assigns the action to a role.
Managing owned resources and operational processes also qualifies.
The same exact quote may ALSO support workflow_steps if it describes a journey;
that is a valid different view, not a reason to omit the responsibility.
Do not turn prohibited actions into things the actor performs. Preserve conditions.
Do not infer actions from actor identity, domain conventions, or other actors.
Actor identity alone never implies responsibilities. Merely saying an actor uses
the platform identifies participation, not a specific product activity; return
{"items": []} if that is all the response says. Do not expand generic usage into
stereotypical actions from product/domain knowledge, actor context, or examples.
Before emitting each candidate, read its own evidence quote independently: it
must explicitly support every proposed action, its owner, and any conditions.
A verbatim actor-only quote cannot support an invented action. Omit unsupported
actions even if they would be plausible for this actor or product.
The founder's intention to build the product is not an actor responsibility.
Every item uses key="responsibilities", exactly one known canonical actor in role,
and exact contiguous evidence from the latest response. Do not invent owners.
Return {"items": []} only if no role actions are explicitly supported.
Do not copy actions from instructional examples into an unrelated response.
"""
),

(
    "PERMISSION",
    PermissionFact,
    DiscoveryTopic.USER_ROLES,
    """
Extract only explicit actor PERMISSIONS from the latest response.

A permission is an explicit authorization, prohibition, restriction,
exclusivity rule, access-control boundary, or role-specific authority.

Ask:

"Does this statement establish what this actor is specifically allowed,
forbidden, restricted, uniquely authorized, or limited to doing or accessing?"

A normal product capability is NOT a permission.
Every permission item must use a known canonical actor ID in role.

Statements that only identify which actors exist, or say there are no other
actors, are actor facts. They do not grant those actors permission to exist,
join, or use the product. For example, "No, just patients and providers for
now" contains no permission; return {"items": []}.

The fact that an actor can perform an action through the product does not
by itself establish permission semantics.

The phrases:
- "can"
- "should be able to"
- "will be able to"

do NOT by themselves mean permission.

If the response merely describes ordinary product usage — such as searching,
comparing, booking, communicating, submitting information, receiving
notifications, or paying — return no permission item.

Only extract a permission when the response explicitly communicates an
authorization boundary, restriction, prohibition, exclusivity condition,
access limitation, or role-specific authority.

Be conservative.

If no explicit permission semantics are present, return:

{"items": []}

Do not convert workflow/capability statements into permissions.

Every returned item must:
- use key="permissions"
- identify exactly one functional actor in `role`
- have a non-empty value
- be explicitly supported by the latest response
- use exact contiguous evidence copied from the latest response

Never reconstruct or paraphrase evidence.
Never infer authorization merely because the product supports an action.
Never use instruction text as evidence.
"""
),
    ("WORKFLOW", WorkflowFact, DiscoveryTopic.CORE_WORKFLOW, """Extract the explicitly stated normal interaction/process sequence and its structure.
Preserve stated order. A journey action list such as describe need, search, compare,
book, communicate, receive reminders, pay supports workflow_steps even if partial.
The same evidence may also support role responsibilities. Do not suppress a journey
because its actor actions were already extracted as responsibilities.
An isolated capability, actor list, or unordered inventory of managed resources
is not a process sequence. Do not invent ordering, triggers, dependencies or end states.
Apply these independent thresholds before emitting each non-step field:
- trigger: the quote explicitly identifies what starts the interaction. A product
  purpose or founder intention does not state a start event.
- completion_condition: the quote explicitly defines when the process is complete.
  The last listed action, a reminder, or payment capability is not that definition.
- end_state: the quote explicitly states a resulting status after completion.
  A list of things users can do establishes no final status.
- downstream_dependency: the quote explicitly names a required prerequisite,
  external action, approval, system, or event. Omit if unmentioned; silence is
  never evidence of none. Explicitly denying such dependencies is different.
"I want to build a platform" is founder intent, not the workflow trigger.
"Providers manage profiles, specialties and schedules" alone is not a journey.
Extract explicit trigger, completion_condition, downstream_dependency and end_state
whenever stated, even alongside a capability list. Apply each field's definition.
An approval may also be a business rule; a completion signal may also be a user
success criterion. Shared evidence is allowed when both meanings are explicit."""),
    ("GOAL", GoalFact, DiscoveryTopic.USER_GOALS, """Extract explicit desired outcomes, success signals, and motivations.
primary_user_goals / secondary_user_goals: a desired RESULT that an actor wants
or the product helps them achieve. Role-specific goals need one known canonical owner.
Every primary_user_goals or secondary_user_goals item MUST include the JSON field
"role" with that actor's canonical ID. Mentioning the actor in value/evidence is
not a substitute for role. Never omit role or use null for these two keys.
One product-purpose outcome is one goal. Do not break a feature or workflow sentence into individual goals.
Capabilities alone are not goals. However, an explicitly desired result may share
its evidence AND wording with a responsibility/workflow fact. Prior extraction
never disqualifies a supported outcome; assess this category independently.
STRICT OUTCOME REQUIREMENT: the selected evidence must itself express the actor's
desired result, purpose, or problem to solve. "Should be able to" followed by an
action list does not express separate desired outcomes for every action. Return
ZERO goal items from that list unless it also explicitly expresses an outcome.
"Manage profiles/specialties/schedules/appointments/payments" alone establishes
provider actions, not provider goals. A primary provider does not become a
secondary actor merely because its activities were described second.
For a product-purpose sentence followed by patient and provider capability lists,
extract the stated purpose outcome only. Never turn each list entry into a goal.
success_criteria: an explicitly stated condition showing the desired result was achieved.
motivations: an explicitly stated reason or problem explaining WHY users want the result.
For product-wide success_criteria/motivations, role may be null; do not invent an owner.
Resolve explicit pronouns from the source/question without assigning one actor's goal to another.
Do not invent convenience, speed, access, trust, or efficiency from features.
Example: "The platform helps patients find and book appointments. Patients should
be able to describe their concern, compare providers, message them and pay."
Extract the find-and-book outcome from the purpose sentence. Extract zero goals
from the ensuing capability list unless it explicitly states a desired outcome.
Example: "Buyers browse listings, select an item and pay. Their goal is to safely
complete a purchase." The first sentence is actions/workflow; the second is a goal.
Before returning an item, ask whether its own evidence states the proposed result,
completion condition, or motivation. If it only states an available action, omit it.
Copy evidence verbatim; a contextual pronoun may use the full supporting sentences."""),
    ("RULES", RemainingFact, None, """Extract remaining EXPLICIT knowledge only.
Only use a listed topic/key combination supplied in the prompt. Do not invent arbitrary keys.
Normal features such as search, booking, communication, appointment reminders, payments, and provider profile management yield {"items": []} unless the user explicitly states a governing rule, constraint, MVP requirement, exception, or edge case.
Use the supplied definitions for EVERY allowed field. A sentence may also support
responsibilities, permissions, goals, or workflow in another pass; that does not
exclude a supported rule, constraint, scope decision, exception, or edge case.
EXCEPTIONS describe cancellation, timeout, invalid-action handling, or recovery.
EDGE_CASES describe duplicate, boundary, concurrent, or other unusual scenarios.
Use both when both meanings are stated; do not put every exception in rare_scenarios.
"A repeated payment submission is rejected" supports duplicate_actions AND invalid_actions.
"An idle checkout expires after 10 minutes" supports time_constraints AND timeouts.
"No payments in version one" supports out_of_scope with value="payments excluded
from version one"; it is not absence of out_of_scope items.
Only emit whole-field absence when explicitly stated; use absence="none", value="none"
or absence="not_applicable", value="not applicable". Do not confuse a denied
feature, forbidden action, limit, or specific exception with whole-field absence."""),
)


def normalize_fact(fact: Fact, topic: DiscoveryTopic, scope, turn: int) -> KnowledgeItem:
    data = fact.model_dump()
    if isinstance(fact, RemainingFact):
        topic = DiscoveryTopic(data.pop("topic"))
    if isinstance(fact, ActorFact):
        aliases = data.pop("aliases")
        data["roles"] = [canonical_role(role) for role in data["roles"]]
        if not data["roles"] and fact.key in ("multiple_roles", "role_transitions"):
            data.pop("roles")
        if aliases:
            data["aliases"] = {data["roles"][0]: [alias.strip().lower() for alias in aliases]}
    if isinstance(fact, OwnedFact):
        data["role"] = canonical_role(data["role"])
    if isinstance(fact, GoalFact) and data.get("role"):
        data["role"] = canonical_role(data["role"])
    return KnowledgeItem(topic=topic, scope=scope, source_turn=turn, **data)
