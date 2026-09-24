"""Small, independent structured extraction contracts for one user response."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.state import DiscoveryTopic, KnowledgeState, KnowledgeItem
from agents.state import BUSINESS_RULES_KEYS, CONSTRAINTS_KEYS, MVP_SCOPE_KEYS, EXCEPTIONS_KEYS, EDGE_CASES_KEYS
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
    key: Literal[BUSINESS_RULES_KEYS, CONSTRAINTS_KEYS, MVP_SCOPE_KEYS, EXCEPTIONS_KEYS, EDGE_CASES_KEYS] = Field(
        description="Bare field name belonging to topic, e.g. approval_rules. Never include a topic prefix; topic is stored separately.")

    @model_validator(mode="before")
    @classmethod
    def normalize_topic_prefix(cls, data):
        # Repair only an exact matching topic prefix. The field enum and
        # topic/key validator still reject unknown or mismatched fields.
        if isinstance(data, dict):
            topic, key = data.get("topic"), data.get("key")
            if isinstance(topic, str) and isinstance(key, str) and key.startswith(topic + "."):
                return {**data, "key": key[len(topic) + 1:]}
        return data

    @model_validator(mode="after")
    def valid_key(self):
        from agents.state import TOPIC_KEY_MAP
        if self.key not in TOPIC_KEY_MAP[DiscoveryTopic(self.topic)]:
            raise ValueError(f"{self.key} is not a key for {self.topic}")
        return self


PASSES = (
    ("ACTOR", ActorFact, DiscoveryTopic.USER_ROLES, """Identify functional product actors only.
Application scope and business-process participation are separate. This boundary
applies on EVERY turn, including initial discovery, before the actor registry exists.
The current scope is an extraction boundary, not evidence of actor membership.
Only declare a CONFIRMED actor in that scope when the source explicitly establishes
interaction with that application. Shared payments, disputes, records, or business
processes do not establish shared application access. Another application,
back-office tool, internal operation, external system, offline process, or
third-party platform must not be silently treated as the current application.
Retain explicit surface distinctions in supporting evidence. If the actual
surface is unknown, keep membership unresolved; do not invent a surface or
default it to the current app. Do not create actors in a different scope here.
Resolve each mentioned label against the confirmed canonical actor registry,
aliases, and source-backed role relationships BEFORE proposing a new actor.
A contextual capacity, subrole, or transaction role is not automatically a
separate application user type. If the response describes one actor acting in
different capacities, emit ONE actor declaration with that canonical ID in roles.
Record the explicitly stated capacity labels in aliases, preserve the relationship
and its conditions in value, and quote the full supporting relationship in evidence.
For later mentions of established capacities, reuse their canonical actor; do not
emit primary_users or secondary_users for the capacity labels themselves.
When the relationship is new for an existing actor, emit an updated declaration
under the existing canonical ID so its aliases and source evidence are retained.
Do not infer a relationship from domain conventions or similar-looking names.
Shared participation or the ability to hold multiple independent user types alone
does not make those types capacities of a single actor. Discover a genuinely new
application actor when explicitly described. If the user explicitly establishes
a label as a separate application user type, preserve that distinction even if
the same label previously denoted a contextual capacity.
Apply this identity resolution before primary/secondary classification: multiple
sides of an interaction do not by themselves establish independent actor types.
When actors have already been discovered, classify each newly mentioned
participant BEFORE adding a confirmed actor: existing actor under another label,
contextual capacity, new current-application user, other application/back-office
participant, external workflow dependency, or ambiguous membership.
Only add a new CONFIRMED actor when its own quote explicitly establishes that it
interacts with the CURRENT application. Explicitly identifying it as a user of
this application also establishes membership. Logging in, using its dashboard,
viewing records in it, submitting information through it, or managing something
through it are evidence when the application reference resolves to the current
scope. These are semantic examples, not a required keyword list.
A named participant performing an action, even a review/approval in a conditional
workflow, does not by itself establish application membership. A job title or
administrative-sounding duty is not sufficient evidence of a secondary user.
An explicit other-application/back-office participant or external dependency is
not an actor of this scope; emit no actor declaration for it here. Do not relabel
an explicitly external participant as an ambiguous current-application user.
If the participant is explicitly mentioned but current-application membership
is genuinely ambiguous, preserve a tentative actor candidate using the existing
knowledge_state="INFERRED". Its value must retain the stated participation and
say current-application membership is unresolved; its evidence must be the exact
supporting quote. Do not invent an in-app action, alias relationship, or definite
membership. This is unresolved information for clarification, not a confirmed
actor or an owner for extracted actions. Never infer a participant from silence.
For later new participants, this membership check takes precedence over the
occupational and primary/secondary classification guidance below.
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
For these two fields, distinguish whole-field absence from a scoped restriction.
"Users never have multiple roles at all" explicitly denies multiple roles in
every context and may use multiple_roles with value="none", absence="none".
"Users can have different roles across transactions but cannot hold both within
one transaction" is a substantive multiple_roles policy, with absence=null.
Preserve BOTH the cross-transaction allowance and the within-transaction ban.
"Users may switch roles" is a positive role_transitions policy. "Users cannot
switch roles once a workflow instance starts" is a scoped role_transitions
restriction, with absence=null; preserve when the restriction starts and applies.
Never use cannot/never/not alone to decide absence. Conditional restrictions and
mixed allowed/prohibited behavior are policies, not whole-field absence.
For a "yes, but" answer, resolve the allowance against the actual question and
retain the stated exception. Extract both fields when both policies are stated;
do not lose a transition restriction merely because the active gap is multiple_roles.
Do not infer either policy merely from a list of actors. A conditional denial is a confirmed
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
Distinguish an action/capability from a statement that only defines authorization.
An exclusivity rule, prohibition, or access limitation alone is a permission,
not an additional responsibility. Do not manufacture an unqualified action by
removing only, cannot, or an access/authority condition from such a statement.
In particular, "can" inside an access-boundary statement does not independently
establish a performed activity. Emit both categories only when the evidence also
affirmatively states the actor's action, duty, or capability independently of
the authorization boundary; preserve each meaning in its own candidate.
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
Before emitting an item, identify the explicit boundary on this actor's proposed
action or access. The candidate value must retain that boundary, including its
polarity, exclusivity, conditions, and scope. Merely paraphrasing an action as
"can" or "may" perform it does not establish authorization semantics.
A capability clause elsewhere in a quote cannot borrow a restriction belonging
to another action or actor. Likewise, an ownership phrase such as managing one's
resources is not an access limitation unless the source actually limits access.
An authorization-only clause belongs here without automatically duplicating it
as a responsibility. Both categories are valid only when the quote independently
states an action/duty/capability and an explicit authorization boundary.
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
For workflow_steps, require actual ordered or process-like actions connected by
a journey, stage, handoff, or temporal relationship in the source. Formal numbering
is unnecessary, but a standalone available action or feature is insufficient.
One stated action within an explicit process stage may be a partial process;
do not require a complete journey or manufacture preceding/following steps.
Apply these independent thresholds before emitting each non-step field:
- trigger: the quote explicitly identifies what starts the interaction. A product
  purpose or founder intention does not state a start event.
  A motivation or reason for choosing the product is not a process-start event.
- completion_condition: the quote explicitly defines when the process is complete.
  The last listed action, a reminder, or payment capability is not that definition.
  The availability of a dispute/exception-handling feature does not say when the
  main process completes. Do not invent a completion test from a feature.
- end_state: the quote explicitly states a resulting status after completion.
  A list of things users can do establishes no final status.
  Desired benefits and arbitrary descriptive statements are not resulting states.
- downstream_dependency: the quote explicitly names a required prerequisite,
  external action, approval, system, or event AND ties it to a process stage that
  needs it before proceeding or completing. A named participant, temporal qualifier,
  eligibility condition, or role restriction alone is not that dependency link.
  Omit if unmentioned; silence is
  never evidence of none. Explicitly denying such dependencies is different.
  Use absence="none" only for an explicit denial of the WHOLE dependency field.
  Denying one kind of approval or prerequisite does not deny all dependencies.
Assess all five fields independently. The current CORE_WORKFLOW topic or gap is
not evidence for any field. A response may support exactly one workflow fact;
return only that fact and leave every unsupported field unmentioned.
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
Apply this requirement to individual actions too, not only capability lists.
A performed workflow step, transaction action, agreement, approval, or dispute
handling procedure does not by itself state a desired result. A product feature,
authorization boundary, or mandatory business rule also establishes no goal.
Do not turn these descriptions into goals by adding "wants", "aims to", or an
unstated benefit to the value. Describing a feature as secure/easy/fast does not
state a user's objective or reason for choosing the product.
For a sentence containing both an action and an explicit desired outcome, extract
the stated outcome, not the action relabeled as a goal. This is a semantic test,
not a ban on particular verbs: an explicitly desired result remains valid even
when the same wording also describes an action elsewhere.
Choose the USER_GOALS key independently: a desired actor result belongs to a
role-specific goal; an explicit condition defining success belongs to
success_criteria; a stated reason for choosing/wanting the product belongs to
motivations. A prerequisite for performing an action is not a success condition
unless the user explicitly defines success that way. Do not move unsupported
actions, features, or rules into success_criteria or motivations as a fallback.
Resolve the owner from the stated outcome and supported identity/reference
context. The current gap's actor is not ownership evidence. If ownership remains
ambiguous, omit the role-specific goal rather than selecting an actor to satisfy
the schema. Do not turn an owner's action into another actor's desired result.
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
Store topic separately and emit key as the bare field name only. For example,
emit topic="BUSINESS_RULES", key="approval_rules", never key="BUSINESS_RULES.approval_rules".
Qualified labels in field definitions identify categories; they are not serialized keys.
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
