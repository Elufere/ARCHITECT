"""Field meanings shared by extraction, grounding, and discovery questions.

These are semantic views of evidence, not mutually exclusive buckets.
"""
from agents.state import DiscoveryTopic as T


FIELD_DEFINITIONS = {
    T.USER_ROLES: {
        "primary_users": "Functional actors directly participating in the core product value exchange, including both service recipients and providers. Do not invent roles or split professions into separate actors.",
        "secondary_users": "Additional actors who interact with the scoped product outside its primary value exchange, such as explicitly named support or oversight roles. Explicit absence of ALL additional actors is valid; denying one actor is not total absence.",
        "responsibilities": "The significant actions, activities, duties, capabilities, or processes that a role performs or manages within the product. Explicit ordinary product actions and 'can'/'should be able to' capabilities qualify. Preserve the exact actor owner. Actor names alone establish no actions.",
        "permissions": "Explicit authorization, prohibition, access restriction, exclusivity, or conditional authority for one role. A capability alone is not authorization. Preserve what is allowed/forbidden and its conditions; no special permissions is a valid explicit absence.",
        "multiple_roles": "Whether ONE person/account can hold multiple roles, simultaneously or in different interactions. 'A provider can also be a patient' and 'users can have both roles' establish this. 'Each account has only one role' establishes the negative. Merely listing several actors does not.",
        "role_transitions": "Whether and under what conditions a user changes or switches roles over time. 'Patients become providers after verification' establishes a conditional transition; 'roles never change' or 'users cannot switch roles' establishes no transitions. Holding two roles does not by itself imply switching.",
    },
    T.USER_GOALS: {
        "primary_user_goals": "A desired result a primary actor wants or the product explicitly helps them achieve, owned by that actor. An action list alone is not a goal. A purpose clause can state an outcome even when it uses action verbs.",
        "secondary_user_goals": "A desired result explicitly sought by an existing secondary actor, owned by that actor. Do not assign a primary actor's goal here. No secondary actors means no secondary-role goals are required.",
        "success_criteria": "An explicit observable condition indicating a user goal was achieved. May be product-wide (role=null) or actor-specific. A completed action list alone does not establish success. Can overlap a workflow completion condition if both meanings are explicit.",
        "motivations": "The explicit reason, need, or problem explaining why users seek the outcome or use the product. May be product-wide (role=null) or actor-specific. Do not infer convenience, safety, or efficiency from features.",
    },
    T.CORE_WORKFLOW: {
        "trigger": "The stated event or action that starts the normal product interaction, not the founder's intention to build it.",
        "workflow_steps": "The stated interaction/process sequence: connected steps of a normal journey, preserving stated order. A journey action list such as search, compare, book, communicate, pay qualifies even if partial. An isolated capability or unordered inventory of resources to manage is not a sequence. Never invent ordering or completeness.",
        "completion_condition": "The explicit condition or event that marks the normal interaction complete. Do not infer it from the last listed capability. May share evidence with a stated success criterion.",
        "downstream_dependency": "A stated external action, service, approval, or handoff required for the workflow to complete outside the current app/user's control. Do not infer dependency from another actor merely existing. Explicit no external dependency is valid.",
        "end_state": "The explicitly stated resulting status or situation after the normal interaction completes, rather than the condition that triggers completion. One sentence may explicitly establish both.",
    },
    T.BUSINESS_RULES: {
        "validation_rules": "System/business conditions that submitted data or actions must satisfy, and stated validation consequences. Ordinary input features alone are not validation rules.",
        "approval_rules": "Required review/approval, who grants it, when it is required, and conditions for approval. May also establish a permission or external workflow dependency.",
        "eligibility_rules": "Qualifications or prerequisites determining who may participate or use a function. Can overlap a role permission when actor authorization is explicit.",
        "limits": "Explicit numerical or categorical limits on transactions or operations. A limit can also support a constraint; boundary_conditions additionally needs behavior at/beyond the boundary.",
        "ownership_rules": "Rules assigning control/ownership of data, resources, or actions. Managing a profile alone does not establish an ownership policy.",
        "visibility_rules": "Rules specifying who can view which information and under what conditions. May also establish an actor-specific access permission.",
    },
    T.CONSTRAINTS: {
        "legal_constraints": "Explicit legal, regulatory, compliance, or mandatory legal requirements/boundaries affecting the product. Never assume regulations from the domain.",
        "business_constraints": "Explicit business policy, budget, commercial, or organizational limitations/requirements. Ordinary product behavior is not a constraint unless stated as a boundary.",
        "operational_constraints": "Explicit operational capacity, resource, availability, integration, or dependency requirements/restrictions. Normal provider management capabilities are not constraints.",
        "geographic_constraints": "Explicit geographic service coverage, jurisdiction, or location restrictions/requirements.",
        "time_constraints": "Explicit deadlines, expiry periods, availability windows, or timing requirements. A time limit is not timeout handling unless the response says what happens on expiry.",
    },
    T.MVP_SCOPE: {
        "must_have_features": "Features explicitly required/included for the MVP, launch, or first version. Do not infer MVP inclusion from a generic feature description. A direct answer to the actual MVP question supplies this context.",
        "nice_to_have_features": "Explicitly optional, desirable, deferred, or later-version features. Explicit no optional/deferred features is valid.",
        "out_of_scope": "Features explicitly excluded from the scoped product or MVP. 'No payments in version one' records payments as excluded, not absence of exclusions. A deferred feature can also be out of the first version.",
        "success_metrics": "Explicit product/MVP success measures, adoption/business targets, or evaluation indicators. Distinct from whether an individual user interaction succeeded; shared evidence is allowed if both are stated.",
    },
    T.EXCEPTIONS: {
        "user_cancellations": "Stated handling or policy when a user cancels a normal interaction, including explicit disallowance of cancellation. Do not infer refunds or recovery.",
        "timeouts": "Stated behavior when an interaction expires or a user/system fails to act in time. A duration alone belongs to time constraints/limits, not handling.",
        "invalid_actions": "Stated rejection/handling of invalid, disallowed, or failing actions. Can share evidence with validation or authorization rules if the failure response is explicit.",
        "recovery": "Stated resumption, retry, restoration, or compensation after interruption/failure. Do not assume recovery from a normal happy path.",
    },
    T.EDGE_CASES: {
        "duplicate_actions": "Stated handling of repeated submissions/actions, including idempotency or duplicate prevention. Can also be invalid-action handling when rejection is stated.",
        "boundary_conditions": "Stated behavior at or beyond minimum/maximum/empty/zero/capacity boundaries. A bare limit alone is not boundary behavior.",
        "simultaneous_actions": "Stated handling of concurrent actions, races, or contention for the same resource. Do not infer concurrency handling from multiple actors existing.",
        "rare_scenarios": "Explicit unusual scenarios and their intended handling outside the more specific duplicate/boundary/concurrency fields. Do not route every ordinary cancellation or timeout here merely because it is an exception.",
    },
}


OVERLAP_RULES = """Evaluate each supported semantic view independently. Evidence is
reusable across topics and keys; never discard a fact merely because another pass
used its quote. Never emit a fact merely because another category is supported.
Every candidate must independently satisfy its own category using its own quote.
Explicit role actions may be BOTH responsibilities and workflow
steps when they form a journey. Desired outcomes still need explicit outcome
meaning; authorization still needs an explicit boundary. Likewise, validation
with a stated rejection can support both validation_rules and invalid_actions.
A concurrency failure with stated recovery may support simultaneous_actions and
recovery. Do not force arbitrary single-topic routing; do not manufacture another
meaning just to fill a gap. Preserve actor, scope, conditions, and negation.
Explicit whole-field absence/inapplicability is knowledge, not missing data.
Silence and descriptions of separate actors establish no whole-field absence.
Specific exclusions/prohibitions are substantive facts, not whole-field absence.
"""


def field_contract(pairs):
    """Render only relevant definitions to avoid bloating each model prompt."""
    return "\n".join(f"{topic.value}.{key}: {FIELD_DEFINITIONS[topic][key]}"
                     for topic, key in dict.fromkeys(pairs))
