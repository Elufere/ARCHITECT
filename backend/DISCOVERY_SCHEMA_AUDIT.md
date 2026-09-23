# USER_APP discovery schema audit

The alignment audit covers all **38 required keys**. The later regression repair
preserves those definitions and extraction paths while tightening own-evidence
support, goals, workflow fields, absence, and user-facing clarification.

## Answers to the ten audit questions

1. The planner's intended meaning is summarized by the shared definition in each
   row below; planner objectives remain short questions about the missing knowledge.
2. The same field definition is sent to the responsible extraction pass.
3. Grounding receives the definitions for its candidates and evaluates each one
   independently against its own evidence quote. Definitions are internal context.
4. Concrete narrower/different interpretations and repairs are listed per row.
   Natural-language model omissions remain possible; deterministic test fixtures
   do not establish model accuracy. Live replays test that separately.
5. Overlap is permitted, never required merely because another category matches.
   Every emitted view must independently satisfy its category. There is no generic
   fallback bucket, and deduplication includes topic/key/owner.
6. All fields can represent confirmed whole-field absence/inapplicability where
   explicit and appropriate. Silence, uncertainty, specific exclusions, and
   unmentioned behavior do not qualify. Active-gap negative resolution preserves
   the exact gap. Serialized `none` cannot bypass the separate absence audit.
7. Responsibilities, permissions, primary goals, and secondary goals require an
   existing owner in the current scope. Goal type must match actor classification.
   Actor declarations use `roles`, not `role`. Other required keys are global;
   success criteria/motivations can optionally retain an owner. Scope/topic/key
   boundaries are checked throughout extraction and gap completion.
8. Identical quotes may support responsibility and workflow, or rule and exception,
   when each view is supported. No cross-topic deduplication suppresses either.
9. No required planner key lacks storage or an extraction path after alignment.
10. No stored discovery key is omitted from planner completeness. An executable
    test asserts equality of planner keys, storage keys, pass coverage, and definitions.

## Per-field findings

The definition column applies to extraction and grounding. Question generation
uses it only as hidden context; clarification never quotes it to the user.

| Gap | Shared semantic definition | Existing pass | Finding / resolution |
| --- | --- | --- | --- |
| `USER_ROLES.primary_users` | Functional actors directly participating in the core product value exchange, including both service recipients and providers. Do not invent roles or split professions into separate actors. | ACTOR | Actor semantics retained. Declarations use roles, not an owner in role; audit logs now identify actor IDs. |
| `USER_ROLES.secondary_users` | Additional actors who interact with the scoped product outside its primary value exchange, such as explicitly named support or oversight roles. Explicit absence of ALL additional actors is valid; denying one actor is not total absence. | ACTOR | Active-gap absence preserved. Typed explicit absence can also be emitted without active focus. Question and clarification include all known primary actors. |
| `USER_ROLES.responsibilities::<role>` | The significant actions, activities, duties, capabilities, or processes that a role performs or manages within the product. Explicit ordinary product actions and 'can'/'should be able to' capabilities qualify. Preserve the exact actor owner. Actor names alone establish no actions. | RESPONSIBILITY | Fixed duty-only definition; explicit role actions/capabilities qualify and may share workflow evidence. |
| `USER_ROLES.permissions::<role>` | Explicit authorization, prohibition, access restriction, exclusivity, or conditional authority for one role. A capability alone is not authorization. Preserve what is allowed/forbidden and its conditions; no special permissions is a valid explicit absence. | PERMISSION | Authorization meaning retained. Question instructions now ask for boundaries, not another capability list. |
| `USER_ROLES.multiple_roles` | Whether ONE person/account can hold multiple roles, simultaneously or in different interactions. 'A provider can also be a patient' and 'users can have both roles' establish this. 'Each account has only one role' establishes the negative. Merely listing several actors does not. | ACTOR | Previously no pass could emit it. Added to ACTOR. Explicit policy required; silence cannot establish none. |
| `USER_ROLES.role_transitions` | Whether and under what conditions a user changes or switches roles over time. 'Patients become providers after verification' establishes a conditional transition; 'roles never change' or 'users cannot switch roles' establishes no transitions. Holding two roles does not by itself imply switching. | ACTOR | Previously no pass could emit it. Added to ACTOR. Preserve conditions and explicit negative policies. |
| `USER_GOALS.primary_user_goals::<role>` | A desired result a primary actor wants or the product explicitly helps them achieve, owned by that actor. An action list alone is not a goal. A purpose clause can state an outcome even when it uses action verbs. | GOAL | Outcome meaning retained and tightened after live over-extraction. Capability lists do not automatically create goals. |
| `USER_GOALS.secondary_user_goals::<role>` | A desired result explicitly sought by an existing secondary actor, owned by that actor. Do not assign a primary actor's goal here. No secondary actors means no secondary-role goals are required. | GOAL | Outcome meaning retained; primary actors cannot own these goals even when the secondary registry is empty. |
| `USER_GOALS.success_criteria` | An explicit observable condition indicating a user goal was achieved. May be product-wide (role=null) or actor-specific. A completed action list alone does not establish success. Can overlap a workflow completion condition if both meanings are explicit. | GOAL | Alignment made owner optional to match the global planner question. No further schema changes in the regression fix. |
| `USER_GOALS.motivations` | The explicit reason, need, or problem explaining why users seek the outcome or use the product. May be product-wide (role=null) or actor-specific. Do not infer convenience, safety, or efficiency from features. | GOAL | Alignment made owner optional to match the global planner question. No further schema changes in the regression fix. |
| `CORE_WORKFLOW.trigger` | The stated event or action that starts the normal product interaction, not the founder's intention to build it. | WORKFLOW | Explicit own-evidence support required for this field. Neither product purpose nor the final listed action establishes it; unmentioned dependencies are not none. |
| `CORE_WORKFLOW.workflow_steps` | The stated interaction/process sequence: connected steps of a normal journey, preserving stated order. A journey action list such as search, compare, book, communicate, pay qualifies even if partial. An isolated capability or unordered inventory of resources to manage is not a sequence. Never invent ordering or completeness. | WORKFLOW | Journey sequence may overlap role actions. No automatic trigger/completion/result/dependency from a capability list. |
| `CORE_WORKFLOW.completion_condition` | The explicit condition or event that marks the normal interaction complete. Do not infer it from the last listed capability. May share evidence with a stated success criterion. | WORKFLOW | Explicit own-evidence support required for this field. Neither product purpose nor the final listed action establishes it; unmentioned dependencies are not none. |
| `CORE_WORKFLOW.downstream_dependency` | A stated external action, service, approval, or handoff required for the workflow to complete outside the current app/user's control. Do not infer dependency from another actor merely existing. Explicit no external dependency is valid. | WORKFLOW | Explicit own-evidence support required for this field. Neither product purpose nor the final listed action establishes it; unmentioned dependencies are not none. |
| `CORE_WORKFLOW.end_state` | The explicitly stated resulting status or situation after the normal interaction completes, rather than the condition that triggers completion. One sentence may explicitly establish both. | WORKFLOW | Explicit own-evidence support required for this field. Neither product purpose nor the final listed action establishes it; unmentioned dependencies are not none. |
| `BUSINESS_RULES.validation_rules` | System/business conditions that submitted data or actions must satisfy, and stated validation consequences. Ordinary input features alone are not validation rules. | RULES | No field-shape mismatch. Added explicit shared policy definition and allowed independently supported overlap with permissions/constraints/handling. |
| `BUSINESS_RULES.approval_rules` | Required review/approval, who grants it, when it is required, and conditions for approval. May also establish a permission or external workflow dependency. | RULES | No field-shape mismatch. Added explicit shared policy definition and allowed independently supported overlap with permissions/constraints/handling. |
| `BUSINESS_RULES.eligibility_rules` | Qualifications or prerequisites determining who may participate or use a function. Can overlap a role permission when actor authorization is explicit. | RULES | No field-shape mismatch. Added explicit shared policy definition and allowed independently supported overlap with permissions/constraints/handling. |
| `BUSINESS_RULES.limits` | Explicit numerical or categorical limits on transactions or operations. A limit can also support a constraint; boundary_conditions additionally needs behavior at/beyond the boundary. | RULES | No field-shape mismatch. Added explicit shared policy definition and allowed independently supported overlap with permissions/constraints/handling. |
| `BUSINESS_RULES.ownership_rules` | Rules assigning control/ownership of data, resources, or actions. Managing a profile alone does not establish an ownership policy. | RULES | No field-shape mismatch. Added explicit shared policy definition and allowed independently supported overlap with permissions/constraints/handling. |
| `BUSINESS_RULES.visibility_rules` | Rules specifying who can view which information and under what conditions. May also establish an actor-specific access permission. | RULES | No field-shape mismatch. Added explicit shared policy definition and allowed independently supported overlap with permissions/constraints/handling. |
| `CONSTRAINTS.legal_constraints` | Explicit legal, regulatory, compliance, or mandatory legal requirements/boundaries affecting the product. Never assume regulations from the domain. | RULES | No field-shape mismatch. Explicit requirements/boundaries required; normal product behavior is not automatically a constraint. |
| `CONSTRAINTS.business_constraints` | Explicit business policy, budget, commercial, or organizational limitations/requirements. Ordinary product behavior is not a constraint unless stated as a boundary. | RULES | No field-shape mismatch. Explicit requirements/boundaries required; normal product behavior is not automatically a constraint. |
| `CONSTRAINTS.operational_constraints` | Explicit operational capacity, resource, availability, integration, or dependency requirements/restrictions. Normal provider management capabilities are not constraints. | RULES | No field-shape mismatch. Explicit requirements/boundaries required; normal product behavior is not automatically a constraint. |
| `CONSTRAINTS.geographic_constraints` | Explicit geographic service coverage, jurisdiction, or location restrictions/requirements. | RULES | No field-shape mismatch. Explicit requirements/boundaries required; normal product behavior is not automatically a constraint. |
| `CONSTRAINTS.time_constraints` | Explicit deadlines, expiry periods, availability windows, or timing requirements. A time limit is not timeout handling unless the response says what happens on expiry. | RULES | No field-shape mismatch. Explicit requirements/boundaries required; normal product behavior is not automatically a constraint. |
| `MVP_SCOPE.must_have_features` | Features explicitly required/included for the MVP, launch, or first version. Do not infer MVP inclusion from a generic feature description. A direct answer to the actual MVP question supplies this context. | RULES | Explicit launch/MVP inclusion required. Actual question may contextualize a short answer; generic feature descriptions do not imply MVP. |
| `MVP_SCOPE.nice_to_have_features` | Explicitly optional, desirable, deferred, or later-version features. Explicit no optional/deferred features is valid. | RULES | Explicit optional/deferred scope; may also establish first-version exclusion. No architectural change. |
| `MVP_SCOPE.out_of_scope` | Features explicitly excluded from the scoped product or MVP. 'No payments in version one' records payments as excluded, not absence of exclusions. A deferred feature can also be out of the first version. | RULES | Excluded feature is a substantive exclusion, not absence of exclusions. MVP scoping is exempted from the generic roadmap prohibition. |
| `MVP_SCOPE.success_metrics` | Explicit product/MVP success measures, adoption/business targets, or evaluation indicators. Distinct from whether an individual user interaction succeeded; shared evidence is allowed if both are stated. | RULES | Product/MVP measures remain distinct from individual interaction success. No shape change. |
| `EXCEPTIONS.user_cancellations` | Stated handling or policy when a user cancels a normal interaction, including explicit disallowance of cancellation. Do not infer refunds or recovery. | RULES | Clarified handling of this interruption/failure class. May overlap a specific edge case or rule when each meaning is explicit. |
| `EXCEPTIONS.timeouts` | Stated behavior when an interaction expires or a user/system fails to act in time. A duration alone belongs to time constraints/limits, not handling. | RULES | Clarified handling of this interruption/failure class. May overlap a specific edge case or rule when each meaning is explicit. |
| `EXCEPTIONS.invalid_actions` | Stated rejection/handling of invalid, disallowed, or failing actions. Can share evidence with validation or authorization rules if the failure response is explicit. | RULES | Clarified handling of this interruption/failure class. May overlap a specific edge case or rule when each meaning is explicit. |
| `EXCEPTIONS.recovery` | Stated resumption, retry, restoration, or compensation after interruption/failure. Do not assume recovery from a normal happy path. | RULES | Clarified handling of this interruption/failure class. May overlap a specific edge case or rule when each meaning is explicit. |
| `EDGE_CASES.duplicate_actions` | Stated handling of repeated submissions/actions, including idempotency or duplicate prevention. Can also be invalid-action handling when rejection is stated. | RULES | Clarified scenario-specific handling; overlap allowed when both meanings are supported. No additional schema changes during regression repair. |
| `EDGE_CASES.boundary_conditions` | Stated behavior at or beyond minimum/maximum/empty/zero/capacity boundaries. A bare limit alone is not boundary behavior. | RULES | Alignment changed question from bare limits to behavior at boundaries. No further schema changes in the regression fix. |
| `EDGE_CASES.simultaneous_actions` | Stated handling of concurrent actions, races, or contention for the same resource. Do not infer concurrency handling from multiple actors existing. | RULES | Clarified scenario-specific handling; overlap allowed when both meanings are supported. No additional schema changes during regression repair. |
| `EDGE_CASES.rare_scenarios` | Explicit unusual scenarios and their intended handling outside the more specific duplicate/boundary/concurrency fields. Do not route every ordinary cancellation or timeout here merely because it is an exception. | RULES | Explicit unusual residual scenarios, not a catch-all for ordinary exceptions. No further schema changes in the regression fix. |

## Completeness and scope checks

Alignment corrected canonical underscore/display-space matching, reopened completed
role topics when new roles create gaps, and allowed explicit empty actor sets to
waive per-role questions while unlocking dependencies. The subsequent surgical
regression repair did not change those behaviors.

Question-generation evidence is scoped and confirmed. Required MVP scope questions
are no longer forbidden as roadmap questions, and exception/edge-case discovery can
ask about intended handling without asserting speculative product facts. Those
alignment changes were retained; no schema redesign was added in regression repair.

## Verification

See `tests/test_schema_alignment.py` (all 38 keys and overlap),
`tests/test_alignment_regressions.py` (own-evidence, absence, actor context, and
clarification), and `tests/replay_schema_alignment.py` (opt-in live local replay).
See `tests/artifacts/schema_alignment/TEST_RESULTS.md` for classified broad failures.
The actual live trace and serialized knowledge are stored alongside that report.

Extraction still uses six passes, one normal audit when candidates exist, and at most one
active-gap absence decision. A malformed audit can receive one bounded missing-verdict
repair. No per-role or per-field model calls were introduced.
Secondary-user question generation now uses confirmed actor labels directly and
saves one generator call for that gap. Clarification remains zero model calls.
