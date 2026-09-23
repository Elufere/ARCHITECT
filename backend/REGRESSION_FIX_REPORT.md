# Discovery regression repair

Live acceptance status: **passed**. The final clean-state local qwen2.5:7b replay completed the initial description, clarification, and short No answer. It stored the six supported initial facts, advanced from secondary_users to permissions::patient, and rejected both injected unsupported claims. The saved careconnect_replay.json records passed=true. Earlier failed replay/probe artifacts are retained as diagnostic history.

This repair preserves the preceding schema alignment. It adds no model call per
role and makes no further changes to success criteria, motivations, boundaries,
rare scenarios, MVP architecture, exception schemas, topic reopening, or role
canonicalization. The earlier alignment audit is in DISCOVERY_SCHEMA_AUDIT.md.

## Regressions, causes, and minimal fixes

| Regression | Cause | Minimal fix |
| --- | --- | --- |
| Action lists emitted as goals | Removing blanket anti-overlap language left insufficient independent outcome requirements; the small local model converted actions into goals. | Require an expressed desired result in each goal's own quote. Product-purpose clauses qualify; capability lists alone do not. Preserve legitimate cross-category overlap. |
| Provider actions classified as secondary goals | Goal ownership/classification was not reliably tied to the confirmed actor registry, especially an empty secondary registry. | Reject a goal whose owner is outside its primary/secondary registry, including empty registries; provide registry context to grounding. Provider paragraph order does not determine actor classification. |
| Whole-field `none` from silence | The model treated separate roles/unmentioned dependencies as negative policies; serialized `value=none, absence=null` could bypass absence-specific checks. | Normalize serialized absence sentinels, forbid silence-based negatives, and require a separate `confirmed_absence_ids` verdict in the existing audit call. A generic supported ID alone is insufficient. |
| Invented workflow trigger/completion/end/dependency | Broadening workflow extraction without independent per-key thresholds allowed the model to reinterpret founder intent and final actions. | Require explicit start events, completion conditions, resulting states, and dependencies independently; a journey list only establishes its stated steps. |
| Correct action attached to the wrong quote | Whole-response grounding could rescue a claim with a fact stated elsewhere. | Require the candidate's own evidence to support value, category and role. Full-response context may resolve references but cannot supply the missing action/outcome. |
| Audit approves a mentioned action in an invalid category | An ID-only support verdict let the model conflate factual mention with category validity. | Require exact typed `TOPIC.key` category verdicts for each own quote, factual support, and the separate absence verdict when applicable. Group candidates beside their own quotes. Reject pure capability-list category contradictions before auditing. |
| Audit silently omits a valid candidate | A live response listed five supported IDs for six candidates and gave no verdict/reason for the patient goal. | Repair missing or structurally incomplete support verdicts once, using local IDs and the same evidence/context. Never retry explicit rejections. Preserve previously verified results if the repair fails. |
| Short-answer audit returns malformed evidence references | The model repeatedly used a category name as a dictionary key instead of the evidence ID, including during repair. | The model-facing response now uses typed entries with a numeric evidence ID and category list. Convert to the internal dictionary only after validation; reject duplicate entries. |
| General audit loses the gap interpreter's answer context | The exact-gap interpreter correctly returned none, but its review was passed along only as another undifferentiated candidate. | Include its exact field, role, scope, quote and resolution in the audit context. The auditor still independently verifies it and cannot use it to justify other fields. |
| Actor rejection logged as `unowned` | The diagnostic printed `item.role or 'unowned'` for every item, including declarations that correctly carry `roles` instead. | Explicitly identify actor declarations in the audit payload and instructions. Log actual actor IDs, own quote, candidate ID, and rejection reason. No role-owner requirement was added to declarations. |
| Malformed actor IDs in a diagnostic replay | The shared extraction prompt mixed declaration and owner-selection instructions; the model emitted `roles=["primary_users"]` for both actors. This schema classification was accepted as an ID. | Distinguish registration from ownership and reject actor field names as IDs. Put the user's source in a separate HumanMessage. The failed call is preserved in `careconnect_malformed_actor_calls.json`. |
| Secondary-user question omits a known actor | Free-form question generation could omit confirmed context. The reported original loss cannot be assigned an exact rejected actor from its old diagnostic alone. | Build this simple question directly from all confirmed scoped primary actor declarations. Test extraction through grounding, storage, planner reconstruction and question context. |
| Clarification leaks ontology | Shared definitions were appended to planner objectives; conversation_manager directly interpolated that objective in `I mean ...` plus generic filler. | Remove definition concatenation from public objectives. Keep semantic definitions as hidden extraction/grounding/question context; use separate plain-language clarification templates for all 38 keys. |

The retained before-fix live replay actually stored both primary actors and
referenced both in its question. It reproduced semantic over-extraction, not the
reported actor disappearance. Therefore the original missing actor/rejection
reason remains unproven; `unowned` itself was a logging label, not a validation
failure. The new diagnostics make a recurrence attributable.

The diagnostic replay also demonstrated that prompt examples could be quoted as
evidence. Exact-substring validation rejected those invented quotes. User text is
now separated from system instructions; earlier action candidates are no longer
repeated inside the goal prompt, reducing both action-to-goal priming and input
size. The local model was running with a 4096-token context; an explicit 8192-token
window now gives the schema, definitions and audit candidates more room. This is
a context-capacity precaution, not proof that the previous output was truncated.

## Surgical files

- `agents/extraction_passes.py`: strict goals/workflow/actor policies; serialized absence normalization.
- `agents/discovery_fields.py`: independent overlap and explicit-negative contract.
- `agents/semantic_validation.py`: typed own-evidence category contract, narrow contradiction checks, separate absence verdict and rejection reasons.
- `agents/knowledge_tracker.py`: evidence groups, actor classification/declaration context, audit enforcement, bounded protocol repair and useful rejection diagnostics.
- `agents/conversation_language.py`: confirmed actor labels and plain-language questions/clarifications.
- `agents/conversation_manager.py`: clarification recognition and replies without objective echo/filler.
- `agents/interview_planner.py`: stop appending internal field definitions to the conversational objective.
- `agents/question_generator.py`: complete actor context for secondary-user questions; semantic definitions remain hidden instructions.
- `tests/test_alignment_regressions.py`: new regression coverage, including wrong evidence, category overlap, goals, workflow separation, silence vs explicit absence, actors and all clarification keys.
- `tests/test_schema_alignment.py`, `tests/test_gap_absence.py`, `tests/test_pm_discovery_regressions.py`: absence mocks declare the independent audit verdict; existing coverage retained.
- `tests/test_extraction_passes.py`: verify source text in its separate message and strict goal semantics without duplicate action candidates.
- `tests/replay_schema_alignment.py`: live clean-state replay with clarification and short `No.`; actual serialized knowledge and diagnostics retained.
- `tests/probe_grounding_contract.py`: opt-in live audit against the real failed candidate batch; no fabricated model fixtures.
- `tests/probe_gap_handoff.py`: opt-in live audit of the actual failed short-No candidates and persisted initial context.

The final audit uses structured JSON-schema output with numeric evidence IDs and exact field-name enums.
JSON-mode, tool-calling, and per-quote-call experiments were not retained. A worked
example was removed after the model copied its rejection reasons onto different
candidate IDs. The final payload groups each source quote with its candidates.

The narrow pre-audit checks reject only capability lists using "should/will/must
be able to" with no separate outcome/lifecycle clause, and founder intent without
a stated start event. They do not manufacture any knowledge or resolve negative
answers. Single capabilities and mixed statements with explicit outcome/lifecycle
signals still go to semantic auditing. Tests cover preserved mixed clauses and
responsibility/workflow overlap. This is a conservative guard, not a general
natural-language entailment parser.
The extractor may still propose action-based goals. The passing live probe proves
those proposals are rejected before storage; it does not claim flawless raw model
output or a universal guarantee for every future model response.

## Preserved behavior

Responsibilities still include explicit product actions and capabilities, including
"should be able to". Shared evidence may independently support responsibilities
and workflow. Permissions still require actual authorization/restriction. Actor
policy extraction, scoped role gaps, gap-aware negatives and centralized field
definitions remain. No keyword rule interprets a user's "No" as absence.

The short-No test checks that the exact active question can be denied, whereas
"No caregivers" does not establish no secondary users. Clarification changes only
language, not knowledge or gap satisfaction. All 38 clarification keys are tested
for internal-term leakage; no new schemas were introduced for that audit.

## Verification

### CareConnect initial turn

The final clean-state replay stored exactly these six confirmed facts:

| Field | Stored content |
| --- | --- |
| USER_ROLES.primary_users | patient (display: patients) |
| USER_ROLES.primary_users | healthcare_provider (display: healthcare professionals) |
| USER_ROLES.responsibilities::patient | Describe concerns, search, compare profiles/availability, book, communicate, receive reminders, pay |
| USER_ROLES.responsibilities::healthcare_provider | Manage profiles, specialties, schedules, appointments, payments |
| CORE_WORKFLOW.workflow_steps | The stated patient journey, in its stated order |
| USER_GOALS.primary_user_goals::patient | Find and book appointments; evidence is the platform-purpose sentence |

The patient responsibility and workflow retain the same exact patient sentence as
evidence. No action-only goal, inferred role policy/absence, permission, invented
workflow lifecycle, rule, constraint, scope, exception or edge case was stored.
Remaining USER_ROLES gaps are secondary_users, permissions::patient,
permissions::healthcare_provider, multiple_roles, and role_transitions.

Exact question: "Besides patients and healthcare professionals, will anyone else use the user app?"

Exact clarification: "I mean, besides patients and healthcare professionals, will anyone else use the user app?"

Clarification made no model calls and changed no knowledge. Neither responsibility
gap was asked. The before-fix replay stored 16 initial facts, including unsupported
multiple-role absence, workflow trigger/completion/end state and action-only goals.

### Answer after clarification

The final replay stored one new item after "No.": USER_ROLES.secondary_users,
value=none, absence=none, scope=USER_APP, roles=[], CONFIRMED, confidence=1.0,
evidence="No.". The planner advanced to permissions::patient. Its remaining
USER_ROLES gaps are permissions::patient, permissions::healthcare_provider,
multiple_roles and role_transitions. No workflow or goal was added on this turn.

Focused: **200 passed**. Broad backend: **219 passed, 12 failed, 3 collection errors**.
All 12 remaining test failures are also present in the frozen baseline: 11 are
outdated expectations and one is a pre-existing compound-role validation failure.
The three collection errors concern missing chromadb, missing bs4, and the already
absent prior_gap_recovery module. See
`tests/artifacts/schema_alignment/TEST_RESULTS.md` for individual classifications
and the XML/text artifacts for complete evidence. Tests were not skipped to conceal
failures. Compile checks and `git diff --check` passed.

Mocked regressions exercise the real validation/grounding/storage code with fixed
model decisions. They prove enforcement, not live LLM classification accuracy.
The separate live replay records actual local qwen2.5:7b responses.

## Model calls

The extraction architecture is unchanged: six extraction passes, one grounding
audit when candidates exist, and at most one active-gap interpretation call.
There are no per-role or per-quote calls. A malformed audit that omits candidate
verdicts or omits the required category/absence fields can trigger **one additional repair call**, limited to those incomplete verdicts;
explicit rejections never trigger it. Clarification remains zero calls.
The secondary-user question now requires zero generation calls
instead of one; other generated questions retain their existing call. The replay's
extra injected-claim audit is a verification probe, not a production turn call.

Measured conversation calls (excluding the final verification-only probe):

| Turn | Saved before-fix run | Final run |
| --- | ---: | ---: |
| Initial description | 8 (six passes, audit, question) | 8 (six passes, audit, one repair; question uses a template) |
| Clarification | 0 | 0 |
| Negative answer | 9 (six passes, gap interpreter, audit, question) | 10 (same plus one bounded audit repair) |
| Total | 17 | 18 |

Thus the measured replay increased by one call overall. The normal six-pass
architecture is preserved; incomplete audit responses can add one repair per
information turn, and the secondary-user template saves one generation call.

The final injected-claim probe accepted zero items. Its two audit calls are excluded from conversation call counts. The saved before-fix run used the longer explicit negative answer; the final run tests the stricter clarification followed by a bare No.
