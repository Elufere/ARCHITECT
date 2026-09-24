This describes the current Product Manager (PM) agent implementation in the Architect project, updated on 2026-09-23. It includes goal-owner repair, role-policy answers, controlled-question contracts, and isolation of direct-answer grounding. It describes the existing code, not a proposed redesign. Paths below are relative to the repository root.

The PM conducts a structured product-discovery interview, stores evidence-backed facts, chooses the next missing requirement, asks one question, and eventually attempts to compile a PRD. It uses Python, LangGraph, LangChain's OpenAI integration, and Pydantic. The named components are functions in one graph, not autonomous agents running concurrently.

1. Execution flow

The terminal entry point is `backend/test_pm.py`. It creates or resumes `AgentState`, persists each submitted human answer, increments the turn counter, and calls `graph.invoke(state)`. Atomic JSON checkpoints record successful node boundaries and the next-node cursor, so recovery can resume without re-extracting committed answers. Planner-owned gap coverage is separate from global knowledge. See `DISCOVERY_RECOVERY.md` for coverage, checkpoint and retry semantics; this uses boundary snapshots rather than a LangGraph database checkpointer.

```text
Human message
    |
    v
Conversation manager
    |-- clarification / rationale / summary / uncertainty --> local reply --> END
    |-- ordinary confirmation -----------------------------> planner
    |-- product information / correction / objection ------> knowledge tracker
    |-- confirmation of an inferred fact ------------------> knowledge tracker
                                                               |
                                                               v
                                                        interview planner
                                                           |         |
                                             gap remains --+         +-- completion flag
                                                   |                         |
                                                   v                         v
                                            question generator          PRD compiler
                                                   |                         |
                                                   v                        END
                                                guardrail
                                              /           \
                                      rejection           acceptance
                                         |                    |
                                  regenerate question        END
```

The graph is defined in `backend/agents/graph.py`. The CLI first runs USER_APP discovery, then offers an optional ADMIN_DASHBOARD phase. Facts are retained between phases and tagged by scope; the planner filters facts by the active scope.

2. Models and call structure

| Component | Implementation |
| --- | --- |
| Conversation intent classification | Python regular expressions; no model call |
| Six extraction passes | Central OpenAI model (`OPENAI_MODEL`, default `gpt-4.1-mini`), temperature 0, timeout 60 seconds |
| Active-gap interpretation and evidence grounding | Same centrally configured OpenAI model, separate structured-output calls |
| Interview planner | Deterministic Python |
| Question generation | OpenAI `gpt-4.1-mini` (configurable through `OPENAI_MODEL`), temperature 0; secondary-user questions have a deterministic template |
| Question evaluator | OpenAI `gpt-4.1-mini` (configurable through `OPENAI_MODEL`), temperature 0, structured output |
| Final PRD compilation | Central OpenAI model (`OPENAI_MODEL`), parsed `PRDDraft` structured output |
| PRD verification | Central OpenAI model (`OPENAI_MODEL`), independent source/claim classification plus per-claim semantic auditing |

Extraction calls execute sequentially. They are not parallel agents. A typical free-form information turn with an active gap and candidate facts uses six extraction calls, one gap-interpretation call, one grounding call, one question-generation call, and one question-evaluation call: approximately ten calls before repairs. Literal choices to the application's controlled additional-users question instead use the answer contract described below. The count varies with empty candidate batches, deterministic questions, compilation, repairs, and guardrail retries.

3. Shared state and knowledge representation

`backend/agents/state.py` defines a TypedDict `AgentState`. Messages use LangGraph's `add_messages` reducer. Other state includes raw idea, discovered knowledge, scope, current topic, per-topic status and maturity, current gap, objective, question hint, current role, known/missing fields, selected context, inferred evidence, conversation intent, correction flag, turn count, completion flags, and PRD output.

Each stored `KnowledgeItem` has:

```text
topic              DiscoveryTopic
scope              USER_APP | ADMIN_DASHBOARD
key                canonical field name belonging to the topic
value              textual fact
evidence           source quotation
roles              actor IDs for actor declarations, otherwise usually null
aliases            optional actor alias mapping
role               one owner for an actor-specific fact, otherwise null
confidence         numeric confidence
knowledge_state    CONFIRMED | INFERRED
source_turn        turn number
absence            null | none | not_applicable
```

Actor declarations use `roles`; responsibilities, permissions, and primary/secondary goals use a singular `role`. Role IDs are normalized and checked against known actors. Primary goals must belong to primary actors, and secondary goals to secondary actors.

`product_model.py` derives a grouped textual view of confirmed facts in the current scope. This is context for summaries and question generation, not a separate independently maintained domain model or database.

4. Discovery schema

The fixed schema contains eight topics and 38 base fields. Actor-specific fields expand into multiple gaps when multiple actors exist.

| Topic | Fields |
| --- | --- |
| USER_ROLES | primary_users, secondary_users, responsibilities, permissions, multiple_roles, role_transitions |
| USER_GOALS | primary_user_goals, secondary_user_goals, success_criteria, motivations |
| CORE_WORKFLOW | trigger, workflow_steps, completion_condition, downstream_dependency, end_state |
| BUSINESS_RULES | validation_rules, approval_rules, eligibility_rules, limits, ownership_rules, visibility_rules |
| CONSTRAINTS | legal_constraints, business_constraints, operational_constraints, geographic_constraints, time_constraints |
| MVP_SCOPE | must_have_features, nice_to_have_features, out_of_scope, success_metrics |
| EXCEPTIONS | user_cancellations, timeouts, invalid_actions, recovery |
| EDGE_CASES | duplicate_actions, boundary_conditions, simultaneous_actions, rare_scenarios |

`discovery_fields.py` supplies shared semantic definitions to extraction, grounding, and question generation. A single quotation may support several categories if it independently meets each definition. The code distinguishes an actor's capability, authorization, desired outcome, workflow step, and business rule.

5. Knowledge extraction and validation

`knowledge_tracker.py` orchestrates six passes defined in `extraction_passes.py`:

| Pass | Responsibility |
| --- | --- |
| ACTOR | Primary/secondary actors and role-combination/transition policies |
| RESPONSIBILITY | Actions and duties performed by known actors |
| PERMISSION | Explicit authorization, restrictions, prohibitions, or conditional authority |
| WORKFLOW | Process sequence, start event, completion, dependencies, and end state |
| GOAL | Desired outcomes, success criteria, and motivations |
| RULES | Remaining five topics: business rules, constraints, MVP scope, exceptions, edge cases |

All six inspect the latest answer, even if the planner is currently asking about only one topic. Each receives the current scope/gap, last question, field definitions, and known actor IDs. Later passes can use actor candidates accepted by the earlier actor pass; final semantic grounding occurs afterward.

The provider-facing envelope is `RawPass(items: list[dict])`. Each dictionary is subsequently validated against its pass-specific Pydantic schema. Thus, the provider schema itself does not enforce every item property; those detailed contracts are embedded in the prompt and enforced locally. One malformed item can be rejected without discarding valid siblings.

Local checks enforce field/topic compatibility, valid owners, actor classification, confidence of at least 0.75, nonempty values, and exact case-sensitive evidence substrings. Some narrow category contradictions are rejected deterministically. These checks do not prove that a fact's meaning follows from its quote.

Rejected goal candidates receive one bounded model repair call. Repairs must pass the same local checks and downstream grounding. The code never automatically assigns the active gap's actor as the missing owner. Other extraction passes do not currently have an equivalent general repair mechanism.

The active-gap interpreter runs separately. Despite the function name `extract_gap_absence`, it now handles both explicit absence and role policies. Its `GapAnswer` can return `none`, `not_applicable`, `policy`, or `unresolved`, with evidence, confidence, and an optional policy value. A focused prompt handles multiple_roles and role_transitions. For example, “no” to a question about holding both roles in one transaction produces a substantive prohibition limited to that transaction, not absence of a policy. The question supplies interpretation context; the answer remains the evidence.

Next, an LLM grounding audit groups candidates by their exact quotation. When both direct answers and incidental candidates exist, the current gap's candidates are audited in a separate batch first. Other candidates are audited independently; their malformed output cannot discard an already-grounded direct answer. This can add one audit call to a turn. Extraction and grounding also receive previously confirmed actor declarations, including their source quotes, to resolve actor identity and established relationships. Those declarations do not serve as evidence for new actions or permissions.

Each audit returns supported candidate IDs, supported semantic categories per quote, explicit-absence IDs, and rejection reasons. A fact must pass both candidate support and category support; absence additionally requires an absence verdict. Missing verdicts can trigger one repair call. Audit failure rejects the affected batch rather than storing it unchecked. Logs print direct/incidental batch counts and the number of accepted facts for the active answer.

Accepted facts are merged and deduplicated. Explicit absence replaces prior values for the same scope/topic/key/owner; substantive facts replace prior absence. Confirmed facts replace inferred facts for the same slot. A correction flag removes matching prior facts before adding replacements. The derived product model is rebuilt afterward.

6. Conversation management and planning

`conversation_manager.py` classifies clarification, rationale requests, summaries, corrections, confirmations, objections, uncertainty, and ordinary product information using regex patterns. Clarification and other conversational replies can end the turn without extraction. A confirmation while a discovery gap is active now reaches the tracker, so a bare “yes” is no longer discarded. For an objection such as “I already told you,” the tracker re-extracts earlier human answers. Confirmation of an inferred fact can promote the existing inference.

`answer_contract.py` handles the application-generated additional-users question. The generator attaches metadata identifying its exact gap and scope. The tracker verifies that metadata, the current gap/scope, and the unchanged question text before interpreting a literal choice. “No” records scoped secondary-user absence directly, using the user's literal answer as evidence. “Yes” supplies no actor identity, so it leaves that gap open and requests the additional actors' names and activities via `answer_followup`. This is deterministic interpretation of a controlled question, not an LLM-grounding bypass for free-form content. Answers containing qualifications, untagged questions, altered questions, and mismatched scopes/gaps still use semantic extraction. Question generation consumes that follow-up instead of asking the existence question again.

`interview_planner.py` chooses the first unresolved schema field. Examples of expanded gaps are `responsibilities::customer`, `permissions::customer`, and `primary_user_goals::customer`. Explicitly empty actor sets can waive corresponding actor-specific fields. Only confirmed scoped facts count toward coverage. One confirmed fact for a slot can satisfy its presence requirement; coverage is not a guarantee of exhaustive requirements quality.

The planner stays on the current topic until its gaps are filled. Completed topics can reopen when newly discovered actors introduce new gaps. It also computes maturity: UNSEEN, MENTIONED, SKETCHED, COHERENT, DECISION_READY. Maturity controls dependency eligibility; completion still requires no missing fields.

Topic prerequisites are: goals depend on roles; workflow depends on roles and goals; business rules depend on workflow; constraints and MVP scope depend on business rules; exceptions depend on workflow and business rules; edge cases depend on exceptions. Topics are visited in enum order when eligible.

7. Question generation and guardrails

`question_generator.py` receives the selected gap, objective, role, scope, confirmed facts, product model, recent conversation, and special permission/inference guidance. It is instructed to ask one natural product question without drifting into technical design. Although recent conversation is summarized in the prompt, the invocation also passes the full message history. Secondary-user discovery uses a deterministic question based on known actors.

`guardrails.py` checks question formatting, conservatively detects repeated delivered questions, limits invented internal roles in USER_APP discovery, and invokes a semantic evaluator for alignment with the exact gap. Rejection appends a SystemMessage and routes back to generation. Regeneration is limited to two retries per user turn. After the third rejected draft, the guardrail returns a deterministic clarification for the unresolved gap and ends the turn without accepting the draft or marking the gap complete. The retry counter resets on a new human turn or accepted question. Secondary-user retries bypass the fixed template, and rejection feedback is included in the generator's main system prompt. If the evaluator throws an exception, the current implementation still allows the question through.

8. PRD compilation and current implementation limitations

`pm_agent.py` is only the compilation node; the rest of the PM behavior lives in the other modules. It builds a frozen snapshot of confirmed facts from the active scope. It supplies that snapshot, stable fact IDs, and canonical definitions to the compiler, without raw chat history, inferred facts, other-scope facts or static engineering requirements. `prd_validation.py` checks references, category alignment, source coverage, independent semantic classifications, and a detailed semantic verdict for every factual claim before saving. Approval authority cannot be turned into visibility merely by citing a valid approval fact.

The version 2.0 PRD schema contains sourced product name/summary claims, scope boundaries, personas, functional requirements with validation criteria, nonfunctional constraints, deferred items, and open questions. Factual claims carry source_fact_ids, a canonical category, actor_ids and conditions. The application attaches the original source snapshot and validation report. A ChromaDB search helper exists in `kb_injection.py`, but this graph does not call it. See PRD_COMPILATION.md for the output schema and verification boundary.

The following are code-review observations and should not be mistaken for completed features or newly verified runtime failures:

- `awaiting_confirmation=True` routes directly to compilation. There is no final user-confirmation turn in that route.
- The planner sets that flag when no eligible topic remains; that branch does not independently assert that every topic is completed.
- Compilation now fails closed on missing/invalid semantic verdicts, unsupported claims, and context-budget overflow. It permits one draft repair and never publishes a rejected draft. This remains model-based semantic verification, not a formal proof.
- The graph ends after compilation; it has no Architect node or handoff edge.
- The phases now write separate USER_APP and ADMIN_DASHBOARD artifacts. The second phase still resets only some state fields, retaining messages, facts, and some planning/completion-related state; phase-boundary behavior needs review.
- The uncertainty reply says it will keep an open decision and continue, but that branch does not persist a deferred-gap record or advance the planner.
- Rejected extraction leaves a slot missing; the planner can ask it again. There is no general no-progress budget or escalation strategy. Guardrail duplicate prevention changes wording but does not itself recover lost facts.
- The same model family proposes facts and judges their grounding, so an audit is a second check, not independent proof of correctness. Recent logs also show unsupported categories, malformed rule keys, and incorrect grounding decisions beyond the specific repaired cases.

Recent verification: 238 targeted deterministic regression tests passed. A live local-model replay of the exact multiple_roles question followed by “no” produced a transaction-specific negative policy and passed grounding. Captured live extraction of the escrow responsibilities answer produced two customer-owned facts; a live focused grounding check accepted both and advanced the planner to permissions::customer. The original reported responsibility rejection was not reproduced: the captured subset also passed a baseline mixed audit. Fault-injection regressions verify that unrelated malformed audit output cannot discard the direct answer, without granting unsupported facts automatic acceptance. This does not establish that a complete multi-phase interview and PRD compilation succeed end to end.
