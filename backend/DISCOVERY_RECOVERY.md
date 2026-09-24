# Deliberate discovery coverage and durable recovery

Implemented without running tests, the application, or OpenAI requests. Static
source parsing and diff inspection are the only verification performed.

## Root causes

- `build_gap_info()` used confirmed fact keys/owners as completion coverage.
- Incidental facts were stored correctly, but their existence skipped required
  questions. Prior confirmed target facts had no confirmation/expansion mode.
- The planner's exhausted-loop path and a stale `awaiting_confirmation` flag
  could reach compilation without an independent all-gaps-covered check.
- CLI state lived only in memory and every invocation entered extraction again.
- Several extraction/audit catch blocks swallowed provider failures. The question
  evaluator even accepted a question if its model call failed.

## Coverage lifecycle

`discovered_knowledge` remains global across the interview. Scope, semantic topic,
key, owner, confidence and evidence retain their existing meaning. No extractor
has been disabled to prevent incidental knowledge.

`fact_acquisition` separately records the first acquisition of each fact by stable
fingerprint as DIRECT or INCIDENTAL, including its source turn and active gap.
This does not change CONFIRMED vs INFERRED. A confirmed incidental fact remains a
confirmed fact and remains eligible for eventual PRD compilation.

`gap_coverage` is keyed by scope/topic/gap/canonical owner. An absent entry is
unresolved. The planner writes RESOLVED only after consuming `active_answer_result`,
a receipt for a successfully committed or already-represented grounded answer to
the actually asked gap. The receipt preserves answer ID, turn, user evidence,
fact IDs, and DIRECT_ANSWER or CONFIRMED_EXISTING resolution. Replaying that
receipt is idempotent; it does not create another fact.

The generator/guardrail boundary records `asked_gap`. A selected gap without a
delivered question is not sufficient to resolve coverage. Existing confirmed
target evidence produces `confirm_existing`; inferred evidence produces
`confirm_inference`. A gap-specific semantic review verifies affirmative answers
to these questions; 'no additions' must not erase the existing facts as absence.
Cross-topic evidence is still supplied in the global product model and scoped
question context. Valid absence can resolve an asked gap; absence of actors waives
inapplicable per-role questions only after the actor-source gap was deliberately
resolved. This retains the existing per-role topic contract.

The planner no longer equates `known_keys` with resolved gaps. Logs label these
as knowledge present versus unresolved discovery gaps. Missing deliberate coverage
invalidates stale completed flags explicitly. Compilation is gated both at routing
and at the compilation node by all required gaps being resolved. Existing confirmed
facts are not discarded when coverage is missing.

## Checkpoints and resume

The CLI creates a UUID session, persisted under `backend/sessions/<UUID>.json`.
`ARCHITECT_SESSION_DIR` can override the directory. Files contain the entire state,
LangChain message history/IDs, knowledge/provenance, scope, turn, gap/coverage/topic
statuses, pending answer/question, and a next-node cursor. JSON has a version and
is restored through the existing message and Pydantic constructors.

Writes use a temporary file in the same directory, flush/fsync and atomic replace.
They occur immediately after a submitted answer and after every successful graph
node. Node execution receives a private state copy. Failed nodes cannot overwrite
the last successful boundary or commit partial extraction. The CLI holds an OS
session lock so two processes cannot write the same interview simultaneously.

| Saved cursor | Resume action |
| --- | --- |
| conversation_manager / extract | Process the saved answer; never request it again. |
| plan | Consume the saved extraction/answer receipt, without re-extraction. |
| generate | Generate a question without reprocessing the previous answer. |
| guardrail | Validate the already-saved question; regenerate only if rejected. |
| waiting | Display the saved question and wait for input; make no model call. |
| compile_prd | Retry compilation only after checking deliberate coverage. |
| phase_complete | Offer the optional admin phase without recompiling USER_APP. |
| completed | No unfinished-session resume offer. |

CLI startup lists unfinished sessions; Enter resumes the latest, a number selects
another, and `new` starts a separate session without deleting prior progress.
USER_APP completion remains AWAITING_PHASE_CHOICE until the user chooses whether
to run ADMIN_DASHBOARD. Knowledge and scope-keyed coverage survive that transition.
Only legitimate completed discovery plus successful existing PRD verification can
reach a completed phase/session; errors and token-summary output cannot.

## Failure handling

All central model invocations use bounded exponential backoff for connection and
timeout errors, temporary server errors, and eligible rate limits. Defaults are
three retries after the first attempt, with 1/2/4 second delays. Authentication,
permission, invalid-request and exhausted-quota failures stop without retry loops.
SDK retries remain zero so retry budgets do not multiply. Usage callbacks remain
on each underlying call; retries still appear in token/cost logs.

`LLMCallFailed` and `ExtractionFailed` propagate through extraction, grounding,
absence replacement, deduplication, correction review, question evaluation and
compilation to the CLI. A provider failure, malformed pass response or failed
grounding protocol is not NO_FACTS_FOUND. Successfully parsed, grounded empty
results remain legitimate empty answers, but cannot resolve a gap without a valid
active-answer receipt. Exhaustion preserves the last checkpoint and prints its
topic/gap and restart instructions. No automatic outer CLI retry loop is added.

## Files changed for this request

This list excludes changes already present from the earlier extraction fixes and
OpenAI migration.

| File | Change |
| --- | --- |
| `.gitignore` | Ignore durable interview files in `backend/sessions/`. |
| `README.md` | Replace the memory-only CLI warning with checkpoint/resume instructions. |
| `backend/.env.example` | Document bounded retry settings and optional checkpoint directory. |
| `backend/OPENAI_CONFIGURATION.md` | Explain centralized retries and process-local usage totals. |
| `backend/PM_ARCHITECTURE_CURRENT.md` | Document separate coverage and durable CLI boundaries. |
| `backend/DISCOVERY_RECOVERY.md` | Record implementation, limitations and manual verification steps. |
| `backend/test_pm.py` | `new_session`, `choose_session`, `run_phase`, `run_session` and `run_cli` persist input, resume saved cursors and report failures. This is the interactive CLI. |
| `backend/agents/state.py` | Add session/cursor, coverage, acquisition, active-answer and known-evidence fields. |
| `backend/agents/discovery_coverage.py` | Add scoped coverage keys, answer receipts, acquisition records and grounded confirmation review. |
| `backend/agents/interview_planner.py` | `build_gap_info` uses deliberate coverage; the planner consumes answer receipts; `all_required_gaps_resolved` gates completion. |
| `backend/agents/knowledge_tracker.py` | Tighten category instructions, emit acquisition/answer receipts and propagate extraction failures. |
| `backend/agents/question_generator.py` | Use prior confirmed or inferred facts in confirmation/expansion questions. |
| `backend/agents/guardrails.py` | Allow informed completeness questions and propagate evaluation failures. |
| `backend/agents/llm.py` | Wrap synchronous/asynchronous central model calls in bounded provider-error retries. |
| `backend/agents/llm_errors.py` | Define safe call/extraction failures and their propagation helper. |
| `backend/agents/interview_checkpoint.py` | Implement atomic serialization/restoration, session locking and durable node wrappers. |
| `backend/agents/graph.py` | Resume at saved cursors, wrap successful nodes with checkpoints and guard compilation with coverage. |
| `backend/agents/absence_supersession.py` | Propagate call/extraction failures through the existing fallback. |
| `backend/agents/knowledge_corrections.py` | Propagate call/extraction failures through the existing fallback. |
| `backend/agents/knowledge_duplicates.py` | Propagate call/extraction failures through the existing fallback. |
| `backend/agents/prd_validation.py` | Propagate call/extraction failures from validation reviews. |
| `backend/agents/pm_agent.py` | Propagate call/extraction failures from compilation. |

## Manual verification next (not executed)

1. Start a new interview with role behaviour, a workflow and an approval policy in
   the initial idea. Inspect JSON: these facts should be INCIDENTAL with no resolved
   future gaps. Each later gap must ask informed confirmation/expansion.
2. Confirm an active known gap with 'yes, that is all', then expand another with
   a substantive answer. Only the asked gap should gain RESOLVED coverage; facts
   learned about other topics must remain available but must not resolve them.
3. Answer the additional-users question negatively. Check scoped absence and
   deliberately resolved secondary_users. Check that inapplicable secondary-user
   goals are waived rather than inventing placeholder actors or goals.
4. Try role occupancy, system recovery, and regulatory statements. Check that
   unrelated responsibility/permission/goal/workflow candidates are not persisted;
   legitimate cross-topic facts should remain in the knowledge store.
5. Submit an answer while offline. Check bounded retries, preserved pending answer,
   EXTRACTION_FAILED, unchanged gap coverage, and no compilation. Restore network
   and restart: select the session and process the answer without retyping it.
6. Interrupt after extraction (cursor plan/generate). Restart: verify extraction
   does not rerun, no duplicate facts are added, and the same gap continues.
7. Interrupt after question generation (cursor guardrail), then while waiting for
   input. Restart should respectively validate the saved question or display it
   without another generation call. Ctrl+C must not mark completion.
8. Use an invalid API credential temporarily: one attempt, clear failure, checkpoint
   retained. Restore the original credential without printing or committing it.
9. Supply incidental knowledge for every remaining key. Compilation must still be
   blocked until all required gaps have deliberate coverage. Finish the interview:
   verify the existing PRD output, then resume/decline the optional admin phase.
10. Open the same session from two CLI processes: the second must refuse the lock.
    Check that token/cost output still appears for successful calls and retries.

## Remaining limits

- The earlier crashed interview predates checkpoints; it cannot be reconstructed
  automatically from lost memory. These changes protect subsequent saved sessions.
- Semantic extraction/confirmation still depends on the configured model. No live
  semantic accuracy or runtime recovery claims have been verified in this task.
- A crash during an unfinished node may repeat that node's API calls on resume;
  local state commits are atomic, but remote calls are not exactly-once billing.
- JSON checkpoints are local, unencrypted interview data. The directory is ignored
  by Git; disk backups, retention, migrations and encryption remain future work.
- Usage session totals remain process-local and reset on restart; the durable
  interview ID and turn number persist.
- Existing offline fixtures that assumed knowledge presence implies completion
  need updated coverage/question receipts before being used as acceptance tests.
  No tests were run or silently rewritten to mask this intentional semantic change.
