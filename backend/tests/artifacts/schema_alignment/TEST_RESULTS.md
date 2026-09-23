# Discovery alignment and surgical regression test results

## Baseline and current results

- Frozen pre-alignment discovery tests: **55 passed, 17 failed, 1 collection error**.
- After alignment, focused tests: **115 passed**. The live replay nevertheless
  exposed semantic regressions; deterministic model fixtures do not prove LLM behavior.
- After surgical regression fixes, focused run: **200 passed**.
- Broad backend run after surgical fixes: **219 passed, 12 failed, 3 collection errors**.
- Every remaining discovery failure also appears in the frozen baseline. No newly
  failing runnable production regression was found by the broad comparison.
- One new test initially used pytest's reserved parameter name `request`; this was
  a REAL REGRESSION in the new test's collection, fixed by renaming it `utterance`.

The earlier live semantic regressions (false none, action-as-goal, invented workflow
completion) are REAL REGRESSIONS despite the initial fixture-based suite passing.
Their before-fix artifacts are retained as `careconnect_before_regression_fix.*`.
The final live artifacts record actual model behavior, not stubbed responses.
Final live acceptance: **PASSED**. The clean-state CareConnect conversation stored six supported initial facts; both responsibilities were satisfied, clarification retained both actors, and No persisted secondary_users=none and advanced to permissions::patient. Both injected unsupported claims were rejected. See careconnect_replay.json (passed=true), careconnect_calls.json and careconnect_trace.txt. Earlier failed artifacts are preserved and are not the final outcome.

## Remaining failures, individually classified

| Test/module | Classification | Reason |
| --- | --- | --- |
| `backend.test_pm` | UNRELATED COLLECTION/ENVIRONMENT ERROR | Declared chromadb dependency is unavailable in the isolated test runtime. |
| `backend.test_scrapper` | UNRELATED COLLECTION/ENVIRONMENT ERROR | Declared BeautifulSoup (bs4) dependency is unavailable in the isolated test runtime. |
| `backend.tests.test_extraction_semantics` | UNRELATED COLLECTION/ENVIRONMENT ERROR | Legacy test imports prior_gap_recovery, a module absent before this change. |
| `test_live_failure_repaired_by_model_before_storage` | OUTDATED TEST EXPECTATION | Expects the retired ExtractedKnowledge/extraction_llm monolithic API; all six fail identically in the frozen baseline. |
| `test_second_invalid_batch_stops_without_salvaging_unsupported_sibling` | OUTDATED TEST EXPECTATION | Expects the retired ExtractedKnowledge/extraction_llm monolithic API; all six fail identically in the frozen baseline. |
| `test_valid_output_does_not_retry` | OUTDATED TEST EXPECTATION | Expects the retired ExtractedKnowledge/extraction_llm monolithic API; all six fail identically in the frozen baseline. |
| `test_provider_failure_does_not_retry` | OUTDATED TEST EXPECTATION | Expects the retired ExtractedKnowledge/extraction_llm monolithic API; all six fail identically in the frozen baseline. |
| `test_repaired_output_still_requires_exact_evidence` | OUTDATED TEST EXPECTATION | Expects the retired ExtractedKnowledge/extraction_llm monolithic API; all six fail identically in the frozen baseline. |
| `test_roles_is_required_in_provider_schema` | OUTDATED TEST EXPECTATION | Expects the retired ExtractedKnowledge/extraction_llm monolithic API; all six fail identically in the frozen baseline. |
| `test_evidence_rejects_ungrounded_value` | OUTDATED TEST EXPECTATION | Expects source-quote structural validation to perform semantic entailment. That responsibility belongs to the separate grounding stage, covered by current tests. |
| `test_coherent_topic_is_not_completed_while_schema_gaps_remain` | OUTDATED TEST EXPECTATION | Expects permissions::guests despite missing secondary_users and preserved source actor order. Same baseline failure. |
| `test_permission_remains_a_gap_after_abstract_responsibility` | OUTDATED TEST EXPECTATION | Expects permissions as the first gap although secondary_users is still unknown. The permission gap does remain missing; same baseline failure. |
| `test_compound_secondary_role_does_not_become_a_combined_planner_gap` | PRE-EXISTING FAILURE | Persisted KnowledgeItem rejects a four-word compound role before the legacy splitting helper runs; reproduced in frozen baseline. Outside this surgical fix. |
| `test_misclassified_known_role_answer_is_repaired_to_atomic_active_gap` | OUTDATED TEST EXPECTATION | Expects the old validator to rewrite an actor declaration into responsibilities based on current_gap. Semantic extraction now owns classification; same baseline failure. |
| `test_direct_role_scoped_goal_is_bound_to_active_role` | OUTDATED TEST EXPECTATION | Expects the old validator to invent an omitted owner from the active gap. Current role goals require an extracted owner; same baseline failure. |

## Tests intentionally updated

- CareConnect fixtures now require both patient and provider responsibilities and
  retain shared patient workflow evidence. Canonical IDs use underscores as before.
- The actor-pass failure test now expects an unregistered owner to be rejected,
  matching the confirmed-actor contract instead of accepting an orphaned fact.
- Goal-classification tests no longer ask a structural-only helper to perform a
  semantic success-criterion check. Grounding tests cover the unsupported value.
- Absence fixtures explicitly provide the new separate confirmed-absence verdict;
  ordinary supported IDs alone cannot admit a whole-field none.
- Prompt-shape assertions now require source text in its own HumanMessage and no
  duplicate workflow/action candidates in the goal instructions. The outcome and
  overlap assertions remain. Temporary failures of those old prompt-shape checks
  were OUTDATED TEST EXPECTATIONS, not evidence that source text was omitted.
- Old monolithic API tests and unrelated compatibility tests were not rewritten,
  skipped, or made green by restoring retired architecture.

## Commands

```powershell
python -m pytest tests/test_alignment_regressions.py tests/test_schema_alignment.py tests/test_gap_absence.py tests/test_extraction_passes.py tests/test_focused_extraction_restoration.py -q -p no:cacheprovider
python -m pytest . --continue-on-collection-errors -q -p no:cacheprovider
python -u tests/replay_schema_alignment.py
```

Commands above run from `backend` in an environment with its dependencies installed.
The recorded runs used an isolated Python 3.12 dependency directory because the
existing workspace virtual environment could not be launched by the sandbox.
No source change was made to hide the missing declared backend dependencies. The production imports also require langchain-ollama, which is absent from the existing requirements.txt; packaging was not changed in this surgical repair.
