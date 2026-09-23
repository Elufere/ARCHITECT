The PM compiler now consumes a confirmed fact snapshot for the active discovery scope. It receives no conversation history, raw product idea, inferred facts, other-scope facts, or static engineering requirements. This prevents compilation from independently reinterpreting the original conversation or promoting suggestions into confirmed requirements.

Each source fact retains its category (topic/key), actor fields, full value, evidence, original question when available, confidence, scope, and turn. Conditions remain verbatim in the value/evidence; generated claims also expose actors and conditions explicitly. Fact IDs are deterministic hashes of source content and provenance. Recompiling the same facts preserves their IDs; correcting a fact changes its ID.

Compilation produces a PRDDraft using parsed structured output, then runs these checks before publication:

1. Every factual claim has nonempty source_fact_ids; IDs exist in the active snapshot and are not duplicated.
2. Each declared canonical category is present among its cited source categories. Approval sources cannot be used as visibility sources merely by referencing a valid ID.
3. Requirement IDs are unique and every snapshot fact is referenced somewhere. A repair cannot silently omit an established rule.
4. Independent classification checks each source quote without seeing its extracted value, category or confidence, and each generated claim without seeing its declared category or citations. The observed meanings must support the source category and the generated claim category, without unsupported extra meanings. This prevents an incomplete quote from borrowing missing meaning from the fact value, and a visibility restriction from borrowing an approval label. Valid overlapping views such as actor responsibilities and ordered workflow steps remain allowed.
5. A separate semantic audit then examines each claim, its cited facts, source evidence, and other confirmed facts for conflicts. It checks source evidence, claim meaning, category, actor ownership, conditions/thresholds, acceptance criteria, and contradictions. All checks must explicitly pass for the exact claim ID.
6. Missing/malformed verdicts, verifier outages, over-budget inputs, and negative verdicts prevent saving. Unsupported drafts get at most one regeneration using the same frozen sources and rejection feedback. Verifier infrastructure failures stop immediately.
7. Only a fully verified artifact is serialized and atomically replaces the scope's output file. On rejection or write failure the previous file stays intact; state has pm_is_complete=false, prd_contract=null, and compilation_errors explaining the failure.

The semantic verifier is a model-based safeguard, not a formal proof of entailment. The structural checks are deterministic. Confidence scores and valid citations do not substitute for semantic verification.

Compilation and verification default to the installed qwen2.5:7b model. Set PM_COMPILER_MODEL and PM_VERIFIER_MODEL to override them. Source/claim classification and final auditing are separate calls with different inputs; they are not necessarily different models. Classification results are cached within one compilation/repair sequence. Validation adds sequential model calls proportional to distinct source quotes and generated claims.

The output is schema_version 2.0. This changes the JSON contract: summaries, scope entries, non-functional constraints, deferred items, and persona behaviors are sourced objects rather than bare strings. Each has text, category, source_fact_ids, actor_ids, and conditions. Functional requirements retain id, description, and validation, adding the provenance fields. product_name is a sourced object or null if unknown; elevator_pitch is a list of sourced statements. Open questions remain questions, not accepted requirements. The application appends discovery_scope, source_facts, and validation_report; the compiler cannot write its own ledger or approve itself.

Example requirement (the ID is illustrative):

```json
{
  "id": "FR-01",
  "description": "Orders over $100 require manager approval.",
  "category": "BUSINESS_RULES.approval_rules",
  "source_fact_ids": ["fact_<content hash>"],
  "actor_ids": ["manager"],
  "conditions": ["order total > $100"],
  "validation": "TBD"
}
```

USER_APP saves to output/requirements_mvp.json. ADMIN_DASHBOARD saves to output/requirements_admin_dashboard.json, so compiling the second scope does not overwrite the first. Source snapshots and verdicts are embedded in each file.

Newly accepted discovery facts retain source_question so short evidence such as “no” can be interpreted without sending raw history to the compiler. Older facts lacking adequate evidence/context may fail compilation validation; the verifier must not manufacture missing support. Very large snapshots are explicitly blocked by a conservative context-budget check instead of silently truncating facts. Compilation has not yet been chunked for arbitrarily large interviews.

Implementation: agents/pm_agent.py, agents/prd_schema.py, agents/prd_validation.py. Deterministic regressions: tests/test_prd_compilation.py. Opt-in local-model verification: python tests/live_prd_validation.py (writes only a test report, not a PRD).
