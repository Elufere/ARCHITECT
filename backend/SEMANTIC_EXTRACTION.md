> Historical design notes for the retired monolithic extractor. The current
> six-pass pipeline uses [shared field definitions](agents/discovery_fields.py)
> and [the schema-alignment audit](DISCOVERY_SCHEMA_AUDIT.md). In particular,
> responsibilities now include explicit role actions/capabilities, and grounding
> permits overlapping evidence. Sections below describe earlier designs.

# Semantic extraction boundary

## Before this patch

Production path: graph conversation routing -> `knowledge_tracker_node` ->
`ExtractedKnowledge` / Ollama structured output -> role expansion -> confidence
and evidence validation -> role-specific value projection -> state overwrite ->
deduplication/storage -> `interview_planner_node` / `build_gap_info` -> recovery.

The LLM supplied topic, key, value, role/roles, quote, confidence and optionally
state. `KnowledgeItem` defaulted missing state to CONFIRMED. Python then made
these additional semantic decisions:

| Location before patch | Responsibility violation |
| --- | --- |
| `knowledge_tracker_node`, state assignment | CONFIRMED only for an exact active-gap answer or opening actor discovery; explicit incidental facts became INFERRED. |
| `validate_extraction`, active-gap binding | Assigned missing owners and replaced different owners when role words occurred in evidence. |
| `validate_extraction`, lexical grounding | Stemmed/fuzzy-matched value words against the whole message, including hardcoded admin synonyms. This neither proved clause ownership nor handled genuine inferences/paraphrases. |
| `validate_extraction`, generic/admin rules | Blocked `admin` values and any primary actor record containing an admin keyword. An aggregate record could lose all its actors. |
| `expand_role_scoped_item` and its unrestricted caller | Copied any item, including actor declarations, to roles mentioned anywhere in evidence; retained the original roles list/value in every copy. |
| `role_scoped_value` in extraction, recovery and review | Reconstructed an owner's value by splitting clauses. |
| `split_role_labels` / `role_identity` | Split `/`, `and`, commas and `&`; removed plural suffixes and hardcoded administrator/admin equivalence. |
| `knowledge_tracker_node`, negative fast path | Recognized phrases with regex/substrings and created a CONFIRMED `None specified` answer for the active key. |
| `knowledge_tracker_node`, correction/promotion merge | Could delete reviewed candidates before replacement validation, or delete unrelated inferred propositions sharing a key/owner. |
| `recover_prior_gap` | Filled missing owners from source records and unconditionally changed recovered state to INFERRED. |
| `get_known_keys` | Interpreted `only`/`no other` in text as completion of secondary-user discovery. |
| Planner role helpers | Reconstructed actor labels from comma-separated values when roles were absent. |

An accepted `primary_users` item under `secondary_users` necessarily took the old
INFERRED branch. Under `primary_users` the same item necessarily became CONFIRMED.
The incident's final item alone cannot establish the original gap or extraction
payload. The old code also cannot give different states to two same-key actor
items accepted in the same turn solely because their labels differ.

## Current contract

`ExtractedItem` extends the backwards-compatible stored `KnowledgeItem`.
New model output must explicitly supply `knowledge_state` and `assertion_type`.
The latter is positive, negative, hypothetical, uncertain or conditional.
CONFIRMED describes provenance, not current existence: an explicit denial is a
CONFIRMED negative proposition. A future possibility can be explicitly stated
without establishing an existing actor.

Two persisted additions are sufficient:

- `assertion_type`: optional only for loading old records; required on extraction.
- `aliases`: canonical actor ID -> explicitly supplied alternate labels.

There is no additional canonical_role field. Actor `roles` already stores canonical
IDs; an atomic actor declaration has exactly one ID, `role=null`. Explicit absence
of all additional actors has `roles=[]` and assertion_type=negative. A denial of a
specific actor names that actor in roles. Owned facts use `role`, with no roles or
alias declarations. Source quotes remain unchanged.

Example provider declaration:

```json
{
  "topic": "USER_ROLES",
  "scope": "USER_APP",
  "key": "primary_users",
  "roles": ["service_provider"],
  "role": null,
  "aliases": {"service_provider": ["service providers", "artisan", "artisans"]},
  "value": "Service providers offer the services",
  "evidence": "service providers/artisans who offer the services",
  "knowledge_state": "CONFIRMED",
  "assertion_type": "positive",
  "confidence": 0.99
}
```

Meaning, ownership, aliases, polarity and provenance now belong to the LLM.
The prompt includes recent conversation and persisted records so pronouns and
previous canonical identities can be resolved there. It explicitly covers both
alias slashes and distinct actor slashes without a global punctuation rule.

Python validates schema/key combinations, bounds, exact source quotes, actor/owner
shape, syntactic references and alias conflicts. It rejects unknown owners,
normalizes case/whitespace and applies supplied alias relationships. It stamps the
current scope and turn, merges and deduplicates, then plans from positive confirmed
actors. Conditional permission rules can satisfy permission gaps; hypothetical or
uncertain facts do not complete current discovery. Negative actor declarations
never create actor gaps. Explicit positive/negative updates supersede opposite
polarity for the same actor slot, with a logged reason.

The tracker no longer expands actor records, repairs owners, projects values or
assigns extracted states. Recovery uses the same required semantic contract and
preserves classification. Confirmed recovery updates the product model and causes
the planner to recompute the remaining gap.

### Invalid structured-output recovery

The live FixMate opening exposed a parser-boundary failure: the model emitted
positive actor declarations with `roles=[]`. Pydantic correctly rejected these,
but the exception occurred before the tracker's per-item validation. The entire
batch was lost. That batch also contained an unsupported `multiple_roles=true`
claim, so blindly salvaging otherwise well-shaped siblings would be unsafe.

`extract_with_repair` now makes at most one additional model call after a structured
parsing/validation error. It supplies the original source and diagnostic data and
requires the LLM to reassess the entire batch, including support for siblings not
named by the validation errors. It does not infer role IDs, split labels, or alter
semantic states in Python. Corrected output still passes the normal evidence and
reference checks. A second failure is rejected; provider timeouts are not retried.

Field descriptions now explain actor IDs, owners and alias references directly in
the JSON schema. The prompt explicitly distinguishes multiple actor types from one
account being allowed to hold multiple roles. This is semantic guidance, not a
Python rule interpreting the user's sentence. `tests/test_extraction_repair.py`
replays the reported malformed batch through LangChain's actual Pydantic parser.
Extraction explicitly allocates an 8192-token model context. The inspected local
Ollama process was using 4096; the initial prompt measured 13,927 characters plus
a 3,489-character schema, with additional diagnostics on repair. The larger context
provides headroom; these character counts alone do not prove a particular call
was truncated. Long histories can still require a future prompt-budget policy.
Repair feedback is sent as an explicit user turn after the original system prompt:
the inspected local Qwen template renders user turns but has no message-loop branch
for system roles. Sending feedback as a second system message cannot be relied upon.
The extraction schema also requires an explicit roles field on every item (null for
non-actor facts), so schema-constrained generation cannot silently omit that field.

Live verification on the installed qwen2.5:7b: replaying the reported invalid batch
through the actual parser, then using the real model for the single repair call,
produced CONFIRMED `users` and `service_provider` actor records. The model supplied
the provider aliases and omitted its unsupported multiple_roles claim. The planner
then produced responsibilities and permissions gaps for both actors, leaving
multiple_roles open. This verifies live repair of this failure, not exhaustive
semantic accuracy or completeness of the model's extraction. The offline suite
after this follow-up passed 123 tests.

### Grounding protection and its limit

Schema validation and exact quote membership are deterministic protections, not a
proof of semantic entailment. An unrelated fabricated quote fails. A hallucinated
claim attached to a real quote cannot always be detected deterministically without
reintroducing a semantic parser. The LLM contract must supply genuinely supported
facts; live model evaluations are still needed to measure adherence. This patch
does not add a second model verification call or claim that structured output
eliminates semantic hallucinations.

## Migration

Existing serialized objects continue to load with assertion_type=None and empty
aliases. Their original state is preserved. Legacy positive actor behavior remains
compatible when explicit roles are available; the planner no longer derives roles
from value text. Legacy slash labels are not mechanically split. Other ambiguous
compound labels cannot be diagnosed reliably without their sources.

No persisted files were rewritten or automatically promoted. Re-extract affected
sessions from original human messages using the new contract, review the result,
then replace the affected actor/owned-fact records together. Do not merely convert
every old INFERRED item to CONFIRMED or merge all historical artisan/provider roles.
Previously separate actor IDs that conflict with a proposed alias cause rejection
instead of an automatic migration. New malformed extraction is rejected even when
an old stored object with a similar shape could still load.

## Provenance and FixMate trace

Enable INFO logging for `agents.knowledge_tracker` and `agents.prior_gap_recovery`.
The following was executed through the production tracker and gap builder using
mocked structured LLM output; it is not a live model accuracy result.

Active gap: secondary_users. User explicitly declares customers, providers/artisans
and admins. For each of customers, service_provider and admin:

```text
Before: parsed/default CONFIRMED -> validation PASS -> gap mismatch -> INFERRED
After:  LLM extracted state=CONFIRMED assertion=positive
        validation=PASS
        state after normalization=CONFIRMED
        state after merge=CONFIRMED
```

Actual resulting role gaps:

```text
responsibilities::customers
responsibilities::service_provider
responsibilities::admin
permissions::customers
permissions::service_provider
permissions::admin
```

There are no artisan discovery gaps. The old admin rejection and alias expansion
were separate defects, so the before line illustrates the customer state path,
not a claim that the old full actor batch would have passed validation.

## Regression coverage

`tests/test_extraction_semantics.py` drives schema parsing, tracker storage and
production planning with deterministic model responses. It covers all nine requested
scenarios plus malformed output, missing provenance, legacy loading, alias conflicts,
unknown owners, repeat extraction, explicit negative updates, failed corrections,
recovery state and stage logs. Existing tests now supply model-resolved actor records
and assert that Python preserves semantic decisions rather than inventing repairs.

Offline suite: `python -m pytest backend/tests -q -p no:cacheprovider` from the repo
root with backend on PYTHONPATH and the required Python packages installed.

## Remaining decisions worth revisiting separately

- `conversation_manager.classify_turn` still uses regex for conversational intent,
  corrections and uncertainty. An uncertainty-labelled turn can bypass extraction;
  mixed-content routing should eventually be model-driven or preserve product content.
- Exact affirmative replies in the existing confirmation-review protocol still
  promote presented inferences. This is now a narrow, logged user-confirmation
  transition; it does not depend on whether incidental extraction matches a gap.
- Correction flags still authorize replacing the same schema slot. Fine-grained
  proposition retractions/additions would need a separate explicit update contract.
- Recovery retains the existing invariant against silently moving confirmed actors
  between primary and secondary classifications.
- Planner internal-role ordering and question guardrails retain role-name and text
  heuristics. They do not rewrite stored extraction state, ownership or aliases.
- `role_evidence.py` remains as an isolated legacy helper; extraction, recovery and
  review no longer call its clause projection. Legacy mixed-owner values need review
  or re-extraction, not automatic text repair.
