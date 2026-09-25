# Architect — Model-Driven Discovery (Experimental Branch)

Branch: `feature/model-driven-discovery`

This branch replaces schema traversal as the interview control system. The
existing discovery fields remain available as extraction/normalization anchors
and for PRD compilation compatibility, but missing schema fields no longer force
questions or block completion.

## Control flow

```text
USER ANSWER
    ↓
CAPTURE NEUTRAL CLAIMS ONCE
    ↓
NORMALIZE REFERENCES / ACTORS
    ↓
DETERMINISTIC SEMANTIC ADMISSION
    ↓
FOCUSED SEMANTIC REVIEW ONLY WHERE REQUIRED
    ↓
RECONCILE WITH EXISTING KNOWLEDGE
    ↓
UPDATE CONFIRMED FACTS
    ↓
UPDATE PRODUCT MODEL
    ↓
INFER IMPLICATIONS
    ↓
ACTIVATE / DEACTIVATE RELEVANT REQUIREMENTS
    ↓
ASSESS REQUIREMENT COVERAGE / DEPENDENCIES
    ↓
VALIDATE CONSISTENCY
    ↓
IDENTIFY UNCERTAINTIES / DECISIONS
    ↓
BUILD QUESTION CANDIDATES
    ↓
FILTER ELIGIBLE CANDIDATES
    ↓
PRIORITIZE
    ↓
SELECT NEXT ACTION
    ├── ASK ONE QUESTION
    └── COMPILE
```

## Runtime state cleanup

`DiscoveryTopic` remains only as a semantic taxonomy for facts and compilation.
The live interview no longer carries `topic_status` or `topic_maturity`, does
not mark topics COMPLETE/PARTIAL, and does not reopen topics when knowledge
changes. The former `topic_lifecycle.py` invalidation engine has been removed.

New knowledge changes the derived product model and inquiry frontier directly.
For example, discovering a new `vendor` participant creates an inquiry about
that participant's actions; it does not reopen USER_ROLES or USER_GOALS.

## Layer boundaries

### Confirmed facts

`KnowledgeItem` remains the evidence ledger. A confirmed fact must be grounded
in user evidence. Reconciliation still handles duplicates, refinements,
corrections, contradictions, absence replacement, and supersession.

A grounded fact is not automatically an interview-completion signal.

### Product model

`product_model.py` remains a derived view of confirmed facts. The interview no
longer walks the model by `DiscoveryTopic` order.

The foundational inquiry frontier currently asks only for enough structure to
make the product coherent:

1. core actor(s)
2. what each primary actor actually does
3. the outcome each primary actor is trying to achieve
4. the normal core workflow
5. what makes that workflow successfully complete

These are not a 38-field checklist. Once this minimum model is coherent,
product-specific requirements and contradictions determine depth.

Incidental cross-category extraction does not automatically satisfy the next
foundational decision. Direct answers and facts volunteered before an active
inquiry do. This prevents a responsibility answer that is also misclassified as
a goal from silently skipping goal discovery.

### Implications

`agents/implications.py` records derived relevance separately from confirmed
product truth.

An implication contains:
- the deterministic rule that fired
- the confirmed fact IDs that justified it
- the requirement(s) made relevant

Implications never become confirmed facts.

### Requirements

Existing active requirements remain the product-specific depth mechanism.
Activation is recomputed from the current confirmed model, so requirements can
activate or deactivate as facts are corrected.

A negative role-transition policy such as "roles cannot change within a
transaction" no longer activates the role-transition lifecycle requirement.
Positive transition behavior still can.

### Inquiries

`agents/inquiries.py` introduces first-class `ProductInquiry` objects.

Inquiry sources:
- `MODEL` — foundational model uncertainty
- `REQUIREMENT` — unresolved active requirement facets
- `VALIDATION` — blocking contradiction requiring founder clarification

An unknown can exist without forcing a question. Only open inquiries are turned
into candidates.

### Candidate filtering and priority

Question candidates now carry their inquiry identity and source.

Priority considers:
- contradiction pressure
- decision impact
- dependency unlock value
- uncertainty
- architecture impact
- business risk
- context relevance

Penalties include:
- question breadth/cost
- repetition
- topic fatigue
- premature depth

Requirement depth is penalized while foundational model inquiries remain open,
so Architect first understands the product broadly enough to reason safely.

### Planner

`interview_planner.py` no longer falls back to the schema planner.

It selects the highest-ranked inquiry candidate. If no material inquiry remains,
no active requirement remains, and consistency is clean, discovery is considered
ready to compile even when legacy schema fields are still uncovered.

`all_required_gaps_resolved()` remains only as a legacy diagnostic.

## Capture architecture

Production discovery no longer runs six category-specific extraction calls over
the same answer. It performs one response-wide `knowledge_tracker.CLAIMS` call
that captures explicit propositions and assigns each exactly one semantic kind.

The capture layer deliberately separates:
- proposition capture — what the founder actually asserted;
- reference normalization — which canonical actor a label/pronoun refers to;
- semantic admission — whether that proposition is allowed into its proposed fact type;
- reconciliation — whether the admitted fact is new, duplicate, refinement, correction, or contradiction.

The legacy six-pass implementation remains only as a deterministic test/
compatibility fallback when a mocked model registry does not expose `CLAIMS`.
It is not the production runtime path.

Admitted neutral claims do not go through the old broad cross-category grounding
audit again. Ambiguous propositions must remain `unclassified` and fail closed
instead of being forced into a field. High-impact semantics retain focused
authority boundaries: active-gap absence still uses the dedicated absence
interpreter; absence supersession/corrections keep their review path; actor
membership has deterministic current-surface admission.

Examples of enforced boundaries:
- product benefit is not actor responsibility;
- capability is not permission without authorization semantics;
- desired future outcome is not an established workflow end state;
- the first narrated workflow action is not automatically the trigger;
- an event after successful completion is not automatically a definition of success;
- process participation alone does not make a later participant an app actor.

## What remains intentionally unchanged

The branch still uses:
- existing `KnowledgeItem` / discovery-field storage contracts for compatibility;
- evidence span validation;
- fact comparison/correction and supersession machinery;
- requirement facet coverage and dependency resolution;
- contradiction validation;
- question guardrails;
- checkpoint durability;
- PRD compilation and verification.

## Experimental invariants

The branch adds tests asserting that:

- discovery starts with one core-actor uncertainty rather than the full schema;
- the minimal model frontier progresses actor → actions → outcome → workflow → completion;
- uncovered legacy schema fields do not block completion when no material inquiry remains;
- a hard prohibition on role switching does not activate role-transition depth;
- a positive role-transition policy still can activate that requirement.

## Review goal

The live review should focus on whether the interview now feels like a PM
following the product rather than a form following fields.

The architecture should be rejected if it:
- re-asks information already materially established;
- deep-dives a low-value branch while the core product model is still vague;
- treats incidental classification as proof that a founder decision was covered;
- asks how an impossible behavior works after the founder has ruled it out;
- stops while an active high-impact requirement or blocking contradiction remains.
