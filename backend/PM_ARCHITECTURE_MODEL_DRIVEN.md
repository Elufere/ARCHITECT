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
EXTRACT CANDIDATE CLAIMS
    ↓
GROUND / NORMALIZE / CLASSIFY
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

## What remains intentionally unchanged for now

This is an architecture migration, not a total rewrite.

The branch still uses:
- the existing six extraction passes
- existing Pydantic discovery-field contracts
- grounding and evidence validation
- fact comparison/correction machinery
- requirement facet coverage
- requirement dependency resolution
- contradiction validation
- question guardrails
- checkpoint durability
- PRD compilation and verification

Those layers can be migrated independently after the interview-control model is
validated in live runs.

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
