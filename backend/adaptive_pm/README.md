# Adaptive PM v2

This package is a clean PM-discovery implementation built independently of the legacy `backend/agents` interview graph. It reuses only generic OpenAI infrastructure (`agents.llm`) for model access, retries, and per-call token/cost logging.

## Pipeline

```text
Founder message
  -> Capture intent + atomic facts
  -> Python source/provenance validation
  -> LLM semantic grounding (batched)
  -> Canonical product knowledge update
  -> Dynamic requirement activation + coverage/depth
  -> Dependency propagation
  -> Proposed implications
  -> Contradiction detection
  -> Candidate question generation
  -> Semantic prioritization
  -> Final question audit
  -> Ask exactly one high-value product decision
```

Capture, grounding, canonicalization, requirement reasoning, depth, question value, and wording are intentionally separate decisions.

## State model

Durable state keeps:

- exact grounded user observations;
- canonical knowledge records with stable semantic keys;
- superseded knowledge history;
- product concepts;
- dynamically activated requirements with separate coverage/depth;
- dependency links;
- proposed implications (never auto-confirmed);
- contradictions;
- persistent discovery boundaries/deferrals;
- persistent semantic decision history.

The planner therefore does not depend on a short sliding chat window to remember whether an old decision was already answered.

## Grounding split

Python verifies provenance only: the evidence quote must exist in the current founder message. The grounding LLM decides whether that evidence semantically supports the extracted fact.

A second invariant protects canonical state: a confirmed knowledge mutation is discarded unless it cites a grounded observation.

## Question policy

The planner creates a small candidate set and scores information value against repetition, premature detail, and fatigue. Python orders eligible candidates. A separate semantic audit rejects or rewrites questions that are bundled, repetitive, too technical, implementation-shaped, or inconsistent with founder boundaries.

Decision history is persistent and semantic. An answered/deferred/rejected decision key is not eligible again just because the wording changes.

## Cost shape

The normal product-information turn is designed around batched calls:

1. capture;
2. semantic grounding;
3. product/requirement reasoning;
4. question planning;
5. final question audit.

There is no per-fact LLM comparison loop.

## Run

From `backend/`:

```bash
python -m adaptive_pm.cli
```

or:

```bash
python test_pm_v2.py
```

Environment configuration continues to use `backend/.env` and the existing `OPENAI_*` settings.

Sessions are stored under `backend/adaptive_pm_sessions/` by default. Set `ADAPTIVE_PM_SESSION_DIR` to override this location.
