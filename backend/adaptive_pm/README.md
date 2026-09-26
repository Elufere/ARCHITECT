# Adaptive PM v2

This package is a clean PM-discovery implementation built independently of the legacy `backend/agents` interview graph. It reuses only generic OpenAI infrastructure (`agents.llm`) for model access, retries, and per-call token/cost logging.

## Pipeline

```text
Founder message
  -> Capture intent + atomic facts
  -> Python source/provenance validation
  -> LLM semantic grounding
  -> Canonicalization + classification
  -> Dynamic requirement activation
  -> Coverage + depth + dependency reasoning
  -> Proposed implications
  -> Contradiction detection
  -> Candidate question generation
  -> Independent prioritization
  -> Final semantic question audit
  -> Ask exactly one high-value product decision
```

Capture, semantic grounding, canonicalization/classification, requirement reasoning, coverage/depth, implications, contradiction detection, candidate generation, prioritization, and question wording are intentionally separate decisions.

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

The planner therefore does not depend on a short sliding chat window to remember whether an old decision was already answered. Grounded observations that have not yet been canonicalized are also retained as durable evidence, so a classification miss does not make the founder repeat themselves.

## Grounding split

Python verifies provenance only: the evidence quote must exist in the current founder message. The grounding LLM decides whether that evidence semantically supports the extracted fact.

A second invariant protects canonical state: a confirmed knowledge mutation is discarded unless it cites a grounded observation.

## Question policy

The planner creates a small candidate set and scores information value against repetition, premature detail, and fatigue. Python orders eligible candidates. A separate semantic audit rejects or rewrites questions that are bundled, repetitive, too technical, implementation-shaped, or inconsistent with founder boundaries.

Decision history is persistent and semantic. An answered/deferred/rejected decision key is not eligible again just because the wording changes.

## Model-call shape

Semantic work is batched by responsibility rather than by individual fact:

1. fact capture;
2. semantic grounding;
3. canonicalization/classification;
4. dynamic requirements + coverage/depth/dependencies;
5. implications;
6. contradiction detection when canonical knowledge changed;
7. candidate generation;
8. prioritization when multiple candidates survive;
9. final question audit.

This intentionally uses more distinct reasoning stages than the legacy agent because the specification requires those decisions to remain independent. There is still no per-fact LLM comparison loop.

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

## Package layout

- `capture_stage.py`: intent, fact capture, provenance and semantic grounding.
- `knowledge_stage.py`: canonical product knowledge and concept classification.
- `requirement_stage.py`: requirements, coverage/depth, dependencies, implications and contradictions as separate model calls.
- `question_stage.py`: candidate generation, independent prioritization and final audit.
- `engine.py`: orchestration and durable state transitions only.
- `models.py`: explicit state and stage contracts.
- `stage_prompts.py`: independent semantic contracts.
