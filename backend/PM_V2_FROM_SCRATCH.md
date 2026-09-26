# Architect PM v2 — clean adaptive discovery

Branch: `feature/pm-agent-v2-from-scratch`

This implementation is intentionally separate from the legacy discovery graph.
It reuses only the shared OpenAI client and usage callback infrastructure.

## Core rule

The engine keeps these decisions independent:

1. What did the founder say?
2. Is the proposed fact semantically supported by the founder's evidence?
3. What does the fact mean in the product model?
4. Is it new, duplicate, refinement, correction, or contradiction?
5. What implications follow?
6. Which requirements become relevant?
7. How much coverage/depth do those requirements have?
8. Which unresolved decision is worth asking now?
9. Is the proposed question actually high quality?
10. Is discovery complete enough to compile an implementation-ready PRD?

## Pipeline

```text
USER TURN
  -> turn interpretation / conversation control
  -> Python exact-source provenance
  -> LLM semantic grounding
  -> LLM semantic classification + canonical key
  -> batched LLM reconciliation
  -> confirmed fact memory
  -> implication engine (PROPOSED)
  -> dynamic requirement activation
  -> coverage + depth assessment
  -> contradiction detection
  -> completion assessment
  -> candidate questions
  -> question audit
  -> deterministic priority selection
  -> ONE founder-facing question
```

## Long-term memory

Raw transcript is not the primary memory mechanism.

The planner receives:
- all active confirmed facts,
- tentative/proposed facts separately,
- all active requirements,
- proposed implications,
- proposed recommendations,
- unresolved contradictions,
- persistent founder discovery boundaries,
- full semantic question history,
- only the most recent raw user turns for conversational tone/context.

This means a decision from many turns ago remains available even when its raw
message is no longer in the recent-conversation window.

## Grounding split

Python is responsible for:
- exact evidence existence in the current submitted turn,
- stable IDs,
- state mutation,
- persistence,
- deterministic ranking.

The LLM is responsible for:
- semantic entailment,
- normalization,
- classification,
- duplicate/refinement/correction/contradiction reasoning,
- requirement relevance,
- coverage/depth,
- question value and quality.

Python does not try to understand words using semantic regex rules.

## Knowledge authority

Founder assertions:
- CONFIRMED when stated as current truth,
- PROPOSED when explicitly tentative.

Derived implications:
- always begin PROPOSED.

PM recommendations:
- always begin PROPOSED.

Requirements:
- UNKNOWN when relevant but unresolved,
- PROPOSED when implication/recommendation-backed,
- CONFIRMED only when founder evidence resolves the atomic requirement,
- REJECTED / NOT_APPLICABLE / DEFERRED only from explicit founder direction.

## Question policy

The agent does not ask the next missing field.

It creates several candidates and scores them using:
- business impact,
- architecture impact,
- dependency unlock,
- uncertainty,
- risk,
- contextual relevance,
minus:
- repetition,
- premature detail,
- user fatigue.

A separate semantic audit rejects questions that are:
- already answered,
- paraphrased repeats,
- compound,
- overly technical,
- UI/design implementation trivia,
- premature,
- low value,
- blocked by founder feedback/deferral.

Only one independently answerable decision is asked.

## Persistent interview-control memory

Statements such as:
- "that is the designer's job",
- "you are asking irrelevant questions",
- "I already answered that",
- "I haven't decided",

become discovery boundaries rather than product facts.

Those boundaries persist across turns and are supplied to both candidate
generation and question audit.

## Cost behavior

All semantic comparisons are batched.

There is no per-fact `FACT_COMPARISON` loop. A long founder answer is reconciled
against existing knowledge in one structured reconciliation call.

## Running

From `backend`:

```powershell
python test_pm_v2.py
```

PM v2 sessions are stored separately under `backend/sessions_v2`.

When discovery completes, the CLI compiles a source-linked PRD and writes it to:

```text
backend/output_v2/<session-id>.json
```

The compiler is allowed to use only active CONFIRMED founder facts for confirmed
PRD statements. Proposed implications and recommendations stay explicitly
separate.
