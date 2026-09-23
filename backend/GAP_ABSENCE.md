# Confirmed gap absence

The focused extraction passes still collect explicit positive facts. Two additional
structured decisions run in `knowledge_tracker_node`:

1. `GAP_ANSWER` interprets the latest answer using the active topic, exact gap,
   scope, last assistant question, and existing facts for that field. It returns
   `none`, `not_applicable`, or `unresolved`. Python binds a resolved answer to the
   validated gap; the model cannot choose a different field or owner.
2. `GROUNDING` independently checks all candidates against the latest response,
   including semantic category, value, owner, and polarity. A verbatim quote is
   necessary but does not itself prove the claim. Unsupported candidates are
   dropped. A failed audit stores no new candidates.

Absence is persisted as a normal confirmed `KnowledgeItem`, with `absence="none"`
and `value="none"`, or `absence="not_applicable"` and `value="not applicable"`.
Actor absence uses `roles=[]`; role-specific absence retains the exact gap owner.
Evidence, confidence, scope, topic, and source turn are retained. The optional
absence field defaults to null when reading older records.

The normal merge replaces prior values for that same field/owner with confirmed
absence. Later explicit positive facts replace absence. Other scopes and owners
remain intact. Planner completion uses stored confirmed keys, with no `only` or
`no other` substring heuristic. An empty actor list never creates a pseudo-role.

Uncertainty, objections, denial of a single example, and answers with exceptions
must remain unresolved. The handler does not mechanically interpret `No` as
absence. Mixed responses still go through all extraction passes so independent
new information can survive the audit.

These decisions use the configured local `qwen2.5:7b` model and add up to two model
calls per turn. They improve grounding but remain model judgments. Deterministic
tests in `tests/test_gap_absence.py` exercise validation, persistence, and planner
handoff; live model checks are needed to assess semantic classification quality.
