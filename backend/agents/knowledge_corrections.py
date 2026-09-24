"""Target explicit supersession to prior confirmed facts, never a whole field."""
from pydantic import BaseModel, ConfigDict, Field

from agents.llm_errors import raise_if_llm_failure
from agents.extraction_passes import absence_label, canonical_role
from agents.state import KnowledgeState


class CorrectionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    superseded_ids: list[int]
    confidence: float = Field(ge=0, le=1)


CORRECTION_INSTRUCTION = """Identify prior facts explicitly superseded by this
grounded candidate's own source evidence. Return only supplied superseded_ids.
An intent flag or the word 'correction' alone cannot replace an entire field.
Require explicit correction, replacement, narrowing, or a clearly incompatible
new decision about the SAME assertion. No special keyword is required, but
ambiguity, normal additional detail, different conditions, and independent rules
are not corrections. A prohibition after one stage need not contradict permission
before another stage; replace only if the user explicitly supersedes the old rule
or their conditions actually conflict. Do not infer conflict from similar wording.
For actors, adding another user type does not remove existing users. Replace an
actor declaration only if its membership, classification or exclusive assertion
is explicitly superseded. Preserve unrelated actors and independent rules.
Do not use domain knowledge or question wording to manufacture conflicting evidence.
Return an empty list if no specific prior assertion is clearly superseded.
"""


def correction_targets(candidate, prior, response, decide):
    if (candidate.knowledge_state != KnowledgeState.CONFIRMED
            or not candidate.evidence.strip() or candidate.evidence not in response):
        return []
    previous = [old for old in prior
                if old.knowledge_state == KnowledgeState.CONFIRMED
                and (old.scope, old.topic, old.key) == (candidate.scope, candidate.topic, candidate.key)
                and canonical_role(old.role or "") == canonical_role(candidate.role or "")
                and not (old.absence or absence_label(old.value))]
    if not previous:
        return []
    try:
        result = decide("CORRECTION_REVIEW", CorrectionReview, CORRECTION_INSTRUCTION,
                        dict(candidate=candidate.model_dump(mode="json"),
                             prior=[dict(id=index, fact=old.model_dump(mode="json"))
                                    for index, old in enumerate(previous)]))
        if result.confidence < .95 or any(index < 0 or index >= len(previous) for index in result.superseded_ids):
            return []
        return [previous[index] for index in dict.fromkeys(result.superseded_ids)]
    except Exception as exc:
        raise_if_llm_failure(exc)
        print(f"CORRECTION REVIEW UNRESOLVED: {exc}")
        return []
