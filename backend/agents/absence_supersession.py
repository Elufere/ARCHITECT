"""Protect explicit absence at the commit boundary, independently of extraction."""
from agents.llm_errors import raise_if_llm_failure
from agents.extraction_passes import absence_label
from agents.semantic_validation import GroundingResult
from agents.state import KnowledgeState


ABSENCE_REPLACEMENT_INSTRUCTION = """Review replacement of a confirmed whole-field absence.
The prior_absences are explicit product decisions, not missing information.
Support the candidate ONLY if its own quote in the latest response clearly
corrects or supersedes those decisions for the SAME scope, topic, field and owner.
A general correction intent flag does not establish correction of this field.
The user need not use a special correction keyword: an explicit incompatible
decision can supersede absence. But a participant merely doing work, an incidental
mention, a hypothetical possibility, or activity in another application cannot.
For actor absence, require explicit interaction with the same application or an
explicit correction of who its users are. Business-process participation alone
does not supersede absence of application users. Do not infer application access.
Use the question only to interpret references or an unambiguous confirmation of
the changed decision, never to manufacture a correction from a generic 'yes'.
The candidate must still be entailed by its own evidence, including conditions,
scope and ownership. Prior evidence cannot establish a new correction. If the
surface or supersession is unclear, reject and preserve the prior absence.
Return supported_ids and the candidate's evidence category only for a verified
replacement. Give a rejection reason otherwise. If the replacement itself is a
different whole-field absence, also require confirmed_absence_ids.
"""


def matching_absences(item, knowledge):
    return [old for old in knowledge
            if old.knowledge_state == KnowledgeState.CONFIRMED
            and (old.absence or absence_label(old.value))
            and (old.scope, old.topic, old.key, old.role)
            == (item.scope, item.topic, item.key, item.role)]


def can_replace_absence(item, previous, response, context, decide):
    """A grounded positive fact is not automatically a correction of absence."""
    if (item.knowledge_state != KnowledgeState.CONFIRMED or item.confidence < .75
            or not item.evidence.strip() or item.evidence not in response):
        return False
    absence = item.absence or absence_label(item.value)
    candidate = dict(id=0, topic=item.topic.value, key=item.key, value=item.value,
                     scope=item.scope.value, role=item.role, roles=item.roles, evidence_id=0)
    if absence:
        candidate["absence"] = absence
    try:
        decision = decide("GROUNDING", GroundingResult, ABSENCE_REPLACEMENT_INSTRUCTION,
                          dict(**context, latest_response=response,
                               prior_absences=[old.model_dump(mode="json") for old in previous],
                               candidates=[candidate], evidence_quotes={"0": item.evidence}))
        return (0 in decision.supported_ids
                and f"{item.topic.value}.{item.key}" in decision.evidence_categories.get("0", [])
                and (not absence or 0 in decision.confirmed_absence_ids))
    except Exception as exc:
        raise_if_llm_failure(exc)
        print(f"ABSENCE REPLACEMENT FAILED: {exc}")
        return False


def supersession_record(previous, replacement):
    return dict(fact=previous.model_dump(mode="json"),
                superseded_by=replacement.model_dump(mode="json"))
