"""Separate fact acquisition from deliberate, scoped interview coverage."""
import hashlib
import json

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.evidence_spans import recover_evidence_span
from agents.llm_errors import ExtractionFailed, raise_if_llm_failure
from agents.role_utils import role_identity, roles_match
from agents.state import DiscoveryScope, KnowledgeState


def fact_id(item):
    payload = json.dumps(item.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_key(scope, topic, gap):
    key, separator, role = gap.partition("::")
    gap = key + ("::" + role_identity(role) if separator else "")
    return f"{scope.value}|{topic.value}|{gap}"


def facts_for_gap(state, topic, gap, *, confirmed_only=True):
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    key, _, role = gap.partition("::")
    return [item for item in state.get("discovered_knowledge", [])
            if item.scope == scope and item.topic == topic and item.key == key
            and (not role or roles_match(item.role, role))
            and (not confirmed_only or item.knowledge_state == KnowledgeState.CONFIRMED)]


def gap_resolved(state, topic, gap):
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    record = state.get("gap_coverage", {}).get(coverage_key(scope, topic, gap), {})
    return record.get("status") == "RESOLVED" and bool(facts_for_gap(state, topic, gap))


def active_question_matches(state):
    asked = state.get("asked_gap") or {}
    topic, gap = state.get("current_topic"), state.get("current_gap")
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    return bool(topic and gap and asked.get("scope") == scope.value
                and asked.get("topic") == topic.value and asked.get("gap") == gap
                and asked.get("question"))


def answer_receipt(state, supported, knowledge, *, confirmed_existing=False):
    """Evidence of a successful active answer; only the planner resolves coverage."""
    if not active_question_matches(state) or state.get("conversation_intent") == "objection":
        return None
    topic, gap = state["current_topic"], state["current_gap"]
    eligible = facts_for_gap({**state, "discovered_knowledge": knowledge}, topic, gap)
    ids = [fact_id(item) for item in supported if item in eligible]
    if not ids:
        return None
    message = state["messages"][-1]
    if not isinstance(message, HumanMessage):
        return None
    return dict(scope=state["discovery_scope"].value, topic=topic.value, gap=gap,
                source_turn=state.get("turn_count", 0), answer_id=message.id or str(state.get("turn_count", 0)),
                evidence=message.content, fact_ids=ids,
                resolution="CONFIRMED_EXISTING" if confirmed_existing else "DIRECT_ANSWER")


def acquisition_records(state, knowledge, directly_supported):
    records = dict(state.get("fact_acquisition", {}))
    receipt = answer_receipt(state, directly_supported, knowledge)
    direct_ids = set(receipt["fact_ids"]) if receipt else set()
    for item in knowledge:
        identity = fact_id(item)
        if identity not in records:
            records[identity] = dict(acquisition="DIRECT" if identity in direct_ids else "INCIDENTAL",
                                     source_turn=item.source_turn,
                                     active_topic=state["current_topic"].value if state.get("current_topic") else None,
                                     active_gap=state.get("current_gap"))
    return records


class GapConfirmation(BaseModel):
    confirmed: bool
    evidence: str
    confidence: float = Field(ge=0, le=1)


CONFIRMATION_INSTRUCTION = """Determine whether the latest answer explicitly
confirms the supplied prior facts as the complete answer to the asked active gap.
The question must actually present that information for confirmation/expansion.
Affirmations like 'yes, that is all' or 'nothing else' may confirm completeness;
a bare yes/no is valid only if the question makes its meaning unambiguous.
Do not interpret 'no additions' as whole-field absence of the existing facts.
Corrections, uncertainty, additional details or an ambiguous response do not
confirm all prior facts. Do not infer agreement from silence or intent labels.
Use only the supplied question, prior facts and latest response. Evidence must
be copied verbatim from the latest response. Return confirmed=false if uncertain.
"""


def confirms_existing(state, decide):
    if not active_question_matches(state) or state.get("next_discovery_move") not in ("confirm_existing", "confirm_inference"):
        return False
    facts = facts_for_gap(state, state["current_topic"], state["current_gap"], confirmed_only=False)
    if not facts:
        return False
    response = state["messages"][-1].content
    try:
        decision = decide("GAP_CONFIRMATION", GapConfirmation, CONFIRMATION_INSTRUCTION,
                          dict(question=state["asked_gap"]["question"], latest_response=response,
                               gap=state["current_gap"], prior_facts=[item.model_dump(mode="json") for item in facts]))
    except Exception as exc:
        raise_if_llm_failure(exc)
        raise ExtractionFailed("Active-gap confirmation did not return a valid decision") from exc
    return (decision.confirmed and decision.confidence >= .95
            and recover_evidence_span(decision.evidence, response) is not None)
