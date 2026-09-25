"""Resolve an explicit validation clarification against only the recorded conflict facts."""
from __future__ import annotations

import json
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.absence_supersession import supersession_record
from agents.discovery_coverage import fact_id
from agents.evidence_spans import recover_evidence_span
from agents.llm import get_structured_model
from agents.llm_errors import ExtractionFailed, raise_if_llm_failure
from agents.product_model import build_product_model
from agents.state import AgentState, KnowledgeState


class ConflictResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolved: bool
    superseded_fact_ids: List[str]
    retained_fact_ids: List[str]
    evidence: str
    confidence: float = Field(ge=0, le=1)


CONFLICT_RESOLUTION_INSTRUCTION = """Resolve ONLY the supplied recorded product contradiction.

You are given:
- the exact clarification question,
- the user's latest answer,
- the previously conflicting confirmed facts,
- newly grounded confirmed facts from this answer, if any.

Decide whether the answer clearly establishes the CURRENT rule/decision.

If the user clearly chooses one prior fact, retain that fact and supersede the
other conflicting fact(s).

If the user replaces the prior conflict with a new qualified rule, the old
conflicting facts may all be superseded ONLY when the supplied newly_grounded_facts
explicitly capture the replacement.

If the answer says both rules apply under different conditions, mark resolved
only when newly_grounded_facts explicitly preserve those conditions well enough
to replace the old incompatible broad facts.

Do not resolve from silence, uncertainty, vague preference, or your own domain
assumptions. Do not supersede facts outside conflict_facts. The question helps
interpret references such as "the first one"; it is not evidence by itself.

When resolved=false:
- superseded_fact_ids must be []
- retained_fact_ids must be []
- evidence must still quote the relevant latest answer if possible.

When resolved=true:
- superseded_fact_ids and retained_fact_ids may contain ONLY supplied conflict IDs.
- every supplied conflict ID must be accounted for as either superseded or retained.
- at least one conflict fact must be superseded.
- evidence must be an exact contiguous quote from latest_response supporting the
  decision.

Return only the structured decision.
"""


_resolution_model = None


def resolution_model():
    global _resolution_model
    if _resolution_model is None:
        _resolution_model = get_structured_model(
            call_name="consistency.resolve_conflict",
            schema=ConflictResolutionDecision,
        )
    return _resolution_model


def _latest_exchange(state: AgentState) -> tuple[str, str] | None:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    answer = messages[-1].content
    question = next(
        (message.content for message in reversed(messages[:-1]) if isinstance(message, AIMessage)),
        "",
    )
    return question, answer


def _replacement_for(previous, retained, current_turn):
    same_field = [
        item for item in current_turn
        if (
            item.scope == previous.scope
            and item.topic == previous.topic
            and item.key == previous.key
            and item.role == previous.role
        )
    ]
    if same_field:
        return same_field[0]
    if retained:
        return retained[0]
    if current_turn:
        return current_turn[0]
    return None


def validation_resolution_node(state: AgentState) -> dict:
    if state.get("planner_source") != "validation":
        return {}

    issue = state.get("selected_validation_issue") or {}
    conflict_ids = list(dict.fromkeys(issue.get("fact_ids") or []))
    exchange = _latest_exchange(state)
    if len(conflict_ids) < 2 or exchange is None:
        return {}

    knowledge = list(state.get("discovered_knowledge", []))
    by_id = {fact_id(item): item for item in knowledge}
    live_conflicts = [by_id[identity] for identity in conflict_ids if identity in by_id]

    # The normal same-field correction path may already have removed one side.
    # If fewer than two recorded conflict facts remain, validation will clear the
    # issue on the upcoming consistency pass.
    if len(live_conflicts) < 2:
        return {}

    question, answer = exchange
    current_turn = [
        item for item in knowledge
        if item.knowledge_state == KnowledgeState.CONFIRMED
        and item.source_turn == state.get("turn_count", 0)
        and item not in live_conflicts
    ]

    payload = {
        "question": question,
        "latest_response": answer,
        "conflict_facts": [
            {
                "fact_id": fact_id(item),
                "fact": item.model_dump(mode="json"),
            }
            for item in live_conflicts
        ],
        "newly_grounded_facts": [
            {
                "fact_id": fact_id(item),
                "fact": item.model_dump(mode="json"),
            }
            for item in current_turn
        ],
    }

    try:
        decision = resolution_model().invoke([
            SystemMessage(content=CONFLICT_RESOLUTION_INSTRUCTION),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        if not isinstance(decision, ConflictResolutionDecision):
            decision = ConflictResolutionDecision.model_validate(decision)
    except Exception as exc:
        raise_if_llm_failure(exc)
        raise ExtractionFailed("Contradiction clarification could not be safely resolved") from exc

    supplied = {fact_id(item) for item in live_conflicts}
    superseded = set(decision.superseded_fact_ids)
    retained_ids = set(decision.retained_fact_ids)

    if not decision.resolved:
        if superseded or retained_ids:
            raise ExtractionFailed("Unresolved contradiction returned fact mutations")
        return {}

    if decision.confidence < 0.95:
        return {}
    if superseded & retained_ids:
        raise ExtractionFailed("Contradiction resolution both retained and superseded the same fact")
    if superseded | retained_ids != supplied:
        raise ExtractionFailed("Contradiction resolution did not account for every live conflict fact")
    if not superseded:
        raise ExtractionFailed("Contradiction resolution must supersede at least one conflict fact")
    if not superseded.issubset(supplied) or not retained_ids.issubset(supplied):
        raise ExtractionFailed("Contradiction resolution referenced facts outside the selected issue")

    evidence = recover_evidence_span(decision.evidence, answer)
    if evidence is None:
        raise ExtractionFailed("Contradiction resolution evidence is not grounded in the latest answer")

    retained = [by_id[identity] for identity in retained_ids if identity in by_id]
    if not retained and not current_turn:
        # Replacing every conflicting fact without a grounded replacement would
        # erase product knowledge rather than resolve it.
        return {}

    superseded_knowledge = list(state.get("superseded_knowledge", []))
    for identity in superseded:
        previous = by_id[identity]
        replacement = _replacement_for(previous, retained, current_turn)
        if replacement is None:
            return {}
        superseded_knowledge.append(supersession_record(previous, replacement))

    updated = [
        item for item in knowledge
        if fact_id(item) not in superseded
    ]
    acquisition = {
        identity: record
        for identity, record in state.get("fact_acquisition", {}).items()
        if identity not in superseded
    }
    return {
        "discovered_knowledge": updated,
        "superseded_knowledge": superseded_knowledge,
        "fact_acquisition": acquisition,
        "product_model": build_product_model(
            updated,
            state.get("discovery_scope"),
        ),
    }
