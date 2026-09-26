from __future__ import annotations

from .models import DiscoveryState, KnowledgeStatus


def active_fact_payload(state: DiscoveryState) -> list[dict]:
    """Return all active confirmed facts.

    Long-term semantic memory is intentionally not a sliding transcript window.
    Old confirmed knowledge remains available to planning regardless of turn age.
    """
    return [
        {
            "id": item.id,
            "canonical_key": item.canonical_key,
            "statement": item.statement,
            "domains": item.domains,
            "entities": item.entities,
            "negative": item.negative,
        }
        for item in state.facts
        if item.active and item.status == KnowledgeStatus.CONFIRMED
    ]


def requirement_payload(state: DiscoveryState) -> list[dict]:
    return [
        {
            "key": req.key,
            "label": req.label,
            "description": req.description,
            "status": req.status.value,
            "coverage": req.coverage.value,
            "depth": req.depth.value,
            "evidence_fact_ids": req.evidence_fact_ids,
            "implication_ids": req.implication_ids,
            "dependencies": req.dependencies,
            "missing_decisions": req.missing_decisions,
            "business_impact": req.business_impact,
            "architecture_impact": req.architecture_impact,
            "risk": req.risk,
        }
        for req in state.requirements.values()
    ]


def implication_payload(state: DiscoveryState) -> list[dict]:
    return [
        {
            "id": item.id,
            "statement": item.statement,
            "based_on_fact_ids": item.based_on_fact_ids,
            "domains": item.domains,
            "status": item.status.value,
            "needs_validation": item.needs_validation,
            "importance": item.importance,
        }
        for item in state.implications
        if item.status in (KnowledgeStatus.PROPOSED, KnowledgeStatus.CONFIRMED)
    ]


def contradiction_payload(state: DiscoveryState) -> list[dict]:
    return [
        {
            "id": item.id,
            "fact_ids": item.fact_ids,
            "issue": item.issue,
            "blocking": item.blocking,
        }
        for item in state.contradictions
        if not item.resolved
    ]


def boundary_payload(state: DiscoveryState) -> list[dict]:
    return [
        {
            "kind": item.kind,
            "evidence": item.evidence,
            "decision_key": item.decision_key,
            "question": item.question,
            "instruction": item.instruction,
        }
        for item in state.boundaries
    ]


def history_payload(state: DiscoveryState) -> list[dict]:
    """Keep semantic history for every delivered decision, not only recent turns."""
    return [
        {
            "turn": item.turn,
            "decision_key": item.decision_key,
            "objective": item.objective,
            "question": item.question,
            "requirement_keys": item.requirement_keys,
            "status": item.status,
        }
        for item in state.question_history
    ]


def recent_turn_payload(state: DiscoveryState, count: int = 6) -> list[dict]:
    return [
        {"number": turn.number, "text": turn.text}
        for turn in state.source_turns[-count:]
    ]


def product_context(state: DiscoveryState) -> dict:
    return {
        "raw_idea": state.raw_idea,
        "confirmed_facts": active_fact_payload(state),
        "requirements": requirement_payload(state),
        "proposed_implications": implication_payload(state),
        "open_contradictions": contradiction_payload(state),
        "discovery_boundaries": boundary_payload(state),
        "question_history": history_payload(state),
        "recent_user_turns": recent_turn_payload(state),
    }
