"""Focused tests for resolving recorded contradictions after user clarification."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import validation_resolution as vr
from agents.discovery_coverage import fact_id
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState
from agents.validation_resolution import ConflictResolutionDecision, validation_resolution_node


def fact(topic, key, value, *, turn=1):
    return KnowledgeItem(
        topic=topic,
        scope=S.USER_APP,
        key=key,
        value=value,
        evidence=value,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
    )


def validation_state(first, second, *, answer="Use the second rule.", extra=None):
    question = (
        f"I have '{first.value}' and '{second.value}' recorded. "
        "Which rule should apply now?"
    )
    return {
        "planner_source": "validation",
        "selected_validation_issue": {
            "id": "conflict",
            "fact_ids": [fact_id(first), fact_id(second)],
        },
        "messages": [AIMessage(content=question), HumanMessage(content=answer)],
        "discovered_knowledge": [first, second] + list(extra or []),
        "superseded_knowledge": [],
        "fact_acquisition": {
            fact_id(first): {"source_turn": first.source_turn},
            fact_id(second): {"source_turn": second.source_turn},
        },
        "turn_count": 3,
        "discovery_scope": S.USER_APP,
    }


def test_non_validation_turn_is_noop():
    assert validation_resolution_node({"planner_source": "schema"}) == {}


def test_cross_field_clarification_can_retain_one_fact_and_supersede_the_other(monkeypatch):
    must = fact(T.MVP_SCOPE, "must_have_features", "Card payments are required in MVP")
    excluded = fact(T.MVP_SCOPE, "out_of_scope", "Card payments are excluded from MVP", turn=2)
    state = validation_state(must, excluded)

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=True,
            superseded_fact_ids=[fact_id(must)],
            retained_fact_ids=[fact_id(excluded)],
            evidence="Use the second rule.",
            confidence=1,
        )),
    )
    result = validation_resolution_node(state)
    active_ids = {fact_id(item) for item in result["discovered_knowledge"]}
    assert fact_id(must) not in active_ids
    assert fact_id(excluded) in active_ids
    assert fact_id(must) not in result["fact_acquisition"]
    assert result["superseded_knowledge"][0]["fact"]["value"] == must.value
    assert result["superseded_knowledge"][0]["superseded_by"]["value"] == excluded.value
    assert "MVP_SCOPE" in result["product_model"]


def test_both_old_facts_can_be_replaced_only_with_new_grounded_fact(monkeypatch):
    old_five = fact(T.BUSINESS_RULES, "limits", "Maximum five requests")
    old_ten = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", turn=2)
    replacement = fact(
        T.BUSINESS_RULES,
        "limits",
        "Free users may have five requests and premium users may have ten",
        turn=3,
    )
    state = validation_state(
        old_five,
        old_ten,
        answer="Free users get five, premium users get ten.",
        extra=[replacement],
    )

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=True,
            superseded_fact_ids=[fact_id(old_five), fact_id(old_ten)],
            retained_fact_ids=[],
            evidence="Free users get five, premium users get ten.",
            confidence=1,
        )),
    )
    result = validation_resolution_node(state)
    assert result["discovered_knowledge"] == [replacement]
    assert len(result["superseded_knowledge"]) == 2
    assert all(
        record["superseded_by"]["value"] == replacement.value
        for record in result["superseded_knowledge"]
    )


def test_cannot_erase_all_conflict_facts_without_grounded_replacement(monkeypatch):
    first = fact(T.BUSINESS_RULES, "limits", "Maximum five requests")
    second = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", turn=2)
    state = validation_state(first, second, answer="Neither.")

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=True,
            superseded_fact_ids=[fact_id(first), fact_id(second)],
            retained_fact_ids=[],
            evidence="Neither.",
            confidence=1,
        )),
    )
    assert validation_resolution_node(state) == {}


def test_unresolved_clarification_does_not_mutate_knowledge(monkeypatch):
    first = fact(T.BUSINESS_RULES, "limits", "Maximum five requests")
    second = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", turn=2)
    state = validation_state(first, second, answer="I'm not sure.")

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=False,
            superseded_fact_ids=[],
            retained_fact_ids=[],
            evidence="I'm not sure.",
            confidence=1,
        )),
    )
    assert validation_resolution_node(state) == {}


def test_resolution_cannot_reference_unrelated_fact_id(monkeypatch):
    first = fact(T.BUSINESS_RULES, "limits", "Maximum five requests")
    second = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", turn=2)
    unrelated = fact(T.CONSTRAINTS, "time_constraints", "Requests expire after one day", turn=3)
    state = validation_state(first, second, extra=[unrelated])

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=True,
            superseded_fact_ids=[fact_id(first), fact_id(unrelated)],
            retained_fact_ids=[fact_id(second)],
            evidence="Use the second rule.",
            confidence=1,
        )),
    )
    with pytest.raises(Exception):
        validation_resolution_node(state)


def test_low_confidence_resolution_preserves_conflict(monkeypatch):
    first = fact(T.BUSINESS_RULES, "limits", "Maximum five requests")
    second = fact(T.BUSINESS_RULES, "limits", "Maximum ten requests", turn=2)
    state = validation_state(first, second)

    monkeypatch.setattr(
        vr,
        "resolution_model",
        lambda: SimpleNamespace(invoke=lambda _: ConflictResolutionDecision(
            resolved=True,
            superseded_fact_ids=[fact_id(first)],
            retained_fact_ids=[fact_id(second)],
            evidence="Use the second rule.",
            confidence=0.8,
        )),
    )
    assert validation_resolution_node(state) == {}
