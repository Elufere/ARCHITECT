"""Regression for the escrow interview repeatedly rejecting ownerless goals."""
from types import SimpleNamespace

import pytest

from agents import knowledge_tracker as tracker
from agents.interview_planner import build_gap_info
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


SOURCE = "The customer wants confidence that the other party will honor the agreement."


def goal(**extra):
    return dict(key="primary_user_goals", value="confidence that the agreement is honored",
                evidence=SOURCE, confidence=1, **extra)


def replay(monkeypatch, repaired, initial=None):
    calls = []
    def model(name):
        def invoke(messages):
            if name != "GOAL":
                return {"items": []}
            calls.append(messages)
            if len(calls) == 1:
                return {"items": initial if initial is not None else [goal()]}
            if isinstance(repaired, Exception):
                raise repaired
            return {"parsed": {"items": repaired}}
        return SimpleNamespace(invoke=invoke)
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: model(name) for name, *_ in tracker.PASSES})
    actor = KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key="primary_users",
                          value="customers", roles=["customer"], evidence="customers",
                          confidence=1, knowledge_state="CONFIRMED", source_turn=1)
    state = dict(discovered_knowledge=[actor], current_topic=T.USER_GOALS,
                 current_gap="primary_user_goals::customer", turn_count=6)
    items = tracker.extract_passes(SOURCE, state, S.USER_APP)
    return items, calls, state


def test_missing_owner_repaired_without_fabricating_deliberate_gap_coverage(monkeypatch):
    items, calls, state = replay(monkeypatch, [goal(role="customer")])
    assert len(calls) == 2
    assert [item.role for item in items] == ["customer"]
    assert "requires an owner" in calls[1][0].content
    assert calls[1][-1].content == SOURCE
    state["discovered_knowledge"].extend(items)
    assert "primary_user_goals::customer" in build_gap_info(state, T.USER_GOALS)["missing_keys"]


@pytest.mark.parametrize("repair", [
    [goal()], [goal(role="seller")],
    [dict(goal(role="customer"), key="secondary_user_goals")],
    [dict(goal(role="customer"), evidence="fabricated quote")],
    TimeoutError("provider unavailable"),
])
def test_invalid_repair_stops_without_fabricating_owner(monkeypatch, repair):
    items, calls, _ = replay(monkeypatch, repair)
    assert items == []
    assert len(calls) == 2


def test_valid_goals_need_no_repair(monkeypatch):
    items, calls, _ = replay(monkeypatch, [], initial=[goal(role="customer")])
    assert len(items) == 1
    assert len(calls) == 1


def test_valid_sibling_survives_failed_repair(monkeypatch):
    items, calls, _ = replay(monkeypatch, [goal()], initial=[goal(role="customer"), goal()])
    assert len(items) == 1
    assert len(calls) == 2
