"""Own evidence must state a goal meaning and its owner, not a plausible benefit.

Offline tests use adversarial extraction and audit fixtures. Enable
RUN_LIVE_GOAL_BOUNDARIES=1 for production model checks on the same source cases.
"""
import json
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


NEGATIVES = [
    "The buyer funds the escrow.",
    "The customer agrees to transaction terms.",
    "The buyer approves completion.",
    "The customer opens a dispute, the auditor reviews evidence, and the customer receives the decision.",
    "The app holds funds securely.",
    "KYC must be completed before transactions are created.",
    "Only the buyer may confirm completion.",
    "The seller is responsible for uploading proof of delivery.",
]
POSITIVES = [
    ("The buyer wants to transact without risking payment before fulfillment.",
     "primary_user_goals", "buyer", "transact without risking payment before fulfillment"),
    ("The transaction is successful when the seller fulfills the terms and payment is released.",
     "success_criteria", None, "seller fulfills the terms and payment is released"),
    ("The customer chooses the app to reduce fraud risk.",
     "motivations", "customer", "reduce fraud risk"),
    ("The buyer funds the escrow so that payment stays protected until fulfillment.",
     "primary_user_goals", "buyer", "payment stays protected until fulfillment"),
]
GOAL_KEYS = {"primary_user_goals", "secondary_user_goals", "success_criteria", "motivations"}


def state(text):
    return dict(messages=[HumanMessage(content=text)], current_topic=T.USER_GOALS,
                current_gap="primary_user_goals::seller", discovery_scope=S.USER_APP,
                topic_status={}, turn_count=2,
                discovered_knowledge=[KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP,
                    key="secondary_users" if role == "auditor" else "primary_users",
                    roles=[role], value=role, evidence=f"{role} uses this application", confidence=1)
                    for role in ("buyer", "seller", "customer", "auditor")])


def proposal(text, key, value, role=None):
    return dict(key=key, value=value, role=role, evidence=text, confidence=1)


def install(monkeypatch, proposals, expected, *, repaired=None):
    calls = {}

    def invoke(name, messages):
        calls.setdefault(name, []).append(messages)
        if name == "GOAL":
            prompt = messages[0].content
            assert "Apply this requirement to individual actions too" in prompt
            assert "The current gap's actor is not ownership evidence" in prompt
            assert "not a success condition" in prompt
            if len(calls[name]) > 1 and repaired is not None:
                assert "Do not assign the active gap's owner automatically" in prompt
                return {"items": repaired}
            return {"items": proposals}
        if name == "GROUNDING":
            assert "For USER_GOALS, an individual action" in messages[0].content
            assert "Reject ambiguous or reassigned owners" in messages[0].content
            groups = json.loads(messages[-1].content)["evidence_groups"]
            candidates = [c for group in groups for c in group["candidates"]]
            supported = [c["id"] for c in candidates
                         if (c["key"], c.get("role"), c["value"]) in expected]
            return dict(
                evidence_categories=[dict(evidence_id=int(group["evidence_id"]),
                    categories=sorted({f"USER_GOALS.{key}" for key, _, _ in expected})) for group in groups],
                supported_ids=supported, confirmed_absence_ids=[],
                rejection_reasons={str(c["id"]): "Own evidence does not support this goal meaning or owner"
                                   for c in candidates if c["id"] not in supported})
        return {"items": []}

    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*(entry[0] for entry in tracker.PASSES), "GROUNDING"]})
    return calls


@pytest.mark.parametrize("text", NEGATIVES)
def test_actions_rules_and_features_cannot_be_saved_under_any_goal_key(monkeypatch, text):
    candidates = [proposal(text, key, "complete a safe transaction", role)
                  for key, role in [("primary_user_goals", "buyer"), ("secondary_user_goals", "auditor"),
                                    ("success_criteria", None), ("motivations", None)]]
    calls = install(monkeypatch, candidates, set())
    initial = state(text)
    result = tracker.knowledge_tracker_node(initial)
    assert result["discovered_knowledge"] == initial["discovered_knowledge"]
    assert "GROUNDING" in calls


@pytest.mark.parametrize("text,key,role,value", POSITIVES)
def test_explicit_goal_meanings_retain_the_correct_key_and_owner(monkeypatch, text, key, role, value):
    expected = {(key, role, value)}
    install(monkeypatch, [proposal(text, key, value, role)], expected)
    result = tracker.knowledge_tracker_node(state(text))
    goals = [item for item in result["discovered_knowledge"] if item.topic == T.USER_GOALS]
    assert {(item.key, item.role, item.value) for item in goals} == expected
    assert all(item.evidence == text for item in goals)


def test_current_gap_cannot_reassign_a_named_actors_goal(monkeypatch):
    text, key, role, value = POSITIVES[0]
    install(monkeypatch, [proposal(text, key, value, "seller"), proposal(text, key, value, role)],
            {(key, role, value)})
    result = tracker.knowledge_tracker_node(state(text))
    goals = [item for item in result["discovered_knowledge"] if item.topic == T.USER_GOALS]
    assert len(goals) == 1 and goals[0].role == "buyer"


@pytest.mark.parametrize("repair_owner", ["seller", "buyer"])
def test_owner_repair_must_still_match_the_outcomes_actor(monkeypatch, repair_owner):
    text, key, role, value = POSITIVES[0]
    calls = install(monkeypatch, [proposal(text, key, value)], {(key, role, value)},
                    repaired=[proposal(text, key, value, repair_owner)])
    result = tracker.knowledge_tracker_node(state(text))
    goals = [item for item in result["discovered_knowledge"] if item.topic == T.USER_GOALS]
    assert len(calls["GOAL"]) == 2
    assert len(goals) == (1 if repair_owner == "buyer" else 0)
    assert all(item.role == "buyer" for item in goals)


def test_unknown_owner_cannot_pass_schema_repair(monkeypatch):
    text, key, _, value = POSITIVES[0]
    candidate = proposal(text, key, value, "unregistered")
    calls = install(monkeypatch, [candidate], {(key, "unregistered", value)}, repaired=[candidate])
    initial = state(text)
    assert tracker.knowledge_tracker_node(initial)["discovered_knowledge"] == initial["discovered_knowledge"]
    assert len(calls["GOAL"]) == 2 and "GROUNDING" not in calls


@pytest.mark.skipif(os.environ.get("RUN_LIVE_GOAL_BOUNDARIES") != "1",
                    reason="Requires the configured live OpenAI model (uses API credits)")
@pytest.mark.parametrize("text", NEGATIVES)
def test_live_non_goal_statements(text):
    initial = state(text)
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    assert not any(item.topic == T.USER_GOALS for item in extracted)


@pytest.mark.skipif(os.environ.get("RUN_LIVE_GOAL_BOUNDARIES") != "1",
                    reason="Requires the configured live OpenAI model (uses API credits)")
@pytest.mark.parametrize("text,key,role,value", POSITIVES)
def test_live_explicit_goal_meanings(text, key, role, value):
    initial = state(text)
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    accepted = tracker.ground_items(extracted, text, initial)
    assert any(item.key == key and item.role == role for item in accepted)
    assert all(item.role != "seller" for item in accepted if item.key in ("primary_user_goals", "secondary_user_goals"))
