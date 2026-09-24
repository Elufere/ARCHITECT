"""Role policy recovery preserves scope and polarity, with independent grounding."""
import json
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


CASES = [
    ("multiple_roles", "Can users have different roles across transactions?",
     "Users can act as different roles across transactions but cannot hold both within the same transaction.",
     "policy"),
    ("multiple_roles", "Can users have multiple roles at all?",
     "Users never have multiple roles at all.", "none"),
    ("role_transitions", "When may users change roles?",
     "Users can switch roles between workflow instances.", "policy"),
    ("role_transitions", "Can users switch roles during a workflow instance?",
     "Users cannot switch roles once a workflow instance starts.", "policy"),
]
MIXED = ("Yes, but they can never be a buyer and seller in the same transaction. "
         "Also, they cannot switch role in a single transaction from the one they started with.")
MIXED_RULES = {
    "multiple_roles": "Customers can have different roles across transactions, but cannot be a buyer and seller in the same transaction.",
    "role_transitions": "Customers cannot switch from their starting role within a single transaction.",
}


def initial_state(text, question, gap):
    return dict(messages=[AIMessage(content=question), HumanMessage(content=text)],
                current_topic=T.USER_ROLES, current_gap=gap, discovery_scope=S.USER_APP,
                discovered_knowledge=[KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP,
                    key="primary_users", roles=["customer"], value="customers",
                    evidence="Customers use the application.", confidence=1)],
                topic_status={}, turn_count=2)


def install_models(monkeypatch, text, decisions, *, absence_verified=True, fail_review=False):
    calls = []

    def invoke(name, messages):
        calls.append((name, messages))
        if name == "ACTOR":
            # Reproduce the failure: each substantive rule was serialized as none.
            return {"items": [dict(key=key, value="none", absence="none",
                                   evidence=text, confidence=1) for key in decisions]}
        if name == "GAP_ANSWER":
            assert "Negative words" in messages[0].content
            assert "mixed \"yes, but\"" in messages[0].content
            if fail_review:
                raise ValueError("role-policy reviewer unavailable")
            payload = json.loads(messages[-1].content)
            resolution, value = decisions[payload["gap"]]
            return dict(resolution=resolution, value=value, evidence=text, confidence=1)
        if name == "GROUNDING":
            assert "ALL contexts" in messages[0].content
            assert "drops its allowed" in messages[0].content
            groups = json.loads(messages[-1].content)["evidence_groups"]
            candidates = [candidate for group in groups for candidate in group["candidates"]]
            return dict(
                evidence_categories=[dict(evidence_id=int(group["evidence_id"]),
                    categories=list({f"{c['topic']}.{c['key']}" for c in group["candidates"]}))
                    for group in groups],
                supported_ids=[c["id"] for c in candidates],
                confirmed_absence_ids=[c["id"] for c in candidates if c.get("absence") and absence_verified],
                rejection_reasons={} if absence_verified else {
                    str(c["id"]): "Whole-field absence is not established" for c in candidates if c.get("absence")})
        return {"items": []}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*(entry[0] for entry in tracker.PASSES), "GAP_ANSWER", "GROUNDING"]})
    return calls


@pytest.mark.parametrize("key,question,text,resolution", CASES)
def test_role_policy_absence_is_reinterpreted_without_losing_the_rule(
        monkeypatch, key, question, text, resolution):
    expected_value = text if resolution == "policy" else "none"
    calls = install_models(monkeypatch, text, {key: (resolution, text if resolution == "policy" else None)})
    result = tracker.knowledge_tracker_node(initial_state(text, question, key))
    policies = [item for item in result["discovered_knowledge"] if item.key == key]
    assert len(policies) == 1
    assert policies[0].value == expected_value and policies[0].evidence == text
    assert policies[0].absence == ("none" if resolution == "none" else None)
    assert any(name == "GROUNDING" for name, _ in calls)
    assert not any(item.topic == T.BUSINESS_RULES for item in result["discovered_knowledge"])


@pytest.mark.parametrize("gap", ["multiple_roles", "role_transitions", None])
def test_exact_mixed_answer_preserves_both_policies_even_outside_active_gap(monkeypatch, gap):
    install_models(monkeypatch, MIXED, {key: ("policy", value) for key, value in MIXED_RULES.items()})
    result = tracker.knowledge_tracker_node(initial_state(
        MIXED, "Can customers have different roles across transactions?", gap))
    policies = {item.key: item for item in result["discovered_knowledge"] if item.key in MIXED_RULES}
    assert set(policies) == set(MIXED_RULES)
    for key, item in policies.items():
        assert item.value == MIXED_RULES[key]
        assert item.absence is None and item.evidence == MIXED
        assert not item.roles and not item.aliases


def test_scoped_prohibition_cannot_bypass_whole_field_absence_audit(monkeypatch):
    key, question, text, _ = CASES[-1]
    install_models(monkeypatch, text, {key: ("none", None)}, absence_verified=False)
    initial = initial_state(text, question, key)
    assert tracker.knowledge_tracker_node(initial)["discovered_knowledge"] == initial["discovered_knowledge"]


def test_role_policy_review_failure_does_not_persist_none(monkeypatch):
    key, question, text, _ = CASES[0]
    install_models(monkeypatch, text, {key: ("policy", text)}, fail_review=True)
    initial = initial_state(text, question, key)
    assert tracker.knowledge_tracker_node(initial)["discovered_knowledge"] == initial["discovered_knowledge"]


@pytest.mark.skipif(os.environ.get("RUN_LIVE_ROLE_POLICY") != "1",
                    reason="Requires the configured live OpenAI model (uses API credits)")
@pytest.mark.parametrize("key,question,text,resolution", CASES)
def test_live_role_policy_scope(key, question, text, resolution):
    result = tracker.knowledge_tracker_node(initial_state(text, question, key))
    policies = [item for item in result["discovered_knowledge"] if item.key == key]
    assert policies and all(item.evidence in text for item in policies)
    if resolution == "policy":
        assert all(item.absence is None and item.value != "none" for item in policies)
    else:
        assert any(item.absence == "none" for item in policies)
