"""Shared admission instructions and opt-in pre-grounding model regressions."""
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


FOCUSES = [
    (T.USER_ROLES, "secondary_users"),
    (T.USER_ROLES, "responsibilities::member"),
    (T.USER_ROLES, "permissions::member"),
    (T.CORE_WORKFLOW, "trigger"),
    (T.USER_GOALS, "primary_user_goals::member"),
    (T.BUSINESS_RULES, "approval_rules"),
]


def state(text, topic, gap):
    return dict(messages=[AIMessage(content="Who approves requests, what starts the process, and what restrictions apply?"),
                          HumanMessage(content=text)],
                current_topic=topic, current_gap=gap, discovery_scope=S.USER_APP,
                turn_count=2, discovered_knowledge=[KnowledgeItem(
                    topic=T.USER_ROLES, scope=S.USER_APP, key="primary_users",
                    value="member", roles=["member"], evidence="Members use this app.", confidence=1)])


@pytest.mark.parametrize("topic,gap", FOCUSES)
@pytest.mark.parametrize("text", ["Members want to reduce wasted time.", "I am not sure.", "Yes."])
def test_every_pass_receives_admission_contract_before_generation(monkeypatch, topic, gap, text):
    seen = []

    def invoke(name, messages):
        prompt = messages[0].content
        seen.append(name)
        assert f"Category admission for this {name} pass happens BEFORE generating candidates" in prompt
        assert "Does this exact statement explicitly express a" in prompt
        assert "fact belonging to MY category?" in prompt
        assert "even for INFERRED items" in prompt
        assert "Question context may resolve references" in prompt
        assert "does not make an ambiguous answer confirm all of them" in prompt
        assert "each category's\nmeaning independently" in prompt
        assert 'Return {"items": []}' in prompt
        assert "Do not emit speculative candidates" in prompt
        assert messages[-1].content == text
        return {"items": []}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name, *_ in tracker.PASSES})
    assert tracker.extract_passes(text, state(text, topic, gap), S.USER_APP) == []
    assert seen == [name for name, *_ in tracker.PASSES]


CASES = [
    ("Members want to reduce wasted time.", {"primary_user_goals", "motivations"}),
    ("Members can upload documents.", {"responsibilities"}),
    ("The review process starts when a request arrives.", {"trigger"}),
    ("The interface is blue.", set()),
    ("I am not sure.", set()),
    ("Members upload documents because they want to reduce wasted time.",
     {"responsibilities", "primary_user_goals", "motivations"}),
]


@pytest.mark.skipif(os.environ.get("RUN_LIVE_CROSS_PASS_ISOLATION") != "1",
                    reason="Requires the configured live extraction model")
@pytest.mark.parametrize("topic,gap", FOCUSES)
@pytest.mark.parametrize("text,allowed", CASES)
def test_live_categories_before_grounding(text, allowed, topic, gap):
    # Deliberately inspect generation without calling the final auditor.
    candidates = tracker.extract_passes(text, state(text, topic, gap), S.USER_APP)
    keys = {item.key for item in candidates}
    assert keys <= allowed
    if allowed:
        assert keys
    if text.startswith("Members upload"):
        assert "responsibilities" in keys
        assert keys & {"primary_user_goals", "motivations"}
