"""Responsibility prompt contract and own-quote grounding regressions.

Run live model checks with RUN_LIVE_RESPONSIBILITY=1 and the configured Ollama
model available. Deterministic tests exercise the pipeline with audit fixtures;
they do not claim to establish a language model's semantic accuracy.
"""
import os
from types import SimpleNamespace

import pytest

from agents import knowledge_tracker as tracker
from agents.semantic_validation import GroundingResult
from agents.state import DiscoveryScope, DiscoveryTopic


CASES = [
    ("The primary users are customers.", "customer", "can browse items and pay", False),
    ("Doctors use the platform.", "doctor", "review patient records", False),
    ("Doctors can review patient records.", "doctor", "review patient records", True),
    ("The primary users are astronomers.", "astronomer", "observe stars and track planets", False),
]


def state():
    return dict(discovered_knowledge=[], current_topic=DiscoveryTopic.USER_ROLES,
                discovery_scope=DiscoveryScope.USER_APP, turn_count=1)


@pytest.mark.parametrize("text,role,value,supported", CASES)
def test_responsibility_own_quote_audit(monkeypatch, text, role, value, supported):
    def invoke(name, messages):
        if name == "RESPONSIBILITY":
            prompt = messages[0].content
            assert "Actor identity alone never implies responsibilities" in prompt
            assert "Merely saying an actor uses" in prompt
            assert "must explicitly support every proposed action" in prompt
            assert "Buyers can browse items and pay" not in prompt
            return {"items": [dict(key="responsibilities", value=value,
                                   evidence=text, confidence=1, role=role)]}
        if name == "ACTOR":
            return {"items": [dict(key="primary_users", value=role,
                                   evidence=text, confidence=1, roles=[role])]}
        return {"items": []}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name, *_ in tracker.PASSES})

    def audit(name, schema, instruction, payload):
        assert name == "GROUNDING"
        assert "Its quote supports no responsibility category" in instruction
        assert "explicitly state every proposed action" in instruction
        candidates = payload["candidates"]
        assert all(payload["evidence_quotes"][str(c["evidence_id"])] == text
                   for c in candidates)
        return GroundingResult(
            evidence_categories={"0": ["USER_ROLES.primary_users"] +
                                 (["USER_ROLES.responsibilities"] if supported else [])},
            supported_ids=[c["id"] for c in candidates
                           if c["key"] == "primary_users" or supported])

    monkeypatch.setattr(tracker, "semantic_decision", audit)
    current = state()
    candidates = tracker.extract_passes(text, current, DiscoveryScope.USER_APP)
    # A verbatim quote and valid owner alone must not bypass semantic grounding.
    assert any(c.key == "responsibilities" for c in candidates)
    accepted = tracker.ground_items(candidates, text, current)
    assert any(c.key == "primary_users" and c.roles == [role] for c in accepted)
    duties = [c for c in accepted if c.key == "responsibilities"]
    assert len(duties) == int(supported)
    if supported:
        assert duties[0].role == role and duties[0].value == value


@pytest.mark.skipif(os.environ.get("RUN_LIVE_RESPONSIBILITY") != "1",
                    reason="Requires the configured live Ollama model")
@pytest.mark.parametrize("text,role,value,supported", CASES)
def test_live_responsibility_extraction(text, role, value, supported):
    current = state()
    candidates = tracker.extract_passes(text, current, DiscoveryScope.USER_APP)
    assert any(c.key == "primary_users" and role in (c.roles or []) for c in candidates)
    duties = [c for c in candidates if c.key == "responsibilities"]
    if not supported:
        assert duties == []
    else:
        assert any(c.role == role for c in duties)
    accepted = tracker.ground_items(candidates, text, current)
    assert any(c.key == "primary_users" for c in accepted)
    assert bool([c for c in accepted if c.key == "responsibilities"]) == supported
