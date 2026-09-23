"""Canonical capacities: context propagation, auditing, and optional live checks.

RUN_LIVE_ACTOR_RESOLUTION=1 enables semantic regressions against the configured
Ollama model. Offline fixtures test the plumbing, not model semantic accuracy.
"""
import json
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.extraction_passes import ActorFact, normalize_fact
from agents.semantic_validation import GroundingResult
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


CASES = [
    ("customer", ["buyer", "seller"],
     "Customers can act as buyers or sellers depending on the transaction.",
     "The buyer submits the request. The seller reviews it."),
    ("employee", ["manager", "approver"],
     "Employees act as managers or approvers depending on the assignment.",
     "The manager submits the request. The approver reviews it."),
    ("account_holder", ["sender", "recipient"],
     "Account holders act as senders or recipients depending on the transfer.",
     "The sender submits the request. The recipient reviews it."),
]


def actor(role, aliases, quote):
    return dict(key="primary_users", roles=[role], aliases=aliases,
                value=quote, evidence=quote, confidence=1)


def knowledge(raw):
    return normalize_fact(ActorFact.model_validate(raw), T.USER_ROLES, S.USER_APP, 1)


def state(text, existing=()):
    return dict(messages=[HumanMessage(content=text)], current_topic=T.USER_ROLES,
                discovery_scope=S.USER_APP, discovered_knowledge=list(existing),
                topic_status={}, turn_count=2)


@pytest.mark.parametrize("role,aliases,quote,later", CASES)
def test_current_and_later_passes_receive_canonical_capacity_context(
        monkeypatch, role, aliases, quote, later):
    declaration = actor(role, aliases, quote)
    prompts = {}

    def invoke(name, messages):
        prompts[name] = messages[0].content
        return {"items": [declaration] if name == "ACTOR" else []}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name, *_ in tracker.PASSES})
    extracted = tracker.extract_passes(quote, state(quote), S.USER_APP)
    assert {r for item in extracted for r in item.roles or []} == {role}
    assert extracted[0].evidence == quote
    assert extracted[0].aliases == {role: aliases}
    assert "BEFORE proposing a new actor" in prompts["ACTOR"]
    context = json.dumps(tracker.confirmed_actor_context(
        {"discovered_knowledge": extracted}, S.USER_APP))
    assert context in prompts["RESPONSIBILITY"]

    tracker.extract_passes(later, state(later, extracted), S.USER_APP)
    assert context in prompts["ACTOR"]
    assert f'Canonical actor IDs: ["{role}"]' in prompts["ACTOR"]


@pytest.mark.parametrize("role,aliases,quote,later", CASES)
def test_audit_sees_aliases_and_rejects_capacity_actor_proposals(
        monkeypatch, role, aliases, quote, later):
    canonical = knowledge(actor(role, aliases, quote))
    proposals = [knowledge(actor(label, [], later)) for label in aliases]
    seen = []

    def audit(name, schema, instruction, payload):
        seen.append(payload)
        assert "resolve labels against confirmed_actor_context" in instruction
        if payload["latest_response"] == quote:
            assert payload["candidates"][0]["aliases"] == {role: aliases}
            return GroundingResult(supported_ids=[0],
                evidence_categories={"0": ["USER_ROLES.primary_users"]})
        assert payload["confirmed_actor_context"][0]["aliases"] == {role: aliases}
        assert payload["confirmed_actor_context"][0]["evidence"] == quote
        return GroundingResult(supported_ids=[], evidence_categories={"0": []})

    monkeypatch.setattr(tracker, "semantic_decision", audit)
    assert tracker.ground_items([canonical], quote, state(quote)) == [canonical]
    assert tracker.ground_items(proposals, later, state(later, [canonical])) == []
    assert len(seen) == 2


def test_new_capacity_aliases_survive_actor_deduplication(monkeypatch):
    role, aliases, quote, _ = CASES[0]
    old = knowledge(actor(role, [], quote))
    updated = knowledge(actor(role, aliases, quote))
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: [updated])
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_: items)
    initial = state(quote, [old])
    result = tracker.knowledge_tracker_node(initial)
    records = result["discovered_knowledge"]
    assert {r for item in records for r in item.roles or []} == {role}
    assert any(item.aliases == {role: aliases} and item.evidence == quote
               for item in records)
    # Repeating the same declaration must not accumulate duplicates.
    repeated = tracker.knowledge_tracker_node({**initial, **result})
    assert repeated["discovered_knowledge"] == records


@pytest.mark.parametrize("role,aliases,quote,later", CASES)
def test_explicit_separate_actor_types_remain_discoverable(
        monkeypatch, role, aliases, quote, later):
    existing = knowledge(actor(role, aliases, quote))
    separate = f"{aliases[0]} and {aliases[1]} are separate primary application user types."
    outputs = [actor(label, [], separate) for label in aliases]

    def invoke(name, messages):
        if name == "ACTOR":
            assert "preserve that distinction even if" in messages[0].content
        return {"items": outputs if name == "ACTOR" else []}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name, *_ in tracker.PASSES})
    extracted = tracker.extract_passes(separate, state(separate, [existing]), S.USER_APP)
    assert {r for item in extracted for r in item.roles or []} == set(aliases)
    assert all(item.evidence == separate for item in extracted)


@pytest.mark.skipif(os.environ.get("RUN_LIVE_ACTOR_RESOLUTION") != "1",
                    reason="Requires the configured live Ollama model")
@pytest.mark.parametrize("role,aliases,quote,later", CASES)
def test_live_capacities_and_explicit_separate_types(role, aliases, quote, later):
    first = state(quote)
    extracted = tracker.extract_passes(quote, first, S.USER_APP)
    actors = [item for item in extracted if item.key in ("primary_users", "secondary_users")
              and not item.absence]
    assert {r for item in actors for r in item.roles or []} == {role}
    assert all(item.evidence in quote for item in actors)
    assert set(aliases) <= {alias for item in actors
                           for alias in (item.aliases or {}).get(role, [])}
    grounded = tracker.ground_items(actors, quote, first)
    assert grounded

    second = state(later, grounded)
    later_items = tracker.extract_passes(later, second, S.USER_APP)
    assert {r for item in later_items for r in item.roles or []} <= {role}

    separate = (f"In this application, {aliases[0]} and {aliases[1]} are separate "
                "primary application user types, not contextual capacities of "
                f"{role.replace('_', ' ')}. Each has its own independent account type.")
    third = state(separate, grounded)
    distinct = tracker.extract_passes(separate, third, S.USER_APP)
    distinct = tracker.ground_items(distinct, separate, third)
    assert set(aliases) <= {r for item in distinct for r in item.roles or []}
