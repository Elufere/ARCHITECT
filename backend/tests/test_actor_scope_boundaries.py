"""Scope membership is independent of participation in the wider process.

Offline tests cover context isolation and independent audit outcomes. Enable
RUN_LIVE_ACTOR_RESOLUTION=1 to check the production model against source prose.
"""
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K


def actor(role, scope, quote):
    return KnowledgeItem(topic=T.USER_ROLES, scope=scope, key="secondary_users",
                         roles=[role], value=quote, evidence=quote, confidence=1)


def state(text, scope=S.USER_APP, existing=()):
    return dict(messages=[HumanMessage(content=text)], discovery_scope=scope,
                current_topic=T.USER_ROLES, discovered_knowledge=list(existing),
                topic_status={}, turn_count=2)


def models(monkeypatch, outputs):
    prompts = {}

    def invoke(name, messages):
        prompts[name] = messages[0].content
        if name == "GROUNDING":
            # Wire-format audit fixture, retaining the real semantic_decision path.
            return {"evidence_categories": [{"evidence_id": 0, "categories": [
                "CORE_WORKFLOW.downstream_dependency", "BUSINESS_RULES.approval_rules"]}],
                "supported_ids": [1, 2], "confirmed_absence_ids": [],
                "rejection_reasons": {"0": "The quote establishes a different surface, not current-app membership"}}
        return {"items": outputs.get(name, [])}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*(entry[0] for entry in tracker.PASSES), "GROUNDING"]})
    return prompts


@pytest.mark.parametrize("scope,other_scope", [
    (S.USER_APP, S.ADMIN_DASHBOARD), (S.ADMIN_DASHBOARD, S.USER_APP),
])
def test_other_scope_actor_is_not_available_as_current_actor(monkeypatch, scope, other_scope):
    quote = f"The fraud team reviews payments only through {other_scope.value}."
    other_actor = actor("fraud_team", other_scope, quote)
    prompts = models(monkeypatch, {"RESPONSIBILITY": [dict(
        key="responsibilities", role="fraud_team", value="review payments",
        evidence=quote, confidence=1)]})
    initial = state(quote, scope, [other_actor])
    assert tracker.extract_passes(quote, initial, scope) == []
    assert tracker.confirmed_actor_context(initial, scope) == []
    assert 'Canonical actor IDs: []' in prompts["ACTOR"]
    assert f"Current scope: {scope.value}" in prompts["ACTOR"]
    assert "applies on EVERY turn, including initial discovery" in prompts["ACTOR"]
    assert "do not invent a surface" in prompts["ACTOR"]
    assert other_actor.scope == other_scope


@pytest.mark.parametrize("surface", [
    "ADMIN_APP", "a separate back-office tool", "an internal operations tool",
    "an external system", "an offline process", "a third-party platform",
])
def test_cross_scope_process_facts_survive_actor_rejection(monkeypatch, surface):
    text = ("Payments in USER_APP remain pending until the fraud team approves them "
            f"through {surface}, outside USER_APP.")
    common = dict(value=text, evidence=text, confidence=1)
    prompts = models(monkeypatch, {
        # Simulate the reported leakage: a real quote attached to a wrongly
        # scoped actor must not suppress independently supported process facts.
        "ACTOR": [dict(key="secondary_users", roles=["fraud_team"], **common)],
        "WORKFLOW": [dict(key="downstream_dependency", **common)],
        "RULES": [dict(topic="BUSINESS_RULES", key="approval_rules", **common)],
    })
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    result = tracker.knowledge_tracker_node(state(text))
    records = result["discovered_knowledge"]
    assert {item.key for item in records} == {"downstream_dependency", "approval_rules"}
    assert all(item.scope == S.USER_APP and item.evidence == text and surface in item.value
               for item in records)
    for name in ("WORKFLOW", "RULES"):
        assert "without declaring that participant a current-app user" in prompts[name]
        assert "Do not invent its surface when unspecified" in prompts[name]
    assert "candidate's assigned scope is\n  not evidence" in prompts["GROUNDING"]
    assert "Evaluate those process candidates independently" in prompts["GROUNDING"]


def test_explicit_current_scope_access_is_not_blocked_by_other_scope_membership(monkeypatch):
    other = actor("fraud_team", S.ADMIN_DASHBOARD, "The fraud team uses the admin dashboard.")
    text = "The fraud team also uses USER_APP to review suspicious payments."
    models(monkeypatch, {"ACTOR": [dict(key="secondary_users", roles=["fraud_team"],
        value=text, evidence=text, confidence=1)]})
    extracted = tracker.extract_passes(text, state(text, existing=[other]), S.USER_APP)
    assert len(extracted) == 1
    assert extracted[0].scope == S.USER_APP and extracted[0].roles == ["fraud_team"]
    assert extracted[0].evidence == text
    assert other.scope == S.ADMIN_DASHBOARD


@pytest.mark.parametrize("initial_discovery", [True, False])
def test_unspecified_fraud_team_surface_remains_unresolved(monkeypatch, initial_discovery):
    text = "A fraud team reviews suspicious payments."
    value = text + " Current-application membership is unresolved."
    models(monkeypatch, {"ACTOR": [dict(
        key="secondary_users", roles=["fraud_team"], value=value, evidence=text,
        confidence=1, knowledge_state=K.INFERRED)]})
    existing = [] if initial_discovery else [
        actor("customer", S.USER_APP, "Customers use USER_APP.")]
    initial = state(text, existing=existing)
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    assert len(extracted) == 1 and extracted[0].knowledge_state == K.INFERRED
    assert extracted[0].value == value and extracted[0].evidence == text
    context = tracker.confirmed_actor_context(
        {**initial, "discovered_knowledge": [*existing, *extracted]}, S.USER_APP)
    assert not any("fraud_team" in entry["roles"] for entry in context)


LIVE_CASES = [
    ("A fraud team reviews suspicious payments.", "unknown"),
    ("An admin reviews disputes.", "unknown"),
    ("A fraud team reviews suspicious payments in ADMIN_APP.", "other"),
    ("A fraud team reviews suspicious payments through a separate back-office tool.", "other"),
    ("A fraud team reviews suspicious payments as an offline process.", "other"),
    ("A fraud team reviews suspicious payments in an external system.", "other"),
    ("A fraud team reviews suspicious payments on a third-party platform.", "other"),
    ("The fraud team uses USER_APP to review suspicious payments.", "current"),
]


@pytest.mark.skipif(os.environ.get("RUN_LIVE_ACTOR_RESOLUTION") != "1",
                    reason="Requires the configured live OpenAI model (uses API credits)")
@pytest.mark.parametrize("initial_discovery", [True, False])
@pytest.mark.parametrize("text,expected", LIVE_CASES)
def test_live_actor_surface_membership(text, expected, initial_discovery):
    existing = [] if initial_discovery else [
        actor("customer", S.USER_APP, "Customers use USER_APP.")]
    initial = state(text, existing=existing)
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    accepted = tracker.ground_items(extracted, text, initial)
    for items in (extracted, accepted):
        participants = [item for item in items if item.key in ("primary_users", "secondary_users")
                        and item.roles != ["customer"] and not item.absence]
        if expected == "current":
            assert any(item.roles == ["fraud_team"] and item.knowledge_state == K.CONFIRMED
                       for item in participants)
        elif expected == "unknown":
            assert participants and all(item.knowledge_state == K.INFERRED for item in participants)
            assert all(item.evidence in text for item in participants)
        else:
            assert participants == []


@pytest.mark.skipif(os.environ.get("RUN_LIVE_ACTOR_RESOLUTION") != "1",
                    reason="Requires the configured live OpenAI model (uses API credits)")
def test_live_cross_surface_approval_is_process_knowledge():
    text = ("Payments in USER_APP remain pending until a fraud team approves them "
            "offline; the team does not use USER_APP.")
    initial = state(text)
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    accepted = tracker.ground_items(extracted, text, initial)
    assert not any(item.roles for item in accepted)
    assert any(item.key in ("downstream_dependency", "approval_rules") for item in accepted)
    assert all(item.scope == S.USER_APP and item.evidence in text for item in accepted)


def test_existing_actor_is_not_redeclared_from_non_identity_rule(monkeypatch):
    text = (
        "Before the escrow is funded, either customer can propose changes to the "
        "transaction terms, but the other party has to accept those changes."
    )
    existing = actor("customer", S.USER_APP, "Customers use USER_APP.").model_copy(
        update={"key": "primary_users"}
    )
    models(monkeypatch, {"ACTOR": [dict(
        key="primary_users", roles=["customer"], aliases=[],
        value=text, evidence=text, confidence=1,
    )]})
    initial = state(text, existing=[existing])
    initial.update(current_topic=T.BUSINESS_RULES, current_gap="ownership_rules")

    extracted = tracker.extract_passes(text, initial, S.USER_APP)

    assert not any(item.key in ("primary_users", "secondary_users") for item in extracted)


def test_existing_actor_alias_only_capability_is_not_identity_evidence(monkeypatch):
    text = "A buyer can fund the escrow and confirm completion."
    existing = actor("customer", S.USER_APP, "Customers use USER_APP.").model_copy(
        update={"key": "primary_users"}
    )
    models(monkeypatch, {"ACTOR": [dict(
        key="primary_users", roles=["customer"], aliases=["buyer"],
        value=text, evidence=text, confidence=1,
    )]})
    initial = state(text, existing=[existing])
    initial.update(current_topic=T.USER_ROLES, current_gap="permissions::customer")

    extracted = tracker.extract_passes(text, initial, S.USER_APP)

    assert not any(item.key in ("primary_users", "secondary_users") for item in extracted)


def test_explicit_existing_actor_alias_relationship_remains_admissible(monkeypatch):
    text = "A customer can act as a buyer in one transaction."
    existing = actor("customer", S.USER_APP, "Customers use USER_APP.").model_copy(
        update={"key": "primary_users"}
    )
    models(monkeypatch, {"ACTOR": [dict(
        key="primary_users", roles=["customer"], aliases=["buyer"],
        value=text, evidence=text, confidence=1,
    )]})
    initial = state(text, existing=[existing])
    initial.update(current_topic=T.BUSINESS_RULES, current_gap="ownership_rules")

    extracted = tracker.extract_passes(text, initial, S.USER_APP)

    actors = [item for item in extracted if item.key == "primary_users"]
    assert len(actors) == 1
    assert actors[0].roles == ["customer"]
    assert actors[0].aliases == {"customer": ["buyer"]}
