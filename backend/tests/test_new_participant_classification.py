"""Later participants must not become confirmed app users from job titles alone.

Offline fixtures exercise state/registry boundaries and audit payloads. Set
RUN_LIVE_ACTOR_RESOLUTION=1 to also check the production model's classification.
"""
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.extraction_passes import ActorFact, normalize_fact
from agents.interview_planner import inferred_evidence_for_gap
from agents.semantic_validation import GroundingResult
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeState as K


def actor(role, text, status=K.CONFIRMED):
    return dict(key="secondary_users", roles=[role], evidence=text, confidence=1,
                value=text if status == K.CONFIRMED else
                f"{text} Current-application membership is unresolved.",
                knowledge_state=status)


def existing_actor():
    return normalize_fact(ActorFact.model_validate(dict(
        key="primary_users", roles=["customer"], value="customers",
        evidence="Customers use this application.", confidence=1)),
        T.USER_ROLES, S.USER_APP, 1)


def state(text):
    return dict(messages=[HumanMessage(content=text)], current_topic=T.USER_ROLES,
                current_gap="secondary_users", discovery_scope=S.USER_APP,
                discovered_knowledge=[existing_actor()], topic_status={}, turn_count=2)


def models(monkeypatch, outputs):
    prompts = {}

    def invoke(name, messages):
        prompts[name] = messages[0].content
        return {"items": outputs.get(name, [])}

    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name, *_ in tracker.PASSES})
    return prompts


@pytest.mark.parametrize("role", [
    "admin", "mediator", "courier", "auditor", "manager", "agent", "vendor",
    "reviewer", "technician", "payment_processor",
])
def test_unresolved_participant_is_preserved_but_cannot_own_actions(monkeypatch, role):
    text = f"If there is a dispute, a {role.replace('_', ' ')} reviews the evidence."
    tentative = actor(role, text, K.INFERRED)
    prompts = models(monkeypatch, {
        "ACTOR": [tentative],
        "RESPONSIBILITY": [dict(key="responsibilities", role=role,
                                value="reviews the evidence if there is a dispute",
                                evidence=text, confidence=1)],
    })
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)

    def audit(name, schema, instruction, payload):
        assert name == "GROUNDING"
        assert "actor_classification includes candidate proposals" in instruction
        assert payload["actor_classification"] == {
            "primary_users": ["customer"], "secondary_users": []}
        assert len(payload["candidates"]) == 1
        candidate = payload["candidates"][0]
        assert candidate["knowledge_state"] == "INFERRED"
        assert candidate["roles"] == [role]
        assert payload["evidence_quotes"]["0"] == text
        return GroundingResult(supported_ids=[0],
                               evidence_categories={"0": ["USER_ROLES.secondary_users"]})

    monkeypatch.setattr(tracker, "semantic_decision", audit)
    initial = state(text)
    result = tracker.knowledge_tracker_node(initial)
    records = result["discovered_knowledge"]
    unresolved = [item for item in records if item.knowledge_state == K.INFERRED]
    assert len(unresolved) == 1
    assert unresolved[0].roles == [role] and unresolved[0].evidence == text
    assert not any(item.key == "responsibilities" for item in records)
    assert 'Canonical actor IDs: ["customer"]' in prompts["RESPONSIBILITY"]
    assert "knowledge_state=\"INFERRED\"" in prompts["ACTOR"]
    assert "other application/back-office" in prompts["ACTOR"]
    assert "external workflow dependency" in prompts["ACTOR"]
    updated = {**initial, **result}
    assert inferred_evidence_for_gap(updated, T.USER_ROLES, "secondary_users") == [tentative["value"]]
    assert [item["roles"] for item in tracker.confirmed_actor_context(updated, S.USER_APP)] == [["customer"]]
    # Persisted uncertainty also stays out of the registry on a later turn.
    tracker.extract_passes(text, updated, S.USER_APP)
    assert 'Canonical actor IDs: ["customer"]' in prompts["ACTOR"]


def test_explicit_current_app_user_can_own_actions(monkeypatch):
    text = "Vendors also log into this app and manage assigned orders."
    prompts = models(monkeypatch, {
        "ACTOR": [actor("vendor", text)],
        "RESPONSIBILITY": [dict(key="responsibilities", role="vendor",
                                value="manage assigned orders", evidence=text, confidence=1)],
    })
    extracted = tracker.extract_passes(text, state(text), S.USER_APP)
    assert any(item.roles == ["vendor"] and item.knowledge_state == K.CONFIRMED
               for item in extracted)
    assert any(item.key == "responsibilities" and item.role == "vendor" for item in extracted)
    assert 'Canonical actor IDs: ["customer", "vendor"]' in prompts["RESPONSIBILITY"]


@pytest.mark.parametrize("text,status", [
    ("If there is a dispute, a mediator reviews the evidence.", K.CONFIRMED),
    ("A mediator uses a separate back-office app, not this app.", K.CONFIRMED),
    ("A mediator reviews evidence offline and never uses this app.", K.INFERRED),
])
def test_unsupported_membership_still_requires_grounding(monkeypatch, text, status):
    item = normalize_fact(ActorFact.model_validate(actor("mediator", text, status)),
                          T.USER_ROLES, S.USER_APP, 2)

    def audit(name, schema, instruction, payload):
        assert "Reject\n  an unsupported CONFIRMED declaration" in instruction
        assert "explicitly places it outside this application" in instruction
        assert payload["confirmed_actor_context"][0]["roles"] == ["customer"]
        return GroundingResult(supported_ids=[], evidence_categories={"0": []},
                               rejection_reasons={"0": "No current-application membership"})

    monkeypatch.setattr(tracker, "semantic_decision", audit)
    assert tracker.ground_items([item], text, state(text)) == []


LIVE_CASES = [
    ("If there is a dispute, a mediator reviews the evidence.", "mediator", "ambiguous"),
    ("A technician inspects faulty equipment.", "technician", "ambiguous"),
    ("Vendors also log into this app and manage assigned orders.", "vendor", "current"),
    ("Auditors view records in this application's dashboard.", "auditor", "current"),
    ("Reviewers submit decisions through this app.", "reviewer", "current"),
    ("Managers use a separate back-office dashboard and do not use this app.", "manager", "other"),
    ("The workflow waits for an external payment processor's API response; it has no app account.",
     "payment_processor", "external"),
    ("In this app, customers are also called account holders.", "account_holder", "alias"),
    ("Customers act as senders or recipients depending on the transfer.", "sender", "capacity"),
]


@pytest.mark.skipif(os.environ.get("RUN_LIVE_ACTOR_RESOLUTION") != "1",
                    reason="Requires the configured live Ollama model")
@pytest.mark.parametrize("text,role,resolution", LIVE_CASES)
def test_live_new_participant_membership(text, role, resolution):
    initial = state(text)
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    accepted = tracker.ground_items(extracted, text, initial)
    for items in (extracted, accepted):
        actors = [item for item in items if item.key in ("primary_users", "secondary_users")
                  and not item.absence]
        new_actors = [item for item in actors if item.roles != ["customer"]]
        if resolution == "current":
            assert any(item.roles == [role] and item.knowledge_state == K.CONFIRMED
                       for item in new_actors)
        elif resolution == "ambiguous":
            assert any(item.roles == [role] and item.knowledge_state == K.INFERRED
                       for item in new_actors)
            assert not any(item.knowledge_state == K.CONFIRMED for item in new_actors)
        else:
            assert new_actors == []
        assert all(item.evidence in text for item in actors)
