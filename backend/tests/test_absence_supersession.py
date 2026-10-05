"""Confirmed absence requires an independently supported superseding decision."""
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.semantic_validation import GroundingResult
from agents.discovery_coverage import GapConfirmation
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K


QUESTION = "Besides customers, will anyone else use the user app?"
ABSENCE_QUOTE = "We only have customers."


def absent(**changes):
    fields = dict(topic=T.USER_ROLES, scope=S.USER_APP, key="secondary_users",
                  value="none", absence="none", evidence=ABSENCE_QUOTE,
                  source_question=QUESTION, source_turn=1, confidence=1)
    return KnowledgeItem(**{**fields, **changes})


def participant(role, text, **changes):
    fields = dict(topic=T.USER_ROLES, scope=S.USER_APP, key="secondary_users",
                  value=f"{role} uses the app", roles=[role], evidence=text,
                  source_turn=2, confidence=1)
    return KnowledgeItem(**{**fields, **changes})


def state(text):
    customer = participant("customer", "Customers use the app", key="primary_users", source_turn=1)
    return dict(messages=[AIMessage(content=QUESTION), HumanMessage(content=text)],
                discovered_knowledge=[customer, absent()], current_topic=T.USER_ROLES,
                discovery_scope=S.USER_APP, turn_count=2)


def setup(monkeypatch, items, allowed=False, failure=False):
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: items)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    # Facts arrive grounded; this suite specifically tests replacement authority.
    monkeypatch.setattr(tracker, "ground_items", lambda candidates, *_: candidates)
    calls = []

    def decide(name, schema, instruction, payload):
        if name == "GAP_CONFIRMATION":
            return GapConfirmation(
                confirmed=True,
                evidence=payload["latest_response"],
                confidence=1,
            )
        assert name == "GROUNDING"
        assert "Review replacement of a confirmed whole-field absence" in instruction
        assert "Business-process participation alone" in instruction
        assert "general correction intent flag" in instruction
        calls.append(payload)
        if failure:
            raise ValueError("audit unavailable")
        candidate = payload["candidates"][0]
        support = allowed(candidate) if callable(allowed) else allowed
        return GroundingResult(supported_ids=[0] if support else [],
            evidence_categories={"0": [f"{candidate['topic']}.{candidate['key']}"] if support else []},
            confirmed_absence_ids=[0] if support and candidate.get("absence") else [],
            rejection_reasons={} if support else {"0": "No superseding decision"})

    monkeypatch.setattr(tracker, "semantic_decision", decide)
    return calls


def test_customers_only_answer_establishes_confirmed_absence(monkeypatch):
    from test_gap_absence import models

    initial = state(ABSENCE_QUOTE)
    initial["discovered_knowledge"] = initial["discovered_knowledge"][:1]
    initial["current_gap"] = "secondary_users"
    models(monkeypatch, evidence=ABSENCE_QUOTE)
    result = tracker.knowledge_tracker_node(initial)
    absence = next(item for item in result["discovered_knowledge"] if item.key == "secondary_users")
    assert absence.absence == "none" and absence.knowledge_state == K.CONFIRMED
    assert absence.evidence == ABSENCE_QUOTE and absence.source_question == QUESTION


@pytest.mark.parametrize("correction_flag", [False, True])
def test_incidental_participant_cannot_override_explicit_absence(monkeypatch, correction_flag):
    text = "A courier delivers the package."
    duty = KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key="responsibilities",
                         role="courier", value="deliver package", evidence=text, confidence=1)
    process = duty.model_copy(update={"topic": T.CORE_WORKFLOW, "key": "downstream_dependency", "role": None})
    calls = setup(monkeypatch, [participant("courier", text), duty, process])
    initial = {**state(text), "is_correction": correction_flag}
    result = tracker.knowledge_tracker_node(initial)
    assert absent() in result["discovered_knowledge"]
    assert not any(item.roles == ["courier"] or item.role == "courier"
                   for item in result["discovered_knowledge"])
    assert process in result["discovered_knowledge"] or any(
        item.key == "downstream_dependency" for item in result["discovered_knowledge"])
    assert "topic_status" not in result
    assert "topic_maturity" not in result
    assert result["superseded_knowledge"] == []
    assert len(calls) == 1 and calls[0]["prior_absences"] == [absent().model_dump(mode="json")]


@pytest.mark.parametrize("text", [
    "Actually, vendors also log into the same app.",
    "Vendors also log into USER_APP.",
])
def test_explicit_supersession_archives_old_provenance(monkeypatch, text):
    calls = setup(monkeypatch, [participant("vendor", text)], allowed=True)
    initial = state(text)
    original = deepcopy(initial)
    result = tracker.knowledge_tracker_node(initial)
    active = [item for item in result["discovered_knowledge"] if item.key == "secondary_users"]
    assert len(active) == 1 and active[0].roles == ["vendor"] and not active[0].absence
    history = result["superseded_knowledge"]
    assert history == [dict(fact=absent().model_dump(mode="json"),
                            superseded_by=active[0].model_dump(mode="json"))]
    assert history[0]["fact"]["source_question"] == QUESTION
    assert history[0]["fact"]["evidence"] == ABSENCE_QUOTE
    assert history[0]["superseded_by"]["evidence"] == text
    assert history[0]["superseded_by"]["source_turn"] == 2
    json.dumps(history)  # Durable, JSON-compatible provenance snapshots.
    assert not any("secondary_users: none" in line for line in result["product_model"]["USER_ROLES"])
    assert initial == original
    repeated = tracker.knowledge_tracker_node({**initial, **result})
    assert repeated["superseded_knowledge"] == history and len(calls) == 1


@pytest.mark.parametrize("mode", ["unavailable", "missing_category", "fabricated_quote", "inferred"])
def test_replacement_fails_closed_and_inference_does_not_delete_absence(monkeypatch, mode):
    text = "Actually, vendors also log into the same app."
    candidate = participant("vendor", text)
    if mode == "fabricated_quote":
        candidate.evidence = "A fabricated correction"
    if mode == "inferred":
        candidate.knowledge_state = K.INFERRED
    calls = setup(monkeypatch, [candidate], allowed=True, failure=mode == "unavailable")
    if mode == "missing_category":
        monkeypatch.setattr(tracker, "semantic_decision", lambda *_:
                            GroundingResult(supported_ids=[0], evidence_categories={}))
    result = tracker.knowledge_tracker_node({**state(text), "is_correction": True})
    assert absent() in result["discovered_knowledge"]
    assert not any(item.roles == ["vendor"] and item.knowledge_state == K.CONFIRMED
                   for item in result["discovered_knowledge"])
    assert result["superseded_knowledge"] == []
    if mode in ("inferred", "fabricated_quote"):
        assert calls == []


@pytest.mark.parametrize("reverse", [False, True])
def test_each_sibling_needs_authority_even_after_first_replaces_absence(monkeypatch, reverse):
    correction = "Actually, vendors also log into the same app."
    incidental = "A courier delivers the package."
    items = [participant("vendor", correction), participant("courier", incidental)]
    if reverse:
        items.reverse()
    calls = setup(monkeypatch, items, allowed=lambda candidate: candidate["roles"] == ["vendor"])
    result = tracker.knowledge_tracker_node(state(correction + " " + incidental))
    active = [item for item in result["discovered_knowledge"] if item.key == "secondary_users"]
    assert len(active) == 1 and active[0].roles == ["vendor"]
    assert len(calls) == 2 and len(result["superseded_knowledge"]) == 1


def test_repeat_absence_keeps_original_source_and_no_archive(monkeypatch):
    text = "There are still no other users."
    calls = setup(monkeypatch, [absent(evidence=text, source_turn=2)])
    result = tracker.knowledge_tracker_node(state(text))
    assert absent() in result["discovered_knowledge"]
    assert len([item for item in result["discovered_knowledge"] if item.key == "secondary_users"]) == 1
    assert result["superseded_knowledge"] == [] and calls == []


@pytest.mark.parametrize("boundary", ["scope", "owner"])
def test_absence_protection_is_scoped_to_exact_field_and_owner(monkeypatch, boundary):
    text = "Vendors use the admin dashboard."
    initial = state(text)
    if boundary == "scope":
        item = participant("vendor", text, scope=S.ADMIN_DASHBOARD)
    else:
        initial["discovered_knowledge"].append(absent(key="permissions", role="customer"))
        item = KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key="permissions",
                             role="vendor", value="edit orders", evidence=text, confidence=1)
    calls = setup(monkeypatch, [item])
    result = tracker.knowledge_tracker_node(initial)
    assert absent() in result["discovered_knowledge"]
    assert any(candidate.value == item.value for candidate in result["discovered_knowledge"])
    assert calls == [] and result["superseded_knowledge"] == []


@pytest.mark.parametrize("allowed", [False, True])
def test_inference_confirmation_cannot_bypass_replacement_review(monkeypatch, allowed):
    question = "Should vendors also use the same app, replacing the earlier customers-only decision?"
    initial = state("Yes.")
    initial["messages"] = [AIMessage(content=question), HumanMessage(content="Yes.")]
    initial["discovered_knowledge"].append(participant("vendor", "Vendors may use the app", knowledge_state=K.INFERRED))
    initial.update(conversation_intent="confirmation", next_discovery_move="confirm_inference",
                   current_gap="secondary_users",
                   asked_gap=dict(scope=S.USER_APP.value, topic=T.USER_ROLES.value,
                                  gap="secondary_users", question=question))
    calls = setup(monkeypatch, [], allowed=allowed)
    result = tracker.knowledge_tracker_node(initial)
    assert len(calls) == 1 and calls[0]["question"] == question
    if allowed:
        assert absent() not in result["discovered_knowledge"]
        replacement = next(item for item in result["discovered_knowledge"] if item.roles == ["vendor"])
        assert replacement.knowledge_state == K.CONFIRMED
        assert replacement.evidence == "Yes." and replacement.source_question == question
        assert result["superseded_knowledge"][0]["superseded_by"] == replacement.model_dump(mode="json")
    else:
        assert absent() in result["discovered_knowledge"]
        assert result["superseded_knowledge"] == []


def test_replaying_historical_answer_cannot_supersede_newer_absence(monkeypatch):
    historical = "Vendors also use this app."
    calls = setup(monkeypatch, [participant("vendor", historical)], allowed=True)
    initial = state("I already answered this.")
    initial["messages"].insert(0, HumanMessage(content=historical))
    initial["conversation_intent"] = "objection"
    result = tracker.knowledge_tracker_node(initial)
    assert absent() in result["discovered_knowledge"] and calls == []


def test_serialized_legacy_absence_is_protected_and_superseded(monkeypatch):
    text = "Actually, vendors also log into the same app."
    setup(monkeypatch, [participant("vendor", text)], allowed=True)
    initial = state(text)
    legacy = absent(absence=None)
    initial["discovered_knowledge"][-1] = legacy
    result = tracker.knowledge_tracker_node(initial)
    assert legacy not in result["discovered_knowledge"]
    assert result["superseded_knowledge"][0]["fact"] == legacy.model_dump(mode="json")


@pytest.mark.parametrize("topic,key,role,label,old_quote,text", [
    (T.USER_ROLES, "permissions", "customer", "none", "No special permissions.",
     "Actually, customers can view only their own orders."),
    (T.CONSTRAINTS, "time_constraints", None, "not_applicable", "Time constraints do not apply.",
     "Actually, orders must now be completed within two days."),
])
def test_whole_field_absence_replacement_is_not_actor_specific(
        monkeypatch, topic, key, role, label, old_quote, text):
    previous = absent(topic=topic, key=key, role=role, absence=label,
                      value="none" if label == "none" else "not applicable", evidence=old_quote)
    replacement = KnowledgeItem(topic=topic, scope=S.USER_APP, key=key, role=role,
                                value=text, evidence=text, source_turn=2, confidence=1)
    calls = setup(monkeypatch, [replacement], allowed=True)
    initial = state(text)
    initial["discovered_knowledge"].append(previous)
    result = tracker.knowledge_tracker_node(initial)
    assert previous not in result["discovered_knowledge"]
    assert absent() in result["discovered_knowledge"]
    assert calls[0]["prior_absences"] == [previous.model_dump(mode="json")]
    assert result["superseded_knowledge"][0]["fact"] == previous.model_dump(mode="json")
