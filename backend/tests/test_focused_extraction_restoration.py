"""Deterministic checks for focused extraction and planner handoff."""
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.interview_planner import build_gap_info
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeState


CARE = (
    "I want to build CareConnect, a platform that helps patients find and book appointments "
    "with healthcare professionals such as general doctors, dermatologists, dentists, "
    "physiotherapists, and nutritionists. Patients should be able to describe their health "
    "concern, search for suitable healthcare providers, compare their profiles and availability, "
    "book appointments, communicate with the provider, receive appointment reminders, and pay "
    "through the platform. Healthcare providers should be able to manage their profiles, "
    "specialties, schedules, appointments, and payments."
)


def fact(key, value, evidence, **fields):
    return dict(key=key, value=value, evidence=evidence, confidence=0.95,
                knowledge_state="CONFIRMED", **fields)


def run(monkeypatch, message, outputs, gap=None):
    # Keep these pass contract tests independent of the separate semantic audit.
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_: items)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    calls = []
    def model(name):
        def invoke(messages):
            calls.append(name)
            return {"parsed": {"items": outputs.get(name, [])}, "raw": None}
        return SimpleNamespace(invoke=invoke)
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: model(name) for name, *_ in tracker.PASSES})
    state = dict(messages=[HumanMessage(content=message)], current_topic=None,
                 current_gap=gap, discovery_scope=DiscoveryScope.USER_APP,
                 discovered_knowledge=[], topic_status={}, turn_count=1)
    result = tracker.knowledge_tracker_node(state)
    assert calls == ["ACTOR", "RESPONSIBILITY", "PERMISSION", "WORKFLOW", "GOAL", "RULES"]
    return result


def test_careconnect_initial_turn(monkeypatch):
    workflow = CARE[CARE.index("Patients should"):CARE.index(" Healthcare providers should")]
    provider = CARE[CARE.index("Healthcare providers should"):]
    result = run(monkeypatch, CARE, {
        "ACTOR": [fact("primary_users", "patient", "patients find and book appointments",
                       roles=["patient"]),
                  fact("primary_users", "healthcare provider", provider,
                       roles=["healthcare_provider"])],
        "RESPONSIBILITY": [fact("responsibilities", "describe concern, search, compare, book, communicate, receive reminders, pay", workflow, role="patient"),
                           fact("responsibilities", "manage profiles, specialties, schedules, appointments, and payments",
                                provider, role="healthcare_provider")],
        "WORKFLOW": [fact("workflow_steps", "describe concern, search, compare, book, communicate, receive reminders, pay",
                          workflow)],
        "GOAL": [fact("primary_user_goals", "find and book appointments",
                      "helps patients find and book appointments", role="patient")],
    })
    items = result["discovered_knowledge"]
    assert {i.roles[0] for i in items if i.key == "primary_users"} == {"patient", "healthcare_provider"}
    assert [(i.key, i.role) for i in items if i.key == "responsibilities"] == [
        ("responsibilities", "patient"), ("responsibilities", "healthcare_provider")]
    assert not any(i.key == "permissions" for i in items)
    assert any(i.key == "workflow_steps" for i in items)
    assert any(i.key == "primary_user_goals" and i.role == "patient" for i in items)
    assert all(i.knowledge_state == KnowledgeState.CONFIRMED for i in items)
    gaps = build_gap_info(result, DiscoveryTopic.USER_ROLES)
    assert "primary_users" in gaps["known_keys"]
    assert "responsibilities::patient" in gaps["missing_keys"]
    assert "responsibilities::healthcare_provider" in gaps["missing_keys"]


def test_permission_and_capability_are_separate(monkeypatch):
    permission = "Only verified healthcare providers can accept appointments."
    result = run(monkeypatch, permission, {
        "ACTOR": [fact("primary_users", "healthcare provider", permission,
                       roles=["healthcare_provider"])],
        "PERMISSION": [
        fact("permissions", "only verified providers can accept appointments",
             permission, role="healthcare_provider")]})
    assert any(i.key == "permissions" for i in result["discovered_knowledge"])
    capability = "Patients can search providers and book appointments."
    result = run(monkeypatch, capability, {})
    assert not any(i.key == "permissions" for i in result["discovered_knowledge"])


def test_functional_actor_and_exact_evidence(monkeypatch):
    message = "People should be able to find and book artisans."
    result = run(monkeypatch, message, {"ACTOR": [
        fact("primary_users", "customer", message, roles=["customer"]),
        fact("primary_users", "artisan", message, roles=["artisan"]),
        fact("primary_users", "people", message, roles=["people"]),
        fact("primary_users", "artisan", "fabricated evidence", roles=["artisan"])]})
    assert {i.roles[0] for i in result["discovered_knowledge"]} == {"customer", "artisan"}
