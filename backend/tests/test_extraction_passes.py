"""Pass isolation and canonical merge with deterministic model responses."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.llm_errors import ExtractionFailed
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeItem


CARE = ("I want to build CareConnect, a platform that helps patients find and book "
        "appointments with healthcare professionals such as general doctors, "
        "dermatologists, dentists, physiotherapists, and nutritionists. Patients "
        "should be able to describe their health concern, search for suitable "
        "healthcare providers, compare their profiles and availability, book "
        "appointments, communicate with the provider, receive appointment reminders, "
        "and pay through the platform. Healthcare providers should be able to manage "
        "their profiles, specialties, schedules, appointments, and payments.")
FIX = ("I want to build FixMate, a platform that helps people find and book trusted "
       "local artisans like electricians, plumbers, cleaners, painters, and appliance "
       "repair technicians. Users should be able to describe what they need, find "
       "suitable service providers nearby, compare them, book a service, communicate "
       "with the provider, and pay through the platform.")
ADMIN = "Customers book artisans. Artisans accept jobs and complete them. Admins verify artisans and resolve disputes."


def item(key, value, evidence, **extra):
    return dict(key=key, value=value, evidence=evidence, confidence=1, **extra)


def run(monkeypatch, text, outputs, prompt_sink=None):
    # This helper tests the extraction passes; semantic auditing has its own suite.
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_: items)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    calls = {}
    def model(name):
        def invoke(messages):
            calls[name] = messages[0].content
            assert isinstance(messages[-1], HumanMessage)
            assert messages[-1].content == text
            output = outputs.get(name, [])
            if isinstance(output, Exception):
                raise output
            return {"items": output}
        return SimpleNamespace(invoke=invoke)
    monkeypatch.setattr(tracker, "extraction_models", lambda: {name: model(name) for name, *_ in tracker.PASSES})
    result = tracker.knowledge_tracker_node(dict(messages=[HumanMessage(content=text)],
        current_topic=DiscoveryTopic.USER_ROLES, discovery_scope=DiscoveryScope.USER_APP,
        discovered_knowledge=[], topic_status={}, turn_count=1))
    assert len(calls) == len(tracker.PASSES)
    if prompt_sink is not None:
        prompt_sink.update(calls)
    return result["discovered_knowledge"]


def test_careconnect(monkeypatch):
    prompts = {}
    patient = CARE[CARE.index("Patients should"):CARE.index(" Healthcare providers should")]
    records = run(monkeypatch, CARE, {
        "ACTOR": [item("primary_users", "patient", "patients find and book appointments", roles=["patient"]),
                  item("primary_users", "healthcare provider", "healthcare professionals such as general doctors", roles=["healthcare provider"])],
        "RESPONSIBILITY": [item("responsibilities", "describe concern, search, compare, book, communicate, receive reminders, pay", patient, role="patient"),
                           item("responsibilities", "manage profiles, specialties, schedules, appointments, and payments", "Healthcare providers should be able to manage their profiles, specialties, schedules, appointments, and payments.", role="healthcare provider")],
        "WORKFLOW": [item("workflow_steps", "describe concern, search, compare, book, communicate, receive reminders, pay", "Patients should be able to describe their health concern, search for suitable healthcare providers, compare their profiles and availability, book appointments, communicate with the provider, receive appointment reminders, and pay through the platform.")],
    }, prompts)
    assert {r.roles[0] for r in records if r.key == "primary_users"} == {"patient", "healthcare_provider"}
    assert any(r.key == "responsibilities" and r.role == "healthcare_provider" for r in records)
    assert any(r.key == "responsibilities" and r.role == "patient" for r in records)
    assert any(r.key == "workflow_steps" for r in records)
    assert "actions, activities, duties, capabilities, or processes" in prompts["RESPONSIBILITY"]
    assert '{"items": []}' in prompts["RESPONSIBILITY"]


def test_fixmate(monkeypatch):
    records = run(monkeypatch, FIX, {
        "ACTOR": [item("primary_users", "customer", "people find and book trusted local artisans", roles=["customer"]),
                  item("primary_users", "artisan", "local artisans like electricians, plumbers, cleaners, painters, and appliance repair technicians", roles=["artisan"], aliases=["service provider"])],
        "WORKFLOW": [item("workflow_steps", "describe need, find, compare, book, communicate, pay", "Users should be able to describe what they need, find suitable service providers nearby, compare them, book a service, communicate with the provider, and pay through the platform.")],
    })
    assert {r.roles[0] for r in records if r.key == "primary_users"} == {"customer", "artisan"}
    assert not any(r.key == "secondary_users" for r in records)


def test_admin(monkeypatch):
    records = run(monkeypatch, ADMIN, {
        "ACTOR": [item("primary_users", role, quote, roles=[role]) for role, quote in
                  [("customer", "Customers book artisans"), ("artisan", "Artisans accept jobs and complete them")]] +
                 [item("secondary_users", "admin", "Admins verify artisans and resolve disputes", roles=["admin"])],
        "RESPONSIBILITY": [item("responsibilities", "accept and complete jobs", "Artisans accept jobs and complete them", role="artisan"),
                           item("responsibilities", "verify artisans and resolve disputes", "Admins verify artisans and resolve disputes", role="admin")],
    })
    assert [(r.key, r.roles[0]) for r in records if r.roles] == [
        ("primary_users", "customer"), ("primary_users", "artisan"), ("secondary_users", "admin")]
    assert {r.role for r in records if r.key == "responsibilities"} == {"artisan", "admin"}


def test_structured_pass_failure_stops_turn_at_durable_boundary(monkeypatch, capsys):
    with pytest.raises(ExtractionFailed):
        run(monkeypatch, ADMIN, {
            "ACTOR": ValueError("malformed structured output"),
            "RESPONSIBILITY": [item("responsibilities", "accept jobs", "Artisans accept jobs", role="artisan"),
                               item("permissions", "invented", "fabricated quote", role="admin")],
        })
    output = capsys.readouterr().out
    assert "ACTOR EXTRACTION FAILED" in output
    assert "RESPONSIBILITY REJECTED" not in output


@pytest.mark.parametrize("text,actors,goal_role,goal_value,goal_evidence", [
    (CARE, [("patient", "patients find and book appointments"),
            ("healthcare_provider", "Healthcare providers should be able to manage their profiles, specialties, schedules, appointments, and payments.")],
     "patient", "find and book appointments with healthcare professionals",
     "patients find and book appointments with healthcare professionals"),
    (FIX, [("customer", "people find and book trusted local artisans"),
           ("artisan", "local artisans like electricians, plumbers, cleaners, painters, and appliance repair technicians")],
     "customer", "find and book trusted local artisans",
     "people find and book trusted local artisans"),
])
def test_goals_use_current_actor_classification(monkeypatch, text, actors, goal_role, goal_value, goal_evidence):
    prompts = {}
    records = run(monkeypatch, text, {
        "ACTOR": [item("primary_users", role, quote, roles=[role]) for role, quote in actors],
        "GOAL": [item("primary_user_goals", goal_value, goal_evidence, role=goal_role),
                 item("secondary_user_goals", "manage profiles", actors[1][1], role=actors[1][0])],
    }, prompts)
    assert [r.role for r in records if r.key == "primary_user_goals"] == [goal_role]
    assert not any(r.key == "secondary_user_goals" for r in records)
    assert not any(r.key == "success_criteria" for r in records)
    assert all(role in prompts["GOAL"] for role, _ in actors)
    assert "Confirmed secondary roles: none" in prompts["GOAL"]
    assert "a desired RESULT that an actor wants" in prompts["GOAL"]
    assert "an explicitly stated reason or problem explaining WHY" in prompts["GOAL"]


def test_responsibility_evidence_is_never_repaired(monkeypatch, capsys):
    sentence = "Healthcare providers should be able to manage their profiles, specialties, schedules, appointments, and payments."
    records = run(monkeypatch, CARE, {
        "ACTOR": [item("primary_users", "healthcare_provider", sentence, roles=["healthcare_provider"])],
        "RESPONSIBILITY": [item("responsibilities", "manage schedules",
                                "Healthcare providers should be able to manage their schedules",
                                role="healthcare_provider"),
                           item("responsibilities", "manage profiles, specialties, schedules, appointments, and payments",
                                sentence, role="healthcare_provider")],
    })
    responsibilities = [r for r in records if r.key == "responsibilities"]
    assert len(responsibilities) == 1 and responsibilities[0].evidence == sentence
    assert "Evidence is not an exact substring" in capsys.readouterr().out


def test_goal_prompt_uses_actor_context_without_replaying_action_candidates(monkeypatch):
    quote = "Healthcare providers should be able to manage their profiles, specialties, schedules, appointments, and payments."
    prompts = {}
    records = run(monkeypatch, CARE, {
        "ACTOR": [item("primary_users", "patient", "patients find and book appointments", roles=["patient"]),
                  item("primary_users", "healthcare_provider", quote, roles=["healthcare_provider"])],
        "RESPONSIBILITY": [item("responsibilities", "manage profiles, specialties, schedules, appointments, and payments",
                                quote, role="healthcare_provider")],
        "GOAL": [item("primary_user_goals", "find and book appointments with healthcare professionals",
                      "helps patients find and book appointments with healthcare professionals", role="patient")],
    }, prompts)
    goal_prompt = prompts["GOAL"]
    assert 'Confirmed primary roles: patient, healthcare_provider' in goal_prompt
    assert 'Responsibilities/permissions already extracted' not in goal_prompt
    assert 'STRICT OUTCOME REQUIREMENT' in goal_prompt
    assert 'Responsibilities/permissions already extracted' not in prompts["WORKFLOW"]
    assert [record.role for record in records if record.key == "primary_user_goals"] == ["patient"]


def test_goal_prompt_keeps_workflow_actions_out_of_goals(monkeypatch):
    prompts = {}
    run(monkeypatch, CARE, {
        "ACTOR": [item("primary_users", "patient", "patients find and book appointments", roles=["patient"])],
        "WORKFLOW": [item("workflow_steps", "describe, compare, book, message and pay",
                          "Patients should be able to describe their health concern, search for suitable healthcare providers, compare their profiles and availability, book appointments, communicate with the provider, receive appointment reminders, and pay through the platform.")],
    }, prompts)
    assert "Do not break a feature or workflow sentence into individual goals" in prompts["GOAL"]
    assert "If it only states an available action, omit it" in prompts["GOAL"]
    assert '"key": "workflow_steps"' not in prompts["GOAL"]
    assert "extract the stated purpose outcome only" in prompts["GOAL"]


def test_permission_prompt_excludes_actor_only_answer(monkeypatch):
    prompts = {}
    run(monkeypatch, "No, just patients and healthcare providers for now.", {}, prompts)
    assert "Statements that only identify which actors exist" in prompts["PERMISSION"]
    assert "contains no permission" in prompts["PERMISSION"]


def test_distinct_explicit_provider_outcome_can_coexist_with_responsibility(monkeypatch):
    text = "Healthcare providers want to keep their schedules accurate so patients only book available times."
    records = run(monkeypatch, text, {
        "ACTOR": [item("primary_users", "healthcare_provider", text, roles=["healthcare_provider"])],
        "RESPONSIBILITY": [item("responsibilities", "keep schedules accurate", text,
                                role="healthcare_provider")],
        "GOAL": [item("primary_user_goals", "keep schedules accurate so patients only book available times",
                      text, role="healthcare_provider")],
    })
    assert {record.key for record in records if record.role == "healthcare_provider"} == {
        "responsibilities", "primary_user_goals"}


def test_explicit_success_signal_is_still_accepted():
    quote = "The appointment is confirmed."
    criterion = KnowledgeItem(topic=DiscoveryTopic.USER_GOALS, scope=DiscoveryScope.USER_APP,
                              key="success_criteria", value=quote, evidence=quote,
                              role="patient", confidence=1)
    assert tracker.validate_extraction(criterion, quote)[0]


def test_success_criterion_semantic_support_is_audited(monkeypatch):
    from agents.semantic_validation import GroundingResult
    quote = "Success means the appointment is confirmed."
    unsupported = KnowledgeItem(topic=DiscoveryTopic.USER_GOALS, scope=DiscoveryScope.USER_APP,
                                key="success_criteria", value="The payment is complete.",
                                evidence=quote, role="patient", confidence=1)
    supported = unsupported.model_copy(update={"value": "The appointment is confirmed."})
    # Source-quote validation alone cannot prove that a different value follows.
    assert tracker.validate_extraction(unsupported, quote)[0]
    monkeypatch.setattr(tracker, "semantic_decision", lambda *_: GroundingResult(evidence_categories={"0": ["USER_GOALS.success_criteria"]}, supported_ids=[1]))
    assert tracker.ground_items([unsupported, supported], quote, {}) == [supported]


def test_explicit_motivation_with_confirmed_actor_is_retained(monkeypatch):
    text = "Customers use the app because calling several restaurants to compare prices takes too long."
    records = run(monkeypatch, text, {
        "ACTOR": [item("primary_users", "customer", "Customers use the app", roles=["customer"])],
        "GOAL": [item("motivations", "calling several restaurants to compare prices takes too long",
                      text, role="customer")],
    })
    assert [(record.key, record.role) for record in records if record.key == "motivations"] == [
        ("motivations", "customer")]
