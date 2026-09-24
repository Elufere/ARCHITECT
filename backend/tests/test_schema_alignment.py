"""Semantic contracts and extraction -> grounding -> planner integration.

Model responses are deterministic here; replay_schema_alignment.py exercises the
actual configured OpenAI model separately. No stage after model invocation is bypassed.
"""
import json
from types import SimpleNamespace
from typing import get_args

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.discovery_fields import FIELD_DEFINITIONS, OVERLAP_RULES
from agents.extraction_passes import PASSES
from agents.interview_planner import DISCOVERY_TASKS, build_gap_info, interview_planner_node
from agents.question_generator import permission_discovery_guidance
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K, TOPIC_KEY_MAP, TopicStatus
from replay_schema_alignment import CARE, NO_OTHERS


def raw(key, value, evidence, **fields):
    return dict(key=key, value=value, evidence=evidence, confidence=1, **fields)


def actor(role, secondary=False, scope=S.USER_APP):
    return KnowledgeItem(topic=T.USER_ROLES, scope=scope, key="secondary_users" if secondary else "primary_users",
                         value=role, roles=[role], evidence=role, confidence=1)


def run(monkeypatch, text, outputs, *, existing=(), topic=None, gap=None, question="",
        reject=(), absence="unresolved", scope=S.USER_APP):
    calls = []
    def invoke(name, messages):
        calls.append((name, messages))
        if name == "GAP_ANSWER":
            return dict(resolution=absence, evidence=text, confidence=1,
                        value=text if absence == "policy" else None)
        if name == "GROUNDING":
            groups = json.loads(messages[-1].content)["evidence_groups"]
            candidates = [{**item, "evidence_id": group["evidence_id"]} for group in groups for item in group["candidates"]]
            accepted = [i["id"] for i in candidates if (i["key"], i["value"]) not in reject]
            categories = {str(c["evidence_id"]): [f"{i['topic']}.{i['key']}" for i in candidates
                          if i["evidence_id"] == c["evidence_id"]] for c in candidates}
            return dict(evidence_categories=categories, supported_ids=accepted,
                        rejection_reasons={str(i["id"]): "Unsupported claim" for i in candidates if i["id"] not in accepted},
                        confirmed_absence_ids=[i["id"] for i in candidates if i.get("absence") and i["id"] in accepted])
        return dict(items=outputs.get(name, []))
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*[p[0] for p in PASSES], "GAP_ANSWER", "GROUNDING"]})
    state = dict(messages=[AIMessage(content=question), HumanMessage(content=text)],
                 current_topic=topic, current_gap=gap, discovered_knowledge=list(existing),
                 discovery_scope=scope, topic_status={}, topic_maturity={}, turn_count=0)
    state.update(tracker.knowledge_tracker_node(state))
    return state, calls


def test_careconnect_shared_evidence_and_two_turn_planner_progression(monkeypatch):
    purpose, patient, provider = CARE.split("\n\n")
    actions = "describe health concern, search, compare, book, communicate, receive reminders, pay"
    outputs = {
        "ACTOR": [raw("primary_users", "patient", purpose, roles=["patient"]),
                  raw("primary_users", "healthcare_provider", purpose, roles=["healthcare_provider"])],
        "RESPONSIBILITY": [raw("responsibilities", actions, patient, role="patient"),
                           raw("responsibilities", "manage profiles, specialties, schedules, appointments and payments", provider, role="healthcare_provider")],
        "WORKFLOW": [raw("workflow_steps", actions, patient)],
        "GOAL": [raw("primary_user_goals", "find and book appointments", purpose, role="patient")],
    }
    state, calls = run(monkeypatch, CARE, outputs)
    state.update(interview_planner_node(state))
    assert state["current_gap"] == "secondary_users"
    assert all(f"responsibilities::{r}" not in state["missing_keys"] for r in ("patient", "healthcare_provider"))
    shared = [i for i in state["discovered_knowledge"] if i.evidence == patient]
    assert {(i.topic, i.key) for i in shared} == {(T.USER_ROLES, "responsibilities"), (T.CORE_WORKFLOW, "workflow_steps")}
    assert len(state["discovered_knowledge"]) == 6
    assert [name for name, _ in calls] == [*[p[0] for p in PASSES], "GROUNDING"]
    prompts = {name: messages[0].content for name, messages in calls}
    definition = FIELD_DEFINITIONS[T.USER_ROLES]["responsibilities"]
    assert definition in prompts["RESPONSIBILITY"] and definition in prompts["GROUNDING"]
    assert OVERLAP_RULES in prompts["GROUNDING"]
    polluted = {"WORKFLOW": [raw("workflow_steps", "patients and professionals use the app", NO_OTHERS)],
                "GOAL": [raw("primary_user_goals", "find and book appointments", NO_OTHERS, role="patient")]}
    next_state, _ = run(monkeypatch, NO_OTHERS, polluted, existing=state["discovered_knowledge"],
        topic=T.USER_ROLES, gap="secondary_users", question="Are there any other users?", absence="none",
        reject=[("workflow_steps", "patients and professionals use the app"), ("primary_user_goals", "find and book appointments")])
    next_state.update(interview_planner_node(next_state))
    assert next_state["current_gap"] == "permissions::patient"
    assert len(next_state["discovered_knowledge"]) == 7
    assert next_state["discovered_knowledge"][-1].absence == "none"


@pytest.mark.parametrize("text,key", [
    ("Users can have both roles.", "multiple_roles"),
    ("Each account has only one role.", "multiple_roles"),
    ("A provider can also be a patient.", "multiple_roles"),
    ("No, users cannot have multiple roles.", "multiple_roles"),
    ("Patients can become providers after verification.", "role_transitions"),
    ("Roles never change.", "role_transitions"),
    ("Users cannot switch roles.", "role_transitions"),
])
def test_role_policies_have_an_existing_pass_and_satisfy_gap_without_active_focus(monkeypatch, text, key):
    state, _ = run(monkeypatch, text, {"ACTOR": [raw(key, text, text)]})
    assert key not in build_gap_info(state, T.USER_ROLES)["missing_keys"]
    item = state["discovered_knowledge"][0]
    assert item.roles is None and item.role is None and item.knowledge_state == K.CONFIRMED


def test_short_positive_policy_answer_uses_question_context(monkeypatch):
    state, calls = run(monkeypatch, "Yes.", {"ACTOR": [raw("multiple_roles", "One account can hold both roles", "Yes.")]},
        topic=T.USER_ROLES, gap="multiple_roles", question="Can one account hold both roles?")
    assert "multiple_roles" not in build_gap_info(state, T.USER_ROLES)["missing_keys"]
    assert "Can one account hold both roles?" in calls[0][1][0].content


def test_permissions_are_not_capabilities(monkeypatch):
    text = "Sellers can upload shipment evidence, but they cannot release escrow funds."
    state, _ = run(monkeypatch, text, {
        "RESPONSIBILITY": [raw("responsibilities", "upload shipment evidence", text, role="seller")],
        "PERMISSION": [raw("permissions", "cannot release escrow funds", text, role="seller")],
    }, existing=[actor("seller")])
    facts = state["discovered_knowledge"][1:]
    assert {(i.key, i.value) for i in facts} == {("responsibilities", "upload shipment evidence"), ("permissions", "cannot release escrow funds")}
    assert not any(i.absence for i in facts)
    gaps = build_gap_info(state, T.USER_ROLES)["missing_keys"]
    assert "permissions::seller" not in gaps and "responsibilities::seller" not in gaps


def test_workflow_and_outcome_are_distinct(monkeypatch):
    actions = "Buyers browse listings, select an item and pay."
    text = actions + " Their goal is to safely complete a purchase."
    state, _ = run(monkeypatch, text, {
        "RESPONSIBILITY": [raw("responsibilities", "browse listings, select an item and pay", actions, role="buyer")],
        "WORKFLOW": [raw("workflow_steps", "browse listings, select an item and pay", actions)],
        "GOAL": [raw("primary_user_goals", "safely complete a purchase", text, role="buyer")],
    }, existing=[actor("buyer")])
    assert len([i for i in state["discovered_knowledge"] if i.key == "primary_user_goals"]) == 1
    assert "primary_user_goals::buyer" not in build_gap_info(state, T.USER_GOALS)["missing_keys"]


@pytest.mark.parametrize("topic,gap,text", [
    (T.USER_ROLES, "secondary_users", "No other users."),
    (T.USER_ROLES, "role_transitions", "Users cannot switch roles."),
    (T.USER_ROLES, "multiple_roles", "Each account has only one role."),
    (T.USER_ROLES, "permissions::patient", "No special permissions."),
    (T.BUSINESS_RULES, "approval_rules", "No approvals are required."),
    (T.CONSTRAINTS, "business_constraints", "No additional constraints."),
    (T.MVP_SCOPE, "nice_to_have_features", "No optional features."),
    (T.EXCEPTIONS, "recovery", "Recovery does not apply."),
    (T.EDGE_CASES, "rare_scenarios", "No other unusual scenarios."),
])
def test_negative_answers_satisfy_exact_gap(monkeypatch, topic, gap, text):
    is_policy = gap in ("multiple_roles", "role_transitions")
    state, _ = run(monkeypatch, text, {}, existing=[actor("patient")], topic=topic, gap=gap,
                   absence="policy" if is_policy else "none")
    assert gap not in build_gap_info(state, topic)["missing_keys"]
    item = state["discovered_knowledge"][-1]
    assert item.topic == topic and item.scope == S.USER_APP
    assert item.key == gap.split("::")[0] and item.absence == (None if is_policy else "none")
    if is_policy:
        assert item.value == text
    assert item.role == ("patient" if "::" in gap else None)


def test_initial_secondary_absence_is_representable_without_active_gap(monkeypatch):
    text = "There are no secondary users."
    state, _ = run(monkeypatch, text, {"ACTOR": [raw("secondary_users", "none", text, absence="none")]})
    assert state["discovered_knowledge"][0].roles == []
    assert "secondary_users" not in build_gap_info(state, T.USER_ROLES)["missing_keys"]


@pytest.mark.parametrize("key,text", [("success_criteria", "Success means the appointment is confirmed."),
                                      ("motivations", "The product is needed because manual scheduling takes too long.")])
def test_global_goal_fields_do_not_require_invented_actor(monkeypatch, key, text):
    state, _ = run(monkeypatch, text, {"GOAL": [raw(key, text, text)]})
    assert state["discovered_knowledge"][0].role is None
    assert key not in build_gap_info(state, T.USER_GOALS)["missing_keys"]


def test_primary_actor_cannot_get_secondary_goal_when_no_secondary_actors(monkeypatch):
    text = "Patients want appointments."
    state, _ = run(monkeypatch, text, {"GOAL": [raw("secondary_user_goals", "appointments", text, role="patient")]}, existing=[actor("patient")])
    assert len(state["discovered_knowledge"]) == 1


@pytest.mark.parametrize("text,pairs", [
    ("A repeated payment submission is rejected.", [(T.EDGE_CASES, "duplicate_actions"), (T.EXCEPTIONS, "invalid_actions")]),
    ("An idle checkout expires after 10 minutes.", [(T.CONSTRAINTS, "time_constraints"), (T.EXCEPTIONS, "timeouts")]),
    ("If two users book the last slot simultaneously, the losing request is rejected and can be retried.", [(T.EDGE_CASES, "simultaneous_actions"), (T.EXCEPTIONS, "recovery")]),
    ("At the ten-booking limit, additional requests are rejected.", [(T.BUSINESS_RULES, "limits"), (T.EDGE_CASES, "boundary_conditions"), (T.EXCEPTIONS, "invalid_actions")]),
])
def test_exception_edge_case_and_rule_overlap(monkeypatch, text, pairs):
    state, calls = run(monkeypatch, text, {"RULES": [raw(key, text, text, topic=topic.value) for topic, key in pairs]})
    assert {(i.topic, i.key) for i in state["discovered_knowledge"]} == set(pairs)
    for topic, key in pairs:
        assert key not in build_gap_info(state, topic)["missing_keys"]
        assert FIELD_DEFINITIONS[topic][key] in dict((n, m[0].content) for n, m in calls)["GROUNDING"]


def test_mvp_exclusion_is_not_absence_of_exclusions(monkeypatch):
    text = "No payments in version one."
    state, _ = run(monkeypatch, text, {"RULES": [raw("out_of_scope", "payments excluded from version one", text, topic="MVP_SCOPE")]})
    item = state["discovered_knowledge"][0]
    assert item.absence is None and "payments" in item.value
    assert "out_of_scope" not in build_gap_info(state, T.MVP_SCOPE)["missing_keys"]


def test_actor_topic_scope_ownership_boundaries_and_legacy_spelling(monkeypatch):
    text = "Healthcare providers can manage schedules."
    state, _ = run(monkeypatch, text, {"RESPONSIBILITY": [raw("responsibilities", "manage schedules", text, role="healthcare_provider")]},
        existing=[actor("healthcare provider"), actor("patient")])
    missing = build_gap_info(state, T.USER_ROLES)["missing_keys"]
    assert "responsibilities::healthcare provider" not in missing
    assert "responsibilities::patient" in missing
    other_scope = {**state, "discovery_scope": S.ADMIN_DASHBOARD}
    assert "responsibilities" in build_gap_info(other_scope, T.USER_ROLES)["missing_keys"]
    assert "manage schedules" not in permission_discovery_guidance(other_scope, "healthcare_provider")


def test_existing_gaps_do_not_silently_reopen_a_completed_topic():
    # Only a new commit may invalidate completion, not a planner snapshot.
    state = dict(discovery_scope=S.USER_APP, current_topic=None,
                 discovered_knowledge=[actor("patient")], topic_maturity={},
                 topic_status={T.USER_ROLES: TopicStatus.COMPLETED})
    plan = interview_planner_node(state)
    assert plan["current_topic"] != T.USER_ROLES
    assert plan["topic_status"][T.USER_ROLES] == TopicStatus.COMPLETED


def test_every_planner_field_has_storage_extraction_and_shared_semantics():
    extracted = set()
    for name, schema, topic, _ in PASSES:
        if topic:
            extracted.update((topic, key) for key in get_args(schema.model_fields["key"].annotation))
        else:
            for label in get_args(schema.model_fields["topic"].annotation):
                extracted.update((T(label), key) for key in TOPIC_KEY_MAP[T(label)])
    planned = {(topic, key) for topic, tasks in DISCOVERY_TASKS.items() for key in tasks}
    stored = {(topic, key) for topic, keys in TOPIC_KEY_MAP.items() for key in keys}
    defined = {(topic, key) for topic, definitions in FIELD_DEFINITIONS.items() for key in definitions}
    assert planned == stored == extracted == defined


# One explicit natural-language assertion for every required USER_APP field.
# The model fixtures test routing/validation/completion, not model accuracy.
FIELD_EXAMPLES = {
    T.USER_ROLES: {
        "primary_users": "Patients use the service.",
        "secondary_users": "Support staff help resolve disputes.",
        "responsibilities": "Patients can book appointments.",
        "permissions": "Patients cannot edit provider schedules.",
        "multiple_roles": "A provider can also be a patient.",
        "role_transitions": "Patients become providers after verification.",
    },
    T.USER_GOALS: {
        "primary_user_goals": "Patients want to obtain appropriate medical care.",
        "secondary_user_goals": "Support staff want disputes resolved fairly.",
        "success_criteria": "Success means the appointment is confirmed.",
        "motivations": "Manual scheduling currently takes too long.",
    },
    T.CORE_WORKFLOW: {
        "trigger": "The journey starts when a patient submits a health concern.",
        "workflow_steps": "Patients search, compare providers, then book and pay.",
        "completion_condition": "The interaction completes when payment settles.",
        "downstream_dependency": "Completion requires approval from the external insurer.",
        "end_state": "After completion, the appointment is marked confirmed.",
    },
    T.BUSINESS_RULES: {
        "validation_rules": "The system requires a valid phone number before booking.",
        "approval_rules": "The clinic manager must approve new provider profiles.",
        "eligibility_rules": "Only licensed professionals are eligible to offer care.",
        "limits": "Each patient may hold at most three active bookings.",
        "ownership_rules": "Patients own their uploaded health records.",
        "visibility_rules": "Health records are visible only to the assigned provider.",
    },
    T.CONSTRAINTS: {
        "legal_constraints": "The service must comply with the stated local privacy law.",
        "business_constraints": "The operating budget cannot exceed 5000 per month.",
        "operational_constraints": "The service depends on the clinic's scheduling API being available.",
        "geographic_constraints": "The service is restricted to Lagos.",
        "time_constraints": "Booking requests expire after 15 minutes.",
    },
    T.MVP_SCOPE: {
        "must_have_features": "Version one must include provider search and booking.",
        "nice_to_have_features": "Video consultations are optional and can wait until later.",
        "out_of_scope": "Payments are excluded from version one.",
        "success_metrics": "MVP success is measured by 100 confirmed bookings per month.",
    },
    T.EXCEPTIONS: {
        "user_cancellations": "When patients cancel, their appointment slot is released.",
        "timeouts": "On expiry, unpaid bookings are cancelled.",
        "invalid_actions": "Invalid payment attempts are rejected with an error.",
        "recovery": "Interrupted checkouts can resume from the saved booking.",
    },
    T.EDGE_CASES: {
        "duplicate_actions": "Repeated payment submissions reuse the first receipt.",
        "boundary_conditions": "At zero available slots, the system displays a waiting list.",
        "simultaneous_actions": "When two patients book the last slot, only the first booking succeeds.",
        "rare_scenarios": "If a clinic closes permanently during a booking, patients are referred elsewhere.",
    },
}


@pytest.mark.parametrize("topic,key,text", [(topic, key, text) for topic, examples in FIELD_EXAMPLES.items() for key, text in examples.items()])
def test_every_required_field_round_trips_through_its_pass_and_planner(monkeypatch, topic, key, text):
    assert set(FIELD_EXAMPLES[topic]) == TOPIC_KEY_MAP[topic]
    fields = {}
    existing = [actor("patient"), actor("support", secondary=True)]
    role = "support" if key == "secondary_user_goals" else "patient"
    if key in ("primary_users", "secondary_users"):
        fields["roles"] = ["patient" if key == "primary_users" else "support"]
        existing = []
    elif key in ("responsibilities", "permissions", "primary_user_goals", "secondary_user_goals"):
        fields["role"] = role
    pass_name = None
    for name, schema, pass_topic, _ in PASSES:
        if pass_topic == topic and key in get_args(schema.model_fields["key"].annotation):
            pass_name = name
    if pass_name is None:
        pass_name = "RULES"
        fields["topic"] = topic.value
    state, calls = run(monkeypatch, text, {pass_name: [raw(key, text, text, **fields)]}, existing=existing)
    assert any(i.key == key and i.topic == topic and i.evidence == text for i in state["discovered_knowledge"])
    gap = f"{key}::{role}" if "role" in fields else key
    assert gap not in build_gap_info(state, topic)["missing_keys"]
    prompt = dict((name, messages[0].content) for name, messages in calls)[pass_name]
    assert FIELD_DEFINITIONS[topic][key] in prompt


def test_empty_actor_sets_waive_role_gaps_and_unlock_next_topic():
    from agents.interview_planner import assess_topic_maturity
    from agents.state import TopicMaturity
    items = [KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key=key,
                           value="none", absence="none", evidence="None apply.",
                           roles=[] if key.endswith("users") else None, confidence=1)
             for key in ("primary_users", "secondary_users", "multiple_roles", "role_transitions")]
    state = dict(discovery_scope=S.USER_APP, discovered_knowledge=items)
    assert not build_gap_info(state, T.USER_ROLES)["missing_keys"]
    assert assess_topic_maturity(state, T.USER_ROLES) == TopicMaturity.DECISION_READY
