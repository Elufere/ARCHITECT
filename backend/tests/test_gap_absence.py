"""Gap resolution, grounding, and planner handoff using deterministic model replies."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.llm_errors import ExtractionFailed
from agents.interview_planner import build_gap_info, get_confirmed_roles_for_source, interview_planner_node
from agents.state import DiscoveryScope as Scope, DiscoveryTopic as Topic, KnowledgeItem, KnowledgeState


ANSWER = "No. For the user app, the only users are patients and healthcare professionals."


def actor(role="patient", scope=Scope.USER_APP):
    return KnowledgeItem(topic=Topic.USER_ROLES, scope=scope, key="primary_users",
                         value=role, roles=[role], evidence=role, confidence=1)


def state(text=ANSWER, gap="secondary_users", topic=Topic.USER_ROLES):
    question = "Are there any other users besides patients and healthcare professionals?"
    return dict(messages=[AIMessage(content=question), HumanMessage(content=text)], current_topic=topic, current_gap=gap,
                discovery_scope=Scope.USER_APP, discovered_knowledge=[actor(), actor("healthcare_provider")],
                topic_status={}, turn_count=1,
                asked_gap=(dict(scope=Scope.USER_APP.value, topic=topic.value, gap=gap, question=question)
                           if topic and gap else None))


def models(monkeypatch, resolution="none", evidence=ANSWER, outputs=None, supported=None,
           confidence=1, failure=None, value=None):
    calls = {}
    def invoke(name, messages):
        calls[name] = messages
        if name == failure:
            raise ValueError("model unavailable")
        if name == "GAP_ANSWER":
            return dict(resolution=resolution, evidence=evidence, confidence=confidence, value=value)
        if name == "GROUNDING":
            calls.setdefault("GROUNDING_CALLS", []).append(messages)
            groups = json.loads(messages[-1].content)["evidence_groups"]
            candidates = [{**item, "evidence_id": group["evidence_id"]} for group in groups for item in group["candidates"]]
            accepted = ([c["id"] for c in candidates if supported(c)] if callable(supported)
                        else supported if supported is not None else [c["id"] for c in candidates])
            categories = {str(c["evidence_id"]): [f"{i['topic']}.{i['key']}" for i in candidates
                          if i["evidence_id"] == c["evidence_id"]] for c in candidates}
            return dict(evidence_categories=categories, supported_ids=accepted,
                        rejection_reasons={str(c["id"]): "Unsupported claim" for c in candidates if c["id"] not in accepted},
                        confirmed_absence_ids=[c["id"] for c in candidates if c.get("absence") and c["id"] in accepted])
        return dict(items=(outputs or {}).get(name, []))
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*[p[0] for p in tracker.PASSES], "GAP_ANSWER", "GROUNDING"]})
    return calls


def test_careconnect_absence_closes_gap_and_rejects_unrelated_claims(monkeypatch):
    workflow = dict(key="workflow_steps", value="patients and healthcare professionals use the app",
                    evidence=ANSWER, confidence=1)
    goal = dict(key="primary_user_goals", value="find and book appointments", role="patient",
                evidence=ANSWER, confidence=1)
    calls = models(monkeypatch, outputs={"WORKFLOW": [workflow], "GOAL": [goal]},
                   supported=lambda c: c["key"] == "secondary_users")
    initial = state()
    result = tracker.knowledge_tracker_node(initial)
    added = result["discovered_knowledge"][2:]
    assert len(added) == 1
    absence = added[0]
    assert (absence.key, absence.value, absence.absence, absence.roles) == ("secondary_users", "none", "none", [])
    assert absence.evidence == ANSWER and absence.source_turn == 1
    assert absence.knowledge_state == KnowledgeState.CONFIRMED
    merged = {**initial, **result}
    merged.update(interview_planner_node(merged))
    assert "secondary_users" not in build_gap_info(merged, Topic.USER_ROLES)["missing_keys"]
    assert get_confirmed_roles_for_source(merged, "secondary_users") == []
    assert "secondary_user_goals" not in build_gap_info(merged, Topic.USER_GOALS)["missing_keys"]
    assert "secondary_users: none" in str(result["product_model"])
    audits = [json.loads(messages[-1].content) for messages in calls["GROUNDING_CALLS"]]
    audit = audits[0]
    assert sum(len(group["candidates"]) for a in audits for group in a["evidence_groups"]) == 3
    assert all(c["key"] == "secondary_users" for group in audit["evidence_groups"] for c in group["candidates"])
    assert audit["latest_response"] == ANSWER
    assert audit["active_gap_review"]["key"] == "secondary_users"
    assert audit["active_gap_review"]["evidence"] == ANSWER
    assert audit["active_gap_review"]["absence"] == "none"


@pytest.mark.parametrize("gap,topic,text,resolution", [
    ("permissions::patient", Topic.USER_ROLES, "No special permissions.", "none"),
    ("business_constraints", Topic.CONSTRAINTS, "No additional constraints.", "none"),
    ("downstream_dependency", Topic.CORE_WORKFLOW, "Not applicable.", "not_applicable"),
])
def test_generic_gap_mapping(monkeypatch, gap, topic, text, resolution):
    models(monkeypatch, resolution=resolution, evidence=text)
    initial = state(text, gap, topic)
    result = tracker.knowledge_tracker_node(initial)
    item = result["discovered_knowledge"][-1]
    assert item.key == gap.split("::")[0]
    assert item.role == (gap.split("::")[1] if "::" in gap else None)
    assert item.topic == topic and item.scope == Scope.USER_APP
    assert item.absence == resolution
    merged = {**initial, **result}
    merged.update(interview_planner_node(merged))
    assert gap not in build_gap_info(merged, topic)["missing_keys"]


@pytest.mark.parametrize("text", ["I don't know.", "No admins, but support staff will use it.",
    "No, I already told you.", "No other constraints.", "Maybe later."])
def test_unresolved_answers_do_not_close_active_gap(monkeypatch, text):
    calls = models(monkeypatch, resolution="unresolved", evidence=text)
    initial = state(text)
    result = tracker.knowledge_tracker_node(initial)
    assert result["discovered_knowledge"] == initial["discovered_knowledge"]
    assert "secondary_users" in build_gap_info({**initial, **result}, Topic.USER_ROLES)["missing_keys"]
    assert "GROUNDING" not in calls  # An unresolved review creates no candidate to audit.


@pytest.mark.parametrize("gap,topic", [(None, Topic.USER_ROLES), ("workflow", Topic.CORE_WORKFLOW),
    ("secondary_users", None), ("secondary_users::patient", Topic.USER_ROLES),
    ("permissions", Topic.USER_ROLES), ("permissions::unknown", Topic.USER_ROLES)])
def test_invalid_or_unowned_gap_cannot_create_absence(monkeypatch, gap, topic):
    calls = models(monkeypatch)
    assert tracker.extract_gap_absence(ANSWER, state(gap=gap, topic=topic), Scope.USER_APP) is None
    assert not calls


@pytest.mark.parametrize("options", [dict(evidence="invented"), dict(confidence=0.5), dict(supported=[])])
def test_rejected_or_low_confidence_absence_does_not_persist(monkeypatch, options):
    models(monkeypatch, **options)
    initial = state()
    assert tracker.knowledge_tracker_node(initial)["discovered_knowledge"] == initial["discovered_knowledge"]


@pytest.mark.parametrize("failure", ["GAP_ANSWER", "GROUNDING"])
def test_validator_outage_fails_closed_without_persisting_absence(monkeypatch, failure):
    models(monkeypatch, failure=failure)
    initial = state()
    with pytest.raises(ExtractionFailed):
        tracker.knowledge_tracker_node(initial)
    assert len(initial["discovered_knowledge"]) == 2


def test_repeat_absence_is_idempotent_and_positive_correction_replaces_it(monkeypatch):
    models(monkeypatch)
    initial = state()
    first = tracker.knowledge_tracker_node(initial)
    second = tracker.knowledge_tracker_node({**initial, **first})
    assert second["discovered_knowledge"] == first["discovered_knowledge"]
    text = "Support agents also use the app."
    models(monkeypatch, resolution="unresolved", evidence=text, outputs={"ACTOR": [
        dict(key="secondary_users", value="support agents", roles=["support_agent"],
             evidence=text, confidence=1)]})
    revised = tracker.knowledge_tracker_node({**state(text), **second})
    secondary = [i for i in revised["discovered_knowledge"] if i.key == "secondary_users"]
    assert len(secondary) == 1 and secondary[0].roles == ["support_agent"]
    assert secondary[0].absence is None


def test_other_scope_and_role_are_preserved(monkeypatch):
    models(monkeypatch, evidence="No special permissions.")
    initial = state("No special permissions.", "permissions::patient")
    old = [KnowledgeItem(topic=Topic.USER_ROLES, scope=scope, key="permissions", role=role,
                         value="may edit profiles", evidence="may edit profiles", confidence=1)
           for scope, role in [(Scope.ADMIN_DASHBOARD, "patient"), (Scope.USER_APP, "healthcare_provider")]]
    initial["discovered_knowledge"].extend(old)
    result = tracker.knowledge_tracker_node(initial)
    assert all(item in result["discovered_knowledge"] for item in old)


def test_only_in_primary_evidence_is_not_a_planner_absence_rule():
    initial = state()
    initial["discovered_knowledge"][0].evidence = "Only verified patients can book."
    assert "secondary_users" in build_gap_info(initial, Topic.USER_ROLES)["missing_keys"]


def test_absence_replaces_prior_values_but_keeps_independent_new_facts(monkeypatch):
    text = "No special permissions. Patients want shorter waiting times."
    models(monkeypatch, evidence="No special permissions.", outputs={"GOAL": [
        dict(key="primary_user_goals", role="patient", value="shorter waiting times",
             evidence="Patients want shorter waiting times.", confidence=1)]})
    initial = state(text, "permissions::patient")
    initial["discovered_knowledge"].append(KnowledgeItem(topic=Topic.USER_ROLES,
        scope=Scope.USER_APP, key="permissions", role="patient", value="edit profiles",
        evidence="edit profiles", confidence=1))
    result = tracker.knowledge_tracker_node(initial)
    permissions = [i for i in result["discovered_knowledge"] if i.key == "permissions"]
    assert len(permissions) == 1 and permissions[0].absence == "none"
    assert any(i.value == "shorter waiting times" for i in result["discovered_knowledge"])


def test_grounding_failure_preserves_existing_fact_by_failing_before_commit(monkeypatch):
    models(monkeypatch, failure="GROUNDING")
    initial = state()
    initial["is_correction"] = True
    initial["discovered_knowledge"].append(KnowledgeItem(topic=Topic.USER_ROLES,
        scope=Scope.USER_APP, key="secondary_users", roles=["admin"], value="admin",
        evidence="admin", confidence=1))
    before = list(initial["discovered_knowledge"])
    with pytest.raises(ExtractionFailed):
        tracker.knowledge_tracker_node(initial)
    assert initial["discovered_knowledge"] == before
