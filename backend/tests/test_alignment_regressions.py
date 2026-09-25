"""Surgical checks for the live schema-alignment regressions.

Uses deterministic semantic verdicts to test enforcement and context plumbing;
the opt-in live replay separately verifies actual model judgments.
"""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.conversation_manager import classify_turn, conversation_manager_node
from agents.conversation_language import CLARIFICATION_QUESTIONS
from agents.extraction_passes import ActorFact, PASSES
from agents.interview_planner import build_gap_info, get_confirmed_roles_for_source, interview_planner_node
from agents.question_generator import question_generator_node
from agents.semantic_validation import GroundingResult, category_contradiction
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, TOPIC_KEY_MAP
from test_schema_alignment import actor, raw, run


def item(topic, key, value, quote, **kwargs):
    return KnowledgeItem(topic=topic, scope=S.USER_APP, key=key, value=value,
                         evidence=quote, confidence=1, **kwargs)


def audit(monkeypatch, items, text, supported, absences=()):
    captured = {}
    def decide(name, schema, instruction, payload):
        captured.update(instruction=instruction, payload=payload)
        captured.setdefault("calls", []).append(payload)
        candidates = payload["candidates"]
        original_ids = {c["id"]: next(i for i, original in enumerate(items)
                        if original.key == c["key"] and original.value == c["value"]
                        and original.role == c.get("role")
                        and original.evidence == payload["evidence_quotes"][str(c["evidence_id"])])
                        for c in candidates}
        local_supported = [candidate_id for candidate_id, original_id in original_ids.items() if original_id in supported]
        local_absences = [candidate_id for candidate_id, original_id in original_ids.items() if original_id in absences]
        categories = {str(c["evidence_id"]): [f"{i['topic']}.{i['key']}" for i in candidates
                      if i["evidence_id"] == c["evidence_id"]] for c in candidates}
        return GroundingResult(evidence_categories=categories, supported_ids=local_supported, confirmed_absence_ids=local_absences,
                               rejection_reasons={str(i): "Own quote does not support this category/value"
                                                  for i in original_ids if i not in local_supported})
    monkeypatch.setattr(tracker, "semantic_decision", decide)
    result = tracker.ground_items(items, text, dict(discovery_scope=S.USER_APP))
    return result, captured


def test_own_evidence_cannot_be_rescued_by_another_sentence(monkeypatch):
    purpose = "The platform helps patients find and book appointments."
    actions = "Patients should be able to describe their health concern and book appointments."
    unsupported = item(T.USER_ROLES, "responsibilities", "describe their health concern", purpose, role="patient")
    supported = unsupported.model_copy(update={"evidence": actions})
    accepted, captured = audit(monkeypatch, [unsupported, supported], purpose + " " + actions, [1])
    assert accepted == [supported]
    assert captured["payload"]["evidence_quotes"] == {"0": purpose, "1": actions}
    assert "cannot rescue" in captured["instruction"]


def test_overlap_requires_each_category_to_be_supported(monkeypatch):
    text = "Patients search for providers, compare profiles and book appointments."
    candidates = [item(T.USER_ROLES, "responsibilities", "search, compare and book", text, role="patient"),
                  item(T.CORE_WORKFLOW, "workflow_steps", "search, compare and book", text),
                  *[item(T.USER_GOALS, "primary_user_goals", verb, text, role="patient") for verb in ("search", "compare", "book")]]
    accepted, captured = audit(monkeypatch, candidates, text, [0, 1])
    assert accepted == candidates[:2]
    assert len(captured["payload"]["evidence_quotes"]) == 1
    assert len(captured["payload"]["candidates"]) == 5
    assert "An action list alone is not a goal" in captured["instruction"]


def test_explicit_patient_outcome_is_preserved(monkeypatch):
    text = "The patient's goal is to find and book a suitable healthcare appointment."
    state, _ = run(monkeypatch, text, {"GOAL": [raw("primary_user_goals", "find and book a suitable healthcare appointment", text, role="patient")]}, existing=[actor("patient")])
    assert any(i.key == "primary_user_goals" for i in state["discovered_knowledge"])


@pytest.mark.parametrize("key,topic,text", [
    ("multiple_roles", T.USER_ROLES, "Patients book appointments. Healthcare providers manage appointments."),
    ("downstream_dependency", T.CORE_WORKFLOW, "Patients search, compare, book and pay."),
    ("role_transitions", T.USER_ROLES, "No role transition was described."),
    ("business_constraints", T.CONSTRAINTS, "No constraint was mentioned."),
])
def test_none_without_verified_explicit_negation_cannot_be_stored(monkeypatch, key, topic, text):
    # Even a generic supported verdict cannot bypass the explicit absence gate.
    candidate = item(topic, key, "none", text)
    accepted, captured = audit(monkeypatch, [candidate], text, [0])
    assert not accepted
    assert captured["payload"]["candidates"][0]["absence"] == "none"


@pytest.mark.parametrize("topic,key,text", [
    (T.USER_ROLES, "multiple_roles", "Each account can only have one role. A healthcare provider cannot also act as a patient."),
    (T.CORE_WORKFLOW, "downstream_dependency", "No external approval or external system is required before booking completes."),
])
def test_explicit_whole_field_absence_survives_separate_check(monkeypatch, topic, key, text):
    candidate = item(topic, key, "none", text)
    accepted, _ = audit(monkeypatch, [candidate], text, [0], [0])
    assert len(accepted) == 1 and accepted[0].absence == "none"
    assert key in build_gap_info(dict(discovered_knowledge=accepted), topic)["missing_keys"]


def test_actor_pass_cannot_hide_none_behind_null_absence():
    fact = ActorFact.model_validate(raw("multiple_roles", "none", "Patients book appointments.", absence=None))
    assert fact.absence == "none"  # Still needs semantic verification; not auto-accepted.


def test_workflow_sequence_does_not_automatically_establish_other_fields(monkeypatch):
    text = "Patients search, compare, book and pay."
    candidates = [item(T.CORE_WORKFLOW, key, "search, compare, book and pay", text)
                  for key in ("workflow_steps", "trigger", "completion_condition", "end_state")]
    candidates.append(item(T.CORE_WORKFLOW, "downstream_dependency", "none", text))
    accepted, captured = audit(monkeypatch, candidates, text, [0])
    assert [i.key for i in accepted] == ["workflow_steps"]
    assert "Neither follows from a final listed action" in captured["instruction"]


def test_both_actor_declarations_survive_to_question_context(monkeypatch):
    text = "Patients find and book appointments with healthcare professionals."
    state, calls = run(monkeypatch, text, {"ACTOR": [
        raw("primary_users", "patients", text, roles=["patient"]),
        raw("primary_users", "healthcare professionals", text, roles=["healthcare_provider"])]})
    assert get_confirmed_roles_for_source(state, "primary_users") == ["patient", "healthcare_provider"]
    audit_payload = json.loads(dict(calls)["GROUNDING"][-1].content)
    assert all(c["kind"] == "actor_declaration" and "role" not in c
               for group in audit_payload["evidence_groups"] for c in group["candidates"])
    state.update(current_topic=T.USER_ROLES, current_gap="secondary_users")
    question = question_generator_node(state)["messages"][-1].content
    assert question == "Besides patients and healthcare professionals, will anyone else use the user app?"


@pytest.mark.parametrize("utterance", ["what do you mean?", "I don't understand", "can you explain?", "what are you asking?"])
def test_clarification_never_echoes_internal_objective(utterance):
    assert classify_turn(utterance) == "clarification"
    patient = actor("patient").model_copy(update={"value": "patients"})
    provider = actor("healthcare_provider").model_copy(update={"value": "healthcare professionals"})
    state = dict(current_topic=T.USER_ROLES, current_gap="secondary_users", discovery_scope=S.USER_APP,
                 current_objective="primary value exchange scoped product explicit absence gap field planner objective",
                 discovered_knowledge=[patient, provider], messages=[HumanMessage(content=utterance)])
    reply = conversation_manager_node(state)["messages"][-1].content
    assert reply == "I mean, besides patients and healthcare professionals, will anyone else use the user app?"
    for term in ("primary value exchange", "scoped product", "explicit absence", "gap", "field", "planner objective", "Please describe"):
        assert term not in reply
    assert state["current_gap"] == "secondary_users"


@pytest.mark.parametrize("key", [key for keys in TOPIC_KEY_MAP.values() for key in keys])
def test_all_gap_clarifications_have_user_facing_wording(key):
    state = dict(current_gap=key, messages=[HumanMessage(content="what do you mean?")],
                 current_objective="INTERNAL ONTOLOGY DEFINITION DO NOT REPEAT")
    reply = conversation_manager_node(state)["messages"][-1].content
    assert "INTERNAL" not in reply and "ontology" not in reply and "Please describe" not in reply
    assert reply.endswith("?")
    assert key == "secondary_users" or key in CLARIFICATION_QUESTIONS


def test_rejected_actor_log_names_actor_and_does_not_claim_owner_required(monkeypatch, capsys):
    text = "Patients use the app."
    candidate = item(T.USER_ROLES, "primary_users", "patients", text, roles=["patient"])
    audit(monkeypatch, [candidate], text, [])
    log = capsys.readouterr().out
    assert "actors=['patient']" in log and "id=0" in log
    assert "unowned" not in log and "Own quote" in log


def test_short_no_after_clarification_resolves_gap_but_named_denial_does_not(monkeypatch):
    actors = [actor("patient").model_copy(update={"value": "patients"}),
              actor("healthcare_provider").model_copy(update={"value": "healthcare professionals"})]
    state = dict(current_topic=T.USER_ROLES, current_gap="secondary_users", discovered_knowledge=actors,
                 messages=[HumanMessage(content="what do you mean?")], discovery_scope=S.USER_APP)
    clarification = conversation_manager_node(state)["messages"][-1].content
    resolved, _ = run(monkeypatch, "No.", {}, existing=actors, topic=T.USER_ROLES,
        gap="secondary_users", question=clarification, absence="none")
    resolved.update(interview_planner_node(resolved))
    assert "secondary_users" not in build_gap_info(resolved, T.USER_ROLES)["missing_keys"]
    partial, _ = run(monkeypatch, "No caregivers.", {}, existing=actors, topic=T.USER_ROLES,
        gap="secondary_users", question=clarification, absence="unresolved")
    assert "secondary_users" in build_gap_info(partial, T.USER_ROLES)["missing_keys"]


def test_prompts_keep_strict_goal_workflow_and_absence_thresholds():
    prompts = {name: instruction for name, _, _, instruction in PASSES}
    assert "STRICT OUTCOME REQUIREMENT" in prompts["GOAL"]
    assert "The last listed action" in prompts["WORKFLOW"]
    assert "Omit\nunmentioned policies entirely" in prompts["ACTOR"]


def test_actor_declaration_cannot_use_its_classification_as_an_id():
    with pytest.raises(ValueError, match="actor field name"):
        ActorFact(key="primary_users", value="patients", roles=["primary_users"],
                  evidence="Patients use the app.", confidence=1)


def test_factual_support_without_category_support_cannot_be_stored(monkeypatch):
    quote = "Patients search, compare and book appointments."
    action = item(T.USER_ROLES, "responsibilities", "search, compare and book appointments", quote, role="patient")
    goal = item(T.USER_GOALS, "primary_user_goals", "search", quote, role="patient")
    monkeypatch.setattr(tracker, "semantic_decision", lambda *_: GroundingResult(
        evidence_categories={"0": ["USER_ROLES.responsibilities"]}, supported_ids=[0, 1]))
    assert tracker.ground_items([action, goal], quote, {}) == [action]


def test_audit_requires_explicit_category_verdict():
    with pytest.raises(ValueError):
        GroundingResult.model_validate({"supported_ids": [0]})
    with pytest.raises(ValueError):
        GroundingResult.model_validate({"supported_ids": [0], "evidence_categories": {"0": ["USER_GOALS"]}})


def test_audit_quote_order_follows_source_not_extractor_order(monkeypatch):
    purpose = "The platform helps patients find appointments."
    actions = "Patients search and book appointments."
    candidates = [item(T.USER_ROLES, "responsibilities", "search and book appointments", actions, role="patient"),
                  item(T.USER_GOALS, "primary_user_goals", "find appointments", purpose, role="patient")]
    accepted, captured = audit(monkeypatch, candidates, purpose + " " + actions, [0, 1])
    assert accepted == candidates
    assert captured["payload"]["evidence_quotes"] == {"0": purpose, "1": actions}
    assert [c["evidence_id"] for c in captured["payload"]["candidates"]] == [1, 0]


@pytest.mark.parametrize("key", ["primary_user_goals", "secondary_user_goals", "trigger", "completion_condition", "end_state", "downstream_dependency"])
def test_pure_capability_list_cannot_establish_outcome_or_lifecycle(key):
    assert category_contradiction(key, "Buyers should be able to browse, compare and pay.")


@pytest.mark.parametrize("key,quote", [
    ("responsibilities", "Buyers should be able to browse, compare and pay."),
    ("workflow_steps", "Buyers should be able to browse, compare and pay."),
    ("primary_user_goals", "Buyers should be able to browse and compare so that they can find suitable housing."),
    ("primary_user_goals", "Patients should be able to describe and submit their concerns to find suitable care."),
    ("completion_condition", "Buyers should be able to browse and pay. Once payment settles, checkout is complete."),
    ("end_state", "Buyers should be able to browse and pay. After checkout, the order status is confirmed."),
    ("trigger", "I want to build a marketplace. The process starts when a buyer submits a request."),
    ("primary_user_goals", "Patients should be able to obtain medical care."),
])
def test_mixed_semantics_and_single_capability_remain_for_auditor(key, quote):
    assert category_contradiction(key, quote) is None


@pytest.mark.parametrize("first_rejection,repair_failure", [(False, False), (True, False), (False, True)])
def test_audit_repairs_only_missing_verdicts_once(monkeypatch, first_rejection, repair_failure):
    calls = []
    def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            return GroundingResult(evidence_categories={"0": ["USER_ROLES.responsibilities", "CORE_WORKFLOW.workflow_steps"]},
                supported_ids=[0], rejection_reasons={"1": "Not supported"} if first_rejection else {})
        if repair_failure:
            raise RuntimeError("unavailable")
        return GroundingResult(evidence_categories={"0": ["CORE_WORKFLOW.workflow_steps"]}, supported_ids=[0])
    monkeypatch.setattr(tracker, "extraction_models", lambda: {"GROUNDING": SimpleNamespace(invoke=invoke)})
    payload = dict(evidence_quotes={"0": "Buyers browse and pay."}, candidates=[
        dict(id=0, evidence_id=0, topic="USER_ROLES", key="responsibilities", value="browse and pay", role="buyer"),
        dict(id=1, evidence_id=0, topic="CORE_WORKFLOW", key="workflow_steps", value="browse and pay")])
    result = tracker.semantic_decision("GROUNDING", GroundingResult, "Audit own evidence", payload)
    assert result.supported_ids == ([0, 1] if not first_rejection and not repair_failure else [0])
    assert len(calls) == (1 if first_rejection else 2)
    if len(calls) == 2:
        remaining = json.loads(calls[1][-1].content)["evidence_groups"][0]["candidates"]
        assert len(remaining) == 1 and remaining[0]["key"] == "workflow_steps" and remaining[0]["id"] == 0


@pytest.mark.parametrize("malformed_categories,missing_absence", [(True, False), (False, True), (True, True)])
def test_incomplete_support_is_reaudited_not_accepted(monkeypatch, malformed_categories, missing_absence):
    calls = []
    def invoke(messages):
        calls.append(messages)
        return GroundingResult(
            evidence_categories={"USER_ROLES.secondary_users" if len(calls) == 1 and malformed_categories else "0": ["USER_ROLES.secondary_users"]},
            supported_ids=[0], confirmed_absence_ids=[] if len(calls) == 1 and missing_absence else [0])
    monkeypatch.setattr(tracker, "extraction_models", lambda: {"GROUNDING": SimpleNamespace(invoke=invoke)})
    result = tracker.semantic_decision("GROUNDING", GroundingResult, "Audit own evidence", dict(
        evidence_quotes={"0": "No."}, candidates=[dict(id=0, evidence_id=0,
            topic="USER_ROLES", key="secondary_users", value="none", absence="none")]))
    assert len(calls) == 2
    assert result.supported_ids == [0]
    assert 0 in result.confirmed_absence_ids
    assert result.evidence_categories["0"] == ["USER_ROLES.secondary_users"]


def test_user_evidence_is_separate_from_instructions(monkeypatch):
    text = "Patients book appointments."
    _, calls = run(monkeypatch, text, {})
    for name, messages in calls:
        if name in {entry[0] for entry in PASSES}:
            assert isinstance(messages[-1], HumanMessage)
            assert messages[-1].content == text
            assert "Actor declarations introduce IDs" in messages[0].content


def test_grounding_wire_response_preserves_shared_evidence(monkeypatch):
    response = dict(evidence_categories=[dict(evidence_id=0,
        categories=["USER_ROLES.responsibilities", "CORE_WORKFLOW.workflow_steps"])],
        supported_ids=[0, 1], confirmed_absence_ids=[], rejection_reasons={})
    monkeypatch.setattr(tracker, "extraction_models", lambda: {"GROUNDING": SimpleNamespace(invoke=lambda _: response)})
    result = tracker.semantic_decision("GROUNDING", GroundingResult, "Audit", dict(
        evidence_quotes={"0": "Buyers browse and pay."}, candidates=[
            dict(id=0, evidence_id=0, topic="USER_ROLES", key="responsibilities", value="browse and pay", role="buyer"),
            dict(id=1, evidence_id=0, topic="CORE_WORKFLOW", key="workflow_steps", value="browse and pay")]))
    assert result.supported_ids == [0, 1]
    assert result.evidence_categories["0"] == response["evidence_categories"][0]["categories"]


def test_grounding_wire_response_rejects_category_name_as_evidence_id():
    from agents.semantic_validation import GroundingResponse
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GroundingResponse.model_validate(dict(evidence_categories=[dict(
            evidence_id="USER_ROLES.secondary_users", categories=["USER_ROLES.secondary_users"])],
            supported_ids=[0], confirmed_absence_ids=[0], rejection_reasons={}))


def test_grounding_wire_response_rejects_duplicate_evidence_verdicts():
    from agents.semantic_validation import GroundingResponse
    result = GroundingResponse.model_validate(dict(evidence_categories=[
        dict(evidence_id=0, categories=[]), dict(evidence_id=0, categories=["USER_ROLES.secondary_users"])],
        supported_ids=[0], confirmed_absence_ids=[0], rejection_reasons={}))
    with pytest.raises(ValueError, match="Duplicate"):
        result.decision()
