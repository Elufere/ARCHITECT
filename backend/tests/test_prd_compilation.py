"""Compilation boundary: immutable scoped sources, semantic checks, atomic output."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import pm_agent as pm
from agents.prd_schema import ClaimVerdict, PRDDraft
from agents.prd_validation import build_source_snapshot, draft_claims
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K


QUOTE = "Managers approve orders over $100"


def fact(**changes):
    values = dict(topic=T.BUSINESS_RULES, scope=S.USER_APP, key="approval_rules",
                  value="Orders over $100 require manager approval", evidence=QUOTE,
                  role="manager", confidence=0.95, knowledge_state=K.CONFIRMED, source_turn=2)
    values.update(changes)
    return KnowledgeItem(**values)


def state(*facts, scope=S.USER_APP):
    return dict(discovery_scope=scope, discovered_knowledge=list(facts or [fact()]),
                messages=[HumanMessage(content="UNTRUSTED_CHAT_ONLY: only managers can see orders")],
                raw_idea="UNTRUSTED_IDEA_ONLY", pm_is_complete=False)


def draft(sources):
    return dict(product_name=None, elevator_pitch=[], scope=dict(in_scope=[], out_of_scope=[]),
        personas=[], non_functional_constraints=[], deferred_items=[], open_questions=[],
        functional_requirements=[dict(id="FR-01", description="Orders over $100 require manager approval.",
            category="BUSINESS_RULES.approval_rules", actor_ids=["manager"], conditions=["order total > $100"],
            validation="TBD", source_fact_ids=[sources[0]["fact_id"]])])


def verdict(payload, **changes):
    result = {key: True for key in ClaimVerdict.model_fields if key not in ("claim_id", "explanation")}
    result.update(claim_id=payload["claim_id"], explanation="Claim preserves the cited approval rule and threshold.")
    result.update(changes)
    return result


def setup(monkeypatch, tmp_path, make_draft=None, judge=None, classify=None):
    calls = {"compile": [], "audit": [], "classify": []}
    def compile_(messages):
        payload = json.loads(messages[-1].content)
        calls["compile"].append((messages, payload))
        output = (make_draft or draft)(payload["confirmed_facts"])
        return {"parsed": output}
    def audit_(messages):
        payload = json.loads(messages[-1].content)
        calls["audit"].append(payload)
        return {"parsed": (judge or verdict)(payload)}
    monkeypatch.setattr(pm, "structured_llm", SimpleNamespace(invoke=compile_))
    monkeypatch.setattr(pm, "audit_llm", SimpleNamespace(invoke=audit_))
    def classify_(messages):
        payload = json.loads(messages[-1].content)
        calls["classify"].append(payload)
        return {"parsed": dict(categories=(classify(payload) if classify else ["BUSINESS_RULES.approval_rules"]),
                               explanation="Independent test classification")}
    monkeypatch.setattr(pm, "category_llm", SimpleNamespace(invoke=classify_))
    monkeypatch.setattr(pm, "OUTPUT_DIR", tmp_path)
    return calls


def test_compiles_only_confirmed_active_scope_and_saves_provenance(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    initial = state(fact(), fact(scope=S.ADMIN_DASHBOARD, value="ADMIN_ONLY"),
                    fact(knowledge_state=K.INFERRED, value="INFERRED_ONLY"))
    result = pm.pm_compile_node(initial)
    assert result["pm_is_complete"] and not result["compilation_errors"]
    sent = " ".join(m.content for messages, _ in calls["compile"] for m in messages)
    assert all(text not in sent for text in ("UNTRUSTED_CHAT_ONLY", "UNTRUSTED_IDEA_ONLY", "ADMIN_ONLY", "INFERRED_ONLY"))
    saved = json.loads((tmp_path / "requirements_mvp.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == "2.0" and saved["discovery_scope"] == "USER_APP"
    assert len(saved["source_facts"]) == 1
    source = saved["source_facts"][0]
    assert source["evidence"] == QUOTE and source["source_turn"] == 2 and source["role"] == "manager"
    assert saved["functional_requirements"][0]["source_fact_ids"] == [source["fact_id"]]
    assert saved["functional_requirements"][0]["conditions"] == ["order total > $100"]
    assert saved["validation_report"][0]["claim_supported"] is True
    assert not list(tmp_path.glob("*.tmp"))


def test_source_ids_are_stable_and_corrections_get_new_ids():
    first = build_source_snapshot(state())[0]
    assert first.fact_id == build_source_snapshot(state(fact(confidence=1)))[0].fact_id
    assert first.fact_id != build_source_snapshot(state(fact(value="Orders over $200 require approval")))[0].fact_id
    assert len(build_source_snapshot(state(fact(), fact()))) == 1


def test_short_answer_retains_its_original_question_in_snapshot():
    source = build_source_snapshot(state(fact(evidence="yes", source_question="Must managers approve orders over $100?")))[0]
    assert source.evidence == "yes"
    assert source.source_question == "Must managers approve orders over $100?"


@pytest.mark.parametrize("mutation", ["missing", "empty", "unknown", "wrong_category", "duplicate_ref", "duplicate_req"])
def test_structural_failure_never_reaches_auditor_or_overwrites_file(monkeypatch, tmp_path, mutation):
    def corrupt(sources):
        value = draft(sources)
        req = value["functional_requirements"][0]
        if mutation == "missing": del req["source_fact_ids"]
        if mutation == "empty": req["source_fact_ids"] = []
        if mutation == "unknown": req["source_fact_ids"] = ["invented-fact"]
        if mutation == "wrong_category": req["category"] = "BUSINESS_RULES.visibility_rules"
        if mutation == "duplicate_ref": req["source_fact_ids"] *= 2
        if mutation == "duplicate_req": value["functional_requirements"] *= 2
        return value
    calls = setup(monkeypatch, tmp_path, corrupt)
    output = tmp_path / "requirements_mvp.json"
    output.write_text("PREVIOUS_VERIFIED_PRD", encoding="utf-8")
    result = pm.pm_compile_node(state())
    assert not result["pm_is_complete"] and result["prd_contract"] is None
    assert output.read_text() == "PREVIOUS_VERIFIED_PRD"
    assert not calls["audit"] and len(calls["compile"]) == 2


def test_visibility_restriction_disguised_as_approval_fails_semantic_gate(monkeypatch, tmp_path):
    def corrupt(sources):
        value = draft(sources)
        value["functional_requirements"][0]["description"] = "Only managers can see orders over $100."
        return value
    def reject(payload):
        assert payload["cited_facts"][0]["evidence"] == QUOTE
        assert payload["claim"]["category"] == "BUSINESS_RULES.approval_rules"
        return verdict(payload, category_preserved=False, claim_supported=False,
                       explanation="Approval authority does not establish exclusive visibility.")
    calls = setup(monkeypatch, tmp_path, corrupt, reject)
    result = pm.pm_compile_node(state())
    assert len(calls["audit"]) == 2
    assert not result["pm_is_complete"] and not (tmp_path / "requirements_mvp.json").exists()
    assert "exclusive visibility" in result["compilation_errors"][0]


def test_blind_classification_blocks_visibility_even_if_auditor_would_approve(monkeypatch, tmp_path):
    def corrupt(sources):
        value = draft(sources)
        value["functional_requirements"][0]["description"] = "Only managers can see orders over $100."
        return value
    def classify(payload):
        assert "category" not in payload and "source_fact_ids" not in payload and "value" not in payload
        return ["BUSINESS_RULES.approval_rules"] if "evidence" in payload else ["BUSINESS_RULES.visibility_rules"]
    calls = setup(monkeypatch, tmp_path, corrupt, classify=classify)
    result = pm.pm_compile_node(state())
    assert not result["pm_is_complete"] and not calls["audit"]
    assert "independently classified" in result["compilation_errors"][0]


def test_blind_evidence_check_cannot_use_the_fact_value_to_fill_missing_words(monkeypatch, tmp_path):
    def classify(payload):
        assert payload == {"evidence": "orders over $100", "source_question": None}
        return []
    calls = setup(monkeypatch, tmp_path, classify=classify)
    result = pm.pm_compile_node(state(fact(evidence="orders over $100")))
    assert not result["pm_is_complete"] and not calls["audit"]
    assert len(calls["classify"]) == 1  # Immutable source classification reused on repair.
    assert "source evidence does not independently support" in result["compilation_errors"][0]


def test_overlapping_supported_categories_remain_valid(monkeypatch, tmp_path):
    quote = "Customers create a transaction, agree terms, and then fund escrow."
    first = fact(topic=T.USER_ROLES, key="responsibilities", role="customer", evidence=quote, value=quote)
    second = fact(topic=T.CORE_WORKFLOW, key="workflow_steps", role=None, evidence=quote, value=quote)
    def make(sources):
        value = draft(sources)
        req = value["functional_requirements"][0]
        req.update(description=quote, category="USER_ROLES.responsibilities", actor_ids=["customer"],
                   conditions=[], source_fact_ids=[source["fact_id"] for source in sources])
        return value
    calls = setup(monkeypatch, tmp_path, make, classify=lambda _: ["USER_ROLES.responsibilities", "CORE_WORKFLOW.workflow_steps"])
    assert pm.pm_compile_node(state(first, second))["pm_is_complete"]
    assert len(calls["classify"]) == 2  # One shared quote and one claim, not one call per duplicate quote.


@pytest.mark.parametrize("failed_check", ["source_evidence_supports_facts", "actors_preserved",
    "conditions_preserved", "validation_supported", "no_conflict_with_confirmed_facts"])
def test_each_semantic_check_is_required(monkeypatch, tmp_path, failed_check):
    setup(monkeypatch, tmp_path, judge=lambda p: verdict(p, **{failed_check: False}, explanation="Unsupported change"))
    initial = state(fact(evidence="orders over $100")) if failed_check == "source_evidence_supports_facts" else state()
    result = pm.pm_compile_node(initial)
    assert not result["pm_is_complete"] and failed_check in result["compilation_errors"][0]
    assert not (tmp_path / "requirements_mvp.json").exists()


@pytest.mark.parametrize("mode", ["timeout", "missing", "wrong_id", "non_boolean"])
def test_invalid_or_unavailable_auditor_fails_closed_without_retrying_compiler(monkeypatch, tmp_path, mode):
    def broken(payload):
        if mode == "timeout": raise TimeoutError("offline")
        result = verdict(payload)
        if mode == "missing": del result["conditions_preserved"]
        if mode == "wrong_id": result["claim_id"] = "other claim"
        if mode == "non_boolean": result["claim_supported"] = "true"
        return result
    calls = setup(monkeypatch, tmp_path, judge=broken)
    result = pm.pm_compile_node(state())
    assert not result["pm_is_complete"] and len(calls["compile"]) == 1
    assert not (tmp_path / "requirements_mvp.json").exists()


def test_repaired_draft_is_reaudited_before_save(monkeypatch, tmp_path):
    count = 0
    def revise(sources):
        nonlocal count
        count += 1
        value = draft(sources)
        if count == 1:
            value["functional_requirements"][0]["description"] = "Only managers can see orders over $100."
        return value
    def judge(payload):
        if "see orders" in payload["claim"]["description"]:
            return verdict(payload, claim_supported=False, explanation="Visibility is unsupported")
        return verdict(payload)
    calls = setup(monkeypatch, tmp_path, revise, judge)
    result = pm.pm_compile_node(state())
    assert result["pm_is_complete"] and len(calls["audit"]) == 2
    assert "repair_errors" in calls["compile"][1][1]
    assert "see orders" not in (tmp_path / "requirements_mvp.json").read_text()


def test_nonfunctional_and_scope_claims_cannot_bypass_audit(monkeypatch, tmp_path):
    def other_section(sources):
        value = draft(sources)
        reference = {key: val for key, val in value["functional_requirements"][0].items()
                     if key not in ("id", "description", "validation")}
        value["non_functional_constraints"] = [dict(**reference, text="Only managers can view orders.")]
        return value
    calls = setup(monkeypatch, tmp_path, other_section, lambda p:
        verdict(p, claim_supported=False, explanation="Unsupported visibility restriction"))
    result = pm.pm_compile_node(state())
    assert calls["audit"][0]["claim_id"] == "non_functional_constraints/0"
    assert not result["pm_is_complete"] and not (tmp_path / "requirements_mvp.json").exists()


def test_other_scope_source_id_is_unknown(monkeypatch, tmp_path):
    other = fact(scope=S.ADMIN_DASHBOARD)
    foreign_id = build_source_snapshot(state(other, scope=S.ADMIN_DASHBOARD))[0].fact_id
    def corrupt(sources):
        value = draft(sources)
        value["functional_requirements"][0]["source_fact_ids"] = [foreign_id]
        return value
    calls = setup(monkeypatch, tmp_path, corrupt)
    assert not pm.pm_compile_node(state(fact(), other))["pm_is_complete"]
    assert not calls["audit"]


def test_admin_scope_has_separate_artifact(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    result = pm.pm_compile_node(state(fact(scope=S.ADMIN_DASHBOARD), scope=S.ADMIN_DASHBOARD))
    assert result["pm_is_complete"] and (tmp_path / "requirements_admin_dashboard.json").exists()
    assert not (tmp_path / "requirements_mvp.json").exists()


def test_omitted_fact_cannot_disappear_during_repair(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    result = pm.pm_compile_node(state(fact(), fact(key="visibility_rules", value="Customers see only their own orders",
                                                 evidence="Customers see only their own orders", role="customer")))
    assert not result["pm_is_complete"] and "omitted" in result["compilation_errors"][0]
    assert not calls["audit"]


def test_failed_atomic_replace_keeps_previous_prd(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    path = tmp_path / "requirements_mvp.json"
    path.write_text("PREVIOUS_VERIFIED_PRD", encoding="utf-8")
    def fail(*_):
        raise OSError("disk error")
    monkeypatch.setattr(pm.os, "replace", fail)
    result = pm.pm_compile_node(state())
    assert not result["pm_is_complete"] and path.read_text() == "PREVIOUS_VERIFIED_PRD"
    assert not list(tmp_path.glob(".prd-*.tmp"))


def test_graph_returns_blocked_compilation_without_saving(monkeypatch, tmp_path):
    from agents import graph
    setup(monkeypatch, tmp_path, judge=lambda p: verdict(p, claim_supported=False, explanation="Unsupported claim"))
    monkeypatch.setattr(graph, "knowledge_tracker_node", lambda _: {})
    monkeypatch.setattr(graph, "interview_planner_node", lambda _: {"awaiting_confirmation": True})
    result = graph.build_graph().invoke(state())
    assert not result["pm_is_complete"] and not result["awaiting_confirmation"]
    assert result["compilation_errors"] and result["prd_contract"] is None
    assert not (tmp_path / "requirements_mvp.json").exists()


def test_missing_evidence_or_empty_snapshot_blocks_before_generation(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    assert not pm.pm_compile_node(state(fact(evidence="")))["pm_is_complete"]
    assert not pm.pm_compile_node(state(fact(knowledge_state=K.INFERRED)))["pm_is_complete"]
    assert not calls["compile"]


def test_long_snapshot_is_not_silently_truncated(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    result = pm.pm_compile_node(state(fact(evidence="x" * 40000)))
    assert not calls["compile"] and "context budget" in result["compilation_errors"][0]


def test_all_factual_sections_are_enumerated_for_validation():
    source = build_source_snapshot(state())[0].model_dump(mode="json")
    value = draft([source])
    ref = dict(source_fact_ids=[source["fact_id"]], category="BUSINESS_RULES.approval_rules",
               actor_ids=["manager"], conditions=["over $100"])
    claim = dict(**ref, text=QUOTE)
    value.update(product_name=claim, elevator_pitch=[claim], deferred_items=[claim],
                 non_functional_constraints=[claim], scope=dict(in_scope=[claim], out_of_scope=[claim]),
                 personas=[dict(**ref, name="Manager", description=QUOTE, key_behaviors=[claim])])
    assert len(list(draft_claims(PRDDraft.model_validate(value)))) == 9
