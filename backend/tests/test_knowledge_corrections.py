"""Explicit supersession replaces only targeted prior facts and preserves history."""
import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.knowledge_corrections import CorrectionReview, correction_targets
from agents.knowledge_duplicates import FactComparison
from agents.inquiries import identify_open_inquiries
from agents.state import DiscoveryTopic as T, DiscoveryScope as S, KnowledgeState as K
from test_knowledge_duplicates import fact


def model_foundation():
    return [
        fact("Customers use the app", key="primary_users", role=None, roles=["customer"]),
        fact("Customers create orders", key="responsibilities", role="customer"),
        fact(
            "Customers complete purchases safely",
            topic=T.USER_GOALS,
            key="primary_user_goals",
            role="customer",
        ),
        fact(
            "Customers create, pay for, and complete orders",
            topic=T.CORE_WORKFLOW,
            key="workflow_steps",
            role=None,
        ),
        fact(
            "The order is complete when fulfillment is confirmed",
            topic=T.CORE_WORKFLOW,
            key="completion_condition",
            role=None,
        ),
    ]


def commit(monkeypatch, prior, candidates, *, relation="new", targets=(), initial=None, flag=False):
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: candidates)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_: items)
    def decide(name, *_):
        if name == "FACT_COMPARISON":
            return FactComparison(relation=relation, existing_id=0, confidence=1)
        if name == "CORRECTION_REVIEW":
            return CorrectionReview(superseded_ids=list(targets), confidence=1)
        raise AssertionError(name)
    monkeypatch.setattr(tracker, "semantic_decision", decide)
    state = dict(initial or {}, discovered_knowledge=prior,
        messages=[HumanMessage(content="\n".join(item.evidence for item in candidates))],
        discovery_scope=S.USER_APP, is_correction=flag)
    return tracker.knowledge_tracker_node(state)


@pytest.mark.parametrize("flag", [False, True])
def test_targeted_correction_preserves_independent_rules(monkeypatch, flag):
    old = fact("buyer can cancel before funding", key="permissions")
    other = fact("buyer can view only their own invoices", key="permissions")
    new = fact("buyer cannot cancel after seller acceptance", key="permissions",
               evidence="Correction: replace the cancellation rule; buyer cannot cancel after seller acceptance")
    result = commit(monkeypatch, [old, other], [new], relation="correction", flag=flag)
    assert result["discovered_knowledge"] == [other, new]
    model = result["product_model"][T.USER_ROLES.value]
    assert f"USER_ROLES.permissions[buyer]: {new.value}" in model
    assert f"USER_ROLES.permissions[buyer]: {old.value}" not in model
    assert result["superseded_knowledge"] == [dict(
        fact=old.model_dump(mode="json"), superseded_by=new.model_dump(mode="json"))]


def test_intent_flag_alone_does_not_delete_facts(monkeypatch):
    old, new = fact("buyer uploads receipts"), fact("buyer also downloads invoices")
    assert commit(monkeypatch, [old], [new], flag=True)["discovered_knowledge"] == [old, new]


def test_different_conditions_are_not_automatically_conflicts(monkeypatch):
    old = fact("buyer can cancel before funding", key="permissions")
    new = fact("buyer cannot cancel after seller acceptance", key="permissions")
    assert commit(monkeypatch, [old], [new])["discovered_knowledge"] == [old, new]


def test_explicit_conflicting_decision_supersedes_without_intent_flag(monkeypatch):
    old, new = fact("buyer can cancel", key="permissions"), fact("buyer cannot cancel", key="permissions")
    result = commit(monkeypatch, [old], [new], relation="contradiction", targets=[0])
    assert result["discovered_knowledge"] == [new]


def test_ambiguous_conflict_does_not_destroy_prior_fact(monkeypatch):
    old, new = fact("first assertion"), fact("second assertion")
    assert commit(monkeypatch, [old], [new], relation="contradiction")["discovered_knowledge"] == [old, new]


def test_multi_actor_correction_keeps_same_turn_siblings(monkeypatch):
    old = fact("old group uses the app", key="primary_users", role=None, roles=["old_group"])
    new = [fact(f"Correction: {role} uses the app instead", key="primary_users", role=None, roles=[role])
           for role in ("editor", "reviewer")]
    result = commit(monkeypatch, [old], new, targets=[0], flag=True)
    assert result["discovered_knowledge"] == new
    assert len(result["superseded_knowledge"]) == 1


def test_multiple_specific_prior_rules_can_be_replaced(monkeypatch):
    old = [fact("first old rule"), fact("second old rule"), fact("independent rule")]
    new = fact("Correction: replace the first two rules with this one")
    result = commit(monkeypatch, old, [new], targets=[0, 1], flag=True)
    assert result["discovered_knowledge"] == [old[2], new]
    assert len(result["superseded_knowledge"]) == 2


def test_correction_preserving_model_state_emits_no_topic_lifecycle(monkeypatch, capsys):
    old = fact("customers may view only their own orders", key="permissions", role="customer")
    prior = [*model_foundation(), old]
    new = fact(
        "Correction: customers may view only assigned orders",
        key="permissions",
        role="customer",
    )
    result = commit(monkeypatch, prior, [new], relation="correction")

    assert old not in result["discovered_knowledge"]
    assert new in result["discovered_knowledge"]
    assert "topic_status" not in result
    assert "topic_maturity" not in result
    output = capsys.readouterr().out
    assert "TOPIC INVALIDATION" not in output
    assert "TOPIC STATUS" not in output


def test_new_actor_changes_inquiry_frontier_without_topic_reopening(monkeypatch, capsys):
    prior = model_foundation()
    new = fact(
        "Editors are also primary app users",
        key="primary_users",
        role=None,
        roles=["editor"],
    )
    result = commit(monkeypatch, prior, [new])

    inquiry_state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": result["discovered_knowledge"],
        "fact_acquisition": result.get("fact_acquisition", {}),
        "active_requirements": {},
        "requirement_coverage": {},
        "eligible_requirement_keys": [],
        "validation_issues": [],
        "validation_candidate_blocking": False,
    }
    inquiries = identify_open_inquiries(inquiry_state)

    assert any(
        item.anchor_gap == "responsibilities::editor"
        and item.role == "editor"
        for item in inquiries
    )
    assert "topic_status" not in result
    assert "topic_maturity" not in result
    output = capsys.readouterr().out
    assert "TOPIC INVALIDATION" not in output
    assert "TOPIC STATUS" not in output


@pytest.mark.parametrize("changes", [{"scope": S.ADMIN_DASHBOARD}, {"role": "seller"},
    {"key": "permissions"}, {"knowledge_state": K.INFERRED}])
def test_correction_review_cannot_remove_other_scope_field_owner_or_inference(changes):
    old = fact(**changes)
    def unexpected(*_):
        pytest.fail("Unrelated fact sent to correction reviewer")
    assert correction_targets(fact(), [old], fact().evidence, unexpected) == []


def test_invalid_review_ids_fail_closed():
    assert correction_targets(fact(), [fact()], fact().evidence,
        lambda *_: CorrectionReview(superseded_ids=[0, 4], confidence=1)) == []


def test_historical_replay_cannot_supersede_current_fact(monkeypatch):
    old, new = fact("current decision"), fact("historical decision")
    result = commit(monkeypatch, [old], [new], relation="correction",
                    initial={"conversation_intent": "objection"}, flag=True)
    assert old in result["discovered_knowledge"]
    assert result["superseded_knowledge"] == []


def test_confirmed_whole_field_absence_archives_replaced_positive(monkeypatch):
    old = fact("an approval is required", topic=T.BUSINESS_RULES, key="approval_rules", role=None)
    new = fact("none", topic=T.BUSINESS_RULES, key="approval_rules", role=None,
               absence="none", evidence="Actually, no approvals are required.")
    result = commit(monkeypatch, [old], [new])
    assert result["discovered_knowledge"] == [new]
    assert result["superseded_knowledge"][0]["fact"]["value"] == old.value
