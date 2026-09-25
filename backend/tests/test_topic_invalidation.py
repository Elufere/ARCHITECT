"""Completed coverage changes only at a confirmed knowledge commit boundary."""
from copy import deepcopy

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.interview_planner import build_gap_info, interview_planner_node
from coverage_test_utils import coverage_for_facts
from agents.state import (
    DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K,
    TopicStatus as Status, TopicMaturity,
)


def fact(topic, key, value, **fields):
    return KnowledgeItem(topic=topic, scope=fields.pop("scope", S.USER_APP),
                         key=key, value=value, evidence=value, confidence=1, **fields)


def actor(role="vendor", **fields):
    return fact(T.USER_ROLES, "secondary_users", f"{role} uses this app",
                roles=[role], **fields)


def completed_state():
    knowledge = [
        fact(T.USER_ROLES, "primary_users", "Customers use the app", roles=["customer"]),
        fact(T.USER_ROLES, "secondary_users", "none", absence="none"),
        fact(T.USER_ROLES, "responsibilities", "submit orders", role="customer"),
        fact(T.USER_ROLES, "permissions", "view own orders only", role="customer"),
        fact(T.USER_ROLES, "multiple_roles", "one role per account"),
        fact(T.USER_ROLES, "role_transitions", "roles do not change"),
        fact(T.USER_GOALS, "primary_user_goals", "receive ordered goods", role="customer"),
        fact(T.USER_GOALS, "success_criteria", "goods received"),
        fact(T.USER_GOALS, "motivations", "avoid travel"),
    ]
    initial = dict(messages=[HumanMessage(content="New information")], current_topic=None,
                   discovery_scope=S.USER_APP, discovered_knowledge=knowledge,
                   topic_status={T.USER_ROLES: Status.COMPLETED, T.USER_GOALS: Status.COMPLETED},
                   topic_maturity={T.USER_ROLES: TopicMaturity.DECISION_READY,
                                   T.USER_GOALS: TopicMaturity.DECISION_READY}, turn_count=3)
    initial["gap_coverage"] = coverage_for_facts(initial, T.USER_ROLES, T.USER_GOALS)
    assert not build_gap_info(initial, T.USER_ROLES)["missing_keys"]
    assert not build_gap_info(initial, T.USER_GOALS)["missing_keys"]
    return initial


def commit(monkeypatch, initial, candidates, accepted=None):
    # Lifecycle fixtures model committed facts; absence supersession is audited
    # independently in test_absence_supersession.py.
    monkeypatch.setattr(tracker, "can_replace_absence", lambda *_: True)
    # These fixtures supply distinct valid facts; semantic comparison has its
    # own tests and must not invoke a live model during lifecycle checks.
    monkeypatch.setattr(tracker, "semantic_decision", lambda *_:
                        tracker.FactComparison(relation="new", confidence=1))
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: candidates)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_:
                        items if accepted is None else accepted)
    original = deepcopy(initial)
    result = tracker.knowledge_tracker_node(initial)
    assert initial == original  # No mutation of incoming status or knowledge.
    return {**initial, **result}


def test_new_confirmed_same_scope_actor_reopens_only_affected_topics(monkeypatch, capsys):
    initial = completed_state()
    initial["topic_status"][T.BUSINESS_RULES] = Status.COMPLETED
    updated = commit(monkeypatch, initial, [actor()])
    assert updated["topic_status"][T.USER_ROLES] == Status.PARTIAL
    assert updated["topic_status"][T.USER_GOALS] == Status.PARTIAL
    assert updated["topic_status"][T.BUSINESS_RULES] == Status.COMPLETED
    output = capsys.readouterr().out
    assert output.index("TOPIC STATUS MERGE") < output.index("TOPIC INVALIDATION")
    assert "USER_ROLES: COMPLETED -> PARTIAL" in output
    assert "reason: new confirmed same-scope actor vendor" in output
    assert "new gaps: responsibilities::vendor, permissions::vendor" in output
    assert "new gaps: secondary_user_goals::vendor" in output
    plan = interview_planner_node(updated)
    assert plan["current_topic"] == T.USER_ROLES
    assert plan["current_gap"] == "responsibilities::vendor"
    repeated = commit(monkeypatch, updated, [actor()])
    assert "TOPIC INVALIDATION" not in capsys.readouterr().out
    assert repeated["topic_status"] == updated["topic_status"]


@pytest.mark.parametrize("kind", [
    "duplicate", "alias", "capacity", "incidental", "cross_scope", "inferred",
    "rejected", "hallucinated", "covered_fact",
])
def test_noninvalidating_information_keeps_completed_topics(monkeypatch, capsys, kind):
    initial = completed_state()
    accepted = None
    if kind == "duplicate":
        candidates = [initial["discovered_knowledge"][0]]
    elif kind in ("alias", "capacity"):
        candidates = [initial["discovered_knowledge"][0].model_copy(update={
            "aliases": {"customer": ["account_holder"] if kind == "alias" else ["sender", "recipient"]}})]
    elif kind == "incidental":
        candidates = [fact(T.BUSINESS_RULES, "approval_rules", "A mediator reviews disputes")]
    elif kind == "cross_scope":
        candidates = [actor(scope=S.ADMIN_DASHBOARD)]
    elif kind == "inferred":
        candidates = [actor(knowledge_state=K.INFERRED)]
    elif kind in ("rejected", "hallucinated"):
        candidates, accepted = [actor()], []
    else:
        candidates = [fact(T.USER_ROLES, "responsibilities", "track orders", role="customer")]
    updated = commit(monkeypatch, initial, candidates, accepted)
    assert updated["topic_status"][T.USER_ROLES] == Status.COMPLETED
    assert updated["topic_status"][T.USER_GOALS] == Status.COMPLETED
    plan = interview_planner_node(updated)
    assert plan.get("topic_status", updated["topic_status"])[T.USER_ROLES] == Status.COMPLETED
    assert plan.get("topic_status", updated["topic_status"])[T.USER_GOALS] == Status.COMPLETED
    assert "TOPIC INVALIDATION" not in capsys.readouterr().out


def test_planner_cannot_reopen_from_existing_gaps_alone(capsys):
    initial = completed_state()
    initial["discovered_knowledge"].append(actor())
    assert build_gap_info(initial, T.USER_ROLES)["missing_keys"]
    plan = interview_planner_node(initial)
    assert plan["topic_status"][T.USER_ROLES] == Status.COMPLETED
    assert plan["topic_status"][T.USER_GOALS] == Status.COMPLETED
    assert "TOPIC INVALIDATION" not in capsys.readouterr().out


def test_new_actor_with_full_coverage_in_same_commit_needs_no_reopening(monkeypatch, capsys):
    updated = commit(monkeypatch, completed_state(), [
        actor(), fact(T.USER_ROLES, "responsibilities", "fulfil orders", role="vendor"),
        fact(T.USER_ROLES, "permissions", "manage assigned orders only", role="vendor"),
        fact(T.USER_GOALS, "secondary_user_goals", "complete deliveries", role="vendor"),
    ])
    assert all(status == Status.COMPLETED for status in updated["topic_status"].values())
    assert "TOPIC INVALIDATION" not in capsys.readouterr().out


def test_correction_reopens_only_newly_invalidated_goal_coverage(monkeypatch, capsys):
    initial = {**completed_state(), "is_correction": True}
    updated = commit(monkeypatch, initial, [
        fact(T.USER_ROLES, "primary_users", "none", absence="none"),
        actor("customer"),
    ])
    assert updated["topic_status"][T.USER_ROLES] == Status.COMPLETED
    assert updated["topic_status"][T.USER_GOALS] == Status.PARTIAL
    output = capsys.readouterr().out
    assert "reason: explicit correction changed confirmed completion coverage" in output
    assert "new gaps: secondary_user_goals::customer" in output


def test_correction_that_preserves_coverage_stays_completed(monkeypatch, capsys):
    initial = {**completed_state(), "is_correction": True}
    updated = commit(monkeypatch, initial, [
        fact(T.USER_ROLES, "permissions", "view own and shared orders", role="customer")])
    assert updated["topic_status"] == initial["topic_status"]
    assert "TOPIC INVALIDATION" not in capsys.readouterr().out


def test_new_role_source_coverage_reopens_goals_without_new_actor_identity(monkeypatch, capsys):
    initial = completed_state()
    # The same canonical actor gains a confirmed secondary classification;
    # the goal contract now requires secondary_user_goals for that actor too.
    updated = commit(monkeypatch, initial, [actor("customer")])
    assert updated["topic_status"][T.USER_ROLES] == Status.COMPLETED
    assert updated["topic_status"][T.USER_GOALS] == Status.PARTIAL
    output = capsys.readouterr().out
    assert "committed confirmed fact changed required completion coverage" in output
    assert "new gaps: secondary_user_goals::customer" in output


def test_invalidation_logs_only_new_gaps_not_preexisting_missing_coverage(monkeypatch, capsys):
    initial = completed_state()
    initial["discovered_knowledge"] = [item for item in initial["discovered_knowledge"]
                                       if item.key != "responsibilities"]
    # Serialized enum keys/statuses must retain the same lifecycle semantics.
    initial["topic_status"] = {topic.value: status.value
                               for topic, status in initial["topic_status"].items()}
    updated = commit(monkeypatch, initial, [actor()])
    assert updated["topic_status"][T.USER_ROLES] == Status.PARTIAL
    output = capsys.readouterr().out
    gap_lines = [line for line in output.splitlines() if line.startswith("new gaps:")]
    assert "new gaps: responsibilities::vendor, permissions::vendor" in gap_lines
    assert all("responsibilities::customer" not in line for line in gap_lines)


def test_spelling_change_does_not_turn_existing_gaps_into_new_coverage(monkeypatch, capsys):
    initial = {**completed_state(), "is_correction": True}
    initial["discovered_knowledge"] = [item for item in initial["discovered_knowledge"]
                                       if item.key != "responsibilities"]
    declaration = initial["discovered_knowledge"][0].model_copy(update={
        "roles": ["customers"], "value": "Customers use this application"})
    updated = commit(monkeypatch, initial, [declaration])
    assert updated["topic_status"] == initial["topic_status"]
    assert "TOPIC INVALIDATION" not in capsys.readouterr().out


def test_explicit_confirmation_of_inferred_actor_uses_merge_boundary(monkeypatch, capsys):
    monkeypatch.setattr(tracker, "can_replace_absence", lambda *_: True)
    initial = completed_state()
    initial["discovered_knowledge"].append(actor(knowledge_state=K.INFERRED))
    initial.update(conversation_intent="confirmation", next_discovery_move="confirm_inference",
                   current_topic=T.USER_ROLES, current_gap="secondary_users")
    result = tracker.knowledge_tracker_node(initial)
    assert result["topic_status"][T.USER_ROLES] == Status.PARTIAL
    assert result["topic_status"][T.USER_GOALS] == Status.PARTIAL
    assert initial["topic_status"][T.USER_ROLES] == Status.COMPLETED
    output = capsys.readouterr().out
    assert output.index("TOPIC STATUS MERGE") < output.index("TOPIC INVALIDATION")
