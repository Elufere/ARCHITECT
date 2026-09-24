"""Duplicate comparison and active-store growth, with deterministic verdicts."""
import os

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.knowledge_duplicates import FactComparison, compare_candidate
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState as K


def fact(value="buyer confirms completion", **changes):
    return KnowledgeItem(**(dict(scope=S.USER_APP, topic=T.USER_ROLES,
        key="responsibilities", role="buyer", value=value, evidence=value,
        confidence=1, source_turn=1) | changes))


def verdict(relation, confidence=1, existing_id=0):
    return lambda *_: FactComparison(relation=relation, existing_id=existing_id, confidence=confidence)


def commit(monkeypatch, existing, candidates, relation="semantic_duplicate", **state_changes):
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: candidates)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "ground_items", lambda items, *_: items)
    monkeypatch.setattr(tracker, "semantic_decision", verdict(relation))
    state = dict(messages=[HumanMessage(content="\n".join(item.evidence for item in candidates))],
        discovered_knowledge=existing, discovery_scope=S.USER_APP,
        current_topic=T.USER_ROLES, topic_status={}, turn_count=2)
    state.update(state_changes)
    return tracker.knowledge_tracker_node(state)


def test_exact_duplicates_need_no_semantic_call():
    def unexpected(*_):
        pytest.fail("Exact duplicates must not invoke a model")
    old = fact()
    candidate = fact("BUYER CONFIRMS COMPLETION", evidence="Another exact source", source_turn=7)
    assert compare_candidate(candidate, [old], unexpected) == ("exact_duplicate", old)


@pytest.mark.parametrize("changes", [
    {"scope": S.ADMIN_DASHBOARD}, {"key": "permissions"},
    {"topic": T.USER_GOALS, "key": "primary_user_goals"},
    {"role": "seller"}, {"knowledge_state": K.INFERRED},
])
def test_different_buckets_never_compared(changes):
    def unexpected(*_):
        pytest.fail("Different buckets must not be compared")
    assert compare_candidate(fact(**changes), [fact()], unexpected) == ("new", None)


def test_canonical_owner_formatting_matches():
    assert compare_candidate(fact(role="account_holder"), [fact(role="account holder")],
                             verdict("new"))[0] == "exact_duplicate"


def test_actor_ids_and_new_aliases_are_not_discarded():
    actor = fact("customer", key="primary_users", role=None, roles=["customer"])
    assert compare_candidate(actor.model_copy(update={"roles": ["vendor"]}), [actor],
                             verdict("semantic_duplicate")) == ("new", None)
    assert compare_candidate(actor.model_copy(update={"aliases": {"customer": ["buyer"]}}),
                             [actor], verdict("semantic_duplicate")) == ("new", None)


def test_repeated_mentions_do_not_inflate_count(monkeypatch, capsys):
    old = fact()
    knowledge = [old]
    for turn in range(8):
        result = commit(monkeypatch, knowledge, [fact(
            "the buyer confirms the transaction is complete", source_turn=turn + 2)])
        knowledge = result["discovered_knowledge"]
    assert knowledge == [old]
    assert result["superseded_knowledge"] == []
    assert "Knowledge count: 2" not in capsys.readouterr().out


def test_same_turn_paraphrases_collapse(monkeypatch):
    result = commit(monkeypatch, [], [fact(), fact("the buyer confirms the transaction is complete")])
    assert result["discovered_knowledge"] == [fact()]


def test_refinement_replaces_and_archives(monkeypatch):
    old = fact()
    refined = fact("buyer confirms completion after inspecting the delivery")
    result = commit(monkeypatch, [old], [refined], "refinement")
    assert result["discovered_knowledge"] == [refined]
    assert result["superseded_knowledge"][0]["fact"]["evidence"] == old.evidence


def test_less_specific_repeat_keeps_refinement(monkeypatch):
    old = fact("buyer confirms completion after inspecting the delivery")
    result = commit(monkeypatch, [old], [fact()], "already_refined")
    assert result["discovered_knowledge"] == [old]


@pytest.mark.parametrize("relation", ["new", "contradiction"])
def test_similar_but_distinct_rules_coexist(monkeypatch, relation):
    old = fact("Requests over 100 require approval", topic=T.BUSINESS_RULES, key="approval_rules", role=None)
    new = old.model_copy(update={"value": "Requests from unverified accounts require approval"})
    assert commit(monkeypatch, [old], [new], relation)["discovered_knowledge"] == [old, new]


def test_explicit_correction_still_replaces(monkeypatch):
    old, new = fact(), fact("buyer cannot confirm completion")
    assert commit(monkeypatch, [old], [new], "correction", is_correction=True)["discovered_knowledge"] == [new]


@pytest.mark.parametrize("decision", [verdict("semantic_duplicate", .5),
    verdict("semantic_duplicate", existing_id=99), verdict("semantic_duplicate", existing_id=-1)])
def test_uncertain_or_invalid_verdict_preserves_new_information(decision):
    assert compare_candidate(fact("different action"), [fact()], decision) == ("new", None)


def test_comparison_failure_does_not_delete_facts():
    def fail(*_):
        raise ValueError("unavailable")
    assert compare_candidate(fact("different action"), [fact()], fail) == ("new", None)


@pytest.mark.skipif(os.environ.get("RUN_LIVE_KNOWLEDGE_DUPLICATES") != "1",
                    reason="Requires the configured live comparison model")
@pytest.mark.parametrize("old,new,expected", [
    ("buyer confirms completion", "the buyer confirms the transaction is complete", "semantic_duplicate"),
    ("buyer approves payments above 100", "buyer approves payments from unverified accounts", "new"),
    ("buyer can confirm completion", "buyer cannot confirm completion", "contradiction"),
])
def test_live_semantic_comparison(old, new, expected):
    assert compare_candidate(fact(new), [fact(old)], tracker.semantic_decision)[0] == expected
