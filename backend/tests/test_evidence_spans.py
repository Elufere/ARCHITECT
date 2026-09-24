"""Formatting recovery stores original source bytes and rejects ambiguity."""
import pytest

from agents.evidence_spans import recover_evidence_span
from agents import knowledge_tracker as tracker
from agents.state import DiscoveryTopic as T, DiscoveryScope as S
from test_schema_alignment import raw, run


@pytest.mark.parametrize("source,evidence,expected", [
    ("we only have customers", "we only have customers.", "we only have customers"),
    ("we only have customers", '"We only have customers."', "we only have customers"),
    ("we only have customers", "  We only have customers.  ", "we only have customers"),
    ("First line.\nwe only\n have customers\nLast line.", "We only have customers.", "we only\n have customers"),
    ("we only have customers", "we only, have customers.", "we only have customers"),
    ("We only have customers.", "We only have customers.", "We only have customers."),
    ("customers wait; customers pay", "Customers pay.", "customers pay"),
    ("customers wait; customers wait", "Customers wait.", None),
    ("customers wait; customers wait", "customers wait", None),
    ("we only have customers", "we have only customers.", None),
    ("we only have customers", "we only have clients.", None),
    ("we only have customers", "we only have paying customers.", None),
    ("customers cannot pay", "customers can pay", None),
    ("customers", "customer.", None),
    ("cant", "can't", None),
    ("15", "1.5", None),
    ("1 5", "1.5", None),
    ("a b", "a + b", None),
    ("customers", "...", None),
])
def test_recovery(source, evidence, expected):
    result = recover_evidence_span(evidence, source)
    assert result == expected
    if result is not None:
        assert result in source


def test_absence_candidate_stores_original_source(monkeypatch):
    source = "we only have customers"
    state, calls = run(monkeypatch, source, {"ACTOR": [
        raw("secondary_users", "none", source + ".", roles=[], absence="none")]})
    assert len(state["discovered_knowledge"]) == 1
    assert state["discovered_knowledge"][0].evidence == source
    assert any(name == "GROUNDING" for name, _ in calls)


def test_gap_absence_recovers_before_exact_evidence_check(monkeypatch):
    from agents.semantic_validation import GapAnswer
    monkeypatch.setattr(tracker, "semantic_decision", lambda *_: GapAnswer(
        resolution="none", evidence="We only have customers.", confidence=1))
    item = tracker.extract_gap_absence("we only have customers", dict(
        current_topic=T.USER_ROLES, current_gap="secondary_users",
        discovered_knowledge=[], messages=[]), S.USER_APP)
    assert item is not None
    assert item.evidence == "we only have customers"


def test_ambiguous_candidate_is_not_committed(monkeypatch):
    source = "we only have customers\nwe only have customers"
    state, _ = run(monkeypatch, source, {"ACTOR": [
        raw("secondary_users", "none", "we only have customers.", roles=[], absence="none")]})
    assert state["discovered_knowledge"] == []
