"""One deterministic active-answer repair; coverage still requires a commit."""
import pytest

from agents.interview_planner import build_gap_info, interview_planner_node
from agents.state import DiscoveryTopic as T
from test_schema_alignment import raw, run


TEXT = "The clinic manager must approve new provider profiles."


def answer(monkeypatch, candidate, **kwargs):
    return run(monkeypatch, TEXT, {"RULES": [candidate]},
               topic=T.BUSINESS_RULES, gap="approval_rules",
               question="What approvals are required?", **kwargs)


def test_repaired_answer_commits_and_planner_moves_past_gap(monkeypatch, capsys):
    candidate = raw("BUSINESS_RULES.approval_rules", TEXT, TEXT, topic="BUSINESS_RULES")
    state, calls = answer(monkeypatch, candidate)
    assert "approval_rules" not in build_gap_info(state, T.BUSINESS_RULES)["missing_keys"]
    state.update(interview_planner_node(state))
    assert state["current_gap"] != "approval_rules"
    item = state["discovered_knowledge"][0]
    assert item.key == "approval_rules" and item.value == TEXT and item.evidence == TEXT
    assert candidate["key"] == "BUSINESS_RULES.approval_rules"
    assert [name for name, _ in calls].count("RULES") == 1
    output = capsys.readouterr().out
    labels = ["ACTIVE ANSWER FORMAT ORIGINAL:", "ACTIVE ANSWER FORMAT REJECTION:",
              "ACTIVE ANSWER FORMAT REPAIR ATTEMPT: 1/1", "ACTIVE ANSWER FORMAT REPAIRED:",
              "CANDIDATE FINAL COMMIT:"]
    positions = [output.index(label) for label in labels]
    assert positions == sorted(positions)
    assert output.count("ACTIVE ANSWER FORMAT REPAIR ATTEMPT:") == 1


@pytest.mark.parametrize("changes", [
    {"key": "BUSINESS_RULES.BUSINESS_RULES.approval_rules"},
    {"confidence": 2},
    {"evidence": "The system automatically approves every profile."},
    {"value": ""},
])
def test_failed_repair_is_bounded_and_gap_stays_open(monkeypatch, capsys, changes):
    candidate = raw("BUSINESS_RULES.approval_rules", TEXT, TEXT, topic="BUSINESS_RULES") | changes
    state, calls = answer(monkeypatch, candidate)
    assert state["discovered_knowledge"] == []
    assert "approval_rules" in build_gap_info(state, T.BUSINESS_RULES)["missing_keys"]
    assert [name for name, _ in calls].count("RULES") == 1
    output = capsys.readouterr().out
    assert output.count("ACTIVE ANSWER FORMAT REPAIR ATTEMPT:") == 1
    assert "ACTIVE ANSWER FORMAT FINAL REJECT:" in output


def test_repaired_schema_does_not_bypass_grounding(monkeypatch, capsys):
    state, _ = answer(monkeypatch,
        raw("BUSINESS_RULES.approval_rules", TEXT, TEXT, topic="BUSINESS_RULES"),
        reject=[("approval_rules", TEXT)])
    assert state["discovered_knowledge"] == []
    assert "approval_rules" in build_gap_info(state, T.BUSINESS_RULES)["missing_keys"]
    output = capsys.readouterr().out
    assert "ACTIVE ANSWER FORMAT REPAIRED:" in output
    assert "CANDIDATE FINAL REJECT (grounding):" in output
    assert "CANDIDATE FINAL COMMIT:" not in output


@pytest.mark.parametrize("changes", [
    {"key": "BUSINESS_RULES.unknown_field"},
    {"key": "EXCEPTIONS.approval_rules"},
    {"topic": "EXCEPTIONS"},
])
def test_active_gap_cannot_supply_missing_or_incompatible_semantics(monkeypatch, changes):
    state, calls = answer(monkeypatch,
        raw("BUSINESS_RULES.approval_rules", TEXT, TEXT, topic="BUSINESS_RULES") | changes)
    assert state["discovered_knowledge"] == []
    assert "approval_rules" in build_gap_info(state, T.BUSINESS_RULES)["missing_keys"]
    assert [name for name, _ in calls].count("RULES") == 1
