"""Short role-policy answers retain their meaning instead of becoming absence."""
import json

import pytest
from langchain_core.messages import AIMessage

from agents import knowledge_tracker as tracker
from agents.interview_planner import build_gap_info
from agents.state import DiscoveryTopic as T, DiscoveryScope as S
from test_gap_absence import models, state


QUESTION = "Can a customer have multiple roles in a single transaction, such as being both a buyer and a seller?"
POLICY = "A customer cannot be both buyer and seller in the same transaction."


@pytest.mark.parametrize("gap,question,answer,value", [
    ("multiple_roles", QUESTION, "no", POLICY),
    ("multiple_roles", "Can a person hold both roles?", "yes", "A person can hold both roles."),
    ("role_transitions", "Can customers switch roles during a transaction?", "no",
     "Customers cannot switch roles during a transaction."),
])
def test_policy_answer_closes_only_its_gap(monkeypatch, gap, question, answer, value):
    initial = state(answer, gap)
    initial["messages"][0] = AIMessage(content=question)
    calls = models(monkeypatch, resolution="policy", evidence=answer, value=value,
        outputs={"ACTOR": [dict(key=gap, value="none", evidence=answer, confidence=1, absence="none")]})
    result = tracker.knowledge_tracker_node(initial)
    added = result["discovered_knowledge"][2:]
    assert len(added) == 1
    assert added[0].key == gap and added[0].value == value
    assert added[0].evidence == answer and added[0].absence is None
    gaps = build_gap_info({**initial, **result}, T.USER_ROLES)["missing_keys"]
    assert gap not in gaps
    assert ("role_transitions" if gap == "multiple_roles" else "multiple_roles") in gaps
    audit = json.loads(calls["GROUNDING"][-1].content)
    assert audit["question"] == question
    assert audit["active_gap_review"]["value"] == value


@pytest.mark.parametrize("value", [None, "", "none", "not applicable"])
def test_policy_requires_substantive_value(monkeypatch, value):
    models(monkeypatch, resolution="policy", evidence="no", value=value)
    assert tracker.extract_gap_absence("no", state("no", "multiple_roles"), S.USER_APP) is None


def test_policy_cannot_fill_other_fields(monkeypatch):
    models(monkeypatch, resolution="policy", evidence="no", value=POLICY)
    assert tracker.extract_gap_absence("no", state("no", "secondary_users"), S.USER_APP) is None


def test_policy_still_requires_independent_grounding(monkeypatch):
    initial = state("no", "multiple_roles")
    initial["messages"][0] = AIMessage(content=QUESTION)
    models(monkeypatch, resolution="policy", evidence="no", value=POLICY, supported=[])
    result = tracker.knowledge_tracker_node(initial)
    assert result["discovered_knowledge"] == initial["discovered_knowledge"]


def test_incorrect_none_still_requires_whole_field_grounding(monkeypatch):
    initial = state("no", "multiple_roles")
    initial["messages"][0] = AIMessage(content=QUESTION)
    models(monkeypatch, resolution="none", evidence="no", supported=[])
    result = tracker.knowledge_tracker_node(initial)
    assert result["discovered_knowledge"] == initial["discovered_knowledge"]
