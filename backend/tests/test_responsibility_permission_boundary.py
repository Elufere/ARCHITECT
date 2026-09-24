"""Separate performance/capability from explicit authorization semantics.

Offline tests exercise adversarial candidate filtering and prompt contracts.
RUN_LIVE_PERMISSION_BOUNDARY=1 additionally checks production model behavior.
"""
import json
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


CASES = [
    ("The buyer funds the escrow.", "buyer", "funds the escrow", None),
    ("Only the buyer may confirm completion.", "buyer", None, "only the buyer may confirm completion"),
    ("The seller cannot release funds to themselves.", "seller", None, "cannot release funds to themselves"),
    ("The seller uploads proof of delivery.", "seller", "uploads proof of delivery", None),
    ("Managers can view only their assigned region.", "manager", None, "can view only their assigned region"),
    ("The buyer can fund the escrow.", "buyer", "can fund the escrow", None),
    ("Managers manage their own reports.", "manager", "manage their own reports", None),
    ("The seller uploads proof of delivery but cannot release funds to themselves.",
     "seller", "uploads proof of delivery", "cannot release funds to themselves"),
    ("Managers review records and are the only users authorized to review records.",
     "manager", "review records", "only managers are authorized to review records"),
]


def state(text, roles):
    return dict(messages=[HumanMessage(content=text)], current_topic=T.USER_ROLES,
                discovery_scope=S.USER_APP, turn_count=2, topic_status={},
                discovered_knowledge=[KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP,
                    key="primary_users", roles=[role], value=role,
                    evidence=f"{role} uses this application", confidence=1) for role in roles])


def install(monkeypatch, text, rows, expected, *, support_all=False):
    calls = {}

    def invoke(name, messages):
        calls[name] = messages
        if name in ("RESPONSIBILITY", "PERMISSION"):
            key = "responsibilities" if name == "RESPONSIBILITY" else "permissions"
            return {"items": [dict(key=key, role=role, value=value, evidence=text, confidence=1)
                              for candidate_key, role, value in rows if candidate_key == key]}
        if name == "GROUNDING":
            assert "Audit responsibilities and permissions independently" in messages[0].content
            assert "another action or\n  actor in the same quote" in messages[0].content
            payload = json.loads(messages[-1].content)
            groups = payload["evidence_groups"]
            return dict(
                evidence_categories=[dict(evidence_id=int(group["evidence_id"]),
                    categories=sorted({f"USER_ROLES.{key}" for key, _, _ in expected}))
                    for group in groups],
                supported_ids=[c["id"] for group in groups for c in group["candidates"]
                               if support_all or (c["key"], c["role"], c["value"]) in expected],
                confirmed_absence_ids=[],
                rejection_reasons={str(c["id"]): "This action/owner has no stated boundary"
                    for group in groups for c in group["candidates"]
                    if not support_all and (c["key"], c["role"], c["value"]) not in expected})
        return {"items": []}

    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)
    monkeypatch.setattr(tracker, "extraction_models", lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name: invoke(name, messages))
        for name in [*(entry[0] for entry in tracker.PASSES), "GROUNDING"]})
    return calls


@pytest.mark.parametrize("text,role,responsibility,permission", CASES)
def test_category_gate_rejects_duplicate_candidates_but_keeps_genuine_overlap(
        monkeypatch, text, role, responsibility, permission):
    # The extractor proposes both even when only one category is supported.
    rows = [("responsibilities", role, responsibility or permission),
            ("permissions", role, permission or responsibility)]
    expected = {(key, role, value) for key, value in
                [("responsibilities", responsibility), ("permissions", permission)] if value}
    calls = install(monkeypatch, text, rows, expected, support_all=True)
    result = tracker.knowledge_tracker_node(state(text, [role]))
    accepted = [item for item in result["discovered_knowledge"] if item.key in ("responsibilities", "permissions")]
    assert {(item.key, item.role, item.value) for item in accepted} == expected
    assert all(item.evidence == text for item in accepted)
    assert "not an additional responsibility" in calls["RESPONSIBILITY"][0].content
    assert "The candidate value must retain that boundary" in calls["PERMISSION"][0].content
    assert "can" in calls["PERMISSION"][0].content


def test_boundary_cannot_be_borrowed_from_another_actors_action(monkeypatch):
    text = "The buyer funds the escrow, and only the seller may release funds."
    expected = {("responsibilities", "buyer", "funds the escrow"),
                ("permissions", "seller", "only the seller may release funds")}
    rows = [*expected, ("permissions", "buyer", "funds the escrow"),
            ("responsibilities", "seller", "release funds")]
    install(monkeypatch, text, rows, expected)
    result = tracker.knowledge_tracker_node(state(text, ["buyer", "seller"]))
    assert {(item.key, item.role, item.value) for item in result["discovered_knowledge"]
            if item.role} == expected


def test_unknown_owners_are_rejected_before_semantic_audit(monkeypatch):
    text = "Managers review records and only managers may approve changes."
    rows = [("responsibilities", "unregistered", "review records"),
            ("permissions", "unregistered", "only managers may approve changes")]
    calls = install(monkeypatch, text, rows, set(rows), support_all=True)
    initial = state(text, ["manager"])
    result = tracker.knowledge_tracker_node(initial)
    assert result["discovered_knowledge"] == initial["discovered_knowledge"]
    assert "GROUNDING" not in calls


@pytest.mark.skipif(os.environ.get("RUN_LIVE_PERMISSION_BOUNDARY") != "1",
                    reason="Requires the configured live Ollama model")
@pytest.mark.parametrize("text,role,responsibility,permission", CASES)
def test_live_responsibility_permission_boundary(text, role, responsibility, permission):
    initial = state(text, [role])
    expected = {key for key, value in [("responsibilities", responsibility), ("permissions", permission)] if value}
    extracted = tracker.extract_passes(text, initial, S.USER_APP)
    grounded = tracker.ground_items(extracted, text, initial)
    for records in (extracted, grounded):
        owned = [item for item in records if item.key in ("responsibilities", "permissions")]
        assert {item.key for item in owned} == expected
        assert all(item.role == role and item.evidence in text for item in owned)
