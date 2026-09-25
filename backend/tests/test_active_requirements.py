"""Focused tests for product-specific active requirement state."""
from uuid import uuid4

import pytest

from agents.interview_checkpoint import load_checkpoint, save_checkpoint
from agents.requirements import (
    ActiveRequirement,
    RequirementActivationSource,
    RequirementEvidenceRef,
    RequirementStatus,
    attach_requirement_evidence,
    get_requirement,
    list_requirements,
    register_requirement,
    update_requirement_status,
)
from agents.state import DiscoveryScope, DiscoveryTopic


def requirement(**overrides):
    data = dict(
        id="identity.anonymous_order_ownership",
        topic=DiscoveryTopic.BUSINESS_RULES,
        parent_gap="ownership_rules",
        label="Anonymous order ownership",
        description="Determine how an unauthenticated order is owned before account linking.",
        activation_sources=[
            RequirementActivationSource(
                source_type="confirmed_fact",
                source_key="checkout.authentication_required",
                source_value="false",
                source_turn=4,
                evidence_ref="fact-checkout-auth",
                activation_rule_id="anonymous-checkout-v1",
            )
        ],
    )
    data.update(overrides)
    return ActiveRequirement(**data)


def test_register_and_retrieve_requirement():
    item = requirement()
    store = register_requirement({}, item)
    assert get_requirement(store, item.id, scope=DiscoveryScope.USER_APP) == item
    assert list_requirements(store, topic=DiscoveryTopic.BUSINESS_RULES) == [item]


def test_requirement_status_transitions_preserve_record():
    item = requirement()
    store = register_requirement({}, item)
    for status in (
        RequirementStatus.DEFERRED,
        RequirementStatus.RESOLVED,
        RequirementStatus.NOT_APPLICABLE,
        RequirementStatus.INACTIVE,
        RequirementStatus.ACTIVE,
    ):
        store = update_requirement_status(store, item.id, status, scope=DiscoveryScope.USER_APP)
        scoped = store[f"USER_APP|{item.id}"]
        assert scoped.status == status
        assert scoped.activation_sources == item.activation_sources


def test_inactive_requirement_is_preserved_and_filterable():
    item = requirement()
    store = update_requirement_status(
        register_requirement({}, item), item.id, RequirementStatus.INACTIVE,
        scope=DiscoveryScope.USER_APP,
    )
    key = f"USER_APP|{item.id}"
    assert key in store
    assert list_requirements(store, status=RequirementStatus.INACTIVE) == [store[key]]
    assert list_requirements(store, status=RequirementStatus.ACTIVE) == []


def test_evidence_is_referenced_not_copied_as_knowledge():
    item = requirement()
    store = register_requirement({}, item)
    evidence = RequirementEvidenceRef(fact_id="abc123", source_turn=7)
    store = attach_requirement_evidence(
        store, item.id, [evidence, evidence], scope=DiscoveryScope.USER_APP
    )
    scoped = store[f"USER_APP|{item.id}"]
    assert scoped.evidence_refs == [evidence]
    dumped = scoped.model_dump()
    assert "value" not in dumped["evidence_refs"][0]
    assert "evidence" not in dumped["evidence_refs"][0]


def test_registration_is_idempotent_but_rejects_accidental_replacement():
    item = requirement()
    store = register_requirement({}, item)
    assert register_requirement(store, item) == store
    with pytest.raises(ValueError):
        register_requirement(store, item.model_copy(update={"label": "Different requirement"}))


def test_checkpoint_round_trip_rehydrates_active_requirements(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path))
    session_id = str(uuid4())
    item = requirement()
    state = dict(
        session_id=session_id,
        checkpoint_cursor="waiting",
        interview_status="WAITING_FOR_USER",
        messages=[],
        discovery_scope=DiscoveryScope.USER_APP,
        discovered_knowledge=[],
        active_requirements={f"USER_APP|{item.id}": item},
        topic_status={},
        topic_maturity={},
    )
    save_checkpoint(state)
    loaded = load_checkpoint(session_id)
    key = f"USER_APP|{item.id}"
    assert loaded["active_requirements"][key] == item
    assert isinstance(loaded["active_requirements"][key], ActiveRequirement)


def test_legacy_checkpoint_without_requirements_loads_empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path))
    session_id = str(uuid4())
    state = dict(
        session_id=session_id,
        checkpoint_cursor="waiting",
        interview_status="WAITING_FOR_USER",
        messages=[],
        discovery_scope=DiscoveryScope.USER_APP,
        discovered_knowledge=[],
        topic_status={},
        topic_maturity={},
    )
    save_checkpoint(state)
    loaded = load_checkpoint(session_id)
    assert loaded["active_requirements"] == {}
    assert loaded["requirement_coverage"] == {}
    assert loaded["requirement_dependency_state"] == {}
    assert loaded["eligible_requirement_keys"] == []
    assert loaded["question_candidates"] == []
    assert loaded["eligible_question_candidates"] == []
    assert loaded["question_candidate_eligibility"] == {}
    assert loaded["ranked_question_candidates"] == []
    assert loaded["question_candidate_priority"] == {}
    assert loaded["requirement_question_history"] == []
    assert loaded["planner_source"] == "schema"
    assert loaded["selected_requirement_candidate"] is None
    assert loaded["selected_requirement_priority"] is None
    assert loaded["validation_issues"] == []
    assert loaded["validation_pair_cache"] == {}
    assert loaded["validation_blocking"] is False
    assert loaded["validation_candidate_blocking"] is False
    assert loaded["selected_validation_issue"] is None
