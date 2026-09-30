from uuid import uuid4

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.interview_checkpoint import save_checkpoint
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeItem, KnowledgeState
from api.routes import get_project_workspace
from services.project_repository import create_project_record
from services.workspace import WorkspaceNotFoundError, build_workspace_snapshot


def _state(session_id: str):
    customer = KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=DiscoveryScope.USER_APP,
        key="primary_users",
        value="customers use the product",
        evidence="customers use the product",
        roles=["customer"],
        confidence=1.0,
        knowledge_state=KnowledgeState.CONFIRMED,
    )
    return {
        "session_id": session_id,
        "checkpoint_cursor": "waiting",
        "interview_status": "WAITING_FOR_USER",
        "messages": [
            HumanMessage(content="I want to build an escrow product.", id="founder-1"),
            SystemMessage(content="internal guardrail feedback", id="system-1"),
            AIMessage(content="Who are the main users?", id="architect-1"),
        ],
        "raw_idea": "I want to build an escrow product.",
        "prd_contract": None,
        "pm_is_complete": False,
        "awaiting_confirmation": False,
        "prd_confirmation_pending": False,
        "ready_to_compile": False,
        "discovery_scope": DiscoveryScope.USER_APP,
        "current_topic": DiscoveryTopic.USER_ROLES,
        "discovered_knowledge": [customer],
        "active_requirements": {},
        "product_concepts": [],
        "captured_observations": [],
        "discovery_boundaries": [],
        "model_implications": [],
        "thread_planning_enabled": True,
        "discovery_threads": {},
        "active_discovery_thread": None,
        "thread_frontier": None,
        "thread_relevant_requirement_ids": [],
        "requirement_coverage": {},
        "requirement_dependency_state": {},
        "eligible_requirement_keys": [],
        "question_candidates": [],
        "eligible_question_candidates": [],
        "question_candidate_eligibility": {},
        "ranked_question_candidates": [],
        "question_candidate_priority": {},
        "requirement_question_history": [],
        "open_inquiries": [],
        "selected_inquiry": None,
        "planner_source": "model",
        "selected_requirement_candidate": None,
        "selected_requirement_priority": None,
        "validation_issues": [],
        "validation_pair_cache": {},
        "validation_blocking": False,
        "validation_candidate_blocking": False,
        "selected_validation_issue": None,
        "question_retry_count": 0,
        "question_retry_exhausted": False,
        "answer_followup": None,
    }


def _save_project(tmp_path, monkeypatch, *, name="Escrow App"):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))
    session_id = str(uuid4())
    save_checkpoint(_state(session_id))
    project = create_project_record(
        name=name,
        description="A customer-to-customer escrow product.",
        discovery_session_id=session_id,
    )
    return project, session_id


def test_workspace_snapshot_resolves_project_to_session_and_keeps_project_metadata(tmp_path, monkeypatch):
    project, session_id = _save_project(tmp_path, monkeypatch)

    snapshot = build_workspace_snapshot(project.id)
    payload = snapshot.model_dump(mode="json")

    assert payload["project"]["id"] == project.id
    assert payload["project"]["name"] == "Escrow App"
    assert payload["project"]["description"] == "A customer-to-customer escrow product."
    assert project.id != session_id
    assert payload["project"]["status"] == "discovering"
    assert payload["discovery"]["status"] == "active"
    assert payload["discovery"]["activePrompt"] == "Who are the main users?"
    assert [message["role"] for message in payload["discovery"]["messages"]] == [
        "founder",
        "architect",
    ]
    assert all(
        "guardrail" not in message["content"]
        for message in payload["discovery"]["messages"]
    )
    assert payload["discovery"]["messages"][0]["createdAt"] is None

    sections = {
        section["id"]: section
        for section in payload["understanding"]["sections"]
    }
    assert sections["users"]["items"][0]["label"] == "Customer"
    assert sections["users"]["items"][0]["state"] == "confirmed"
    assert payload["prd"] == {"status": "not_generated", "sections": []}


def test_workspace_snapshot_uses_confirmation_and_compilation_statuses(tmp_path, monkeypatch):
    project, session_id = _save_project(tmp_path, monkeypatch)

    state = _state(session_id)
    state["prd_confirmation_pending"] = True
    save_checkpoint(state)

    snapshot = build_workspace_snapshot(project.id)
    assert snapshot.project.status == "ready_for_prd"
    assert snapshot.discovery.status == "ready_for_confirmation"

    state["prd_confirmation_pending"] = False
    state["ready_to_compile"] = True
    state["checkpoint_cursor"] = "compile_prd"
    state["interview_status"] = "COMPILING_PRD"
    save_checkpoint(state)

    snapshot = build_workspace_snapshot(project.id)
    assert snapshot.discovery.status == "compiling"
    assert snapshot.prd.status == "generating"


def test_workspace_snapshot_rejects_unknown_project(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    with pytest.raises(WorkspaceNotFoundError):
        build_workspace_snapshot("missing-project")

    with pytest.raises(HTTPException) as exc:
        get_project_workspace("missing-project")
    assert exc.value.status_code == 404


def test_existing_cli_session_uuid_remains_viewable_as_legacy_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    session_id = str(uuid4())
    save_checkpoint(_state(session_id))

    snapshot = build_workspace_snapshot(session_id)
    assert snapshot.project.id == session_id
    assert snapshot.project.name == "Untitled project"
    assert snapshot.project.description == "I want to build an escrow product."
