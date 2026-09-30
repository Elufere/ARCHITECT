from uuid import UUID, uuid4

import pytest
from langchain_core.messages import HumanMessage

from agents.interview_checkpoint import save_checkpoint
from agents.state import DiscoveryScope
from services.project_repository import (
    ProjectConflictError,
    ProjectNotFoundError,
    ProjectSessionNotFoundError,
    create_project_record,
    find_project_by_session,
    get_project,
    list_projects,
    update_project_metadata,
)


def _minimal_state(session_id: str):
    return {
        "session_id": session_id,
        "checkpoint_cursor": "waiting",
        "interview_status": "WAITING_FOR_USER",
        "messages": [HumanMessage(content="idea", id="founder-1")],
        "raw_idea": "idea",
        "prd_contract": None,
        "pm_is_complete": False,
        "awaiting_confirmation": False,
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [],
        "active_requirements": {},
    }


def test_project_record_owns_existing_session_with_distinct_project_id(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    session_id = str(uuid4())
    save_checkpoint(_minimal_state(session_id))

    project = create_project_record(
        name="Escrow App",
        description="A customer escrow product.",
        discovery_session_id=session_id,
    )

    assert project.id.startswith("escrow-app-")
    with pytest.raises(ValueError):
        UUID(project.id)
    assert project.discovery_session_id == session_id

    loaded = get_project(project.id)
    assert loaded == project
    assert find_project_by_session(session_id) == project
    assert list_projects() == [project]


def test_one_session_cannot_belong_to_two_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    session_id = str(uuid4())
    save_checkpoint(_minimal_state(session_id))
    create_project_record(
        name="First project",
        description="",
        discovery_session_id=session_id,
    )

    with pytest.raises(ProjectConflictError):
        create_project_record(
            name="Second project",
            description="",
            discovery_session_id=session_id,
        )


def test_project_cannot_point_to_missing_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    with pytest.raises(ProjectSessionNotFoundError):
        create_project_record(
            name="Missing",
            description="",
            discovery_session_id=str(uuid4()),
        )


def test_project_metadata_updates_without_changing_session_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    session_id = str(uuid4())
    save_checkpoint(_minimal_state(session_id))
    project = create_project_record(
        name="Old name",
        description="Old description",
        discovery_session_id=session_id,
    )

    updated = update_project_metadata(
        project.id,
        name="Escrow App",
        description="Updated description",
    )

    assert updated.name == "Escrow App"
    assert updated.description == "Updated description"
    assert updated.discovery_session_id == session_id
    assert updated.updated_at >= project.updated_at


def test_unknown_project_is_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHITECT_PROJECT_DIR", str(tmp_path / "projects"))

    with pytest.raises(ProjectNotFoundError):
        get_project("unknown-project")
