from uuid import uuid4

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from agents.interview_checkpoint import save_checkpoint
from agents.state import DiscoveryScope
from api.routes import create_project, get_projects
from api.schemas import CreateProjectInput, ProjectSummary, WorkspaceSnapshot
from services.project_service import ProjectInitializationError


def _workspace():
    return WorkspaceSnapshot.model_validate({
        "project": {
            "id": "escrow-app-a1b2c3d4",
            "name": "Escrow App",
            "description": "Escrow product",
            "status": "discovering",
            "updatedAt": "2026-09-30T08:00:00+00:00",
        },
        "discovery": {
            "status": "active",
            "messages": [],
            "activePrompt": None,
        },
        "understanding": {"sections": []},
        "prd": {"status": "not_generated", "sections": []},
    })


def test_create_project_route_returns_workspace(monkeypatch):
    import api.routes as routes

    expected = _workspace()
    monkeypatch.setattr(
        routes,
        "create_project_workspace",
        lambda **kwargs: expected,
    )

    result = create_project(
        CreateProjectInput(name="Escrow App", description="Escrow product")
    )
    assert result == expected


def test_create_project_route_exposes_created_project_on_initialization_failure(monkeypatch):
    import api.routes as routes

    def fail(**kwargs):
        raise ProjectInitializationError(
            "escrow-app-a1b2c3d4",
            retryable=True,
            reason="OpenAI unavailable",
        )

    monkeypatch.setattr(routes, "create_project_workspace", fail)

    with pytest.raises(HTTPException) as exc:
        create_project(CreateProjectInput(name="Escrow App", description="Escrow product"))

    assert exc.value.status_code == 503
    assert exc.value.detail["projectId"] == "escrow-app-a1b2c3d4"
    assert exc.value.detail["retryable"] is True


def test_list_projects_route_returns_summaries(monkeypatch):
    import api.routes as routes

    expected = [
        ProjectSummary(
            id="escrow-app-a1b2c3d4",
            name="Escrow App",
            description="Escrow product",
            status="discovering",
            updatedAt="2026-09-30T08:00:00+00:00",
        )
    ]
    monkeypatch.setattr(routes, "list_project_summaries", lambda: expected)

    assert get_projects() == expected
