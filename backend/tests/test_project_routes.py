import pytest
from fastapi import HTTPException
from api.routes import (
    create_project,
    delete_project,
    generate_prd,
    get_projects,
    retry_discovery,
    submit_discovery_turn,
)
from api.schemas import (
    CreateProjectInput,
    DiscoveryTurnInput,
    ProjectSummary,
    WorkspaceSnapshot,
)
from services.discovery_session import DiscoverySessionStateError
from services.discovery_turn import DiscoveryProcessingError
from services.prd_generation import (
    PrdGenerationProcessingError,
    PrdGenerationStateError,
)
from services.project_repository import ProjectNotFoundError
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



def test_submit_discovery_turn_route_returns_workspace(monkeypatch):
    import api.routes as routes

    expected = _workspace()
    monkeypatch.setattr(
        routes,
        "submit_project_discovery_turn",
        lambda project_id, input: expected,
    )

    result = submit_discovery_turn(
        "escrow-app-a1b2c3d4",
        DiscoveryTurnInput(type="answer", message="Customers."),
    )
    assert result == expected


def test_submit_discovery_turn_route_maps_processing_failure_to_503(monkeypatch):
    import api.routes as routes

    def fail(project_id, input):
        raise DiscoveryProcessingError(
            project_id,
            retryable=True,
            reason="OpenAI unavailable",
        )

    monkeypatch.setattr(routes, "submit_project_discovery_turn", fail)

    with pytest.raises(HTTPException) as exc:
        submit_discovery_turn(
            "escrow-app-a1b2c3d4",
            DiscoveryTurnInput(type="answer", message="Customers."),
        )

    assert exc.value.status_code == 503
    assert exc.value.detail["projectId"] == "escrow-app-a1b2c3d4"
    assert exc.value.detail["retryable"] is True


def test_retry_discovery_route_maps_no_pending_work_to_409(monkeypatch):
    import api.routes as routes

    def no_work(project_id):
        raise DiscoverySessionStateError("There is no interrupted discovery work to retry.")

    monkeypatch.setattr(routes, "retry_project_discovery", no_work)

    with pytest.raises(HTTPException) as exc:
        retry_discovery("escrow-app-a1b2c3d4")

    assert exc.value.status_code == 409


def test_retry_discovery_route_maps_missing_project_to_404(monkeypatch):
    import api.routes as routes

    def missing(project_id):
        raise ProjectNotFoundError(f"Project '{project_id}' was not found.")

    monkeypatch.setattr(routes, "retry_project_discovery", missing)

    with pytest.raises(HTTPException) as exc:
        retry_discovery("missing-project")

    assert exc.value.status_code == 404


def test_generate_prd_route_returns_workspace(monkeypatch):
    import api.routes as routes

    expected = _workspace()
    monkeypatch.setattr(
        routes,
        "generate_project_prd",
        lambda project_id: expected,
    )

    assert generate_prd("escrow-app-a1b2c3d4") == expected


def test_generate_prd_route_requires_founder_approval(monkeypatch):
    import api.routes as routes

    def blocked(project_id):
        raise PrdGenerationStateError(
            "PRD generation has not been authorized by the founder."
        )

    monkeypatch.setattr(routes, "generate_project_prd", blocked)

    with pytest.raises(HTTPException) as exc:
        generate_prd("escrow-app-a1b2c3d4")

    assert exc.value.status_code == 409
    assert "authorized" in exc.value.detail


def test_generate_prd_route_maps_processing_failure_to_503(monkeypatch):
    import api.routes as routes

    def fail(project_id):
        raise PrdGenerationProcessingError(
            project_id,
            retryable=True,
            reason="Verifier rejected unsupported claim.",
        )

    monkeypatch.setattr(routes, "generate_project_prd", fail)

    with pytest.raises(HTTPException) as exc:
        generate_prd("escrow-app-a1b2c3d4")

    assert exc.value.status_code == 503
    assert exc.value.detail["projectId"] == "escrow-app-a1b2c3d4"
    assert exc.value.detail["retryable"] is True
    assert exc.value.detail["message"] == "Verifier rejected unsupported claim."



def test_delete_project_route_returns_204(monkeypatch):
    import api.routes as routes

    deleted = []
    monkeypatch.setattr(
        routes,
        "delete_project_workspace",
        lambda project_id: deleted.append(project_id),
    )

    response = delete_project("escrow-app-a1b2c3d4")

    assert response.status_code == 204
    assert deleted == ["escrow-app-a1b2c3d4"]


def test_delete_project_route_maps_missing_project_to_404(monkeypatch):
    import api.routes as routes

    def missing(project_id):
        raise ProjectNotFoundError(f"Project '{project_id}' was not found.")

    monkeypatch.setattr(routes, "delete_project_workspace", missing)

    with pytest.raises(HTTPException) as exc:
        delete_project("missing-project")

    assert exc.value.status_code == 404


def test_delete_project_route_maps_busy_session_to_409(monkeypatch):
    import api.routes as routes
    from services.discovery_session import DiscoverySessionBusyError

    def busy(project_id):
        raise DiscoverySessionBusyError(
            "This project is currently being processed and cannot be deleted yet."
        )

    monkeypatch.setattr(routes, "delete_project_workspace", busy)

    with pytest.raises(HTTPException) as exc:
        delete_project("escrow-app-a1b2c3d4")

    assert exc.value.status_code == 409
