"""HTTP routes for the product-facing Architect API."""
from fastapi import APIRouter, HTTPException

from api.schemas import CreateProjectInput, ProjectSummary, WorkspaceSnapshot
from services.project_repository import ProjectRepositoryError
from services.project_service import (
    ProjectInitializationError,
    create_project_workspace,
    list_project_summaries,
)
from services.workspace import (
    WorkspaceNotFoundError,
    WorkspaceUnavailableError,
    build_workspace_snapshot,
)


router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/projects", response_model=list[ProjectSummary])
def get_projects() -> list[ProjectSummary]:
    try:
        return list_project_summaries()
    except (ProjectRepositoryError, WorkspaceUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/projects", response_model=WorkspaceSnapshot, status_code=201)
def create_project(input: CreateProjectInput) -> WorkspaceSnapshot:
    try:
        return create_project_workspace(
            name=input.name,
            description=input.description,
        )
    except ProjectInitializationError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": exc.reason,
                "projectId": exc.project_id,
                "retryable": exc.retryable,
            },
        ) from exc
    except ProjectRepositoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WorkspaceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/projects/{project_id}/workspace",
    response_model=WorkspaceSnapshot,
)
def get_project_workspace(project_id: str) -> WorkspaceSnapshot:
    try:
        return build_workspace_snapshot(project_id)
    except WorkspaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkspaceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
