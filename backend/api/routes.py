"""HTTP routes for the product-facing Architect API."""
from fastapi import APIRouter, HTTPException

from api.schemas import WorkspaceSnapshot
from services.workspace import (
    WorkspaceNotFoundError,
    WorkspaceUnavailableError,
    build_workspace_snapshot,
)


router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


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
