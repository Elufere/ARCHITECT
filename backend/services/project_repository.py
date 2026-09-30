"""Atomic file-backed repository for Architect projects.

Project metadata is deliberately separate from LangGraph interview checkpoints.
A Project owns exactly one durable discovery session; a session may belong to at
most one persisted Project.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import UUID, uuid4

from pydantic import ValidationError

from agents.interview_checkpoint import checkpoint_path
from models.project import PROJECT_ID_PATTERN, ProjectRecord


class ProjectRepositoryError(RuntimeError):
    pass


class ProjectNotFoundError(ProjectRepositoryError):
    pass


class ProjectConflictError(ProjectRepositoryError):
    pass


class ProjectSessionNotFoundError(ProjectRepositoryError):
    pass


def project_directory() -> Path:
    return Path(
        os.getenv(
            "ARCHITECT_PROJECT_DIR",
            str(Path(__file__).resolve().parents[1] / "projects"),
        )
    )


def _canonical_session_id(session_id: str) -> str:
    try:
        return str(UUID(session_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ProjectSessionNotFoundError("Discovery session id is invalid.") from exc


def _validated_project_id(project_id: str) -> str:
    value = (project_id or "").strip().lower()
    if not re.fullmatch(PROJECT_ID_PATTERN, value):
        raise ProjectNotFoundError(f"Project '{project_id}' was not found.")
    return value


def project_path(project_id: str) -> Path:
    return project_directory() / f"{_validated_project_id(project_id)}.json"


def _read_project_file(path: Path) -> ProjectRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ProjectRecord.model_validate(payload)
    except FileNotFoundError as exc:
        raise ProjectNotFoundError(
            f"Project '{path.stem}' was not found."
        ) from exc
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ProjectRepositoryError(
            f"Project '{path.stem}' could not be read safely."
        ) from exc


def get_project(project_id: str) -> ProjectRecord:
    path = project_path(project_id)
    if not path.exists():
        raise ProjectNotFoundError(f"Project '{project_id}' was not found.")
    return _read_project_file(path)


def list_projects() -> list[ProjectRecord]:
    directory = project_directory()
    if not directory.exists():
        return []

    result: list[ProjectRecord] = []
    for path in directory.glob("*.json"):
        result.append(_read_project_file(path))
    return sorted(result, key=lambda item: item.updated_at, reverse=True)


def find_project_by_session(session_id: str) -> ProjectRecord | None:
    canonical = _canonical_session_id(session_id)
    for project in list_projects():
        if project.discovery_session_id == canonical:
            return project
    return None


def _slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return (value[:48].rstrip("-") or "project")


def generate_project_id(name: str) -> str:
    return f"{_slug(name)}-{uuid4().hex[:8]}"


def _write_project(record: ProjectRecord, *, require_absent: bool = False) -> None:
    path = project_path(record.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if require_absent and path.exists():
        raise ProjectConflictError(f"Project '{record.id}' already exists.")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".project-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(record.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if require_absent and path.exists():
            raise ProjectConflictError(f"Project '{record.id}' already exists.")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def create_project_record(
    *,
    name: str,
    description: str,
    discovery_session_id: str,
    project_id: str | None = None,
) -> ProjectRecord:
    """Register an existing durable discovery session as one web Project."""

    session_id = _canonical_session_id(discovery_session_id)
    if not checkpoint_path(session_id).exists():
        raise ProjectSessionNotFoundError(
            f"Discovery session '{session_id}' was not found."
        )

    existing = find_project_by_session(session_id)
    if existing is not None:
        raise ProjectConflictError(
            f"Discovery session '{session_id}' already belongs to project '{existing.id}'."
        )

    now = datetime.now(timezone.utc)
    record = ProjectRecord(
        id=project_id or generate_project_id(name),
        name=name,
        description=description,
        discovery_session_id=session_id,
        created_at=now,
        updated_at=now,
    )
    _write_project(record, require_absent=True)
    return record


def update_project_metadata(
    project_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
) -> ProjectRecord:
    """Update product-facing metadata without touching discovery state."""

    current = get_project(project_id)
    updated = current.model_copy(
        update={
            **({"name": name.strip()} if name is not None else {}),
            **({"description": description.strip()} if description is not None else {}),
            "updated_at": datetime.now(timezone.utc),
        }
    )
    # Revalidate updates because model_copy does not rerun field validators.
    updated = ProjectRecord.model_validate(updated.model_dump())
    _write_project(updated)
    return updated
