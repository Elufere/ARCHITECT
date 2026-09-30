"""Persistent web-product metadata for an Architect project."""
from __future__ import annotations

from datetime import datetime
import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{2,79}$"


class ProjectRecord(BaseModel):
    """A user-facing project and the durable discovery session it owns."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str = Field(pattern=PROJECT_ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=10000)
    discovery_session_id: str
    created_at: datetime
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(PROJECT_ID_PATTERN, value):
            raise ValueError("Invalid project id")
        return value

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name cannot be empty")
        return value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("discovery_session_id")
    @classmethod
    def canonical_session_id(cls, value: str) -> str:
        return str(UUID(value))
