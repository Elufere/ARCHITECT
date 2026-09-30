"""HTTP response and request contracts for the Architect API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from agents.understanding_projection import UnderstandingSection


ProjectStatus = Literal["draft", "discovering", "ready_for_prd", "prd_ready"]
DiscoveryStatus = Literal["active", "ready_for_confirmation", "compiling", "complete"]
PrdStatus = Literal["not_generated", "generating", "ready"]
MessageRole = Literal["architect", "founder"]


class CreateProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=10000)

    @field_validator("name", "description")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value


class ProjectSummary(BaseModel):
    id: str
    name: str
    description: str
    status: ProjectStatus
    updatedAt: str


class WorkspaceMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    # Historical CLI checkpoints did not store per-message timestamps. Keep the
    # field nullable rather than inventing creation times.
    createdAt: Optional[str] = None


class DiscoverySnapshot(BaseModel):
    status: DiscoveryStatus
    messages: list[WorkspaceMessage]
    activePrompt: Optional[str] = None


class UnderstandingSnapshot(BaseModel):
    sections: list[UnderstandingSection] = Field(default_factory=list)


class PrdSection(BaseModel):
    id: str
    title: str
    body: str


class PrdSnapshot(BaseModel):
    status: PrdStatus
    sections: list[PrdSection] = Field(default_factory=list)


class WorkspaceSnapshot(BaseModel):
    project: ProjectSummary
    discovery: DiscoverySnapshot
    understanding: UnderstandingSnapshot
    prd: PrdSnapshot
