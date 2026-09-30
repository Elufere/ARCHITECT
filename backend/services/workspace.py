"""Build the founder-facing workspace snapshot from a durable Architect session.

This service is the boundary between product-facing HTTP responses and internal
LangGraph/checkpoint state. React should never need to understand AgentState,
requirement coverage, thread planning, or PRD validation internals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage

from agents.interview_checkpoint import checkpoint_path, load_checkpoint
from agents.prd_schema import PRDContract
from agents.understanding_projection import build_understanding_projection
from api.schemas import (
    DiscoverySnapshot,
    PrdSection,
    PrdSnapshot,
    ProjectSummary,
    UnderstandingSnapshot,
    WorkspaceMessage,
    WorkspaceSnapshot,
)


class WorkspaceNotFoundError(LookupError):
    pass


class WorkspaceUnavailableError(RuntimeError):
    pass


def _resolve_session_id(project_id: str) -> str:
    """Temporary compatibility resolver until first-class Project ownership lands.

    The web API is project-oriented, while the current durable backend is still
    session-oriented. For existing sessions, a canonical checkpoint UUID can act
    as the project id without leaking this compromise into the response contract.
    A future ProjectRepository can replace this function with project->session
    lookup while leaving build_workspace_snapshot and the HTTP route unchanged.
    """
    try:
        return str(UUID(project_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.") from exc


def _load_checkpoint_bundle(project_id: str) -> tuple[dict, dict]:
    session_id = _resolve_session_id(project_id)
    try:
        path = checkpoint_path(session_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.") from exc

    if not path.exists():
        raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.")

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        state = load_checkpoint(session_id)
    except FileNotFoundError as exc:
        raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WorkspaceUnavailableError(
            f"Project '{project_id}' exists but its saved workspace could not be read safely."
        ) from exc

    return document, state


def _project_status(state: dict) -> str:
    if state.get("prd_contract") is not None:
        return "prd_ready"
    if state.get("awaiting_confirmation"):
        return "ready_for_prd"
    return "discovering"


def _discovery_status(state: dict) -> str:
    if state.get("prd_contract") is not None or state.get("checkpoint_cursor") in {
        "phase_complete",
        "completed",
    }:
        return "complete"
    if (
        state.get("checkpoint_cursor") == "compile_prd"
        or state.get("interview_status") == "COMPILING_PRD"
    ):
        return "compiling"
    if state.get("awaiting_confirmation"):
        return "ready_for_confirmation"
    return "active"


def _prd_status(state: dict) -> str:
    if state.get("prd_contract") is not None:
        return "ready"
    if (
        state.get("checkpoint_cursor") == "compile_prd"
        or state.get("interview_status") == "COMPILING_PRD"
    ):
        return "generating"
    return "not_generated"


def _message_timestamp(message) -> str | None:
    for mapping in (
        getattr(message, "additional_kwargs", None),
        getattr(message, "response_metadata", None),
    ):
        if not isinstance(mapping, dict):
            continue
        for key in ("created_at", "createdAt", "timestamp"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _stable_message_id(message, index: int, role: str) -> str:
    if getattr(message, "id", None):
        return str(message.id)
    raw = f"{index}|{role}|{getattr(message, 'content', '')}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"message-{digest}"


def _visible_messages(state: dict) -> list[WorkspaceMessage]:
    result: list[WorkspaceMessage] = []
    for index, message in enumerate(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            role = "founder"
        elif isinstance(message, AIMessage):
            role = "architect"
        else:
            # System/guardrail messages are orchestration detail, not founder chat.
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        result.append(
            WorkspaceMessage(
                id=_stable_message_id(message, index, role),
                role=role,
                content=content,
                createdAt=_message_timestamp(message),
            )
        )
    return result


def _active_prompt(state: dict, messages: list[WorkspaceMessage]) -> str | None:
    if state.get("checkpoint_cursor") != "waiting":
        return None
    for message in reversed(messages):
        if message.role == "architect":
            return message.content
    return None


def _claim_texts(items: Iterable) -> list[str]:
    return [item.text for item in items if getattr(item, "text", "").strip()]


def _bullet_block(items: Iterable[str]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"- {item}" for item in values)


def _prd_sections(contract: PRDContract | None) -> list[PrdSection]:
    if contract is None:
        return []

    sections: list[PrdSection] = []

    overview_parts: list[str] = []
    if contract.product_name is not None:
        overview_parts.append(f"Product: {contract.product_name.text}")
    pitch = _claim_texts(contract.elevator_pitch)
    if pitch:
        overview_parts.append(_bullet_block(pitch))
    if overview_parts:
        sections.append(
            PrdSection(id="overview", title="Overview", body="\n\n".join(overview_parts))
        )

    in_scope = _claim_texts(contract.scope.in_scope)
    out_of_scope = _claim_texts(contract.scope.out_of_scope)
    scope_parts: list[str] = []
    if in_scope:
        scope_parts.append("In scope:\n" + _bullet_block(in_scope))
    if out_of_scope:
        scope_parts.append("Out of scope:\n" + _bullet_block(out_of_scope))
    if scope_parts:
        sections.append(PrdSection(id="scope", title="Scope", body="\n\n".join(scope_parts)))

    if contract.personas:
        personas: list[str] = []
        for persona in contract.personas:
            block = f"{persona.name}: {persona.description}"
            behaviors = _claim_texts(persona.key_behaviors)
            if behaviors:
                block += "\n" + _bullet_block(behaviors)
            personas.append(block)
        sections.append(
            PrdSection(id="users", title="Users and roles", body="\n\n".join(personas))
        )

    if contract.functional_requirements:
        requirements: list[str] = []
        for item in contract.functional_requirements:
            requirements.append(
                f"{item.id} — {item.description}\nAcceptance: {item.validation}"
            )
        sections.append(
            PrdSection(
                id="functional_requirements",
                title="Functional requirements",
                body="\n\n".join(requirements),
            )
        )

    constraints = _claim_texts(contract.non_functional_constraints)
    if constraints:
        sections.append(
            PrdSection(
                id="constraints",
                title="Constraints",
                body=_bullet_block(constraints),
            )
        )

    deferred = _claim_texts(contract.deferred_items)
    if deferred:
        sections.append(
            PrdSection(
                id="deferred",
                title="Deferred items",
                body=_bullet_block(deferred),
            )
        )

    if contract.open_questions:
        sections.append(
            PrdSection(
                id="open_questions",
                title="Open questions",
                body=_bullet_block(contract.open_questions),
            )
        )

    return sections


def _project_name(state: dict) -> str:
    contract = state.get("prd_contract")
    if isinstance(contract, PRDContract) and contract.product_name is not None:
        return contract.product_name.text
    return "Untitled project"


def build_workspace_snapshot(project_id: str) -> WorkspaceSnapshot:
    document, state = _load_checkpoint_bundle(project_id)
    updated_at = document.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise WorkspaceUnavailableError(
            f"Project '{project_id}' has no valid saved update timestamp."
        )

    messages = _visible_messages(state)
    understanding = build_understanding_projection(state)
    contract = state.get("prd_contract")
    if contract is not None and not isinstance(contract, PRDContract):
        contract = PRDContract.model_validate(contract)

    return WorkspaceSnapshot(
        project=ProjectSummary(
            id=project_id,
            name=_project_name(state),
            description=state.get("raw_idea", ""),
            status=_project_status(state),
            updatedAt=updated_at,
        ),
        discovery=DiscoverySnapshot(
            status=_discovery_status(state),
            messages=messages,
            activePrompt=_active_prompt(state, messages),
        ),
        understanding=UnderstandingSnapshot(sections=understanding.sections),
        prd=PrdSnapshot(
            status=_prd_status(state),
            sections=_prd_sections(contract),
        ),
    )
