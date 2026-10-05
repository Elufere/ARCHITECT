"""Build the founder-facing workspace snapshot from a durable Architect project.

This service is the boundary between product-facing HTTP responses and internal
LangGraph/checkpoint state. React should never need to understand AgentState,
requirement coverage, thread planning, or PRD validation internals.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
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
from models.project import ProjectRecord
from services.project_repository import (
    ProjectNotFoundError,
    ProjectRepositoryError,
    get_project,
)


class WorkspaceNotFoundError(LookupError):
    pass


class WorkspaceUnavailableError(RuntimeError):
    pass


def _legacy_session_project(project_id: str) -> ProjectRecord | None:
    """Compatibility only: allow old CLI session UUIDs to remain viewable.

    New web projects must resolve through ProjectRecord. This fallback can be
    removed after existing sessions are migrated into the project repository.
    """
    try:
        session_id = str(UUID(project_id))
    except (ValueError, AttributeError, TypeError):
        return None

    path = checkpoint_path(session_id)
    if not path.exists():
        return None

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        state = load_checkpoint(session_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    timestamp = document.get("updated_at")
    try:
        updated_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    contract = state.get("prd_contract")
    name = "Untitled project"
    if isinstance(contract, PRDContract) and contract.product_name is not None:
        name = contract.product_name.text

    return ProjectRecord(
        id=f"legacy-{session_id.split('-')[0]}",
        name=name,
        description=state.get("raw_idea", ""),
        discovery_session_id=session_id,
        created_at=updated_at,
        updated_at=updated_at,
    )


def _resolve_project(project_id: str) -> tuple[ProjectRecord, bool]:
    try:
        return get_project(project_id), False
    except ProjectNotFoundError:
        legacy = _legacy_session_project(project_id)
        if legacy is not None:
            return legacy, True
        raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.")
    except ProjectRepositoryError as exc:
        raise WorkspaceUnavailableError(
            f"Project '{project_id}' exists but its metadata could not be read safely."
        ) from exc


def _load_checkpoint_bundle(project_id: str) -> tuple[ProjectRecord, bool, dict, dict]:
    project, is_legacy = _resolve_project(project_id)
    session_id = project.discovery_session_id
    path = checkpoint_path(session_id)

    if not path.exists():
        if is_legacy:
            raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.")
        raise WorkspaceUnavailableError(
            f"Project '{project_id}' exists but its discovery session is missing."
        )

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        state = load_checkpoint(session_id)
    except FileNotFoundError as exc:
        if is_legacy:
            raise WorkspaceNotFoundError(f"Project '{project_id}' was not found.") from exc
        raise WorkspaceUnavailableError(
            f"Project '{project_id}' exists but its discovery session is missing."
        ) from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WorkspaceUnavailableError(
            f"Project '{project_id}' exists but its saved workspace could not be read safely."
        ) from exc

    return project, is_legacy, document, state


def _project_status(state: dict) -> str:
    if state.get("prd_contract") is not None:
        return "prd_ready"
    if (
        state.get("prd_confirmation_pending")
        or state.get("ready_to_compile")
        or state.get("checkpoint_cursor") == "compile_prd"
    ):
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
    if state.get("prd_confirmation_pending"):
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


def _source_values(
    contract: PRDContract,
    topic: str,
    keys: set[str] | None = None,
) -> list[str]:
    values: list[str] = []
    for source in contract.source_facts:
        if source.absence is not None or source.topic != topic:
            continue
        if keys is not None and source.key not in keys:
            continue
        value = source.value.strip()
        if value and value not in values:
            values.append(value)
    return values


def _prd_sections(
    contract: PRDContract | None,
    project_description: str | None = None,
) -> list[PrdSection]:
    if contract is None:
        return []

    sections: list[PrdSection] = []

    # Reference-quality PRDs begin with product context, but Architect may only
    # render context the founder actually supplied. The durable project
    # description is founder-authored and therefore safe summary material.
    summary_parts: list[str] = []
    if project_description and project_description.strip():
        summary_parts.append(project_description.strip())
    if contract.product_name is not None:
        summary_parts.append(f"Product: {contract.product_name.text}")
    pitch = _claim_texts(contract.elevator_pitch)
    if pitch:
        summary_parts.append(_bullet_block(pitch))
    if not summary_parts and contract.feature_specifications:
        summary_parts.append(
            "Core product areas:\n"
            + _bullet_block(
                feature.title for feature in contract.feature_specifications
            )
        )
    if summary_parts:
        sections.append(
            PrdSection(
                id="overview",
                title="Executive Summary",
                body="\n\n".join(summary_parts),
            )
        )

    goals = _source_values(
        contract,
        "USER_GOALS",
        {"primary_user_goals", "secondary_user_goals"},
    )
    motivations = _source_values(contract, "USER_GOALS", {"motivations"})
    success = _source_values(contract, "USER_GOALS", {"success_criteria"})
    success.extend(
        value
        for value in _source_values(contract, "MVP_SCOPE", {"success_metrics"})
        if value not in success
    )
    goal_parts: list[str] = []
    if motivations:
        goal_parts.append("Problem / motivation:\n" + _bullet_block(motivations))
    if goals:
        goal_parts.append("Product goals:\n" + _bullet_block(goals))
    if success:
        goal_parts.append("Success metrics:\n" + _bullet_block(success))
    if goal_parts:
        has_goals = bool(goals or motivations)
        goal_title = (
            "Goals and Success Metrics"
            if has_goals and success
            else "Goals"
            if has_goals
            else "Success Metrics"
        )
        sections.append(
            PrdSection(
                id="goals",
                title=goal_title,
                body="\n\n".join(goal_parts),
            )
        )

    if contract.personas:
        personas: list[str] = []
        for persona in contract.personas:
            block = f"{persona.name} — {persona.description}"
            behaviors = _claim_texts(persona.key_behaviors)
            if behaviors:
                block += "\n" + _bullet_block(behaviors)
            personas.append(block)
        sections.append(
            PrdSection(
                id="users",
                title="Target Users and Roles",
                body="\n\n".join(personas),
            )
        )

    in_scope = _claim_texts(contract.scope.in_scope)
    out_of_scope = _claim_texts(contract.scope.out_of_scope)
    scope_parts: list[str] = []
    if in_scope:
        scope_parts.append("In scope:\n" + _bullet_block(in_scope))
    if out_of_scope:
        scope_parts.append("Out of scope:\n" + _bullet_block(out_of_scope))
    if scope_parts:
        sections.append(
            PrdSection(
                id="scope",
                title="Product Scope",
                body="\n\n".join(scope_parts),
            )
        )

    if contract.feature_specifications:
        for index, feature in enumerate(contract.feature_specifications, start=1):
            parts = [feature.overview]
            if feature.details:
                parts.append(
                    _bullet_block(
                        detail.text for detail in feature.details
                    )
                )
            sections.append(
                PrdSection(
                    id=f"feature_{index}",
                    title=f"Feature {index}: {feature.title}",
                    body="\n\n".join(part for part in parts if part.strip()),
                )
            )
    elif contract.functional_requirements:
        requirements: list[str] = []
        for item in contract.functional_requirements:
            requirements.append(
                f"{item.id} — {item.description}\nAcceptance: {item.validation}"
            )
        sections.append(
            PrdSection(
                id="functional_requirements",
                title="Functional Requirements",
                body="\n\n".join(requirements),
            )
        )

    workflow_labels = {
        "trigger": "Trigger",
        "workflow_steps": "Flow",
        "completion_condition": "Completion condition",
        "end_state": "Resulting state",
        "downstream_dependency": "External dependency",
    }
    workflow_parts: list[str] = []
    for key, label in workflow_labels.items():
        values = _source_values(contract, "CORE_WORKFLOW", {key})
        if values:
            workflow_parts.append(f"{label}:\n" + _bullet_block(values))
    if workflow_parts:
        sections.append(
            PrdSection(
                id="user_flow",
                title="Core User Flow",
                body="\n\n".join(workflow_parts),
            )
        )

    if contract.external_systems:
        systems = []
        for system in contract.external_systems:
            statements = list(dict.fromkeys(
                item.value for item in system.statements if item.value.strip()
            ))
            block = system.name
            if statements:
                block += "\n" + _bullet_block(statements)
            systems.append(block)
        sections.append(
            PrdSection(
                id="external_systems",
                title="External Systems and Integrations",
                body="\n\n".join(systems),
            )
        )

    product_sources = [
        source
        for source in contract.source_facts
        if source.topic == "PRODUCT_MODEL"
        and source.absence is None
        and source.subject
    ]
    if product_sources:
        by_subject: dict[str, dict[str, list[str]]] = {}
        subject_order: list[str] = []
        for source in product_sources:
            subject = source.subject.strip().replace("_", " ").title()
            if subject not in by_subject:
                by_subject[subject] = {}
                subject_order.append(subject)

            # Entity declarations establish the heading and need no duplicate line.
            if source.key == "entity":
                continue

            label = {
                "attribute": "Fields",
                "state": "States",
                "ownership": "Ownership",
                "persistence": "Persistence & access",
                "boundary": "MVP boundary",
                "transition": "State transitions",
                "operation_rule": "Operation rules",
                "relationship": "Relationships",
            }.get(source.key, source.key.replace("_", " ").title())

            if source.key in {"attribute", "state"} and source.object:
                value = source.object.strip()
            else:
                value = source.value.strip()
            if not value:
                continue
            values = by_subject[subject].setdefault(label, [])
            if value not in values:
                values.append(value)

        blocks = []
        for subject in subject_order:
            lines: list[str] = []
            for label, values in by_subject[subject].items():
                if values:
                    lines.append(f"{label}: " + ", ".join(values))
            block = subject
            if lines:
                block += "\n" + _bullet_block(lines)
            blocks.append(block)
        sections.append(
            PrdSection(
                id="data_model",
                title="Core Data Model",
                body="\n\n".join(blocks),
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
    for item in contract.deferred_decisions:
        qualifiers = []
        if item.resolution_stage:
            qualifiers.append(f"revisit: {item.resolution_stage}")
        if item.owner:
            qualifiers.append(f"owner: {item.owner}")
        text = item.decision
        if qualifiers:
            text += " (" + "; ".join(qualifiers) + ")"
        deferred.append(text)
    deferred = list(dict.fromkeys(deferred))
    if deferred:
        sections.append(
            PrdSection(
                id="deferred",
                title="Deferred Decisions",
                body=_bullet_block(deferred),
            )
        )

    if contract.open_questions:
        sections.append(
            PrdSection(
                id="open_questions",
                title="Open Questions",
                body=_bullet_block(contract.open_questions),
            )
        )

    return sections

def _workspace_updated_at(project: ProjectRecord, document: dict) -> str:
    raw = document.get("updated_at")
    if not isinstance(raw, str) or not raw:
        raise WorkspaceUnavailableError(
            f"Project '{project.id}' has no valid saved update timestamp."
        )
    try:
        checkpoint_updated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkspaceUnavailableError(
            f"Project '{project.id}' has an invalid saved update timestamp."
        ) from exc
    return max(project.updated_at, checkpoint_updated).isoformat()


def build_project_summary(project_id: str) -> ProjectSummary:
    """Return product-facing metadata with live status from the linked checkpoint."""

    project, is_legacy, document, state = _load_checkpoint_bundle(project_id)
    return ProjectSummary(
        id=project_id if is_legacy else project.id,
        name=project.name,
        description=project.description,
        status=_project_status(state),
        updatedAt=_workspace_updated_at(project, document),
    )


def build_workspace_snapshot(project_id: str) -> WorkspaceSnapshot:
    project, is_legacy, document, state = _load_checkpoint_bundle(project_id)

    messages = _visible_messages(state)
    understanding = build_understanding_projection(state)
    contract = state.get("prd_contract")
    if contract is not None and not isinstance(contract, PRDContract):
        contract = PRDContract.model_validate(contract)

    return WorkspaceSnapshot(
        project=ProjectSummary(
            id=project_id if is_legacy else project.id,
            name=project.name,
            description=project.description,
            status=_project_status(state),
            updatedAt=_workspace_updated_at(project, document),
        ),
        discovery=DiscoverySnapshot(
            status=_discovery_status(state),
            messages=messages,
            activePrompt=_active_prompt(state, messages),
        ),
        understanding=UnderstandingSnapshot(sections=understanding.sections),
        prd=PrdSnapshot(
            status=_prd_status(state),
            sections=_prd_sections(contract, project.description),
        ),
    )
