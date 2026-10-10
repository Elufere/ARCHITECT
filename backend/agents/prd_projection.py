"""Deterministic projection from grounded discovery sources into a PRD skeleton.

The LLM is not allowed to decide PRD structure, provenance, categories, actors,
requirement IDs, or whether a grounded source disappears. Whole-field absence
facts constrain generation but do not need visible prose.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from agents.prd_schema import (
    FunctionalRequirement,
    PRDDraft,
    ScopeBoundary,
    SourcedClaim,
    SourceFact,
    UserPersona,
)
from agents.prd_semantics import (
    is_feature_local_goal_source,
    is_system_behavior_source,
    render_acceptance_criterion,
    render_lifecycle_result,
    render_system_behavior,
)


@dataclass(frozen=True)
class ProjectionResult:
    draft: PRDDraft
    visible_source_ids: frozenset[str]
    constraint_source_ids: frozenset[str]


def source_category(source: SourceFact) -> str:
    return f"{source.topic}.{source.key}"


def source_actor_ids(source: SourceFact) -> list[str]:
    values: list[str] = []
    if source.role:
        values.append(source.role)
    values.extend(source.roles or [])
    return list(dict.fromkeys(value for value in values if value))


def is_constraint_source(source: SourceFact) -> bool:
    """Whole-field absence constrains the PRD without requiring visible prose."""
    return source.absence is not None


def _claim(source: SourceFact, text: str | None = None) -> SourcedClaim:
    claim_text = text
    if claim_text is None:
        claim_text = (
            render_system_behavior(source)
            if is_system_behavior_source(source)
            else source.value
        )
    return SourcedClaim(
        text=claim_text.strip(),
        category=source_category(source),
        actor_ids=source_actor_ids(source),
        conditions=[],
        source_fact_ids=[source.fact_id],
    )


def _persona(source: SourceFact, secondary: bool = False) -> UserPersona:
    actors = source_actor_ids(source)
    name = (
        actors[0].replace("_", " ").strip().title()
        if actors
        else source.value.strip().title()
    )
    persona_name = name or ("Secondary user" if secondary else "User")
    # The persona heading already identifies the actor. Do not project raw
    # extraction prose such as "A user is a functional user..." back into the PRD;
    # it adds no product meaning and produces awkward duplicated sentences.
    description = (
        "Secondary product user."
        if secondary
        else "Primary product user."
    )
    return UserPersona(
        name=persona_name,
        description=description,
        key_behaviors=[],
        category=source_category(source),
        actor_ids=actors,
        conditions=[],
        source_fact_ids=[source.fact_id],
    )


def _requires_functional_requirement(source: SourceFact) -> bool:
    category = source_category(source)
    if is_feature_local_goal_source(source):
        return True
    if source.topic in {
        "CORE_WORKFLOW",
        "BUSINESS_RULES",
        "EXCEPTIONS",
        "EDGE_CASES",
        "PRODUCT_MODEL",
    }:
        return True
    return category in {
        "USER_ROLES.responsibilities",
        "USER_ROLES.permissions",
        "USER_ROLES.multiple_roles",
        "USER_ROLES.role_transitions",
        "MVP_SCOPE.must_have_features",
    }


def _section_for(source: SourceFact) -> str:
    if source.topic == "USER_ROLES" and source.key in {
        "primary_users",
        "secondary_users",
    }:
        return "personas"
    if source.topic == "MVP_SCOPE" and source.key == "out_of_scope":
        return "scope.out_of_scope"
    if source.topic == "MVP_SCOPE" and source.key in {
        "must_have_features",
        "nice_to_have_features",
    }:
        return "scope.in_scope"
    if source.topic == "CONSTRAINTS":
        return "non_functional_constraints"
    if is_feature_local_goal_source(source):
        return "functional_requirements"
    if source.topic == "USER_GOALS" or (
        source.topic == "MVP_SCOPE" and source.key == "success_metrics"
    ):
        return "elevator_pitch"
    if _requires_functional_requirement(source):
        return "functional_requirements"
    return "functional_requirements"


def project_prd(sources: list[SourceFact]) -> ProjectionResult:
    visible = [source for source in sources if not is_constraint_source(source)]
    constraints = [source for source in sources if is_constraint_source(source)]

    elevator_pitch: list[SourcedClaim] = []
    in_scope: list[SourcedClaim] = []
    out_of_scope: list[SourcedClaim] = []
    personas: list[UserPersona] = []
    functional: list[FunctionalRequirement] = []
    non_functional: list[SourcedClaim] = []

    requirement_index = 1
    persona_by_actor: dict[str, UserPersona] = {}

    # Create actor shells first so later responsibility facts can enrich the
    # correct persona deterministically without asking the LLM to join records.
    for source in visible:
        if _section_for(source) != "personas":
            continue
        persona = _persona(
            source,
            secondary=(
                source.topic == "USER_ROLES"
                and source.key == "secondary_users"
            ),
        )
        personas.append(persona)
        for actor_id in persona.actor_ids:
            persona_by_actor.setdefault(actor_id, persona)

    for source in visible:
        section = _section_for(source)
        if section == "personas":
            continue
        if section == "scope.out_of_scope":
            out_of_scope.append(_claim(source))
            continue
        if section == "scope.in_scope":
            in_scope.append(_claim(source))
            if not _requires_functional_requirement(source):
                continue
        if section == "elevator_pitch":
            elevator_pitch.append(_claim(source))
            continue
        if section == "non_functional_constraints":
            non_functional.append(_claim(source))
            continue

        if (
            source.topic == "USER_ROLES"
            and source.key in {"responsibilities", "permissions"}
            and not is_system_behavior_source(source)
        ):
            behavior = _claim(source)
            for actor_id in source_actor_ids(source):
                persona = persona_by_actor.get(actor_id)
                if persona is not None and all(
                    behavior.source_fact_ids != existing.source_fact_ids
                    for existing in persona.key_behaviors
                ):
                    persona.key_behaviors.append(behavior.model_copy(deep=True))

        functional.append(
            FunctionalRequirement(
                id=f"FR-{requirement_index:02d}",
                description=(
                    render_system_behavior(source)
                    if is_system_behavior_source(source)
                    else render_lifecycle_result(source)
                    if source.topic == "CORE_WORKFLOW"
                    and source.key in {"workflow_steps", "end_state"}
                    else source.value.strip()
                ),
                validation=render_acceptance_criterion(source, (
                    render_system_behavior(source)
                    if is_system_behavior_source(source)
                    else render_lifecycle_result(source)
                    if source.topic == "CORE_WORKFLOW"
                    and source.key in {"workflow_steps", "end_state"}
                    else source.value.strip()
                )),
                category=(
                    "CORE_WORKFLOW.workflow_steps"
                    if is_system_behavior_source(source)
                    else source_category(source)
                ),
                actor_ids=[] if is_system_behavior_source(source) else source_actor_ids(source),
                conditions=[],
                source_fact_ids=[source.fact_id],
            )
        )
        requirement_index += 1

    draft = PRDDraft(
        product_name=None,
        elevator_pitch=elevator_pitch,
        scope=ScopeBoundary(in_scope=in_scope, out_of_scope=out_of_scope),
        personas=personas,
        functional_requirements=functional,
        non_functional_constraints=non_functional,
        deferred_items=[],
        open_questions=[],
    )
    return ProjectionResult(
        draft=draft,
        visible_source_ids=frozenset(source.fact_id for source in visible),
        constraint_source_ids=frozenset(source.fact_id for source in constraints),
    )


def claim_slots(draft: PRDDraft) -> list[dict]:
    """Expose text-only edit slots with immutable structural context."""
    slots: list[dict] = []
    for index, claim in enumerate(draft.elevator_pitch):
        slots.append({
            "claim_id": f"elevator_pitch/{index}",
            "text": claim.text,
            "validation": None,
            "category": claim.category,
            "source_fact_ids": claim.source_fact_ids,
        })
    for field in ("in_scope", "out_of_scope"):
        for index, claim in enumerate(getattr(draft.scope, field)):
            slots.append({
                "claim_id": f"scope/{field}/{index}",
                "text": claim.text,
                "validation": None,
                "category": claim.category,
                "source_fact_ids": claim.source_fact_ids,
            })
    for index, persona in enumerate(draft.personas):
        slots.append({
            "claim_id": f"personas/{index}",
            "text": persona.description,
            "validation": None,
            "category": persona.category,
            "source_fact_ids": persona.source_fact_ids,
        })
        for behavior_index, behavior in enumerate(persona.key_behaviors):
            slots.append({
                "claim_id": f"personas/{index}/key_behaviors/{behavior_index}",
                "text": behavior.text,
                "validation": None,
                "category": behavior.category,
                "source_fact_ids": behavior.source_fact_ids,
            })
    for requirement in draft.functional_requirements:
        slots.append({
            "claim_id": f"functional_requirements/{requirement.id}",
            "text": requirement.description,
            "validation": requirement.validation,
            "category": requirement.category,
            "source_fact_ids": requirement.source_fact_ids,
        })
    for index, claim in enumerate(draft.non_functional_constraints):
        slots.append({
            "claim_id": f"non_functional_constraints/{index}",
            "text": claim.text,
            "validation": None,
            "category": claim.category,
            "source_fact_ids": claim.source_fact_ids,
        })
    return slots


def apply_prose_edits(draft: PRDDraft, edits) -> PRDDraft:
    """Apply wording only; structural fields are impossible for edits to mutate."""
    result = deepcopy(draft)
    by_id = {}
    for edit in edits:
        if edit.claim_id in by_id:
            continue
        by_id[edit.claim_id] = edit

    for index, claim in enumerate(result.elevator_pitch):
        edit = by_id.get(f"elevator_pitch/{index}")
        if edit:
            claim.text = edit.text.strip()
    for field in ("in_scope", "out_of_scope"):
        for index, claim in enumerate(getattr(result.scope, field)):
            edit = by_id.get(f"scope/{field}/{index}")
            if edit:
                claim.text = edit.text.strip()
    for index, persona in enumerate(result.personas):
        edit = by_id.get(f"personas/{index}")
        if edit:
            persona.description = edit.text.strip()
        for behavior_index, behavior in enumerate(persona.key_behaviors):
            behavior_edit = by_id.get(
                f"personas/{index}/key_behaviors/{behavior_index}"
            )
            if behavior_edit:
                behavior.text = behavior_edit.text.strip()
    for requirement in result.functional_requirements:
        edit = by_id.get(f"functional_requirements/{requirement.id}")
        if edit:
            requirement.description = edit.text.strip()
            if edit.validation:
                requirement.validation = edit.validation.strip()
    for index, claim in enumerate(result.non_functional_constraints):
        edit = by_id.get(f"non_functional_constraints/{index}")
        if edit:
            claim.text = edit.text.strip()
    return result


def validate_projection(
    result: ProjectionResult,
    sources: list[SourceFact],
) -> None:
    """Fail only on deterministic projection bugs, never on LLM bookkeeping."""
    source_ids = {source.fact_id for source in sources}
    if result.visible_source_ids | result.constraint_source_ids != source_ids:
        raise ValueError("Projection source partition does not cover the source ledger.")
    if result.visible_source_ids & result.constraint_source_ids:
        raise ValueError("A source cannot be both visible and constraint-only.")

    cited = {
        ref
        for slot in claim_slots(result.draft)
        for ref in slot["source_fact_ids"]
    }
    missing = result.visible_source_ids - cited
    unknown = cited - result.visible_source_ids
    if missing:
        raise ValueError(
            f"Visible sources omitted from deterministic projection: {sorted(missing)}"
        )
    if unknown:
        raise ValueError(
            "Constraint/unknown sources leaked into visible PRD claims: "
            f"{sorted(unknown)}"
        )
