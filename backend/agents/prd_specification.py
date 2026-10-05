"""Canonical product model and founder-facing feature specification projection.

This layer sits between the immutable discovery source ledger and PRD rendering.
It groups related grounded facts into coherent product capabilities without
changing coverage or inventing behavior. Atomic source-linked requirements remain
in PRDDraft for provenance/verification; feature specifications are the readable
product document view.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from agents.prd_schema import (
    FeatureSpecification,
    PRDDraft,
    SourceFact,
    SourcedClaim,
)


WORD_RE = re.compile(r"[a-z0-9]+")

CAPABILITY_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "persistence",
        re.compile(
            r"\b(?:persist|persistent|stored?|cloud|across\s+devices?|sync|"
            r"remain\s+(?:available|saved)|account[- ]based|save(?:d)?)\b",
            re.I,
        ),
    ),
    (
        "authentication",
        re.compile(
            r"\b(?:sign[- ]?in|log[- ]?in|login|register|registration|"
            r"account\s+creation|authenticate|authentication|password|oauth)\b",
            re.I,
        ),
    ),
    (
        "payments",
        re.compile(
            r"\b(?:payment|pay|billing|subscription|invoice|checkout|refund|"
            r"escrow|fee|payout)\b",
            re.I,
        ),
    ),
    (
        "scheduling",
        re.compile(
            r"\b(?:schedule|calendar|appointment|booking|session\s+time|"
            r"availability)\b",
            re.I,
        ),
    ),
    (
        "communication",
        re.compile(
            r"\b(?:message|chat|call|communicat|conversation|comment)\w*\b",
            re.I,
        ),
    ),
    (
        "notifications",
        re.compile(
            r"\b(?:notification|notify|reminder|alert)\w*\b",
            re.I,
        ),
    ),
    (
        "reporting",
        re.compile(
            r"\b(?:report|analytics|dashboard|metric|insight|trend)\w*\b",
            re.I,
        ),
    ),
    (
        "submission",
        re.compile(
            r"\b(?:submit|submission|upload|assignment|document\s+upload)\w*\b",
            re.I,
        ),
    ),
    (
        "approval",
        re.compile(
            r"\b(?:approve|approval|review|moderate|verification)\w*\b",
            re.I,
        ),
    ),
    (
        "search",
        re.compile(
            r"\b(?:search|filter|discover|browse|sort)\w*\b",
            re.I,
        ),
    ),
)


@dataclass(frozen=True)
class CanonicalProductModel:
    sources: tuple[SourceFact, ...]
    visible_sources: tuple[SourceFact, ...]
    constraint_sources: tuple[SourceFact, ...]
    actors: tuple[str, ...]
    entities: tuple[str, ...]


def _words(value: str) -> set[str]:
    return set(WORD_RE.findall((value or "").lower()))


def _singular(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if normalized.endswith("s") and not normalized.endswith("ss") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


def _humanize(value: str) -> str:
    return re.sub(r"[_\-]+", " ", value).strip().title()


def build_product_model(sources: list[SourceFact]) -> CanonicalProductModel:
    visible = tuple(source for source in sources if source.absence is None)
    constraints = tuple(source for source in sources if source.absence is not None)

    actors: list[str] = []
    entities: list[str] = []
    for source in visible:
        for actor in [source.role, *(source.roles or [])]:
            if actor and actor not in actors:
                actors.append(actor)
        if source.topic == "PRODUCT_MODEL" and source.subject:
            subject = _singular(source.subject)
            if subject and subject not in entities:
                entities.append(subject)

    return CanonicalProductModel(
        sources=tuple(sources),
        visible_sources=visible,
        constraint_sources=constraints,
        actors=tuple(actors),
        entities=tuple(entities),
    )


def _entity_for_source(source: SourceFact, entities: tuple[str, ...]) -> str | None:
    if source.subject:
        return _singular(source.subject)

    semantic_text = " ".join(
        value
        for value in (
            source.value,
            source.subject or "",
            source.relation or "",
            source.object or "",
        )
        if value
    )
    text_words = _words(semantic_text)
    for entity in entities:
        entity_words = _words(entity)
        plural_words = {_singular(word) for word in text_words}
        if entity_words and entity_words <= plural_words:
            return entity
        # Single-token domain entities are common and should match plural forms.
        if len(entity_words) == 1:
            token = next(iter(entity_words))
            if token in {_singular(word) for word in text_words}:
                return entity
    return None


def _family_for_source(source: SourceFact) -> str | None:
    # Group by the source's asserted meaning, never by its evidence span. One
    # founder sentence can support several atomic facts and may mention unrelated
    # capabilities; using evidence here would contaminate bundle classification.
    text = " ".join(
        value
        for value in (
            source.value,
            source.subject or "",
            source.relation or "",
            source.object or "",
        )
        if value
    )
    for family, pattern in CAPABILITY_FAMILIES:
        if pattern.search(text):
            return family
    return None


def _bundle_key(source: SourceFact, model: CanonicalProductModel) -> tuple[str, str | None, str | None]:
    entity = _entity_for_source(source, model.entities)
    family = _family_for_source(source)

    # Persistence/account availability is product-visible enough to deserve its
    # own feature even when it concerns an existing entity.
    if family in {"persistence", "authentication", "payments", "scheduling", "communication",
                  "notifications", "reporting", "submission", "approval", "search"}:
        return (family, entity, source.role)

    if entity:
        return ("entity_core", entity, source.role)

    if source.role:
        return ("actor_capability", None, source.role)

    if source.topic == "CORE_WORKFLOW":
        return ("workflow", None, None)
    if source.topic in {"BUSINESS_RULES", "EXCEPTIONS", "EDGE_CASES"}:
        return ("rules", None, None)
    return ("product_capability", None, None)


def _bundle_title(kind: str, entity: str | None, role: str | None) -> str:
    subject = _humanize(entity) if entity else None
    role_name = _humanize(role) if role else None

    if kind == "persistence":
        return f"{subject} Persistence & Access" if subject else "Persistence & Access"
    if kind == "authentication":
        return "Account & Authentication"
    if kind == "payments":
        return "Payments & Billing"
    if kind == "scheduling":
        return "Scheduling"
    if kind == "communication":
        return "Communication"
    if kind == "notifications":
        return "Notifications & Reminders"
    if kind == "reporting":
        return "Reporting & Analytics"
    if kind == "submission":
        return "Submission Management"
    if kind == "approval":
        return "Review & Approval"
    if kind == "search":
        return "Search & Organization"
    if kind == "entity_core":
        return f"{subject} Management" if subject else "Core Product Management"
    if kind == "actor_capability":
        return f"{role_name} Capabilities" if role_name else "User Capabilities"
    if kind == "workflow":
        return "Core Workflow"
    if kind == "rules":
        return "Product Rules & Exceptions"
    return "Core Product Capabilities"


def _bundle_overview(
    kind: str,
    title: str,
    entity: str | None,
    role: str | None,
    sources: list[SourceFact],
) -> str:
    subject = _humanize(entity) if entity else None
    role_name = _humanize(role) if role else "Users"

    if kind == "entity_core" and subject:
        has_actions = any(
            source.topic == "USER_ROLES"
            and source.key in {"responsibilities", "permissions"}
            for source in sources
        )
        has_model = any(source.topic == "PRODUCT_MODEL" for source in sources)
        if has_actions and has_model:
            return (
                f"Covers how {role_name.lower() if role else 'users'} manage "
                f"{subject.lower()} records and the confirmed data, states, and "
                f"rules that define each {subject.lower()}."
            )
        if has_actions:
            return (
                f"Covers the confirmed actions available for managing "
                f"{subject.lower()} records."
            )
        return (
            f"Defines the confirmed data, states, relationships, and rules for "
            f"{subject.lower()} records."
        )

    if kind == "persistence":
        noun = f"{subject.lower()} data" if subject else "product data"
        return (
            f"Defines how {noun} is retained and made available according to the "
            f"confirmed persistence and access decisions."
        )
    if kind == "authentication":
        return "Defines the confirmed account access and authentication behavior."
    if kind == "payments":
        return "Defines the confirmed payment, billing, and money-movement behavior."
    if kind == "scheduling":
        return "Defines the confirmed scheduling and time-based product behavior."
    if kind == "communication":
        return "Defines the confirmed communication capabilities and boundaries."
    if kind == "notifications":
        return "Defines the confirmed notification and reminder behavior."
    if kind == "reporting":
        return "Defines the confirmed reporting, analytics, and insight capabilities."
    if kind == "submission":
        return "Defines the confirmed submission and content-handling workflow."
    if kind == "approval":
        return "Defines the confirmed review and approval workflow."
    if kind == "search":
        return "Defines the confirmed search, filtering, and organization behavior."
    if kind == "actor_capability":
        return f"Groups the confirmed product capabilities available to {role_name}."
    if kind == "workflow":
        return "Describes the confirmed end-to-end product workflow."
    if kind == "rules":
        return "Groups the confirmed product rules, exceptions, and edge-case behavior."
    return f"Groups the confirmed requirements that make up {title.lower()}."


def _join_items(values: list[str]) -> str:
    cleaned = [value.strip().rstrip(".") for value in values if value.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _attribute_label(source: SourceFact) -> str:
    if source.object:
        return source.object.strip()
    return source.value.strip()


def _group_text(category: str, sources: list[SourceFact]) -> str:
    values = [source.value.strip() for source in sources]
    actors = [source.role for source in sources if source.role]
    actor = actors[0] if actors and all(item == actors[0] for item in actors) else None

    if category == "USER_ROLES.responsibilities":
        prefix = f"{_humanize(actor)} can " if actor else "Users can "
        return prefix + _join_items(values) + "."

    if category == "USER_ROLES.permissions":
        prefix = f"{_humanize(actor)} permissions: " if actor else "Permissions: "
        return prefix + _join_items(values) + "."

    if category == "PRODUCT_MODEL.attribute":
        subjects = {_singular(source.subject or "") for source in sources if source.subject}
        subject = next(iter(subjects)) if len(subjects) == 1 else None
        prefix = f"{_humanize(subject)} details: " if subject else "Data details: "
        return prefix + _join_items([_attribute_label(source) for source in sources]) + "."

    if category == "PRODUCT_MODEL.entity":
        return _join_items(values) + "."

    if category.startswith("CORE_WORKFLOW."):
        return "Workflow: " + _join_items(values) + "."

    if category.startswith("BUSINESS_RULES."):
        return "Rule: " + _join_items(values) + "."

    if category.startswith("EXCEPTIONS.") or category.startswith("EDGE_CASES."):
        return "Exception handling: " + _join_items(values) + "."

    if category == "MVP_SCOPE.must_have_features":
        return "MVP capability: " + _join_items(values) + "."

    return _join_items(values) + "."


def _actor_ids(sources: list[SourceFact]) -> list[str]:
    result: list[str] = []
    for source in sources:
        for actor in [source.role, *(source.roles or [])]:
            if actor and actor not in result:
                result.append(actor)
    return result


def build_feature_specifications(
    draft: PRDDraft,
    model: CanonicalProductModel,
) -> list[FeatureSpecification]:
    requirement_by_source = {
        ref: requirement.id
        for requirement in draft.functional_requirements
        for ref in requirement.source_fact_ids
    }
    operational = [
        source
        for source in model.visible_sources
        if source.fact_id in requirement_by_source
    ]

    bundles: dict[tuple[str, str | None, str | None], list[SourceFact]] = {}
    order: list[tuple[str, str | None, str | None]] = []
    for source in operational:
        key = _bundle_key(source, model)
        if key not in bundles:
            bundles[key] = []
            order.append(key)
        bundles[key].append(source)

    result: list[FeatureSpecification] = []
    for index, key in enumerate(order, start=1):
        kind, entity, role = key
        sources = bundles[key]
        grouped: dict[str, list[SourceFact]] = {}
        category_order: list[str] = []
        for source in sources:
            category = f"{source.topic}.{source.key}"
            if category not in grouped:
                grouped[category] = []
                category_order.append(category)
            grouped[category].append(source)

        details = [
            SourcedClaim(
                text=_group_text(category, grouped[category]),
                category=category,
                actor_ids=_actor_ids(grouped[category]),
                conditions=[],
                source_fact_ids=[source.fact_id for source in grouped[category]],
            )
            for category in category_order
        ]
        title = _bundle_title(kind, entity, role)
        requirement_ids = list(dict.fromkeys(
            requirement_by_source[source.fact_id] for source in sources
        ))
        result.append(
            FeatureSpecification(
                id=f"FEATURE-{index:02d}",
                title=title,
                overview=_bundle_overview(
                    kind,
                    title,
                    entity,
                    role,
                    sources,
                ),
                details=details,
                requirement_ids=requirement_ids,
                source_fact_ids=[source.fact_id for source in sources],
            )
        )

    return result



def validate_feature_specifications(
    features: list[FeatureSpecification],
    draft: PRDDraft,
    model: CanonicalProductModel,
) -> None:
    expected_source_ids = {
        ref
        for requirement in draft.functional_requirements
        for ref in requirement.source_fact_ids
    }
    requirement_ids = {requirement.id for requirement in draft.functional_requirements}
    constraint_ids = {source.fact_id for source in model.constraint_sources}

    assigned: list[str] = []
    for feature in features:
        if len(feature.source_fact_ids) != len(set(feature.source_fact_ids)):
            raise ValueError(f"{feature.id}: duplicate source IDs in feature bundle.")
        if any(ref in constraint_ids for ref in feature.source_fact_ids):
            raise ValueError(
                f"{feature.id}: constraint-only source leaked into feature bundle."
            )
        if any(req not in requirement_ids for req in feature.requirement_ids):
            raise ValueError(f"{feature.id}: unknown functional requirement ID.")
        detail_ids = {
            ref
            for detail in feature.details
            for ref in detail.source_fact_ids
        }
        if detail_ids != set(feature.source_fact_ids):
            raise ValueError(
                f"{feature.id}: feature details do not exactly cover bundle sources."
            )
        assigned.extend(feature.source_fact_ids)

    if len(assigned) != len(set(assigned)):
        raise ValueError("A functional source was assigned to multiple feature bundles.")
    if set(assigned) != expected_source_ids:
        missing = sorted(expected_source_ids - set(assigned))
        extra = sorted(set(assigned) - expected_source_ids)
        raise ValueError(
            "Feature specification coverage does not match functional source ledger: "
            f"missing={missing}, extra={extra}."
        )
