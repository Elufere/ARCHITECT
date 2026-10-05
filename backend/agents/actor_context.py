"""Context-aware resolution of actor labels, aliases and discourse references."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage

from agents.extraction_passes import canonical_role
from agents.role_utils import role_identity
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeState


GENERIC_REFERENCES = {
    "they", "them", "their", "the person", "person", "party", "side",
    "user", "users", "customer/user",
}


@dataclass
class ActorIdentity:
    canonical: str
    labels: set[str] = field(default_factory=set)


def _normalize_label(value: str | None) -> str:
    if not value:
        return ""
    return canonical_role(role_identity(value.replace("_", " ")))


def _label_forms(label: str) -> set[str]:
    phrase = " ".join(_normalize_label(label).split("_"))
    if not phrase:
        return set()
    forms = {phrase}
    parts = phrase.split()
    last = parts[-1]
    if not last.endswith("s"):
        forms.add(" ".join([*parts[:-1], last + "s"]))
    return forms


def actor_registry(state: AgentState, scope: DiscoveryScope) -> dict[str, ActorIdentity]:
    """Return only confirmed application actors and their explicitly proven aliases."""
    registry: dict[str, ActorIdentity] = {}
    for item in state.get("discovered_knowledge", []):
        if (
            item.scope != scope
            or item.topic != DiscoveryTopic.USER_ROLES
            or item.key not in ("primary_users", "secondary_users")
            or item.absence
            or item.knowledge_state != KnowledgeState.CONFIRMED
        ):
            continue
        for raw_role in item.roles or []:
            canonical = _normalize_label(raw_role)
            if not canonical:
                continue
            identity = registry.setdefault(
                canonical,
                ActorIdentity(canonical=canonical, labels={canonical}),
            )
            aliases = [
                *(item.aliases or {}).get(raw_role, []),
                *(item.aliases or {}).get(canonical, []),
            ]
            identity.labels.update(
                normalized
                for normalized in (_normalize_label(alias) for alias in aliases)
                if normalized
            )
    # Role-policy facts may establish capacities for an actor whose membership
    # was already confirmed elsewhere. They enrich identity; they never create it.
    for item in state.get("discovered_knowledge", []):
        if (
            item.scope != scope
            or item.topic != DiscoveryTopic.USER_ROLES
            or item.key not in ("multiple_roles", "role_transitions")
            or item.absence
            or item.knowledge_state != KnowledgeState.CONFIRMED
            or not item.aliases
        ):
            continue
        for raw_role, aliases in item.aliases.items():
            canonical = _normalize_label(raw_role)
            identity = registry.get(canonical)
            if identity is None:
                continue
            identity.labels.update(
                normalized
                for normalized in (_normalize_label(alias) for alias in aliases)
                if normalized and normalized != canonical
            )
    return registry


def canonical_actor_for_label(
    label: str | None,
    state: AgentState,
    scope: DiscoveryScope,
) -> str | None:
    """Resolve an explicit label only when it maps to a confirmed canonical actor."""
    normalized = _normalize_label(label)
    if not normalized:
        return None
    matches = [
        identity.canonical
        for identity in actor_registry(state, scope).values()
        if normalized in identity.labels
    ]
    return matches[0] if len(set(matches)) == 1 else None


def actors_mentioned_in_text(
    text: str,
    state: AgentState,
    scope: DiscoveryScope,
) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    matches: set[str] = set()
    for identity in actor_registry(state, scope).values():
        for label in identity.labels:
            if any(
                re.search(rf"\b{re.escape(form)}\b", normalized)
                for form in _label_forms(label)
            ):
                matches.add(identity.canonical)
                break
    return matches


def _question_text(state: AgentState) -> str:
    return next(
        (
            message.content
            for message in reversed(state.get("messages", [])[:-1])
            if isinstance(message, AIMessage)
        ),
        "",
    )


def _focused_actor(state: AgentState, scope: DiscoveryScope) -> str | None:
    """Resolve a single actor explicitly selected by current discovery state."""
    candidates: list[str | None] = []

    gap = state.get("current_gap") or ""
    if "::" in gap:
        candidates.append(gap.split("::", 1)[1])

    candidates.append(state.get("current_role"))

    for payload_name in ("selected_inquiry", "selected_requirement_candidate", "thread_frontier"):
        payload = state.get(payload_name)
        if isinstance(payload, dict):
            candidates.append(payload.get("role"))

    resolved = {
        canonical
        for candidate in candidates
        if (canonical := canonical_actor_for_label(candidate, state, scope))
    }
    return next(iter(resolved)) if len(resolved) == 1 else None


def _has_contextual_reference(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return bool(re.search(
        r"\b(?:they|them|their|the person|that person|this person|"
        r"the party|that party|this party|either party|either side|"
        r"both parties|both sides|he|she|his|her)\b",
        normalized,
    ))


def resolve_owned_claim_role(
    proposed_role: str | None,
    evidence: str,
    state: AgentState,
    scope: DiscoveryScope,
) -> str | None:
    """Resolve an owned product claim to an already-confirmed canonical actor.

    Resolution order is intentionally conservative:
    1. explicit canonical/alias label proposed by extraction;
    2. exactly one confirmed actor/alias explicitly named in the evidence;
    3. a contextual pronoun/reference tied to one actor selected by the current
       discovery focus or uniquely named in the immediately preceding PM question;
    4. a contextual reference when only one canonical actor exists at all.

    Never returns an unconfirmed label and never guesses across multiple plausible
    canonical actors.
    """
    explicit = canonical_actor_for_label(proposed_role, state, scope)
    if explicit:
        return explicit

    mentioned = actors_mentioned_in_text(evidence, state, scope)
    if len(mentioned) == 1:
        return next(iter(mentioned))
    if len(mentioned) > 1:
        return None

    if not _has_contextual_reference(evidence):
        return None

    focused = _focused_actor(state, scope)
    if focused:
        return focused

    question_matches = actors_mentioned_in_text(_question_text(state), state, scope)
    if len(question_matches) == 1:
        return next(iter(question_matches))

    registry = actor_registry(state, scope)
    if len(registry) == 1:
        return next(iter(registry))

    return None
