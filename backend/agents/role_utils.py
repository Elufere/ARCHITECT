"""Canonical handling for role lists used across discovery nodes."""

import re


def split_role_labels(roles: list[str] | None) -> list[str]:
    """Expand a model-returned compound list entry into atomic role labels.

    The extraction contract requires roles to be individual labels.  This is a
    defensive repair for model output such as ``["administrators and support
    staff"]``; it deliberately preserves both roles rather than inventing a
    combined role in the planner.
    """
    atomic_roles = []
    seen = set()
    for role in roles or []:
        for label in re.split(r"\s*(?:,|/|&|\band\b)\s*", role, flags=re.I):
            label = label.strip().lower()
            if label and label not in seen:
                seen.add(label)
                atomic_roles.append(label)
    return atomic_roles


def role_identity(role: str | None) -> str:
    """Compare harmless aliases without changing the label presented to users."""
    normalized = (role or "").replace("_", " ").strip().lower().rstrip("s")
    return "admin" if normalized in {"admin", "administrator"} else normalized


def roles_match(first: str | None, second: str | None) -> bool:
    return bool(first and second and role_identity(first) == role_identity(second))
