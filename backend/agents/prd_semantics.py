"""Context-sensitive, deterministic wording and classification helpers for PRD compilation."""
from __future__ import annotations

import re


_GOAL_KEYS = {"primary_user_goals", "secondary_user_goals"}


def question_requests_system_behavior(question: str | None) -> bool:
    """Whether the captured question explicitly asks what the app/system should do."""
    text = " ".join((question or "").strip().lower().split())
    return bool(text and re.search(
        r"\b(?:should|would|could|can|does|will)\s+(?:the\s+)?"
        r"(?:app|application|system|product)\b",
        text,
    ))


def is_system_behavior_claim(question: str | None, value: str | None) -> bool:
    """Recognize system-action answers to questions explicitly about app behavior."""
    if not question_requests_system_behavior(question):
        return False
    text = (value or "").strip()
    return bool(re.match(
        r"^(?:(?:the|this)\s+)?(?:app|application|system|product)\b"
        r"|^(?:automatically\s+)?(?:re)?schedule\b"
        r"|^receives?\s+reminders?\b"
        r"|^(?:send|cancel|notify|deliver|remove|store|save|sync|synchronize|show|display)\b",
        text,
        re.I,
    ))


def is_system_behavior_source(source) -> bool:
    """True only for action-like source facts whose recorded question supports system ownership."""
    category_is_action = (
        (source.topic == "USER_ROLES" and source.key == "responsibilities")
        or (source.topic == "CORE_WORKFLOW" and source.key == "workflow_steps")
    )
    return category_is_action and is_system_behavior_claim(
        source.source_question, source.value
    )


def render_system_behavior(source) -> str:
    """Render short system-action answers as clear, conditional product behavior."""
    value = (source.value or "").strip().rstrip(".")
    question = (source.source_question or "").lower()

    if re.match(r"^receives?\s+reminders?\b", value, re.I):
        schedule = re.sub(r"^receives?\s+reminders?\s*", "", value, flags=re.I)
        schedule = re.sub(r"\b1\s*hr\b", "1 hour", schedule, flags=re.I)
        schedule = re.sub(r"\b30\s*mins?\b", "30 minutes", schedule, flags=re.I)
        schedule = re.sub(r"\b5\s*mins?\b", "5 minutes", schedule, flags=re.I)
        schedule = schedule.replace("the due date and time", "the task’s due date and time")
        return f"The app sends reminders {schedule}."

    if re.match(r"^(?:automatically\s+)?reschedule\b", value, re.I):
        if (
            re.search(r"\bcompleted task\b", question)
            and re.search(r"\b(?:active|reopen|returned)\b", question)
            and re.search(r"\bfuture\b", question)
        ):
            return (
                "When a completed task is returned to active while its due date and time "
                "are still in the future, the app reschedules all reminders whose "
                "scheduled times have not passed."
            )
        remainder = re.sub(
            r"^(?:automatically\s+)?reschedule\s+", "", value, flags=re.I
        )
        return "The app automatically reschedules " + remainder[:1].lower() + remainder[1:] + "."

    if re.match(r"^(?:the|this)\s+(?:app|application|system|product)\b", value, re.I):
        return value[:1].upper() + value[1:] + "."
    return value[:1].upper() + value[1:] + "."


def is_feature_local_rationale(kind: str, evidence: str | None, question: str | None) -> bool:
    """A benefit clause attached to a specific behavior decision is not a global goal."""
    if kind not in {"desired_outcome", *_GOAL_KEYS}:
        return False
    quote = (evidence or "").strip()
    rationale_clause = bool(re.match(
        r"^(?:so(?:\s+that)?|because|in order to|therefore)\b", quote, re.I
    ))
    return rationale_clause and question_requests_system_behavior(question)


def is_feature_local_goal_source(source) -> bool:
    return (
        source.topic == "USER_GOALS"
        and source.key in _GOAL_KEYS
        and is_feature_local_rationale(
            source.key, source.evidence, source.source_question
        )
    )


def render_lifecycle_result(source) -> str:
    """Render confirmed workflow steps and resulting states with their trigger context."""
    value = (source.value or "").strip().rstrip(".")
    question = (source.source_question or "").lower()

    if (
        source.topic == "CORE_WORKFLOW"
        and source.key == "workflow_steps"
        and re.search(r"deletion.*confirmation|confirmation.*deletion", value, re.I)
    ):
        return "Deletion requires confirmation before the task is removed."

    if source.topic == "CORE_WORKFLOW" and source.key == "end_state":
        if (
            "reminder" in value.lower()
            and re.search(r"\bcancel", value, re.I)
            and "completed" in question
            and "before" in question
        ):
            return (
                "When a task is marked completed before its due time, the app cancels "
                "all remaining scheduled reminders."
            )
        if (
            re.search(r"\bdeleted tasks?\b", value, re.I)
            and re.search(r"\bremoved from the app\b", value, re.I)
        ):
            return "After deletion, the task is removed from the app."

    return value[:1].upper() + value[1:] + "."


def normalize_action_phrase(value: str) -> str:
    """Normalize common third-person extraction fragments into capability verbs."""
    text = (value or "").strip()
    replacements = {
        "creates": "create", "edits": "edit", "deletes": "delete",
        "marks": "mark", "receives": "receive", "changes": "change",
        "sets": "set", "updates": "update", "completes": "complete",
        "opens": "open", "closes": "close", "cancels": "cancel",
        "sends": "send", "removes": "remove", "shows": "show",
        "adds": "add", "schedules": "schedule", "reschedules": "reschedule",
    }
    for inflected, base in replacements.items():
        if re.match(rf"^{inflected}\b", text, re.I):
            return re.sub(rf"^{inflected}\b", base, text, count=1, flags=re.I)
    return text
