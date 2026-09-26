"""
Production-Ready Stage Classifier & Validator.
Validates AI output against PRODUCT_DISCOVERY stage. Does NOT rewrite.
"""

import logging
import re
from enum import Enum

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from agents.llm_errors import raise_if_llm_failure
from agents.llm_errors import ExtractionFailed
from agents.llm import get_structured_model
from pydantic import BaseModel

from agents.state import DiscoveryScope
from agents.role_utils import role_identity, split_role_labels
from agents.conversation_language import clarification_question

logger = logging.getLogger(__name__)

class Stage(str, Enum):
    PRODUCT_DISCOVERY = "PRODUCT_DISCOVERY"
    SOLUTION_DESIGN = "SOLUTION_DESIGN"
    SYSTEM_DESIGN = "SYSTEM_DESIGN"
    TECHNICAL_IMPLEMENTATION = "TECHNICAL_IMPLEMENTATION"
    ROADMAP = "ROADMAP"
    OTHER = "OTHER"

class EvaluationSchema(BaseModel):
    stage: Stage
    passed: bool
    offending_questions: list[int]
    explanation: str
    guidance: str

evaluator_llm = get_structured_model(call_name="guardrail", schema=EvaluationSchema)

# ─────────────────────────────────────────────────────────────────────
# Deterministic duplicate detection is intentionally conservative. Topic
# relevance is evaluated semantically by the structured evaluator below:
# lexical overlap with planner instructions caused valid questions to fail.
# ─────────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "that", "this", "it", "they", "their", "have", "has",
    "had", "be", "been", "being", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "not", "no", "yes", "so", "if", "as", "just", "also",
    "ask", "user", "users", "about", "what", "who", "when", "how",
    "why", "does", "specifically", "only",
}

def get_content_words(text: str) -> set:
    """Extracts meaningful words, ignoring stop words. Mirrors knowledge_tracker.py's helper."""
    words = re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
    return {w for w in words if w not in STOP_WORDS}


def check_topic_relevance(
    question: str,
    question_hint: str,
    current_objective: str,
) -> tuple[bool, str]:
    """Kept as a compatibility hook; semantic relevance is checked by the LLM."""
    return True, ""


SEMANTIC_ALIASES = {
    "workflow": {"workflow", "journey", "process", "flow", "steps"},
    "completion": {"complete", "completion", "finish", "finished", "end"},
    "approval": {"approval", "approve", "review", "authorise", "authorize"},
}


def _canonical_question_words(question: str) -> set[str]:
    words = get_content_words(question)
    canonical = set(words)
    for concept, aliases in SEMANTIC_ALIASES.items():
        if words & aliases:
            canonical.add(concept)
    return canonical


def questions_are_semantic_duplicates(first: str, second: str) -> bool:
    """Catch repeated end-to-end workflow requests without blocking distinct gaps."""
    first_words = _canonical_question_words(first)
    second_words = _canonical_question_words(second)
    if not first_words or not second_words:
        return False
    # "Workflow from start to finish" and "journey from beginning to end" are
    # equivalent discovery requests.  Do not apply broad word-overlap matching
    # to other topics: it falsely treats separate role-specific goal questions
    # as duplicates merely because they share words such as "goals" and "app".
    repeated_workflow_request = {
        "workflow", "completion"
    }.issubset(first_words) and {"workflow", "completion"}.issubset(second_words)
    if repeated_workflow_request:
        return len(first_words & second_words) >= 3

    return " ".join(first.lower().split()) == " ".join(second.lower().split())


def delivered_prior_questions(messages: list) -> list[str]:
    """Return only questions that reached the user, excluding rejected drafts."""
    questions = []
    for index, message in enumerate(messages[:-1]):
        if not isinstance(message, AIMessage):
            continue
        # A guardrail rejection follows a draft with a SystemMessage. A human
        # answer means the question was actually presented to the user.
        if isinstance(messages[index + 1], HumanMessage):
            questions.append(message.content.strip())
    return questions


EVALUATOR_PROMPT = """
You are validating a Product Manager's interview question.

Current discovery topic:
{current_topic}

Planner source:
{planner_source}

Current objective:
{current_objective}

Current schema gap / extraction anchor:
{current_gap}

Selected requirement context:
{requirement_context}

Validation context:
{validation_context}

Question:
{agent_output}

The planner has already determined that this is the next missing piece of
information.

Known facts for this topic:
{known_facts}
Known facts may have been learned incidentally. A question asking the user to
confirm their completeness or expand them for the Current gap is valid discovery;
knowledge presence alone does not mean the gap has been deliberately resolved.

Previously asked questions for this topic:
{previous_questions}

Your job is to determine whether the question is asking specifically about
the Current objective.

When Planner source is "model", the Current gap is only an extraction anchor.
Validate the question against the Current objective. Do not require neighboring
schema fields to be covered, and do not reject a useful question merely because
its natural wording crosses a field boundary while resolving the selected
product uncertainty.

When Planner source is "requirement", the gap is ONLY an extraction anchor.
Validate the question against the Selected requirement context and target facets
instead. A valid requirement question may naturally span more than one field if
all parts directly serve the selected requirement.

When Planner source is "validation", the question must neutrally resolve the
supplied contradiction. It may quote or summarize the two incompatible confirmed
statements and ask which CURRENT rule/decision applies. It must not choose a side,
silently merge them, or drift into unrelated discovery.

Reject if the question:
- changes to another topic
- drifts to a different product decision that does not help resolve the Current objective
- asks multiple unrelated objectives; closely related target facets of one selected requirement are allowed
- asks implementation
- asks architecture
- asks roadmap planning unrelated to the selected gap (MVP_SCOPE questions about
  launch inclusion, optional/deferred features, and exclusions are valid scope discovery)
- asks technical design
- asks about exceptions/errors when the gap is about goals or workflow
- asks about the happy path when the gap is about exceptions or edge cases

CRITICAL: The question MUST directly discover the "Current objective" stated above.
If Planner source is "requirement", it must directly investigate the selected
requirement and at least one target facet. If the question could be answered
without addressing that specific objective, REJECT IT.

Do NOT reject merely because it mentions users, permissions, workflows,
approvals, validation, or business concepts, PROVIDED it is asking about
the Current gap specifically.

If the question directly discovers the current objective, it MUST pass.

Return the schema.
"""

ROLE_LEAK_REJECTION = """CRITICAL ERROR: Your previous question mentioned role(s) other than '{current_role}'.

The current objective is scoped to the '{current_role}' role ONLY.

Rewrite the question so it asks about '{current_role}' exclusively, and
does not name, compare to, or reference any other role.
"""

EVALUATOR_REJECTION = """CRITICAL ERROR: Your previous question did not correctly discover the current objective.

Reason: {reason}

The current gap is: {current_gap}
The current objective is: {current_objective}

Rewrite the question so it ONLY asks about: {current_objective}
Do NOT ask about any other topic or field.
"""

NOT_A_QUESTION_REJECTION = """CRITICAL ERROR: Your output must be a single interview question ending with a '?'.
Do not make statements or acknowledgments (e.g. "Got it, thanks!"). Ask a question."""

TOPIC_RELEVANCE_REJECTION = """CRITICAL ERROR: Your previous question drifted away from what you were supposed to ask.

Reason: {reason}

The current gap is: {current_gap}
The current objective is: {current_objective}

Rewrite the question so it clearly and directly asks about: {current_objective}
"""

USER_SCOPE_ROLE_REJECTION = """CRITICAL ERROR: This is customer-app discovery.

Do not introduce or use administrators, support staff, moderators, or other
internal roles as examples. Ask only about customer-facing users, or ask about
the handoff point without asking what internal staff do.
"""

INTERNAL_ROLE_PATTERN = re.compile(
    r"\b(admin(?:istrator)?s?|support(?: staff)?|customer support|"
    r"moderators?|back[- ]office|internal staff)\b", re.I
)


def normalize_role(role: str) -> str:
    """Strips trailing 's' to treat 'customer' and 'customers' as the same role."""
    return role_identity(role)

def get_known_roles(state: dict, topic) -> set[str]:
    """Roles derived from primary_users/secondary_users facts for this topic."""
    roles = set()
    for item in state.get("discovered_knowledge", []):
        if item.topic == topic and item.key in ("primary_users", "secondary_users"):
            for r in split_role_labels(item.roles or item.value.split(",")):
                r = r.strip().lower()
                if r:
                    roles.add(r)
    return roles


def question_mentions_other_roles(question: str, current_role: str, all_roles: set[str]) -> list[str]:
    norm_current = normalize_role(current_role)
    other_roles = {r for r in all_roles if normalize_role(r) != norm_current}

    if not other_roles:
        return []

    q_lower = question.lower()
    leaked = []

    for role in other_roles:
        norm_role = normalize_role(role)
        pattern = rf"\b{re.escape(norm_role)}s?\b"
        if re.search(pattern, q_lower):
            leaked.append(role)

    return leaked


def evaluate_question(state: dict) -> dict:
    """Validates the AI's output. If it fails, appends a rejection to force a retry."""
    print(">>> GUARDRAIL")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage):
        return state

    # --------------------------------------------------------
    # Check 1: Deterministic question check (no LLM)
    # --------------------------------------------------------
    if not last_message.content.strip().endswith("?"):
        logger.warning("Planner output is not a question. Forcing retry.")
        rejection = SystemMessage(content=NOT_A_QUESTION_REJECTION)
        return {"messages": [rejection]}

    current_role = state.get("current_role")
    current_topic = state.get("current_topic")
    current_gap = state.get("current_gap")
    current_objective = state.get("current_objective")
    question_hint = state.get("question_hint")
    planner_source = state.get("planner_source", "model")
    selected_requirement = state.get("selected_requirement_candidate") or {}
    requirement_context = "None"
    validation_context = "None"
    if planner_source == "requirement":
        requirement_context = (
            f"requirement_id={selected_requirement.get('requirement_id')}; "
            f"target_facets={selected_requirement.get('target_facets', [])}"
        )
    if planner_source == "validation":
        validation_issue = state.get("selected_validation_issue") or {}
        validation_context = (
            f"issue={validation_issue.get('message')}; "
            f"conflicting_values={validation_issue.get('fact_values', [])}"
        )

    # Internal roles are prohibited only when the model invents them.  Once a
    # user has explicitly named an administrator (or similar) as a role, a
    # question about that confirmed role is legitimate discovery, not a leak.
    if (
        state.get("discovery_scope") == DiscoveryScope.USER_APP
        and INTERNAL_ROLE_PATTERN.search(last_message.content)
        and not any(
            normalize_role(role) in normalize_role(last_message.content)
            or normalize_role(last_message.content) in normalize_role(role)
            for role in get_known_roles(state, current_topic)
        )
    ):
        logger.warning("Internal role introduced during USER_APP discovery. Forcing retry.")
        return {"messages": [SystemMessage(content=USER_SCOPE_ROLE_REJECTION)]}

    # --------------------------------------------------------
    # Check 2: Deterministic duplicate-question check
    # --------------------------------------------------------
    prior_ai_questions = delivered_prior_questions(messages)
    duplicate = next(
        (question for question in prior_ai_questions
         if questions_are_semantic_duplicates(last_message.content, question)),
        None,
    )
    if duplicate:
        logger.warning("Generated question semantically duplicates a prior question. Forcing retry.")
        rejection = SystemMessage(
            content=f"CRITICAL ERROR: You already asked an equivalent question earlier: '{duplicate}'. "
                    f"The current objective is: {current_objective}. "
                    f"Write a NEW question that asks about that objective specifically, "
                    f"not the one you already asked."
        )
        return {"messages": [rejection]}

    # Do not reject a role-specific question simply because it names another
    # role as the object of work.  For example, an administrator may need to
    # "oversee hosts and guests"; that is still an administrator-responsibility
    # question, not a role leak.  The generator is explicitly role-scoped and
    # the objective evaluator below verifies that the selected gap is asked.

    # --------------------------------------------------------
    # Check 4: Deterministic topic-relevance check (no LLM, no
    # hand-authored keyword table — derived from question_hint/objective)
    # --------------------------------------------------------
    relevance_passed, relevance_reason = check_topic_relevance(
        last_message.content, question_hint, current_objective
    )
    if not relevance_passed:
        logger.warning(f"Topic relevance check failed: {relevance_reason}")
        rejection = SystemMessage(
            content=TOPIC_RELEVANCE_REJECTION.format(
                reason=relevance_reason,
                current_gap=current_gap,
                current_objective=current_objective,
            )
        )
        return {"messages": [rejection]}

    # --------------------------------------------------------
    # Check 5: LLM-based stage/topic evaluator (final safety net)
    # --------------------------------------------------------
    try:
        result = evaluator_llm.invoke(
            EVALUATOR_PROMPT.format(
                current_topic=state["current_topic"].value,
                planner_source=planner_source,
                current_gap=current_gap,
                current_objective=current_objective,
                requirement_context=requirement_context,
                validation_context=validation_context,
                agent_output=last_message.content,
                known_facts="\n".join(
                    f"- {item.key}: {item.value}"
                    for item in state.get("discovered_knowledge", [])
                    if item.topic == current_topic
                ) or "None",
                previous_questions="\n".join(f"- {question}" for question in prior_ai_questions[-5:]) or "None",
            )
        )

        if result.passed:
            return state

        logger.warning(
            f"Validation failed: {result.stage}. "
            f"Reason: {result.explanation}. Forcing retry."
        )

        rejection = SystemMessage(
            content=EVALUATOR_REJECTION.format(
                reason=result.explanation,
                current_gap=current_gap,
                current_objective=current_objective,
            )
        )

        return {"messages": [rejection]}

    except Exception as e:
        raise_if_llm_failure(e)
        logger.error(f"Evaluator failed: {e}")
        raise ExtractionFailed("Question evaluation failed; the question has not been approved") from e



def _record_requirement_question(state: dict, question: str) -> list[dict]:
    """Record delivered inquiry questions.

    The legacy state key is retained for checkpoint compatibility, but entries
    now represent model, requirement, and validation inquiries.
    """
    history = list(state.get("requirement_question_history", []))
    candidate = state.get("selected_inquiry") or state.get("selected_requirement_candidate") or {}
    if not candidate:
        return history
    entry = {
        "candidate_id": candidate.get("id"),
        "inquiry_id": candidate.get("inquiry_id") or candidate.get("id"),
        "source": candidate.get("source") or state.get("planner_source"),
        "requirement_key": candidate.get("requirement_key"),
        "requirement_id": candidate.get("requirement_id"),
        "target_facets": list(candidate.get("target_facets") or []),
        "thread_id": candidate.get("thread_id") or state.get("active_discovery_thread"),
        "decision_key": candidate.get("decision_key"),
        "topic": (
            state.get("current_topic").value
            if getattr(state.get("current_topic"), "value", None)
            else state.get("current_topic")
        ),
        "question": question,
        "turn": state.get("turn_count", 0),
    }
    signature = (
        entry["inquiry_id"],
        entry["question"],
        entry["turn"],
    )
    if not any(
        (item.get("inquiry_id") or item.get("candidate_id"), item.get("question"), item.get("turn"))
        == signature
        for item in history
    ):
        history.append(entry)
    return history


MAX_QUESTION_RETRIES = 2


def guardrail_node(state: dict) -> dict:
    """Bound regeneration per user turn; never return a rejected draft as safe."""
    result = evaluate_question(state)
    messages = result.get("messages", [])
    if not messages or not isinstance(messages[-1], SystemMessage):
        last = state.get("messages", [])[-1] if state.get("messages") else None
        history = (
            _record_requirement_question(state, last.content)
            if isinstance(last, AIMessage) else state.get("requirement_question_history", [])
        )
        return {**result, "question_retry_count": 0, "requirement_question_history": history}
    retries = state.get("question_retry_count", 0)
    if retries >= MAX_QUESTION_RETRIES:
        logger.warning("Question retry limit reached; returning a gap clarification.")
        return {"question_retry_count": 0, "messages": [AIMessage(content=(
            "I'm having trouble resolving this part of your answer. "
            + clarification_question(state)
        ))]}
    return {**result, "question_retry_count": retries + 1}
