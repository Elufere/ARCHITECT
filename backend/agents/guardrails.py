"""
Production-Ready Stage Classifier & Validator.
Validates AI output against PRODUCT_DISCOVERY stage. Does NOT rewrite.
"""

import logging
from enum import Enum
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import re

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

evaluator_llm = ChatOllama(model="qwen2.5:7b", temperature=0).with_structured_output(EvaluationSchema)

EVALUATOR_PROMPT = """
You are validating a Product Manager's interview question.

Current discovery topic:
{current_topic}

Current objective:
{current_objective}

Current gap (the exact field being asked about):
{current_gap}

Question:
{agent_output}

The planner has already determined that this is the next missing piece of
information.

Your job is to determine whether the question is asking specifically about
the Current objective and Current gap above — not merely about the same
topic in general.

Reject if the question:
- changes to another topic
- asks about a different field within the same topic (e.g. asks about
  responsibilities when the gap is permissions, or asks about goals when
  the gap is success_criteria)
- asks multiple objectives
- asks implementation
- asks architecture
- asks roadmap
- asks technical design

Do NOT reject merely because it mentions users, permissions, workflows,
approvals, validation, or business concepts, PROVIDED it is asking about
the Current gap specifically.

If the question directly discovers the current objective, it MUST pass.

Return the schema.
"""

# Used by the role-leak check
ROLE_LEAK_REJECTION = """CRITICAL ERROR: Your previous question mentioned role(s) other than '{current_role}'.

The current objective is scoped to the '{current_role}' role ONLY.

Rewrite the question so it asks about '{current_role}' exclusively, and
does not name, compare to, or reference any other role.
"""

# Used by the LLM evaluator check
EVALUATOR_REJECTION = """CRITICAL ERROR: Your previous question did not correctly discover the current objective.

Reason: {reason}

Rewrite the question so it only asks about the current objective.
"""

# Used by the question-mark check
NOT_A_QUESTION_REJECTION = """CRITICAL ERROR: Your output must be a single interview question ending with a '?'.
Do not make statements or acknowledgments (e.g. "Got it, thanks!"). Ask a question."""


def normalize_role(role: str) -> str:
    """Strips trailing 's' to treat 'customer' and 'customers' as the same role."""
    return role.lower().rstrip('s')

def get_known_roles(state: dict, topic) -> set[str]:
    """Roles derived from primary_users/secondary_users facts for this topic."""
    roles = set()
    for item in state.get("discovered_knowledge", []):
        if item.topic == topic and item.key in ("primary_users", "secondary_users"):
            for r in item.value.split(","):
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


def guardrail_node(state: dict) -> dict:
    """Validates the AI's output. If it fails, appends a rejection to force a retry."""
    print(">>> GUARDRAIL")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage):
        return state

    # --------------------------------------------------------
    # Deterministic question check (no LLM) — Fix 3
    # --------------------------------------------------------
    if not last_message.content.strip().endswith("?"):
        logger.warning("Planner output is not a question. Forcing retry.")
        rejection = SystemMessage(content=NOT_A_QUESTION_REJECTION)
        return {"messages": [rejection]}

    current_role = state.get("current_role")
    current_topic = state.get("current_topic")

        # --------------------------------------------------------
    # Deterministic duplicate-question check
    # --------------------------------------------------------
    prior_ai_questions = [
        m.content.strip().lower()
        for m in messages[:-1]
        if isinstance(m, AIMessage)
    ]
    if last_message.content.strip().lower() in prior_ai_questions:
        logger.warning("Generated question duplicates a prior question. Forcing retry.")
        rejection = SystemMessage(
            content=f"CRITICAL ERROR: You already asked this exact question earlier. "
                    f"The current objective is: {state.get('current_objective')}. "
                    f"Write a NEW question that asks about that objective specifically, "
                    f"not the one you already asked."
        )
        return {"messages": [rejection]}

    # --------------------------------------------------------
    # Deterministic role-scoping check (runs next, no LLM)
    # --------------------------------------------------------
    if current_role and current_topic:
        all_roles = get_known_roles(state, current_topic)
        leaked = question_mentions_other_roles(last_message.content, current_role, all_roles)

        if leaked:
            logger.warning(
                f"Role leak detected: question mentions {leaked} "
                f"while scoped to '{current_role}'. Forcing retry."
            )
            rejection = SystemMessage(
                content=ROLE_LEAK_REJECTION.format(current_role=current_role)
            )
            return {"messages": [rejection]}

    # --------------------------------------------------------
    # Existing LLM-based stage/topic evaluator
    # --------------------------------------------------------
    try:
        result = evaluator_llm.invoke(
            EVALUATOR_PROMPT.format(
                current_topic=state["current_topic"].value,
                current_gap=state["current_gap"],
                current_objective=state["current_objective"],
                agent_output=last_message.content,
            )
        )

        if result.passed:
            return state

        logger.warning(
            f"Validation failed: {result.stage}. "
            f"Reason: {result.explanation}. Forcing retry."
        )

        rejection = SystemMessage(
            content=EVALUATOR_REJECTION.format(reason=result.explanation)
        )

        return {"messages": [rejection]}

    except Exception as e:
        logger.error(f"Evaluator failed: {e}")
        return state