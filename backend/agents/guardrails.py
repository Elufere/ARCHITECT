"""
Production-Ready Stage Classifier & Validator.
Validates AI output against PRODUCT_DISCOVERY stage. Does NOT rewrite.
"""

import logging
from enum import Enum
from unittest import result
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

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

evaluator_llm = ChatOllama(model="llama3.1", temperature=0).with_structured_output(EvaluationSchema)

EVALUATOR_PROMPT = """
You are validating a Product Manager's interview question.

Current discovery topic:
{current_topic}

Current objective:
{current_objective}

Current gap:
{current_gap}

Question:
{agent_output}

The planner has already determined that this is the next missing piece of
information.

Your ONLY job is to determine whether the question is asking about that
objective.

Reject ONLY if the question:

- changes to another topic
- asks multiple objectives
- asks implementation
- asks architecture
- asks roadmap
- asks technical design

Do NOT reject merely because it mentions users, permissions, workflows,
approvals, validation, or business concepts.

If the question directly discovers the current objective,
it MUST pass.

Return the schema.
"""

REJECTION_MESSAGE = """
CRITICAL ERROR: Your previous response did not remain within the PRODUCT_DISCOVERY stage.

Reason:
{reason}

Review your previous response before rewriting.

For every rejected question:

1. Determine why the question does NOT reduce uncertainty about the product requirements.
2. Identify what product information is still missing.
3. Replace ONLY the invalid question(s) with better product discovery questions.
4. Keep any valid questions unchanged.

A valid product discovery question should:
- Discover user goals, workflows, business rules, constraints, roles, or MVP scope.
- Reduce uncertainty about WHAT the product should do.
- Stay within the current focus area.
- Avoid repeating information already gathered.

Do NOT ask questions whose primary purpose is to determine:
- How the product will be implemented.
- Technical architecture or engineering decisions.
- Security implementation.
- Compliance or legal strategy unless explicitly required by the product.
- Future roadmap or post-MVP planning.

Your goal is not simply to ask different questions.
Your goal is to ask the next most valuable unanswered product discovery questions.
"""

def guardrail_node(state: dict) -> dict:
    """Validates the AI's output. If it fails, appends a rejection to force a retry."""
    print(">>> GUARDRAIL")
    messages = state["messages"]
    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage):
        return state
        
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
            content=REJECTION_MESSAGE.format(
                reason=result.explanation
            )
        )

        return {
            "messages": [rejection]
        }
        
    except Exception as e:
        logger.error(f"Evaluator failed: {e}")
        return state