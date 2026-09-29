"""
Production-Ready Stage Classifier & Validator.
Validates AI output against PRODUCT_DISCOVERY stage. Does NOT rewrite.
"""

import json
import logging
import re
from enum import Enum

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from agents.llm_errors import raise_if_llm_failure
from agents.llm_errors import ExtractionFailed
from agents.llm import get_structured_model
from pydantic import BaseModel

from agents.state import DiscoveryScope, KnowledgeState
from agents.role_utils import role_identity, split_role_labels
from agents.conversation_language import clarification_question, final_question_text

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
    """Return only delivered interview questions, excluding advisory prose/drafts."""
    questions = []
    for index, message in enumerate(messages[:-1]):
        if not isinstance(message, AIMessage):
            continue
        # A guardrail rejection follows a draft with a SystemMessage. A human
        # answer means the turn was actually presented to the user.
        if isinstance(messages[index + 1], HumanMessage):
            questions.append(final_question_text(message.content))
    return questions


EVALUATOR_PROMPT = """
You are validating a Product Manager's interview question.

STAGE BOUNDARY:
PRODUCT_DISCOVERY asks what the product must accomplish, what actors/rules/states/
outcomes/constraints matter, and which material decisions belong in a PRD.
SOLUTION_DESIGN includes screen flows, control/button sequence, layout/placement,
visual formatting, component choice, microcopy, clickable-vs-plain presentation,
and similar UX interaction mechanics when they do not materially alter a product
rule. Classify/reject those as solution design rather than product discovery.
A UI-adjacent question may remain PRODUCT_DISCOVERY only when the detail itself
materially changes authorization/security/compliance, money/data movement,
lifecycle/state, a major dependency, or another substantive PRD decision.

EXPECTED ANSWER SPACE:
Do not classify the final question from its surface vocabulary alone. Before
passing it, imagine the natural complete answer a competent founder would give.
If that answer would primarily be a procedural walkthrough, sequence of user
interactions, screens/controls, provider-integration mechanics, internal system
steps, or technical execution, the question is below product-discovery depth even
when it contains no explicit UI/technical words.

A broad "how" question is not automatically invalid, and the word "how" is not a
banned term. It is valid when the natural answer is one atomic product rule,
business decision, state/lifecycle condition, authorization rule, money/data rule,
material dependency, or similarly consequential PRD behavior. It is invalid when
a natural answer is mainly "first do X, then Y, then Z" or explains how the
experience/system executes an already-known behavior.

Judge the materiality of the SPECIFIC ANSWER being requested, not the importance
of the surrounding feature or domain. A business-critical product area can still
contain low-value interaction or implementation details that belong below
discovery depth.

FOUNDER-FACING ABSTRACTION:
A product uncertainty can be legitimate while its wording is still wrong.
The founder should be asked to choose or clarify DESIRED PRODUCT BEHAVIOR, not to
describe internal system representation. Reject wording that asks how/when the
system stores, persists, records, finalizes, derives, or internally manages a
decision when the same uncertainty can be expressed as an observable product rule
such as when something should take effect, whether confirmation is required, who
should be allowed to act, or what should happen next.

This is a semantic rule, not a banned-word list. Terms such as "record" or
"store" are acceptable when they are genuinely the founder's product decision
(for example a requirement to retain a legally required record). Reject them only
when they unnecessarily turn a founder-owned product decision into an explanation
of internal mechanics.

Future tense does not automatically make system-shaped wording acceptable:
"How should the system store this?" can still be the wrong abstraction if the
actual founder decision is "When should this become effective?"

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

Conversation intent:
{conversation_intent}

Persistent discovery boundaries:
{discovery_boundaries}

Raw product idea:
{raw_idea}

Question/response:
{agent_output}

Confirmed knowledge established from the founder's latest answer:
{latest_confirmed_understanding}

The planner has already determined that this is the next missing piece of
information.

Known facts for this topic:
{known_facts}
Known facts may have been learned incidentally. A question asking the user to
confirm their completeness or expand them for the Current gap is valid discovery;
knowledge presence alone does not mean the gap has been deliberately resolved.

Previously delivered interview decisions/questions:
{previous_questions}

Your job has TWO checks:

1. UNDERSTANDING CHECK
The PM response may contain a short acknowledgement before the final question.
That acknowledgement must faithfully reflect confirmed founder knowledge.
It may paraphrase or combine supplied confirmed facts, but it must NOT present
a PM inference, recommendation, implementation choice, UI behavior, workflow,
business rule, or technical mechanism as founder-confirmed knowledge.
If the response mentions an implication that is not directly confirmed, it must
be explicitly tentative (for example "this suggests" or "this may mean").
Do not require the acknowledgement to mention every latest fact; concise is good.
If there was no confirmed knowledge on the latest turn, a response containing
no acknowledgement is valid. If it does acknowledge the founder, it may only
paraphrase explicit founder wording from the recent conversation. A product
category or label does NOT make conventional domain mechanics founder-confirmed;
reject statements that derive money flow, approval behavior, actors, assets,
security semantics, or workflow merely from the category name.
Reject the response if its acknowledgement invents or upgrades unconfirmed
information. Explain exactly which claim is unsupported.

2. QUESTION CHECK
Determine whether the FINAL interview question asks specifically about the
Current objective. Ignore the acknowledgement when judging question count and
question scope; only the final interview question should seek information.

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

When Conversation intent is "advice_request", the PM may first reflect confirmed
understanding and then give a brief set of non-authoritative suggestions before
the final interview question. Those suggestions
must be clearly framed as options, must stay within the active product decision,
and must not be treated as confirmed founder choices. Evaluate the FINAL question
against the Current objective. Reject advice that drifts into implementation,
architecture, or a generic feature wishlist.

Reject if the response:
- phrases the discovery question as though the product behavior is already
  implemented/currently operating when the supplied founder evidence does not
  explicitly establish an existing/live implementation. Product discovery should
  ask about intended behavior ("should", "would", "will", "do you want", "what
  should happen") by default. A founder describing requirements in present tense
  does not by itself prove the feature already exists.
- presents unsupported or inferred information in the acknowledgement as though
  the founder confirmed it
- violates a persistent discovery boundary by retrying, paraphrasing, or deepening
  a line of questioning the founder explicitly delegated, rejected, or explicitly
  closed as outside the product / sufficiently specified
- semantically repeats an already delivered product decision, even when the new
  wording, decision key, thread, or schema anchor differs. A genuine next causal
  decision is allowed; a narrower UX refinement of an already-settled product
  decision is not
- descends below product discovery into interaction design merely because the
  product behavior is already known. Reject questions whose remaining answer is
  principally about button/control sequence, exact screen flow, layout/placement,
  visual styling, copy wording, clickable-vs-plain presentation, modal/toast/
  component choice, or similar UX mechanics UNLESS that detail materially changes
  a product rule, authorization/security/compliance boundary, money/data movement,
  lifecycle/state transition, major dependency, or irreversible outcome
- has an expected answer space that is primarily procedural even though the
  question itself sounds product-level. For example, a broad request for how a
  user performs a multi-step workflow can still be solution design if a natural
  complete answer would be a sequence of interactions rather than one material
  product rule. Evaluate what the question asks the founder to SPECIFY, not just
  whether the wording contains UI/technical terminology
- changes to another topic
- drifts to a different product decision that does not help resolve the Current objective
- asks more than ONE independently answerable product question when Planner source
  is "model". Related does not mean atomic: timing + process, process + conditions,
  permissions + features, or several requested changes are still multiple questions.
  If the founder could answer one part without answering another, reject it.
- closely related target facets of one selected requirement may be combined only
  when Planner source is "requirement" and they form one natural decision
- when Planner source is "model", broadens one frontier decision into an end-to-end
  workflow recap, asks for "main steps from X to Y", "main actions from X to Y",
  or combines multiple actors' journeys/stages instead of resolving the selected
  uncertainty. Reject even if this is grammatically one question: if a natural
  answer requires several sequential actions, it is too broad
- uses technical or implementation-shaped wording when the same product decision
  can be asked in simple founder-facing language
- asks the founder to explain internal representation/mechanics (for example how
  a product decision is stored, recorded, persisted, finalized, derived, or
  internally managed) when the SAME uncertainty can be asked as intended,
  observable product behavior. Preserve the decision; reject only the abstraction
  of the wording
- treats "make the question smaller" as permission to make it more screen-specific.
  Atomicity should stop at one material PRODUCT decision/behavior, not one UI click
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

If the question directly discovers the current objective, it should pass ONLY
when it stays at the objective's intended granularity. A question can mention the
objective and still be invalid if it expands into a multi-stage recap, bundles
several dimensions, asks for a checklist-like answer, or naturally elicits a
procedural UX/implementation walkthrough instead of one PRD-level decision.

Before returning passed=true, perform BOTH semantic tests internally:

ATOMICITY TEST:
Decompose the FINAL QUESTION into independently answerable information units.
If the founder could fully answer one requested unit while leaving another
unanswered, the question is not atomic. Do not merge units merely because they
share one actor, feature, workflow, sentence, or broad label.

EXPECTED-ANSWER TEST:
Imagine the shortest natural answer that would FULLY satisfy the final question.
Would that answer primarily specify one founder-owned PRD rule/decision/behavior,
or would it require a procedural interaction walkthrough, interface mechanics,
technical mechanism, or internal execution detail?

Pass only when the final question asks for one independently answerable
PRD-level uncertainty, unless a lower-level detail itself has a demonstrated
material product consequence.

For model-driven discovery, prefer ONE short founder-facing question. Examples of
invalid shapes include "when and how...", "what changes in A, B, and C?",
"...what happens, and are there any conditions?", and "what are the main actions
A and B perform from X to Y?". A valid workflow question should usually ask about
one actor and one MATERIAL PRODUCT event/decision/state transition at a time.
Do not convert that rule into click-by-click screen discovery.

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

NOT_A_QUESTION_REJECTION = """CRITICAL ERROR: Your response must end with exactly ONE interview question ending in '?'.
A brief grounded acknowledgement may come before it, but do not return acknowledgement/explanation without the final question."""

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
    r"\b(admin(?:istrator)?s?|support (?:agent|agents|staff|team|representative|representatives)|"
    r"customer support (?:agent|agents|staff|team|representative|representatives)|"
    r"moderators?|back[- ]office(?: staff)?|internal staff)\b", re.I
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
        rejection_text = (
            "CRITICAL ERROR: Because the founder asked for suggestions, you may give brief "
            "advice first, but you must end with exactly ONE interview question ending in '?'. "
            "Keep suggestions as options, not confirmed product facts."
            if state.get("conversation_intent") == "advice_request"
            else NOT_A_QUESTION_REJECTION
        )
        rejection = SystemMessage(content=rejection_text)
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

    print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 8: GUARDRAIL INPUT =====")
    print(json.dumps({
        "planner_source": planner_source,
        "current_topic": getattr(current_topic, "value", current_topic),
        "current_gap": current_gap,
        "current_role": current_role,
        "current_objective": current_objective,
        "question_hint": question_hint,
        "question_retry_count": state.get("question_retry_count", 0),
        "generated_response": last_message.content,
        "selected_inquiry": state.get("selected_inquiry") or {},
        "selected_requirement": selected_requirement,
    }, ensure_ascii=False, indent=2, default=str))
    print("===== END STAGE 8 =====\n")

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
        print("===== DISCOVERY ABSTRACTION DEBUG | EARLY GUARDRAIL REJECT =====")
        print(json.dumps({
            "check": "internal_role_scope",
            "result": "REJECT",
            "generated_response": last_message.content,
            "reason": USER_SCOPE_ROLE_REJECTION,
        }, ensure_ascii=False, indent=2, default=str))
        print("===== END EARLY GUARDRAIL REJECT =====\n")
        return {"messages": [SystemMessage(content=USER_SCOPE_ROLE_REJECTION)]}

    # --------------------------------------------------------
    # Check 2: Deterministic duplicate-question check
    # --------------------------------------------------------
    prior_ai_questions = delivered_prior_questions(messages)
    current_question = final_question_text(last_message.content)
    duplicate = next(
        (question for question in prior_ai_questions
         if questions_are_semantic_duplicates(current_question, question)),
        None,
    )
    if duplicate:
        logger.warning("Generated question semantically duplicates a prior question. Forcing retry.")
        print("===== DISCOVERY ABSTRACTION DEBUG | EARLY GUARDRAIL REJECT =====")
        print(json.dumps({
            "check": "semantic_duplicate",
            "result": "REJECT",
            "generated_response": last_message.content,
            "matching_prior_question": duplicate,
        }, ensure_ascii=False, indent=2, default=str))
        print("===== END EARLY GUARDRAIL REJECT =====\n")
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
    print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 9: DETERMINISTIC GUARDRAIL CHECKS =====")
    print(json.dumps({
        "ends_with_question_mark": last_message.content.strip().endswith("?"),
        "internal_role_pattern_found": bool(INTERNAL_ROLE_PATTERN.search(last_message.content)),
        "semantic_duplicate_found": bool(duplicate),
        "duplicate_question": duplicate,
        "topic_relevance_passed": relevance_passed,
        "topic_relevance_reason": relevance_reason,
        "note": "Semantic abstraction/stage validation is performed by the LLM evaluator next.",
    }, ensure_ascii=False, indent=2, default=str))
    print("===== END STAGE 9 =====\n")
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
                conversation_intent=state.get("conversation_intent") or "product_information",
                discovery_boundaries="\n".join(
                    f"- {item.get('instruction')} | source={item.get('evidence')}"
                    for item in state.get("discovery_boundaries", [])[-50:]
                    if not item.get("scope")
                    or item.get("scope") == getattr(state.get("discovery_scope"), "value", state.get("discovery_scope"))
                ) or "None",
                raw_idea=state.get("raw_idea") or "",
                planner_source=planner_source,
                current_gap=current_gap,
                current_objective=current_objective,
                requirement_context=requirement_context,
                validation_context=validation_context,
                agent_output=last_message.content,
                latest_confirmed_understanding="\n".join(
                    f"- {item.topic.value}.{item.key}: {item.value}"
                    for item in state.get("discovered_knowledge", [])
                    if item.scope == state.get("discovery_scope")
                    and item.knowledge_state == KnowledgeState.CONFIRMED
                    and item.source_turn == state.get("turn_count", 0)
                ) or "None",
                known_facts="\n".join(
                    f"- {item.key}: {item.value}"
                    for item in state.get("discovered_knowledge", [])
                    if item.topic == current_topic
                    and item.knowledge_state == KnowledgeState.CONFIRMED
                ) or "None",
                previous_questions="\n".join(
                    f"- {entry.get('objective') or 'decision'} :: {entry.get('question')}"
                    for entry in state.get("requirement_question_history", [])[-12:]
                    if entry.get("question")
                ) or "\n".join(f"- {question}" for question in prior_ai_questions[-5:]) or "None",
            )
        )

        print("===== DISCOVERY ABSTRACTION DEBUG | STAGE 10: LLM GUARDRAIL VERDICT =====")
        print(json.dumps({
            "current_objective": current_objective,
            "generated_response": last_message.content,
            "verdict": result.model_dump(mode="json"),
            "final_guardrail_result": "ALLOW" if result.passed else "REJECT",
        }, ensure_ascii=False, indent=2, default=str))
        print("===== END STAGE 10 =====\n")

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
        "objective": state.get("current_objective"),
        "topic": (
            state.get("current_topic").value
            if getattr(state.get("current_topic"), "value", None)
            else state.get("current_topic")
        ),
        "question": final_question_text(question),
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
        return {
            **result,
            "question_retry_count": 0,
            "question_retry_exhausted": False,
            "requirement_question_history": history,
        }
    retries = state.get("question_retry_count", 0)
    if retries >= MAX_QUESTION_RETRIES:
        logger.warning(
            "Question retry limit reached; abandoning the rejected inquiry and replanning."
        )
        boundaries = list(state.get("discovery_boundaries", []))
        last_draft = next(
            (
                message.content
                for message in reversed(state.get("messages", []))
                if isinstance(message, AIMessage)
            ),
            "",
        )
        candidate = state.get("selected_inquiry") or state.get("selected_requirement_candidate") or {}
        boundaries.append({
            "type": "generation_exhausted",
            "scope": getattr(state.get("discovery_scope"), "value", state.get("discovery_scope")),
            "source_turn": state.get("turn_count", 0),
            "evidence": last_draft,
            "question": last_draft,
            "thread_id": candidate.get("thread_id") or state.get("active_discovery_thread"),
            "decision_key": candidate.get("decision_key"),
            "objective": state.get("current_objective"),
            "instruction": (
                "The selected inquiry could not be phrased safely after bounded guardrail "
                "retries. Do not emit a generic schema fallback or retry the same decision "
                "again on this turn; re-plan to a materially different grounded inquiry."
            ),
        })
        return {
            "question_retry_count": 0,
            "question_retry_exhausted": True,
            "discovery_boundaries": boundaries[-50:],
            "messages": [SystemMessage(content=(
                "QUESTION GENERATION EXHAUSTED: abandon this inquiry and re-plan."
            ))],
        }
    return {**result, "question_retry_count": retries + 1, "question_retry_exhausted": False}
