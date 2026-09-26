# agents/question_generator.py
from agents.llm import get_chat_model

from agents.state import AgentState, DiscoveryScope, KnowledgeState
from agents.product_model import format_product_model
from agents.discovery_fields import FIELD_DEFINITIONS
from agents.answer_contract import additional_actors_question
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


SCOPE_RULES = {
    DiscoveryScope.USER_APP: """
CURRENT SCOPE: USER APP (Customer-facing application)
You are ONLY discovering the customer-facing journey.
- Do NOT proactively invent administrators, internal staff, support agents,
  moderators, or dashboards as separate user types.
- If the user explicitly identifies one of those roles, it is a confirmed
  product role: you may ask the planner-selected question about it. Do not use
  an internal role as a speculative example.
- EXCEPTION: if the user's own workflow naturally hands off to someone outside
  this app (e.g. "then it needs to be approved" or "then it gets reviewed"),
  it is fine to ask what marks that handoff point — but do NOT ask how that
  outside step works, who does it, or what happens inside it. That belongs to
  a later phase.
- Follow the selected gap for every explicitly confirmed role in this scope,
  including staff who actually use this app. Do not import roles from another scope.
""",
    DiscoveryScope.ADMIN_DASHBOARD: """
CURRENT SCOPE: ADMIN DASHBOARD
You are ONLY discovering the admin/internal tooling journey.
- Do NOT ask about the customer-facing roles or their workflow — those were
  already discovered in the previous phase (see "Already known from Phase 1" below).
- Focus entirely on admins, internal staff, monitoring, and configuration.
""",
}


def format_recent_messages(messages: list) -> str:
    """Formats LangChain message objects into a readable string for LLM context."""
    formatted_strings = []

    for msg in messages:
        # Skip system messages to save tokens and keep the conversational flow clean
        if isinstance(msg, SystemMessage):
            continue
        elif isinstance(msg, HumanMessage):
            formatted_strings.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_strings.append(f"PM: {msg.content}")

    return "\n".join(formatted_strings)


def latest_confirmed_understanding(state: AgentState, scope: DiscoveryScope) -> list[str]:
    """Return only grounded/canonical knowledge established on the current founder turn.

    This is user-facing acknowledgement context, not a new reasoning layer.
    It deliberately excludes inferred implications and PM recommendations.
    """
    turn = state.get("turn_count", 0)
    items = []

    for fact in state.get("discovered_knowledge", []):
        if (
            fact.scope == scope
            and fact.knowledge_state == KnowledgeState.CONFIRMED
            and fact.source_turn == turn
        ):
            role = f" [{fact.role}]" if fact.role else ""
            items.append(f"{fact.topic.value}.{fact.key}{role}: {fact.value}")

    return list(dict.fromkeys(items))


def _same_role(first: str | None, second: str | None) -> bool:
    """Compare role labels without treating their wording as product knowledge."""
    if not first or not second:
        return False
    return first.lower().strip().rstrip("s") == second.lower().strip().rstrip("s")


def permission_discovery_guidance(state: AgentState, current_role: str | None) -> str:
    """Give the generator evidence and a semantic decision rule for permissions.

    Responsibilities deliberately do not satisfy the permissions schema key.
    They provide context for discovering authorization boundaries, not another
    capability list. Scope and confirmation are required before using this context.
    """
    responsibilities = [
        item.value
        for item in state.get("discovered_knowledge", [])
        if item.topic.value == "USER_ROLES"
        and item.key == "responsibilities"
        and item.scope == state.get("discovery_scope", DiscoveryScope.USER_APP)
        and item.knowledge_state == KnowledgeState.CONFIRMED
        and _same_role(item.role, current_role)
    ]
    evidence = "\n".join(f"- {value}" for value in responsibilities) or "None recorded."
    return f"""
PERMISSION DISCOVERY — EVIDENCE-AWARE MODE
Responsibility evidence already recorded for '{current_role}':
{evidence}

Responsibilities describe significant actions, activities, capabilities, duties, and processes the role performs or manages in the product.
Permissions describe what the system authorizes that role to access or do,
including scope, restrictions, conditions, transaction-state limitations,
approval requirements, and actions reserved for another role.

First, SEMANTICALLY assess the responsibility evidence above. Do not use a
keyword checklist. Decide whether it establishes concrete in-app actions, or
only a high-level purpose/duty.

- If it is high-level or vague, ask about explicit access or authorization
  boundaries, including whether there are any special restrictions. Do not ask
  for ordinary capabilities as a substitute for permissions.
- If it already establishes concrete actions, do NOT ask the user to list
  those actions again. Briefly ground the question in the known actions and
  ask instead about their authorization boundaries: who/what the role can act
  on, when an action is available, restrictions, reversibility, approval, or
  actions the role cannot perform.

The responsibility evidence is context only. It does NOT complete the
permissions gap, and your question must still discover new permission details.
"""


def inference_confirmation_guidance(state: AgentState) -> str:
    """Tell the generator to refine incidental evidence rather than rediscover it."""
    evidence = state.get("known_gap_evidence", []) + state.get("inferred_gap_evidence", [])
    return f"""
CONFIRMATION / REFINEMENT MODE
The following grounded information was provided incidentally while the user
answered a different question:
{chr(10).join(f'- {item}' for item in evidence)}

Do not ask the normal discovery question from scratch. Reference this evidence
in one natural question and ask the user to confirm, correct, qualify, or add
important restrictions/details. Ask whether this is the complete answer for the
active gap or whether anything should be added or corrected. Do not ask only
"Is that correct?". Existing confirmed facts remain confirmed; deliberate coverage
of this gap remains unresolved until the user answers. This applies equally to
facts learned in other topics. Do not imply that incidental knowledge is absent.
"""


def question_generator_node(state: AgentState) -> dict:
    """Generates contextually appropriate questions for the current topic."""
    print(">>> GENERATE")
    current_topic = state.get("current_topic")
    print("current topic:", current_topic)

    current_objective = state.get("current_objective")
    question_hint = state.get("question_hint")
    current_gap = state.get("current_gap")
    current_role = state.get("current_role")
    discovery_move = state.get("next_discovery_move") or "deepen_understanding"
    discovery_scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    planner_source = state.get("planner_source", "model")
    selected_inquiry = state.get("selected_inquiry") or {}
    selected_requirement = state.get("selected_requirement_candidate") or {}

    if not current_topic:
        return {"messages": [SystemMessage(content="I need to understand your product better. Could you start by telling me who the primary users will be?")]}

    followup = state.get("answer_followup")
    if (followup and followup["gap"] == current_gap and followup["scope"] == discovery_scope
            and not state.get("question_retry_count", 0)):
        return {"messages": [AIMessage(content=followup["question"])]}

    # This question needs the complete actor list, not another model decision.
    # Reuse the same wording for clarification so a following "No" still answers
    # whether ANY other users exist, rather than denying one suggested example.
    if (current_gap == "secondary_users"
            and not current_objective
            and discovery_move not in ("confirm_inference", "confirm_existing")
            and not state.get("question_retry_count", 0)):
        return {"messages": [additional_actors_question(state)]}

    chat_llm = get_chat_model(call_name="question_generator")

    # Get knowledge specific to current topic (and role, if applicable)
    topic_knowledge = [
        f"- {item.key}" + (f" [{item.role}]" if item.role else "") + f": {item.value}"
        for item in state.get("discovered_knowledge", [])
        if item.topic == current_topic
        and item.scope == discovery_scope
        and item.knowledge_state == KnowledgeState.CONFIRMED
        and (current_role is None or item.role is None or item.role == current_role)
    ]

    role_constraint = (
        f"\n- This question is ONLY about the '{current_role}' role. "
        f"Do NOT mention, compare, or ask about any other role in this question."
        if current_role else ""
    )

    scope_rules = SCOPE_RULES.get(discovery_scope, "")
    permission_guidance = (
        permission_discovery_guidance(state, current_role)
        if planner_source == "schema" and current_gap and current_gap.startswith("permissions::")
        else ""
    )
    confirmation_guidance = (
        inference_confirmation_guidance(state)
        if planner_source == "schema"
        and discovery_move in ("confirm_inference", "confirm_existing") else ""
    )
    relevant_context = state.get("relevant_context", [])
    latest_understanding = latest_confirmed_understanding(state, discovery_scope)
    understanding_context = (
        "\n".join(f"- {item}" for item in latest_understanding)
        if latest_understanding else "None captured on this turn."
    )
    requirement_guidance = ""
    model_guidance = ""
    validation_guidance = ""
    advice_requested = state.get("conversation_intent") == "advice_request"
    discovery_boundaries = [
        item for item in state.get("discovery_boundaries", [])[-50:]
        if not item.get("scope") or item.get("scope") == discovery_scope.value
    ]
    boundary_guidance = ""
    if discovery_boundaries:
        boundary_guidance = """
PERSISTENT DISCOVERY BOUNDARIES
The founder has given interview-control feedback. These are not product facts:
""" + "\n".join(
            f"- type={item.get('type')} | {item.get('instruction')} | source: {item.get('evidence')}"
            for item in discovery_boundaries
        ) + """
Apply each boundary according to its type.
- For design_deferral/rejected_inquiry: do not retry that rejected detail.
- For question_too_broad: if the latest conversation intent is scope_objection,
  keep the same product thread but ask a much smaller question. On later turns,
  simply avoid repeating the same oversized question; do not let this old
  boundary pin the interview to that thread.
"""
    output_job = (
        "Your job is to briefly reflect what you now understand from the founder's answer, "
        "give brief PM suggestions they explicitly requested, then end with ONE natural "
        "question that resolves or materially reduces the selected uncertainty."
        if advice_requested
        else "Your job is to briefly reflect what you now understand from the founder's "
             "answer, then end with ONE natural question that resolves or materially "
             "reduces the selected uncertainty."
    )
    advice_guidance = """
COLLABORATIVE PM ADVICE MODE
The founder explicitly asked for suggestions in their latest answer.

Before the final interview question, give a SHORT set of practical PM suggestions
that are directly relevant to the decision they were discussing. Treat them as
options, not confirmed requirements. Do not silently add them to the product
model, and do not imply the founder already chose them.

Prefer suggestions that clarify the product decision or prevent obvious ambiguity.
Do not dump a generic feature wishlist. Avoid implementation and architecture.
After the suggestions, continue the SAME active discovery thread with exactly ONE
natural question that resolves the planner-selected objective. End the response
with that question.
""" if advice_requested else ""
    if planner_source == "requirement":
        requirement = state.get("active_requirements", {}).get(
            selected_requirement.get("requirement_key")
        )
        if requirement is not None:
            facet_map = {facet.id: facet for facet in requirement.facets}
            targets = [
                f"- {facet_map[facet_id].label}: {facet_map[facet_id].description}"
                for facet_id in selected_requirement.get("target_facets", [])
                if facet_id in facet_map
            ]
            requirement_guidance = f"""
REQUIREMENT-DRIVEN DISCOVERY
Selected requirement: {requirement.label}
Requirement description: {requirement.description or requirement.label}
Active discovery thread: {selected_requirement.get('thread_id') or state.get('active_discovery_thread') or 'none'}
Unresolved facets this question may cover:
{chr(10).join(targets) if targets else "- Clarify the unresolved requirement."}

The anchor field shown below exists only for extraction normalization. Do NOT
broaden the question to exhaust that field. Ask specifically about the selected
requirement and its unresolved facets. Keep the question connected to the active
discovery thread; do not use this requirement as an excuse to jump into another
part of the product. You may cover closely related facets in one natural question
when that is clearer for the user.
"""

    if planner_source == "model":
        model_guidance = f"""
MODEL-DRIVEN DISCOVERY
Why this inquiry exists: {selected_inquiry.get('reason', 'The current product model has a material uncertainty.')}
Inquiry objective: {current_objective}
Active discovery thread: {selected_inquiry.get('thread_id') or state.get('active_discovery_thread') or 'none'}
Thread decision key: {selected_inquiry.get('decision_key') or 'none'}

Stay inside the active thread. The question should feel like the natural next
decision created by what the founder has already told you, not like another
field from a checklist. When the latest answer introduced an entity, process,
state, relationship, or business-model fork, ask the next causal question needed
to make that part of the product coherent.

Do not repeat an underlying decision in new wording. Do not jump to a globally
important requirement merely because it exists elsewhere in the model.

Ask EXACTLY ONE atomic product question: one independently answerable decision,
not several related questions joined together. Never ask "when and how", "X, Y,
and Z", or "what happens and are there conditions" in one turn. If the planner
objective itself contains several dimensions, choose the ONE dimension with the
highest product impact and ask only that.

For workflows, "one question" is not enough if the expected answer is a long
sequence. Never ask for the main actions/steps/flow from one stage to another
when multiple actions are involved, and never combine multiple actors' workflow
responsibilities in one question. Ask for one small transition or decision, such
as the first meaningful action for one actor, then continue from the answer.

Use founder-friendly product language. Prefer simple phrases such as "what happens
next", "who can do this", "is this the same role", or "what should the user do"
over technical product or engineering terminology.

Do not turn a narrow frontier into an end-to-end workflow recap, a "main steps
from X to Y" request, or a combined journey for multiple actors.

The anchor field below exists only so extraction can normalize the answer.
It is NOT a checklist item and does not need to be exhaustively completed.
Ask only what materially resolves the selected product uncertainty.
"""

    # When in ADMIN_DASHBOARD phase, surface what was learned about the
    # customer-facing roles in Phase 1, so the LLM has something concrete
    # to avoid re-asking about, instead of a hardcoded example.

    if planner_source == "validation":
        issue = state.get("selected_validation_issue") or {}
        validation_guidance = f"""
CONTRADICTION-RESOLUTION MODE
Validation issue: {issue.get('message', 'Confirmed product facts conflict.')}
Conflicting confirmed values:
{chr(10).join(f'- {item}' for item in state.get('known_gap_evidence', [])) or '- See relevant context below.'}

Ask ONE neutral clarification question that briefly states the incompatible
confirmed statements and asks the user to establish the CURRENT rule/decision.
Do not choose which statement is correct. Do not merge incompatible statements.
Do not ask for unrelated discovery. The answer should make it possible to
supersede or qualify the stale fact.
"""
    other_phase_knowledge = ""
    if discovery_scope == DiscoveryScope.ADMIN_DASHBOARD:
        prior_facts = [
            f"- {item.key}: {item.value}"
            for item in state.get("discovered_knowledge", [])
            if item.scope == DiscoveryScope.USER_APP
        ]
        if prior_facts:
            other_phase_knowledge = (
                "\nAlready known from Phase 1 (do not re-ask about these):\n"
                + "\n".join(prior_facts)
            )

    system_prompt = f"""
You are an experienced Product Manager conducting a structured product discovery interview.

{scope_rules}
{other_phase_knowledge}

The Interview Planner has selected the highest-value open product inquiry.
Follow that inquiry and reuse product knowledge learned anywhere in the model.

{output_job}

========================================
WHAT YOU ARE ASKING ABOUT (MEMORIZE THIS)
========================================

Current topic: {current_topic.value}
Extraction anchor: {current_gap}
Current objective: {current_objective}
Question guidance: {question_hint}
Discovery move: {discovery_move}
Planner source: {planner_source}
{model_guidance}
{requirement_guidance}
{validation_guidance}
{advice_guidance}
{boundary_guidance}
Internal field definition (normalization context only; never quote this to the user):
{FIELD_DEFINITIONS.get(current_topic, {}).get((current_gap or '').split('::')[0], '')}

========================================
WHAT YOU MUST DO
========================================

Write a question that resolves: "{current_objective}"

Use this guidance for HOW to phrase it: "{question_hint}"

Your question MUST directly address the selected uncertainty.
Your question MUST ask for ONE substantive answer only.
Your question MUST use vocabulary related to the product decision.
Do not broaden it merely to fill neighboring schema fields.
Use short, simple founder-facing product language. Avoid implementation-shaped
wording and internal PM/engineering terminology. Do not repeat internal terms
such as primary value exchange, scoped product, explicit absence, current gap,
planner objective, authorization model, state transition, or lifecycle.

========================================
WHAT YOU MUST NOT DO
========================================

- Do NOT ask about exceptions, errors, disputes, failures, or edge cases
  when the objective is about goals, motivations, or workflow steps.
- Do NOT ask about the happy path when the objective is about exceptions or edge cases.
- When this is requirement-driven discovery, stay within the selected requirement
  and its target facets even if the answer may map to more than one schema field.
- When this is contradiction-resolution mode, ask only which current rule/decision
  applies; do not continue ordinary schema discovery in the same question.
- Do NOT jump to unrelated product decisions just because they exist elsewhere in the model.
- Do NOT ask for definitions.
- Do NOT ask "what do you mean by..." unless the user explicitly used an ambiguous term.
- Do NOT invent speculative scenarios. When the selected gap is an exception or
  edge case, ask about intended handling of that class without assuming it occurs.
- Do NOT ask implementation questions.
- Do NOT ask architecture questions.
- Do NOT ask roadmap planning questions. MVP_SCOPE questions about launch
  inclusion, optional/deferred features, or exclusions are valid scope discovery.
- Do NOT ask technical design questions.
- Do NOT ask the user whether any part of the workflow is unclear.
- Do NOT ask the user to identify gaps, ambiguities, or areas needing clarification.
- You must identify the gap yourself from the existing knowledge.
- Follow the CURRENT SCOPE rules above strictly.{role_constraint}

========================================
CONTEXT
========================================

Already known about this topic:
{chr(10).join(topic_knowledge) if topic_knowledge else "Nothing yet"}

Confirmed knowledge established from the founder's LATEST answer:
{understanding_context}

Confirmed context selected for this question (do not ask the user to
re-establish any of these facts):
{chr(10).join(f"- {fact}" for fact in relevant_context) if relevant_context else "None"}

Product knowledge learned across all topics:
{format_product_model(state.get("product_model", {}))}

Recent conversation (background only — may be from a previous topic; it does
NOT tell you what to ask next, only the "WHAT YOU ARE ASKING ABOUT" section above does):
{format_recent_messages(state["messages"][-6:])}

========================================
USE EXISTING KNOWLEDGE
========================================

Existing knowledge is authoritative context.

Before the final question, briefly show the founder what you understood from
their latest answer. This acknowledgement is part of the conversation, not a
new product-reasoning step.

UNDERSTANDING RULES:
- Base the acknowledgement primarily on "Confirmed knowledge established from
  the founder's LATEST answer" above.
- You may connect it to older CONFIRMED knowledge only when the connection is
  directly supported by the supplied context.
- Paraphrase naturally; do not merely copy schema keys or dump the raw ledger.
- Do NOT introduce a new workflow, UI behavior, business rule, requirement,
  technical mechanism, or product decision as though the founder confirmed it.
- If you mention a useful implication that is not confirmed, explicitly frame it
  as tentative with language such as "That suggests..." or "That may mean...".
  Never use tentative implications as evidence that a requirement is settled.
- Keep this short: normally 1-3 sentences. Do not produce a mini-PRD or a long
  bullet list after every answer.
- If no confirmed knowledge was captured from the latest answer, do not invent
  an acknowledgement just to satisfy the format; proceed naturally to the question.
- The FINAL line/paragraph must still contain exactly ONE interview question.

Before asking a question, review all known information about the current
topic. Do not ask the user to provide information that is already clearly
established in the known knowledge.
Known facts do not mean this gap was deliberately covered. When the planner
selects confirm_existing, briefly cite the existing facts and ask whether they
are complete or need expansion/correction, even when their content is confirmed.

If some information is already known but important details remain unclear,
ask about the missing or unclear part rather than asking the original broad
question again.

Existing knowledge does NOT create an obligation to exhaust the surrounding
topic or field. Build on what is known and ask only for the information needed
to resolve the selected inquiry.

{permission_guidance}
{confirmation_guidance}

========================================
OUTPUT
========================================

{"Briefly reflect the confirmed understanding first, then give concise advisory options, and end with exactly ONE interview question. Clearly keep suggestions separate from confirmed product facts." if advice_requested else "Briefly reflect the confirmed understanding first, then end with exactly ONE interview question. Do not add any other questions."}
"""
    if not current_objective or not question_hint:
        raise ValueError(
            "Planner did not provide current_objective/question_hint."
        )

    if state.get("question_retry_count", 0) and isinstance(state["messages"][-1], SystemMessage):
        # Some local chat templates ignore system messages after the first one.
        # Put the latest rejection in the primary system prompt as well.
        system_prompt += "\nRevise the rejected draft using this feedback:\n" + state["messages"][-1].content

    response = chat_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    print("response:", response.content)
    return {"messages": [response]}
