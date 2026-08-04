# agents/question_generator.py
from langchain_ollama import ChatOllama

from agents.state import AgentState, DiscoveryScope
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


SCOPE_RULES = {
    DiscoveryScope.USER_APP: """
CURRENT SCOPE: USER APP (Customer-facing application)
You are ONLY discovering the customer-facing journey.
- Do NOT proactively ask about administrators, internal staff, support agents,
  moderators, or dashboards as separate user types.
- Do NOT name any internal/staff role as an example of "other user types" —
  not even as a suggestion or hypothetical.
- EXCEPTION: if the user's own workflow naturally hands off to someone outside
  this app (e.g. "then it needs to be approved" or "then it gets reviewed"),
  it is fine to ask what marks that handoff point — but do NOT ask how that
  outside step works, who does it, or what happens inside it. That belongs to
  a later phase.
- If the user mentions an internal/staff role, acknowledge it but do NOT ask
  follow-up questions about that role's own responsibilities in this phase.
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


def question_generator_node(state: AgentState) -> dict:
    """Generates contextually appropriate questions for the current topic."""
    print(">>> GENERATE")
    current_topic = state.get("current_topic")
    print("current topic:", current_topic)

    current_objective = state.get("current_objective")
    question_hint = state.get("question_hint")
    current_role = state.get("current_role")
    discovery_scope = state.get("discovery_scope", DiscoveryScope.USER_APP)  # NEW

    if not current_topic:
        return {"messages": [SystemMessage(content="I need to understand your product better. Could you start by telling me who the primary users will be?")]}

    chat_llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.0,
    )

    # Get knowledge specific to current topic (and role, if applicable)
    topic_knowledge = [
        f"- {item.key}" + (f" [{item.role}]" if item.role else "") + f": {item.value}"
        for item in state.get("discovered_knowledge", [])
        if item.topic == current_topic
        and (current_role is None or item.role is None or item.role == current_role)
    ]

    role_constraint = (
        f"\n- This question is ONLY about the '{current_role}' role. "
        f"Do NOT mention, compare, or ask about any other role in this question."
        if current_role else ""
    )

    scope_rules = SCOPE_RULES.get(discovery_scope, "")

    # When in ADMIN_DASHBOARD phase, surface what was learned about the
    # customer-facing roles in Phase 1, so the LLM has something concrete
    # to avoid re-asking about, instead of a hardcoded example.
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

The Interview Planner has already determined exactly what information is missing.

Your ONLY job is to write ONE natural question that discovers that missing information.

Current topic:
{current_topic.value}

Current objective:
{current_objective}

Question guidance:
{question_hint}

Already known:
{chr(10).join(topic_knowledge) if topic_knowledge else "Nothing yet"}

Recent conversation:
{format_recent_messages(state["messages"][-6:])}

Rules:

- Ask EXACTLY ONE question.
- Discover ONLY the current objective.
- Use the Question guidance as the intent of the question.
- Do NOT ask about any other missing fields.
- Do NOT ask about future discovery topics.
- Do NOT ask for definitions.
- Do NOT ask "what do you mean by..." unless the user explicitly used an ambiguous term.
- Do NOT ask hypothetical scenarios.
- Do NOT ask implementation questions.
- Follow the CURRENT SCOPE rules above strictly — they override any instinct to
  mention roles outside the current scope, even as examples.
- Phrase the question naturally as if speaking to a founder.{role_constraint}

Return ONLY the question.
"""
    if not current_objective or not question_hint:
        raise ValueError(
            "Planner did not provide current_objective/question_hint."
        )

    response = chat_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    print("response:", response.content)
    return {"messages": [response]}