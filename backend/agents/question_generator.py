# agents/question_generator.py
from langchain_ollama import ChatOllama

from agents.state import AgentState
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


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

    if not current_topic:
        return {"messages": [SystemMessage(content="I need to understand your product better. Could you start by telling me who the primary users will be?")]}

    chat_llm = ChatOllama(
        model="llama3.1",
        temperature=0.0,
    )

    # Get knowledge specific to current topic
    topic_knowledge = [
        f"- {item.key}: {item.value}"
        for item in state.get("discovered_knowledge", [])
        if item.topic == current_topic
    ]

    system_prompt = f"""
You are an experienced Product Manager conducting a structured product discovery interview.

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
- Phrase the question naturally as if speaking to a founder.

Return ONLY the question.
"""
    if not current_objective or not question_hint:
        raise ValueError(
            "Planner did not provide current_objective/question_hint."
        )

    response = chat_llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}