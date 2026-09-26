"""Recognize conversational turns that are not new product knowledge."""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from agents.conversation_language import clarification_reply, clarification_question, final_question_text
from agents.llm import get_chat_model


PATTERNS = {
    "clarification": re.compile(
        r"\b(what do you mean|what are you asking|can you explain|clarify|rephrase|"
        r"i (?:don'?t|do not) (?:understand|get it)|too technical|simpler terms?|explain (?:it )?simply|"
        r"explain in simpler terms?)\b",
        re.I,
    ),
    "rationale_request": re.compile(r"\b(why are you asking|why do you need|why does that matter)\b", re.I),
    "summary_request": re.compile(r"\b(summarize|summary|recap|where are we)\b", re.I),
    "correction": re.compile(
        r"\b(actually|correction|i changed my mind|that's not right|instead|"
        r"wrong|not if|you are still not|update (?:number )?\d+|i said)\b", re.I
    ),
    "advice_request": re.compile(
        r"\b(?:do you have (?:any |more )?suggestions?|any (?:other )?suggestions?|"
        r"what (?:would|do) you suggest|what else (?:should|could) (?:we|i) (?:add|include|consider)|"
        r"can you suggest|what would you recommend|any recommendations?)\b",
        re.I,
    ),
    "confirmation": re.compile(
        r"^\s*(?:yes|yeah|yep|correct|that's correct|this is correct|"
        r"yes,? (?:this|that) is correct|that(?:'s| is) all|that will be all|"
        r"nothing else|no more)\s*[.!]*\s*$",
        re.I,
    ),
    "design_deferral": re.compile(
        r"\b(?:am i (?:the )?(?:product |ui/?ux )?designer|"
        r"(?:that|this|it)(?:'s| is) (?:the )?designer'?s? job|"
        r"(?:that|this|it) is for (?:the )?(?:product |ui/?ux )?designer|"
        r"leave (?:that|this|it) (?:to|for) (?:the )?(?:product |ui/?ux )?designer|"
        r"designer should decide|designer can decide)\b",
        re.I,
    ),
    "objection": re.compile(
        r"\b(i told you already|already told you|you asked (?:me )?already|"
        r"stop asking|you keep asking|you are asking irrelevant questions|"
        r"you're asking irrelevant questions|irrelevant questions?|"
        r"that(?:'s| is) irrelevant|this is irrelevant|not relevant)\b", re.I
    ),
    "uncertainty": re.compile(r"\b(i don't know|not sure|haven't decided|uncertain)\b", re.I),
}


def classify_turn(content: str) -> str:
    for intent, pattern in PATTERNS.items():
        if pattern.search(content.replace("’", "'")):
            return intent
    return "product_information"


def semantic_clarification_reply(state: AgentState, founder_message: str) -> str:
    """Clarify the existing decision without replanning or creating product facts."""
    previous = next(
        (
            message.content
            for message in reversed(state.get("messages", [])[:-1])
            if isinstance(message, AIMessage)
        ),
        "",
    )
    previous_question = final_question_text(previous)
    objective = state.get("current_objective") or ""
    guidance = state.get("question_hint") or ""
    context = "\n".join(
        f"- {item}"
        for item in state.get("relevant_context", [])[:8]
    ) or "None"

    try:
        response = get_chat_model(
            call_name="conversation_manager.clarify",
            max_tokens=140,
        ).invoke([
            SystemMessage(content="""You are a product manager clarifying the exact
question you just asked because the founder said they did not understand it.

Do not choose a new discovery topic or a new product decision.
Do not extract or invent product facts.
Do not deepen the question.
Answer the founder's clarification directly, then rephrase the SAME intended
decision in simpler language. If the founder asks whether you meant a particular
stage or interpretation, explicitly say whether that matches the supplied
previous question/objective. Keep the reply concise. End with at most ONE
question, and that question must still ask only the same decision."""),
            HumanMessage(content=(
                f"Previous PM question: {previous_question}\n"
                f"Selected objective: {objective}\n"
                f"Question guidance: {guidance}\n"
                f"Relevant confirmed context:\n{context}\n"
                f"Founder's clarification request: {founder_message}"
            )),
        ])
        text = response.content.strip()
        if text:
            return text
    except Exception as exc:
        print(f"SEMANTIC CLARIFICATION FALLBACK: {exc}")

    return clarification_reply(state)


def conversation_manager_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {"conversation_intent": None, "is_correction": False}

    intent = classify_turn(messages[-1].content)
    update = {"conversation_intent": intent, "is_correction": intent == "correction",
              "question_retry_count": 0}
    if intent == "clarification":
        return {
            **update,
            "messages": [
                AIMessage(
                    content=semantic_clarification_reply(
                        state,
                        messages[-1].content,
                    )
                )
            ],
        }
    if intent == "rationale_request":
        return {**update, "messages": [AIMessage(content=(
            "It helps us agree how this part of the app should work. " + clarification_question(state)
        ))]}
    if intent == "summary_request":
        facts = [fact for values in state.get("product_model", {}).values() for fact in values]
        summary = "\n".join(f"- {fact}" for fact in facts) or "- No confirmed details yet."
        return {**update, "messages": [AIMessage(content=f"Here is what I understand so far:\n{summary}")]}
    if intent == "uncertainty":
        return {**update, "messages": [AIMessage(content=(
            "That is fine—I’ll keep it as an open decision and continue with the parts that are known."
        ))]}

    if intent in ("design_deferral", "objection"):
        boundaries = list(state.get("discovery_boundaries", []))
        previous_question = next(
            (
                message.content
                for message in reversed(messages[:-1])
                if isinstance(message, AIMessage)
            ),
            "",
        )
        boundary = {
            "type": "design_deferral" if intent == "design_deferral" else "rejected_inquiry",
            "scope": getattr(state.get("discovery_scope"), "value", state.get("discovery_scope")),
            "source_turn": state.get("turn_count", 0),
            "evidence": messages[-1].content,
            "question": previous_question,
            "thread_id": state.get("active_discovery_thread"),
            "decision_key": (state.get("selected_inquiry") or {}).get("decision_key"),
            "objective": state.get("current_objective"),
        }
        if intent == "design_deferral":
            boundary["instruction"] = (
                "Founder delegates UI/interface/navigation/design implementation details "
                "to the designer. Do not ask the founder to specify those details unless "
                "a concrete product decision cannot be made without them."
            )
        else:
            boundary["instruction"] = (
                "Founder rejected the immediately preceding inquiry as irrelevant or repeated. "
                "Do not retry, paraphrase, or deepen that inquiry; choose a materially different "
                "product decision."
            )
        boundaries.append(boundary)
        return {**update, "discovery_boundaries": boundaries[-50:]}

    return update
