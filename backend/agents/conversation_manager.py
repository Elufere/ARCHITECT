"""Recognize conversational turns that are not new product knowledge."""

import re

from langchain_core.messages import AIMessage, HumanMessage

from agents.state import AgentState
from agents.conversation_language import clarification_reply, clarification_question


PATTERNS = {
    "clarification": re.compile(r"\b(what do you mean|what are you asking|can you explain|clarify|rephrase|i (?:don't|do not) understand)\b", re.I),
    "rationale_request": re.compile(r"\b(why are you asking|why do you need|why does that matter)\b", re.I),
    "summary_request": re.compile(r"\b(summarize|summary|recap|where are we)\b", re.I),
    "advice_request": re.compile(
        r"\b(?:do you have (?:any |more )?suggestions?|any (?:other )?suggestions?|"
        r"what (?:would|do) you suggest|what else (?:should|could) (?:we|i) (?:add|include|consider)|"
        r"can you suggest|what would you recommend|any recommendations?)\b",
        re.I,
    ),
    "correction": re.compile(
        r"\b(actually|correction|i changed my mind|that's not right|instead|"
        r"wrong|not if|you are still not|update (?:number )?\d+|i said)\b", re.I
    ),
    "confirmation": re.compile(
        r"^\s*(?:yes|yeah|yep|correct|that's correct|this is correct|"
        r"yes,? (?:this|that) is correct)\s*[.!]*\s*$", re.I
    ),
    "objection": re.compile(
        r"\b(i told you already|already told you|you asked (?:me )?already|"
        r"stop asking|you keep asking)\b", re.I
    ),
    "uncertainty": re.compile(r"\b(i don't know|not sure|haven't decided|uncertain)\b", re.I),
}


def classify_turn(content: str) -> str:
    for intent, pattern in PATTERNS.items():
        if pattern.search(content.replace("’", "'")):
            return intent
    return "product_information"


def conversation_manager_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {"conversation_intent": None, "is_correction": False}

    intent = classify_turn(messages[-1].content)
    update = {"conversation_intent": intent, "is_correction": intent == "correction",
              "question_retry_count": 0}
    if intent == "clarification":
        return {**update, "messages": [AIMessage(content=clarification_reply(state))]}
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
    # An objection is handled by the knowledge tracker: it searches earlier
    # user answers for evidence for the currently repeated gap.  Do not add an
    # acknowledgement here, because that would make the latest message AI and
    # prevent the repair pass from seeing the user's objection.
    if intent == "objection":
        return update
    return update
