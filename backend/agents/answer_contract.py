"""Interpret literal choices only for questions whose meaning the app controls.

Free-form model questions/answers still require semantic extraction. A question
contract is attached by application code, never inferred from a bare yes/no.
"""
from langchain_core.messages import AIMessage, HumanMessage

from agents.conversation_language import secondary_users_question
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeItem


def additional_actors_question(state):
    return AIMessage(content=secondary_users_question(state), additional_kwargs={
        "answer_contract": {"kind": "additional_actors", "gap": "secondary_users",
                            "scope": state.get("discovery_scope", DiscoveryScope.USER_APP)}})


def interpret_closed_answer(state):
    """Return (facts, follow-up) or None when this is not a controlled choice."""
    messages = state.get("messages", [])
    if len(messages) < 2 or not isinstance(messages[-1], HumanMessage) or not isinstance(messages[-2], AIMessage):
        return None
    question, answer = messages[-2:]
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    contract = question.additional_kwargs.get("answer_contract", {})
    if (contract != {"kind": "additional_actors", "gap": "secondary_users", "scope": scope}
            or state.get("current_topic") != DiscoveryTopic.USER_ROLES
            or state.get("current_gap") != "secondary_users"
            or question.content != secondary_users_question(state)):
        return None
    choice = answer.content.strip().lower().rstrip(".! ")
    if choice in {"no", "nope", "none"}:
        return [KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=scope,
            key="secondary_users", value="none", evidence=answer.content, roles=[],
            confidence=1, absence="none", source_turn=state.get("turn_count", 0))], None
    if choice in {"yes", "yeah", "yep"}:
        app = "dashboard" if scope == DiscoveryScope.ADMIN_DASHBOARD else "user app"
        return [], f"Who else will use the {app}, and what will they do?"
    return None
