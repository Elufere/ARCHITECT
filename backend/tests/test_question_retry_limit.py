"""Exercise the actual graph cycle with a generator that never improves."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents import graph, guardrails, question_generator
from agents.conversation_manager import conversation_manager_node
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


QUESTION = "Will anyone else need to use the user app?"


def state():
    return dict(messages=[AIMessage(content=QUESTION), HumanMessage(content="no")],
                current_topic=T.USER_ROLES, current_gap="secondary_users",
                current_objective="Identify any additional users.", question_hint="Ask about other users.",
                discovery_scope=S.USER_APP, discovered_knowledge=[], topic_status={},
                awaiting_confirmation=False, pm_is_complete=False)


def test_repeated_template_exits_graph_with_bounded_retries(monkeypatch):
    calls = []
    def invoke(messages):
        calls.append(messages)
        return AIMessage(content=QUESTION)
    monkeypatch.setattr(question_generator, "get_chat_model", lambda **_: SimpleNamespace(invoke=invoke))
    # Isolate the generation cycle, leaving graph routing, generator, and
    # deterministic duplicate rejection real. No extraction or model server.
    monkeypatch.setattr(graph, "knowledge_tracker_node", lambda _: {})
    monkeypatch.setattr(graph, "interview_planner_node", lambda _: {})
    initial = state()
    result = graph.build_graph().invoke(initial, config={"recursion_limit": 30})
    # MAX_QUESTION_RETRIES counts regeneration attempts after the initial
    # generated draft, so a model-driven question can invoke the generator once
    # initially plus the bounded retry budget.
    assert len(calls) == 1 + guardrails.MAX_QUESTION_RETRIES
    assert any(
        "already asked" in message.content.lower()
        for message in calls[1]
        if isinstance(message, SystemMessage)
    )
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content.startswith("I'm having trouble")
    assert result["messages"][-1].content.endswith(QUESTION)
    assert result["question_retry_count"] == 0
    assert result["current_gap"] == "secondary_users"
    assert result["discovered_knowledge"] == []
    assert not result["pm_is_complete"]


def test_semantic_rejections_are_also_bounded(monkeypatch):
    monkeypatch.setattr(guardrails, "evaluator_llm", SimpleNamespace(invoke=lambda _: SimpleNamespace(
        passed=False, stage="OTHER", explanation="Wrong field", guidance="Stay on topic")))
    initial = state()
    initial["messages"] = [HumanMessage(content="no"), AIMessage(content="What color should the logo be?")]
    for count in range(guardrails.MAX_QUESTION_RETRIES):
        initial["question_retry_count"] = count
        result = guardrails.guardrail_node(initial)
        assert isinstance(result["messages"][-1], SystemMessage)
        assert result["question_retry_count"] == count + 1
    initial["question_retry_count"] = guardrails.MAX_QUESTION_RETRIES
    assert isinstance(guardrails.guardrail_node(initial)["messages"][-1], AIMessage)


def test_new_user_turn_resets_retry_budget():
    initial = state()
    initial["question_retry_count"] = 2
    assert conversation_manager_node(initial)["question_retry_count"] == 0


def test_successful_question_resets_retry_budget(monkeypatch):
    monkeypatch.setattr(guardrails, "evaluator_llm", SimpleNamespace(invoke=lambda _: SimpleNamespace(passed=True)))
    initial = state()
    initial["messages"] = [HumanMessage(content="no"), AIMessage(content="Are there additional users we should include?")]
    initial["question_retry_count"] = 1
    result = guardrails.guardrail_node(initial)
    assert result["question_retry_count"] == 0
    assert result["messages"][-1].content == initial["messages"][-1].content
