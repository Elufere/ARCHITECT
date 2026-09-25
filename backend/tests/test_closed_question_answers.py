"""Full graph progression for generated questions and their literal answers."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import graph, guardrails, knowledge_tracker as tracker, question_generator
from agents.answer_contract import interpret_closed_answer
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem
from coverage_test_utils import coverage_for_facts


def state(answer="no"):
    actor = KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key="primary_users",
                          value="customers", roles=["customer"], evidence="customers", confidence=1)
    initial = dict(current_topic=T.USER_ROLES, current_gap="secondary_users",
        discovery_scope=S.USER_APP, discovered_knowledge=[actor], topic_status={},
        topic_maturity={}, turn_count=2, awaiting_confirmation=False, pm_is_complete=False)
    initial["gap_coverage"] = coverage_for_facts(initial, T.USER_ROLES)
    question = question_generator.question_generator_node(initial)["messages"][0]
    initial["asked_gap"] = dict(
        scope=S.USER_APP.value,
        topic=T.USER_ROLES.value,
        gap="secondary_users",
        question=question.content,
    )
    initial["messages"] = [question, HumanMessage(content=answer)]
    return initial


def no_extraction_calls(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Literal controlled answers must not depend on LLM extraction or grounding")
    monkeypatch.setattr(tracker, "extraction_models", fail)
    monkeypatch.setattr(guardrails, "evaluator_llm", SimpleNamespace(invoke=lambda _: SimpleNamespace(passed=True)))


def test_no_is_stored_and_planner_moves_to_responsibilities(monkeypatch):
    no_extraction_calls(monkeypatch)
    monkeypatch.setattr(question_generator, "get_chat_model", lambda **_: SimpleNamespace(
        invoke=lambda _: AIMessage(content="What should a customer be able to do in the app?")))
    result = graph.build_graph().invoke(state(), config={"recursion_limit": 30})
    assert result["current_gap"] == "responsibilities::customer"
    assert result["question_retry_count"] == 0
    assert result["answer_followup"] is None
    assert result["discovered_knowledge"][-1].absence == "none"
    assert result["discovered_knowledge"][-1].key == "secondary_users"
    assert result["discovered_knowledge"][-1].evidence == "no"
    assert not any("trouble" in str(m.content) for m in result["messages"])


def test_yes_is_processed_and_asks_who_without_reasking_existence(monkeypatch):
    no_extraction_calls(monkeypatch)
    result = graph.build_graph().invoke(state("yes"), config={"recursion_limit": 12})
    assert result["current_gap"] == "secondary_users"
    assert len(result["discovered_knowledge"]) == 1  # Never invent an unnamed actor.
    assert result["messages"][-1].content == "Who else will use the user app, and what will they do?"
    assert result["question_retry_count"] == 0


def test_actor_names_after_yes_are_extracted_and_clear_followup(monkeypatch):
    from test_gap_absence import models
    no_extraction_calls(monkeypatch)
    workflow = graph.build_graph()
    result = workflow.invoke(state("yes"))
    answer = "Support staff use the app to resolve disputes."
    models(monkeypatch, resolution="unresolved", evidence=answer, outputs={"ACTOR": [dict(
        key="secondary_users", value="support staff", roles=["support_staff"],
        evidence=answer, confidence=1)]})
    monkeypatch.setattr(question_generator, "get_chat_model", lambda **_: SimpleNamespace(
        invoke=lambda _: AIMessage(content="What should a customer be able to do in the app?")))
    result["messages"].append(HumanMessage(content=answer))
    result["turn_count"] += 1
    result = workflow.invoke(result)
    assert result["current_gap"] == "responsibilities::customer"
    assert result["answer_followup"] is None
    assert any(i.key == "secondary_users" and i.roles == ["support_staff"]
               for i in result["discovered_knowledge"])


@pytest.mark.parametrize("answer", ["No admins, but support staff.", "No, maybe later.",
    "I don't know", "yes, moderators", "", "no?"])
def test_free_form_or_ambiguous_answers_require_semantic_extraction(answer):
    assert interpret_closed_answer(state(answer)) is None


@pytest.mark.parametrize("change", ["missing_contract", "changed_question", "changed_scope", "changed_gap"])
def test_no_cannot_be_assigned_to_an_unrelated_question(change):
    initial = state()
    if change == "missing_contract":
        initial["messages"][0] = AIMessage(content=initial["messages"][0].content)
    elif change == "changed_question":
        initial["messages"][0].content = "Do you mean there are no other users?"
    elif change == "changed_scope":
        initial["discovery_scope"] = S.ADMIN_DASHBOARD
    else:
        initial["current_gap"] = "multiple_roles"
    assert interpret_closed_answer(initial) is None


def test_yes_to_free_form_discovery_question_reaches_extraction(monkeypatch):
    initial = state("yes")
    initial["messages"][0] = AIMessage(content="Can a customer also be a provider?")
    initial["current_gap"] = "multiple_roles"
    calls = []
    def extract(s):
        calls.append(s["messages"][-1].content)
        return {}
    monkeypatch.setattr(graph, "knowledge_tracker_node", extract)
    monkeypatch.setattr(graph, "interview_planner_node", lambda _: {})
    monkeypatch.setattr(graph, "question_generator_node", lambda _: {"messages": [AIMessage(content="Next question?")]})
    monkeypatch.setattr(graph, "guardrail_node", lambda _: {})
    graph.build_graph().invoke(initial)
    assert calls == ["yes"]
