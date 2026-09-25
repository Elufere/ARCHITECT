"""True multi-turn graph integration test for the discovery interview loop."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agents import graph, guardrails, question_generator
from agents.state import DiscoveryTopic as T
from discovery_invariant_utils import assert_discovery_invariants
from test_closed_question_answers import state as closed_answer_state, no_extraction_calls
from test_gap_absence import models


def test_two_user_turns_flow_through_full_graph(monkeypatch):
    """A closed actor answer followed by a substantive responsibility answer.

    This keeps extraction, activation, coverage, dependency resolution,
    consistency validation, candidate construction/ranking, planner routing,
    question generation and guardrails in the compiled graph.
    """
    no_extraction_calls(monkeypatch)
    monkeypatch.setattr(
        question_generator,
        "get_chat_model",
        lambda **_: SimpleNamespace(
            invoke=lambda _: AIMessage(content="What should a customer be able to do in the app?")
        ),
    )

    workflow = graph.build_graph()
    first = workflow.invoke(
        closed_answer_state("no"),
        config={"recursion_limit": 30},
    )

    assert first["current_topic"] == T.USER_ROLES
    assert first["current_gap"] == "responsibilities::customer"
    assert first["asked_gap"]["gap"] == "responsibilities::customer"
    assert first["discovered_knowledge"][-1].key == "secondary_users"
    assert first["discovered_knowledge"][-1].absence == "none"
    assert_discovery_invariants(first)

    answer = "Customers create tasks, edit their task details, mark tasks complete, and reopen incomplete work."
    models(
        monkeypatch,
        resolution="unresolved",
        evidence=answer,
        outputs={
            "RESPONSIBILITY": [{
                "key": "responsibilities",
                "value": "create, edit, complete, and reopen tasks",
                "role": "customer",
                "evidence": answer,
                "confidence": 1,
            }],
        },
    )
    monkeypatch.setattr(
        question_generator,
        "get_chat_model",
        lambda **_: SimpleNamespace(
            invoke=lambda _: AIMessage(content="What actions must a customer not be allowed to perform?")
        ),
    )
    monkeypatch.setattr(
        guardrails,
        "evaluator_llm",
        SimpleNamespace(invoke=lambda _: SimpleNamespace(passed=True)),
    )

    first["messages"].append(HumanMessage(content=answer))
    first["turn_count"] += 1
    second = workflow.invoke(first, config={"recursion_limit": 30})

    assert second["current_topic"] == T.USER_ROLES
    assert second["current_gap"] == "permissions::customer"
    assert second["asked_gap"]["gap"] == "permissions::customer"
    assert any(
        item.key == "responsibilities"
        and item.role == "customer"
        and "edit" in item.value
        for item in second["discovered_knowledge"]
    )

    # The edit capability also exercises Fix 9 inside the real graph.
    lifecycle_ids = {requirement.id for requirement in second["active_requirements"].values()}
    assert "lifecycle.modification_behavior" in lifecycle_ids
    assert_discovery_invariants(second)
