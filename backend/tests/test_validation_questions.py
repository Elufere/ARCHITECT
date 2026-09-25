"""Question generation and guardrail behavior for contradiction resolution."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agents import guardrails, question_generator
from agents.consistency_validation import (
    DiscoveryValidationIssue,
    ValidationIssueKind,
    ValidationIssueSeverity,
    ValidationResolution,
)
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


def validation_issue():
    return DiscoveryValidationIssue(
        id="conflict",
        kind=ValidationIssueKind.FACT_CONTRADICTION,
        severity=ValidationIssueSeverity.BLOCKING,
        resolution=ValidationResolution.USER_CLARIFICATION,
        scope=S.USER_APP,
        topic=T.BUSINESS_RULES,
        key="limits",
        fact_ids=["a", "b"],
        fact_values=["Maximum five active requests", "Maximum ten active requests"],
        message="The active-request limit is inconsistent.",
    )


def test_generator_gets_neutral_validation_guidance(monkeypatch):
    prompts = []

    def invoke(messages):
        prompts.extend(messages)
        return AIMessage(
            content="You previously set the limit at five active requests and later at ten; which limit should apply now?"
        )

    monkeypatch.setattr(
        question_generator,
        "get_chat_model",
        lambda **_: SimpleNamespace(invoke=invoke),
    )
    issue = validation_issue()
    state = {
        "planner_source": "validation",
        "selected_validation_issue": issue.model_dump(mode="json"),
        "current_topic": T.BUSINESS_RULES,
        "current_gap": "limits",
        "current_objective": "Resolve the conflicting active-request limit.",
        "question_hint": "Ask which current rule applies.",
        "current_role": None,
        "next_discovery_move": "resolve_contradiction",
        "discovery_scope": S.USER_APP,
        "known_gap_evidence": issue.fact_values,
        "relevant_context": [],
        "messages": [HumanMessage(content="There are limits.")],
        "discovered_knowledge": [],
        "product_model": {},
        "question_retry_count": 0,
    }
    result = question_generator.question_generator_node(state)
    assert result["messages"][0].content.endswith("?")
    system = prompts[0].content
    assert "CONTRADICTION-RESOLUTION MODE" in system
    assert "Maximum five active requests" in system
    assert "Maximum ten active requests" in system
    assert "Do not choose which statement is correct" in system


def test_guardrail_supplies_validation_context(monkeypatch):
    captured = {}

    class Evaluator:
        def invoke(self, prompt):
            captured["prompt"] = prompt if isinstance(prompt, str) else prompt[0].content
            return SimpleNamespace(
                passed=True,
                stage=guardrails.Stage.PRODUCT_DISCOVERY,
                explanation="",
                guidance="",
            )

    monkeypatch.setattr(guardrails, "evaluator_llm", Evaluator())
    issue = validation_issue()
    question = (
        "You previously set the limit at five active requests and later at ten; "
        "which limit should apply now?"
    )
    state = {
        "messages": [HumanMessage(content="We have a limit."), AIMessage(content=question)],
        "planner_source": "validation",
        "selected_validation_issue": issue.model_dump(mode="json"),
        "selected_requirement_candidate": None,
        "current_topic": T.BUSINESS_RULES,
        "current_gap": "limits",
        "current_objective": "Resolve the conflicting active-request limit.",
        "question_hint": "Ask which current rule applies.",
        "current_role": None,
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "requirement_question_history": [],
        "turn_count": 2,
        "question_retry_count": 0,
    }
    result = guardrails.guardrail_node(state)
    assert result["question_retry_count"] == 0
    assert "Planner source:" in captured["prompt"]
    assert "validation" in captured["prompt"]
    assert "conflicting_values=" in captured["prompt"]
