from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agents import guardrails
from agents.conversation_language import final_question_text, message_text
from agents.state import DiscoveryScope, DiscoveryTopic


def test_message_text_normalizes_gpt56_content_blocks():
    content = [
        {"type": "text", "text": "What is the main outcome "},
        {"type": "text", "text": "you want this app to give the user?"},
    ]

    assert message_text(content) == (
        "What is the main outcome you want this app to give the user?"
    )
    assert final_question_text(content).endswith("?")


def test_gpt56_content_blocks_are_normalized_before_question_validation(monkeypatch):
    monkeypatch.setattr(
        guardrails,
        "evaluator_llm",
        SimpleNamespace(
            invoke=lambda _: SimpleNamespace(
                passed=True,
                stage="PRODUCT_DISCOVERY",
                explanation="",
                guidance="",
                offending_questions=[],
                model_dump=lambda mode=None: {
                    "passed": True,
                    "stage": "PRODUCT_DISCOVERY",
                    "explanation": "",
                    "guidance": "",
                    "offending_questions": [],
                },
            )
        ),
    )

    state = {
        "messages": [
            AIMessage(content=[
                {
                    "type": "text",
                    "text": "What is the main outcome you want this app to give the user?",
                }
            ])
        ],
        "discovery_scope": DiscoveryScope.USER_APP,
        "current_topic": DiscoveryTopic.USER_GOALS,
        "current_gap": "primary_user_goals::user",
        "current_role": "user",
        "current_objective": "Clarify the user's primary outcome.",
        "question_hint": "Ask what result the user wants from the app.",
        "planner_source": "model",
        "selected_inquiry": {
            "id": "USER_APP|thread|purpose|user_outcome",
            "inquiry_id": "USER_APP|thread|purpose|user_outcome",
            "source": "MODEL",
        },
        "selected_requirement_candidate": None,
        "conversation_intent": "product_information",
        "raw_idea": "A simple todo app.",
        "discovered_knowledge": [],
        "discovery_boundaries": [],
        "founder_gap_guidance": [],
        "requirement_question_history": [],
        "turn_count": 0,
    }

    result = guardrails.evaluate_question(state)

    assert result is state
