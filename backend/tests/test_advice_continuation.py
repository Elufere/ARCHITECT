"""Collaborative PM advice must not break the active discovery thread."""

from langchain_core.messages import AIMessage, HumanMessage

from agents import guardrails, knowledge_tracker, question_generator
from agents.conversation_language import final_question_text
from agents.conversation_manager import classify_turn, conversation_manager_node
from agents.graph import route_after_conversation_manager
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


def test_advice_request_is_classified_and_still_routes_through_capture():
    text = "Picture, description and price. Do you have more suggestions?"
    state = {"messages": [HumanMessage(content=text)]}

    update = conversation_manager_node(state)

    assert classify_turn(text) == "advice_request"
    assert update["conversation_intent"] == "advice_request"
    assert route_after_conversation_manager(update) == "extract"


def test_final_question_text_ignores_advisory_prefix():
    content = (
        "A few useful options are quantity, fulfillment method, and completion terms. "
        "I would keep warranties out until the product needs them.\n\n"
        "Can the invited customer negotiate those terms before accepting?"
    )

    assert final_question_text(content) == (
        "Can the invited customer negotiate those terms before accepting?"
    )


def test_short_answer_context_uses_only_trailing_question():
    state = {
        "messages": [
            AIMessage(content=(
                "You could include quantity and fulfillment terms as options.\n\n"
                "Can the invited customer negotiate the deal before accepting?"
            )),
            HumanMessage(content="yes"),
        ],
        "current_topic": T.CORE_WORKFLOW,
        "current_gap": "workflow_steps",
        "discovery_scope": S.USER_APP,
    }

    context = knowledge_tracker.answer_context(state)

    assert context["question"] == (
        "Can the invited customer negotiate the deal before accepting?"
    )


def test_question_generator_enters_advice_with_continuation_mode(monkeypatch):
    captured = {}

    class FakeModel:
        def invoke(self, messages):
            captured["system"] = messages[0].content
            return AIMessage(content=(
                "I would consider quantity, fulfillment method, expected completion date, "
                "and what counts as successful completion. I would leave warranties and "
                "returns out until the transaction model actually needs them.\n\n"
                "Can the invited customer negotiate those terms before accepting?"
            ))

    monkeypatch.setattr(
        question_generator,
        "get_chat_model",
        lambda **_: FakeModel(),
    )

    state = {
        "messages": [
            AIMessage(content="What terms are defined when the deal is created?"),
            HumanMessage(content=(
                "Picture, description and price. Do you have more suggestions?"
            )),
        ],
        "conversation_intent": "advice_request",
        "current_topic": T.CORE_WORKFLOW,
        "current_gap": "workflow_steps",
        "current_objective": "Understand how the invited customer responds to the proposed deal terms.",
        "question_hint": "Ask whether the invited customer can negotiate before accepting.",
        "current_role": None,
        "next_discovery_move": "advance_discovery_thread",
        "discovery_scope": S.USER_APP,
        "planner_source": "model",
        "selected_inquiry": {
            "thread_id": "core_transaction",
            "decision_key": "counterparty_term_negotiation",
            "reason": "Deal creation terms are known; the next causal decision is how the invitee responds.",
        },
        "discovered_knowledge": [],
        "product_model": {},
        "relevant_context": [],
        "question_retry_count": 0,
    }

    result = question_generator.question_generator_node(state)

    assert "COLLABORATIVE PM ADVICE MODE" in captured["system"]
    assert "options, not confirmed requirements" in captured["system"]
    assert result["messages"][0].content.endswith("?")


def test_history_records_only_the_interview_question_after_advice():
    response = (
        "I would consider quantity and a completion condition as optional terms.\n\n"
        "Can the invited customer negotiate those terms before accepting?"
    )
    state = {
        "requirement_question_history": [],
        "selected_inquiry": {
            "id": "USER_APP|thread|core_transaction|counterparty_term_negotiation",
            "source": "MODEL",
            "thread_id": "core_transaction",
            "decision_key": "counterparty_term_negotiation",
        },
        "current_topic": T.CORE_WORKFLOW,
        "active_discovery_thread": "core_transaction",
        "turn_count": 4,
    }

    history = guardrails._record_requirement_question(state, response)

    assert history[-1]["question"] == (
        "Can the invited customer negotiate those terms before accepting?"
    )
