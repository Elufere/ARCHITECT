"""Replay the reported escrow answer through live extraction and question routing.

Run explicitly from backend: python tests/replay_escrow_responsibilities.py
The trace captures stage outputs so a failure can become a deterministic test.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import AIMessage, HumanMessage
from agents import knowledge_tracker as tracker
from agents.conversation_manager import conversation_manager_node
from agents.interview_planner import interview_planner_node
from agents.question_generator import question_generator_node
from agents.guardrails import guardrail_node
from agents.state import DiscoveryTopic as T, DiscoveryScope as S, KnowledgeItem

QUESTION = "Can you describe the significant actions or capabilities that a customer, whether they are a buyer or a seller, performs or manages within the app during a transaction?"
ANSWER = (
    "The buyer is responsible for creating or joining the transaction, agreeing to the terms, funding the escrow, reviewing whether the seller has fulfilled the agreement, and confirming completion so the funds can be released. The buyer can also raise a dispute if the seller does not fulfill the agreed terms.\n"
    "The seller is responsible for creating or joining the transaction, agreeing to the terms, fulfilling the agreed goods or services, providing any required proof of fulfillment, and waiting for the buyer's confirmation before receiving the escrowed funds. The seller can also raise or respond to a dispute if there is a problem with the transaction."
)
ACTORS = "the primary users are customers who can either be a buyer or seller based on a particular transaction"
NO_OTHERS = "there are no ther users beside customers"


def initial_state():
    return dict(messages=[HumanMessage(content="I want to create an escrow app"),
        AIMessage(content="Who are the primary users of the escrow app?"), HumanMessage(content=ACTORS),
        AIMessage(content="Besides customers, will anyone else use the user app?"), HumanMessage(content=NO_OTHERS),
        AIMessage(content=QUESTION), HumanMessage(content=ANSWER)],
        current_topic=T.USER_ROLES, current_gap="responsibilities::customer", discovery_scope=S.USER_APP,
        discovered_knowledge=[
            KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key="primary_users", value="customers",
                          roles=["customer"], evidence=ACTORS, confidence=1, source_turn=1),
            KnowledgeItem(topic=T.USER_ROLES, scope=S.USER_APP, key="secondary_users", value="none",
                          absence="none", evidence=NO_OTHERS, confidence=1, source_turn=2)],
        topic_status={T.USER_ROLES: "PARTIAL"}, topic_maturity={}, turn_count=3,
        awaiting_confirmation=False, pm_is_complete=False)


def run():
    trace = {}
    output = Path(__file__).resolve().parents[1] / "escrow_responsibilities_replay.json"
    def encode(value):
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else str(value)
    def save():
        output.write_text(json.dumps(trace, default=encode, indent=2), encoding="utf-8")
    real_models = tracker.extraction_models()
    def invoke(name, messages):
        start = time.monotonic()
        result = real_models[name].invoke(messages)
        trace.setdefault("calls", []).append(dict(name=name, seconds=round(time.monotonic()-start, 2),
            messages=messages, result=result))
        save()
        print(f"REPLAY finished {name}", flush=True)
        return result
    tracker.extraction_models = lambda: {name: SimpleNamespace(invoke=lambda m, n=name: invoke(n, m)) for name in real_models}
    state = initial_state()
    for name, node in [("conversation", conversation_manager_node), ("extract", tracker.knowledge_tracker_node),
                       ("plan", interview_planner_node)]:
        state.update(node(state))
        trace[name] = dict(state)
        save()
    update = question_generator_node(state)
    state["messages"] = state["messages"] + update["messages"]
    trace["generated_question"] = update
    trace["guardrail"] = guardrail_node(state)
    save()
    print("REPLAY NEXT GAP:", state["current_gap"], flush=True)
    print("REPLAY QUESTION:", state["messages"][-1].content, flush=True)
    print("REPLAY TRACE:", output, flush=True)


if __name__ == "__main__":
    run()
