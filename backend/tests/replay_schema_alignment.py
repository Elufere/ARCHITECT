"""Opt-in local Ollama replay: python tests/replay_schema_alignment.py.

Runs the two supplied CareConnect turns from an empty state and records actual
planner selections. It also audits injected unsupported claims, without adding
them to the conversation or persisting them as product knowledge.
"""
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.conversation_manager import conversation_manager_node
from agents.interview_planner import interview_planner_node
from agents.question_generator import question_generator_node
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem

CARE = (
    "I want to build CareConnect, a platform that helps patients find and book "
    "appointments with healthcare professionals such as general doctors, "
    "dermatologists, dentists, physiotherapists, and nutritionists.\n\n"
    "Patients should be able to describe their health concern, search for suitable "
    "healthcare providers, compare their profiles and availability, book appointments, "
    "communicate with the provider, receive appointment reminders, and pay through "
    "the platform.\n\nHealthcare providers should be able to manage their profiles, "
    "specialties, schedules, appointments, and payments."
)
NO_OTHERS = "No. For the user app, the only users are patients and healthcare professionals."


def main():
    folder = Path(__file__).parent / "artifacts" / "schema_alignment"
    folder.mkdir(parents=True, exist_ok=True)
    report, trace, calls, decisions = {}, io.StringIO(), [], []
    actual = tracker.extraction_models()
    def invoke(name, model, messages):
        calls.append(name)
        print(f"Calling {name}", file=sys.stderr, flush=True)
        result = model.invoke(messages)
        parsed = result.get("parsed") if isinstance(result, dict) else result
        decisions.append(dict(name=name, parsed=parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else parsed,
                              input_chars=sum(len(message.content) for message in messages)))
        (folder / "careconnect_calls.json").write_text(json.dumps(decisions, indent=2, default=str), encoding="utf-8")
        return result
    tracker.extraction_models = lambda: {
        name: SimpleNamespace(invoke=lambda messages, name=name, model=model: invoke(name, model, messages))
        for name, model in actual.items()}
    state = dict(messages=[], current_topic=None, current_gap=None,
                 discovery_scope=S.USER_APP, discovered_knowledge=[], topic_status={},
                 topic_maturity={}, turn_count=0)
    try:
        for turn, answer in [(0, CARE), (2, "No.")]:
            state["turn_count"] = turn
            state["messages"].append(HumanMessage(content=answer))
            state.update(conversation_manager_node(state))
            start = len(calls)
            with contextlib.redirect_stdout(trace):
                state.update(tracker.knowledge_tracker_node(state))
                state.update(interview_planner_node(state))
                generated = question_generator_node(state)
            state["messages"].extend(generated["messages"])
            report[f"turn_{turn}"] = dict(
                knowledge=[item.model_dump(mode="json") for item in state["discovered_knowledge"]],
                current_gap=state["current_gap"], missing_keys=state["missing_keys"],
                question=generated["messages"][-1].content,
                extraction_calls=calls[start:], question_calls=0 if state["current_gap"] == "secondary_users" else 1)
            (folder / "careconnect_replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (folder / "careconnect_trace.txt").write_text(trace.getvalue(), encoding="utf-8")
            print(f"Turn {turn}: {state['current_gap']} | {generated['messages'][-1].content}", flush=True)
            if turn == 0:
                state["turn_count"] = 1
                state["messages"].append(HumanMessage(content="what do you mean?"))
                before = list(state["discovered_knowledge"])
                clarified = conversation_manager_node(state)
                state.update({key: value for key, value in clarified.items() if key != "messages"})
                state["messages"].extend(clarified["messages"])
                report["clarification"] = dict(text=clarified["messages"][-1].content,
                                              gap=state["current_gap"], model_calls=0)
                assert before == state["discovered_knowledge"]
                print("Clarification: " + clarified["messages"][-1].content, flush=True)
        initial = report["turn_0"]
        facts = initial["knowledge"]
        assert {r for i in facts if i["key"] == "primary_users" for r in i["roles"]} == {"patient", "healthcare_provider"}
        assert {i["role"] for i in facts if i["key"] == "responsibilities"} == {"patient", "healthcare_provider"}
        assert any(i["key"] == "workflow_steps" for i in facts)
        goals = [i for i in facts if i["key"] == "primary_user_goals"]
        assert len(goals) == 1 and goals[0]["role"] == "patient" and "helps patients" in goals[0]["evidence"]
        assert not any(i["key"] in {"multiple_roles", "role_transitions", "secondary_users", "permissions",
                                    "trigger", "completion_condition", "end_state", "downstream_dependency", "secondary_user_goals"} for i in facts)
        assert not any(i["topic"] in {"BUSINESS_RULES", "CONSTRAINTS", "MVP_SCOPE", "EXCEPTIONS", "EDGE_CASES"} for i in facts)
        assert initial["current_gap"] == "secondary_users"
        assert report["turn_2"]["current_gap"] == "permissions::patient"
        assert any(i.key == "secondary_users" and i.absence == "none" for i in state["discovered_knowledge"])
        assert all("responsibilities::patient" not in turn["missing_keys"] for key, turn in report.items() if key.startswith("turn_"))
        assert all(not (i.source_turn == 2 and i.topic in (T.CORE_WORKFLOW, T.USER_GOALS)) for i in state["discovered_knowledge"])
        for text in (initial["question"], report["clarification"]["text"]):
            assert "patient" in text.lower() and "healthcare" in text.lower()
            assert not any(term in text.lower() for term in ("primary value exchange", "scoped product", "explicit absence", "current gap", "planner objective"))
        injected = [KnowledgeItem(topic=topic, scope=S.USER_APP, key=key, value=value,
                                  evidence=NO_OTHERS, role=role, confidence=1)
                    for topic, key, value, role in [
                        (T.CORE_WORKFLOW, "workflow_steps", "patients and healthcare professionals use the app", None),
                        (T.USER_GOALS, "primary_user_goals", "find and book appointments", "patient")]]
        with contextlib.redirect_stdout(trace):
            grounded = tracker.ground_items(injected, NO_OTHERS, state)
        report["injected_claims_accepted"] = [item.model_dump(mode="json") for item in grounded]
        assert not grounded
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        (folder / "careconnect_replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (folder / "careconnect_trace.txt").write_text(trace.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
