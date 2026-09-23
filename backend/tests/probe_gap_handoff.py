"""Opt-in live check of the failing short-answer grounding handoff.

Uses the actual persisted initial facts and actual failed No-turn candidates.
This focused diagnostic does not substitute for a clean conversation replay.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from agents import knowledge_tracker as tracker
from agents.extraction_passes import PASSES, normalize_fact
from agents.interview_planner import build_gap_info
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem
from replay_schema_alignment import CARE


def main():
    folder = Path(__file__).parent / "artifacts" / "schema_alignment"
    replay = json.loads((folder / "careconnect_before_handoff_replay.json").read_text())
    recorded = json.loads((folder / "careconnect_before_handoff_calls.json").read_text())
    state = dict(messages=[HumanMessage(content=CARE), AIMessage(content=replay["turn_0"]["question"]),
                           HumanMessage(content="what do you mean?"), AIMessage(content=replay["clarification"]["text"]),
                           HumanMessage(content="No.")], current_topic=T.USER_ROLES, current_gap="secondary_users",
                 discovery_scope=S.USER_APP, turn_count=2,
                 discovered_knowledge=[KnowledgeItem.model_validate(item) for item in replay["turn_0"]["knowledge"]])
    contracts = {name: (schema, topic) for name, schema, topic, _ in PASSES}
    latest = {call["name"]: call["parsed"] for call in recorded if call["name"] in contracts}
    candidates = []
    for name, output in latest.items():
        schema, topic = contracts[name]
        for data in output["items"]:
            try:
                item = normalize_fact(schema.model_validate(data), topic, S.USER_APP, 2)
                if tracker.validate_extraction(item, "No.")[0]:
                    candidates.append(item)
            except ValueError:
                pass
    actual, calls = tracker.extraction_models(), []
    def invoke(name, messages):
        print("Calling " + name, flush=True)
        result = actual[name].invoke(messages)
        parsed = result["parsed"]
        calls.append(dict(name=name, parsed=parsed.model_dump(mode="json") if parsed else None))
        (folder / "gap_handoff_probe.json").write_text(json.dumps(calls, indent=2))
        return result
    tracker.extraction_models = lambda: {name: SimpleNamespace(invoke=lambda m, name=name: invoke(name, m)) for name in actual}
    reviewed = tracker.extract_gap_absence("No.", state, S.USER_APP)
    assert reviewed is not None
    accepted = tracker.ground_items([*candidates, reviewed], "No.", state, reviewed)
    assert accepted and all(item.key == "secondary_users" and item.absence == "none" for item in accepted)
    state["discovered_knowledge"].extend(accepted)
    assert "secondary_users" not in build_gap_info(state, T.USER_ROLES)["missing_keys"]
    print("LIVE GAP HANDOFF PASSED", flush=True)


if __name__ == "__main__":
    main()
