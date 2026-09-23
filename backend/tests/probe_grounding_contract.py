"""Opt-in live audit of recorded real CareConnect extraction candidates."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agents import knowledge_tracker as tracker
from agents.extraction_passes import PASSES, normalize_fact
from agents.state import DiscoveryScope
from replay_schema_alignment import CARE


def main():
    folder = Path(__file__).parent / "artifacts" / "schema_alignment"
    recorded = json.loads((folder / (sys.argv[1] if len(sys.argv) > 1 else "careconnect_failed_audit_calls.json")).read_text())
    contracts = {name: (schema, topic) for name, schema, topic, _ in PASSES}
    actual_models = tracker.extraction_models()
    if "--refresh-actions" in sys.argv:
        responses = {}
        for call in recorded:
            if call["name"] in contracts:
                responses.setdefault(call["name"], call["parsed"])
        refreshed = []
        def extract(name, messages):
            if name not in ("GOAL", "WORKFLOW"):
                return responses[name]
            print("LIVE " + name, flush=True)
            result = actual_models[name].invoke(messages)
            parsed = result["parsed"]
            refreshed.append(dict(name=name, parsed=parsed.model_dump(mode="json") if parsed else None))
            (folder / "extraction_mode_probe.json").write_text(json.dumps(refreshed, indent=2))
            return result
        tracker.extraction_models = lambda: {name: SimpleNamespace(invoke=lambda m, name=name: extract(name, m)) for name in contracts}
        candidates = tracker.extract_passes(CARE, {"discovered_knowledge": []}, DiscoveryScope.USER_APP)
        tracker.extraction_models = lambda: actual_models
    else:
        candidates = []
    for call in recorded:
        if "--refresh-actions" in sys.argv:
            break
        if call["name"] not in contracts:
            continue
        schema, topic = contracts[call["name"]]
        for data in call["parsed"]["items"]:
            fact = schema.model_validate(data)
            candidate = normalize_fact(fact, topic, DiscoveryScope.USER_APP, 0)
            if tracker.validate_extraction(candidate, CARE)[0]:
                candidates.append(candidate)
    model = tracker.extraction_models()["GROUNDING"]
    def invoke(messages):
        result = model.invoke(messages)
        parsed = result["parsed"]
        (folder / "grounding_contract_probe.json").write_text(json.dumps(
            dict(decision=parsed.model_dump(mode="json") if parsed else None,
                 parsing_error=str(result.get("parsing_error")), raw=str(result.get("raw")),
                 payload=json.loads(messages[-1].content)), indent=2))
        return result
    tracker.extraction_models = lambda: {"GROUNDING": SimpleNamespace(invoke=invoke)}
    state = dict(messages=[HumanMessage(content=CARE)], discovery_scope=DiscoveryScope.USER_APP,
                 discovered_knowledge=[], current_topic=None, current_gap=None)
    accepted = tracker.ground_items(candidates, CARE, state)
    (folder / "grounding_probe_accepted.json").write_text(json.dumps(
        [item.model_dump(mode="json") for item in accepted], indent=2))
    for item in accepted:
        print(item.key, item.role, item.value)
    assert len(accepted) == 6
    assert {item.role for item in accepted if item.key == "responsibilities"} == {"patient", "healthcare_provider"}
    assert len([item for item in accepted if item.key == "primary_user_goals"]) == 1
    assert all(item.key in {"primary_users", "responsibilities", "workflow_steps", "primary_user_goals"} for item in accepted)
    print("LIVE AUDIT PASSED")


if __name__ == "__main__":
    main()
