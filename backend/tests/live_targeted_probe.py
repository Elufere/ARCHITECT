"""Standalone live probes for the responsibility and goal prompt changes."""
import ast
import json
import sys
from pathlib import Path

from live_pass_probe import CARE, FIX, PASSES, call


source = Path(__file__).parents[1] / "agents" / "extraction_passes.py"
module = ast.parse(source.read_text(encoding="utf-8"))
pass_node = next(node for node in module.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "PASSES" for target in node.targets))
for entry in pass_node.value.elts:
    name = entry.elts[0].value
    if name in ("RESPONSIBILITY", "GOAL"):
        _, keys, special = PASSES[name]
        PASSES[name] = (entry.elts[3].value, keys, special)

results = {}
cases = (
    ("CareConnect", CARE, ["patient", "healthcare_provider"]),
    ("FixMate", FIX, ["customer", "artisan"]),
    ("Motivation", "Customers use the app because calling several restaurants to compare prices takes too long.", ["customer"]),
)
for label, message, primary in cases:
    if "care" in sys.argv and label != "CareConnect":
        continue
    if "motivation" in sys.argv and label != "Motivation":
        continue
    if "motivation" not in sys.argv and label == "Motivation":
        continue
    results[label] = {}
    context = f"Confirmed primary roles: {primary}\nConfirmed secondary roles: none\n"
    owned_behavior = ([{
        "role": "healthcare_provider", "key": "responsibilities",
        "value": "manage profiles, specialties, schedules, appointments, and payments",
        "evidence": "Healthcare providers should be able to manage their profiles, specialties, schedules, appointments, and payments.",
    }] if label == "CareConnect" else [])
    post_context = "Responsibilities/permissions already extracted from this response: " + (
        json.dumps(owned_behavior) if owned_behavior else "none") + "\n"
    post_context += "Final GOAL check: omit a goal that repeats the behavior above unless the response explicitly states a distinct desired outcome.\n"
    for name in (("RESPONSIBILITY", "GOAL") if label == "CareConnect" and "goals" not in sys.argv else ("GOAL",)):
        try:
            results[label][name] = call(name, message, context, post_context)
        except Exception as exc:
            results[label][name] = f"ERROR: {exc}"
        print(label, name, results[label][name], flush=True)

Path(__file__).with_name("live_targeted_probe_output.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
