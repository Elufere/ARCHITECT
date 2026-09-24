"""Opt-in OpenAI probe (uses API credits) for the production RESPONSIBILITY instructions."""
import ast
import json
import sys
from pathlib import Path

from live_pass_probe import CARE, FIX, PASSES, call


source = Path(__file__).parents[1] / "agents" / "extraction_passes.py"
module = ast.parse(source.read_text(encoding="utf-8"))
pass_node = next(node for node in module.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "PASSES" for target in node.targets))
entry = next(entry for entry in pass_node.value.elts if entry.elts[0].value == "RESPONSIBILITY")
_, keys, special = PASSES["RESPONSIBILITY"]
PASSES["RESPONSIBILITY"] = (entry.elts[3].value, keys, special)

cases = {
    "CareConnect": (CARE, ["patient", "healthcare_provider"]),
    "FixMate": (FIX, ["customer", "artisan"]),
    "A_restaurant_search": ("Customers can search restaurants, compare menus, order food, and pay.", ["customer"]),
    "B_restaurant_duty": ("Restaurants manage their menus, accept orders, and prepare meals.", ["restaurant"]),
    "C_freelancer_browse": ("Freelancers browse jobs and submit proposals.", ["freelancer"]),
    "D_freelancer_delivery": ("Freelancers deliver accepted work by the agreed deadline.", ["freelancer"]),
    "E_buyer_browse": ("Buyers can browse listings and message sellers.", ["buyer", "seller"]),
    "F_seller_shipping": ("Sellers must ship paid orders within two business days.", ["seller"]),
}

output = Path(__file__).with_name("live_responsibility_probe_output.json")
results = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
for name, (message, roles) in cases.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    try:
        results[name] = call("RESPONSIBILITY", message, f"Known actors: {roles}\n")
    except Exception as exc:
        results[name] = f"ERROR: {exc}"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(name, results[name], flush=True)
