"""Opt-in live check: valid approval, disguised visibility, and incomplete evidence.

Run from backend with python tests/live_prd_validation.py. Does not save a PRD.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.pm_agent import audit_llm, category_llm
from agents.prd_schema import PRDDraft
from agents.prd_validation import build_source_snapshot, validate_prd, PRDValidationError
from agents.state import KnowledgeItem, DiscoveryScope as S, DiscoveryTopic as T


def run():
    results = []
    category_cache = {}
    def classify(messages):
        result = category_llm.invoke(messages)
        parsed = result.get("parsed")
        print("CATEGORY:", parsed.model_dump() if hasattr(parsed, "model_dump") else parsed, flush=True)
        return result
    classifier = SimpleNamespace(invoke=classify)
    cases = [
        ("valid_approval", "Managers approve orders over $100", "Orders over $100 require manager approval.", True),
        ("disguised_visibility", "Managers approve orders over $100", "Only managers can see orders over $100.", False),
        ("incomplete_evidence", "orders over $100", "Orders over $100 require manager approval.", False),
    ]
    for name, quote, description, expected in cases:
        fact = KnowledgeItem(topic=T.BUSINESS_RULES, scope=S.USER_APP, key="approval_rules",
            value="Orders over $100 require manager approval", evidence=quote, role="manager",
            confidence=0.95, source_turn=2)
        sources = build_source_snapshot(dict(discovery_scope=S.USER_APP, discovered_knowledge=[fact]))
        draft = PRDDraft.model_validate(dict(product_name=None, elevator_pitch=[],
            scope=dict(in_scope=[], out_of_scope=[]), personas=[], non_functional_constraints=[],
            functional_requirements=[dict(id="FR-01", description=description, validation="TBD",
                category="BUSINESS_RULES.approval_rules", actor_ids=["manager"], conditions=["order total > $100"],
                source_fact_ids=[sources[0].fact_id])]))
        try:
            reports = validate_prd(draft, sources, audit_llm, classifier, category_cache)
            result = dict(case=name, accepted=True, reports=[r.model_dump() for r in reports])
        except PRDValidationError as exc:
            result = dict(case=name, accepted=False, reason=str(exc))
        result["expected"] = expected
        results.append(result)
        print(json.dumps(result), flush=True)
        Path("prd_live_validation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert all(r["accepted"] == r["expected"] for r in results), "Live verifier did not classify all three cases correctly"


if __name__ == "__main__":
    run()
