"""Helpers for fixtures that intentionally model already-completed discovery coverage."""
from agents.discovery_coverage import coverage_key, fact_id
from agents.interview_planner import PER_ROLE_TASKS
from agents.state import DiscoveryScope, KnowledgeState


def coverage_for_facts(state, *topics):
    """Create deliberate RESOLVED receipts for confirmed fixture facts.

    Use only in tests whose premise is that those facts were already deliberately
    covered. Incidental-extraction tests should not call this helper.
    """
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    wanted = set(topics) if topics else None
    coverage = dict(state.get("gap_coverage", {}))

    for item in state.get("discovered_knowledge", []):
        if item.scope != scope or item.knowledge_state != KnowledgeState.CONFIRMED:
            continue
        if wanted is not None and item.topic not in wanted:
            continue

        gap = item.key
        if item.key in PER_ROLE_TASKS.get(item.topic, set()):
            if not item.role:
                continue
            gap = f"{item.key}::{item.role}"

        key = coverage_key(scope, item.topic, gap)
        record = coverage.setdefault(key, {
            "status": "RESOLVED",
            "fact_ids": [],
            "resolution": "TEST_FIXTURE",
        })
        identity = fact_id(item)
        if identity not in record["fact_ids"]:
            record["fact_ids"].append(identity)

    return coverage
