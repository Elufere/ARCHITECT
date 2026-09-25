"""System-wide invariants for end-to-end Architect discovery tests."""
from agents.discovery_coverage import fact_id
from agents.requirements import RequirementStatus
from agents.state import KnowledgeItem


def assert_discovery_invariants(state):
    knowledge = state.get("discovered_knowledge", [])
    live_fact_ids = {fact_id(item) for item in knowledge}
    superseded = set()
    for record in state.get("superseded_knowledge", []):
        if not isinstance(record, dict):
            continue
        payload = record.get("fact")
        if payload:
            superseded.add(fact_id(KnowledgeItem.model_validate(payload)))
    requirements = state.get("active_requirements", {})
    coverage = state.get("requirement_coverage", {})
    dependencies = state.get("requirement_dependency_state", {})

    # Superseded facts must not remain in the active knowledge set.
    assert live_fact_ids.isdisjoint(superseded)

    # Fact-backed requirement provenance may only reference live confirmed facts.
    # Manual/config activation sources are allowed to have no evidence_ref.
    for key, requirement in requirements.items():
        for source in requirement.activation_sources:
            if source.evidence_ref:
                assert source.evidence_ref in live_fact_ids, (
                    f"{key} activation source references missing fact {source.evidence_ref}"
                )
        for evidence in requirement.evidence_refs:
            assert evidence.fact_id in live_fact_ids, (
                f"{key} requirement evidence references missing fact {evidence.fact_id}"
            )

    # Coverage may only cite live facts. Resolved/active lifecycle state must agree.
    for key, payload in coverage.items():
        requirement = requirements.get(key)
        if requirement is None:
            continue
        for identity in payload.get("candidate_fact_ids", []):
            assert identity in live_fact_ids, (
                f"{key} candidate coverage references missing fact {identity}"
            )
        for facet in payload.get("facets", {}).values():
            for identity in facet.get("fact_ids", []):
                assert identity in live_fact_ids, (
                    f"{key} facet coverage references missing fact {identity}"
                )
        status = payload.get("status")
        assert not (
            requirement.status == RequirementStatus.RESOLVED and status != "RESOLVED"
        )
        assert not (
            requirement.status == RequirementStatus.ACTIVE and status == "RESOLVED"
        )

    # Every eligible/ranked candidate must point at a live ACTIVE requirement and
    # must not bypass dependency blocking.
    for bucket in (
        state.get("eligible_question_candidates", []),
        state.get("ranked_question_candidates", []),
    ):
        for candidate in bucket:
            key = candidate["requirement_key"]
            requirement = requirements.get(key)
            assert requirement is not None, f"candidate references missing requirement {key}"
            assert requirement.status == RequirementStatus.ACTIVE
            decision = dependencies.get(key)
            assert decision and decision.get("eligible") is True, (
                f"candidate bypassed dependency block for {key}"
            )

    # A candidate suppressed by consistency validation must never survive ranking.
    if state.get("validation_candidate_blocking"):
        assert state.get("question_candidates", []) == []
        assert state.get("eligible_question_candidates", []) == []
        assert state.get("ranked_question_candidates", []) == []
