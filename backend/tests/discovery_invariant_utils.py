"""System-wide invariants for end-to-end Architect discovery tests."""
from agents.discovery_coverage import fact_id
from agents.requirements import RequirementStatus


def assert_discovery_invariants(state):
    knowledge = state.get("discovered_knowledge", [])
    live_fact_ids = {fact_id(item) for item in knowledge}
    superseded = {
        record.get("fact_id")
        for record in state.get("superseded_knowledge", [])
        if isinstance(record, dict) and record.get("fact_id")
    }
    requirements = state.get("active_requirements", {})
    coverage = state.get("requirement_coverage", {})
    dependencies = state.get("requirement_dependency_state", {})

    # Superseded facts must not remain in the active knowledge set.
    assert live_fact_ids.isdisjoint(superseded)

    # Requirement activation provenance may only reference live confirmed facts.
    for key, requirement in requirements.items():
        for source in requirement.activation_sources:
            for identity in source.fact_ids:
                assert identity in live_fact_ids, (
                    f"{key} activation source references missing fact {identity}"
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
