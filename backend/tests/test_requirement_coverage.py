"""Focused tests for active-requirement facet coverage."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import requirement_coverage as coverage_module
from agents.discovery_coverage import fact_id
from agents.llm_errors import ExtractionFailed
from agents.requirement_coverage import (
    RequirementCoverageAssessment,
    RequirementCoverageStatus,
    RequirementFacetState,
    apply_requirement_coverage_assessment,
    assess_selected_requirement_answer,
    reconcile_requirement_coverage,
    reconcile_requirement_coverage_record,
)
from agents.requirements import ActiveRequirement, RequirementFacet, RequirementStatus, register_requirement, requirement_store_key
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState


def fact(topic, key, value, *, scope=S.USER_APP, turn=1, absence=None):
    return KnowledgeItem(
        topic=topic, scope=scope, key=key, value=value, evidence=value,
        confidence=1, knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn, absence=absence,
    )


def requirement(*, status=RequirementStatus.ACTIVE, scope=S.USER_APP, facets=True):
    facet_list = [
        RequirementFacet(id="failure_condition", label="Failure condition", description="What fails."),
        RequirementFacet(id="expected_behavior", label="Expected behavior", description="What happens next."),
    ] if facets else []
    return ActiveRequirement(
        id="workflow.external_dependency_failure",
        scope=scope,
        topic=T.EXCEPTIONS,
        parent_gap="recovery",
        label="External dependency failure",
        status=status,
        facets=facet_list,
    )


def test_activation_fact_is_not_requirement_coverage_evidence():
    trigger = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank approval is required")
    record = reconcile_requirement_coverage_record(requirement(), [trigger])
    assert record.status == RequirementCoverageStatus.UNSEEN
    assert record.candidate_fact_ids == []


def test_parent_gap_fact_is_known_shallow_not_resolved():
    recovery = fact(T.EXCEPTIONS, "recovery", "Retry the external request once")
    record = reconcile_requirement_coverage_record(requirement(), [recovery])
    assert record.status == RequirementCoverageStatus.KNOWN_SHALLOW
    assert record.candidate_fact_ids == [fact_id(recovery)]
    assert all(f.state == RequirementFacetState.UNKNOWN for f in record.facets.values())


def test_partial_facet_assessment_requires_expansion():
    recovery = fact(T.EXCEPTIONS, "recovery", "Retry the external request once")
    identity = fact_id(recovery)
    record = apply_requirement_coverage_assessment(
        requirement(), [recovery],
        RequirementCoverageAssessment(covered_facets={"expected_behavior": [identity]}),
    )
    assert record.status == RequirementCoverageStatus.NEEDS_EXPANSION
    assert record.expansion_needed is True
    assert record.facets["expected_behavior"].state == RequirementFacetState.COVERED
    assert record.facets["failure_condition"].state == RequirementFacetState.UNKNOWN


def test_all_required_facets_resolve_requirement_coverage():
    recovery = fact(T.EXCEPTIONS, "recovery", "When approval fails, retry once and then stop")
    identity = fact_id(recovery)
    record = apply_requirement_coverage_assessment(
        requirement(), [recovery],
        RequirementCoverageAssessment(covered_facets={
            "failure_condition": [identity],
            "expected_behavior": [identity],
        }),
    )
    assert record.status == RequirementCoverageStatus.RESOLVED
    assert record.expansion_needed is False


def test_grounded_not_applicable_facet_can_complete_coverage():
    recovery = fact(T.EXCEPTIONS, "recovery", "No retry behavior applies after this failure")
    identity = fact_id(recovery)
    record = apply_requirement_coverage_assessment(
        requirement(), [recovery],
        RequirementCoverageAssessment(
            covered_facets={"failure_condition": [identity]},
            not_applicable_facets={"expected_behavior": [identity]},
        ),
    )
    assert record.status == RequirementCoverageStatus.RESOLVED
    assert record.facets["expected_behavior"].state == RequirementFacetState.NOT_APPLICABLE


def test_unknown_facet_and_out_of_scope_fact_ids_are_rejected():
    recovery = fact(T.EXCEPTIONS, "recovery", "Retry once")
    identity = fact_id(recovery)
    with pytest.raises(ValueError):
        apply_requirement_coverage_assessment(
            requirement(), [recovery],
            RequirementCoverageAssessment(covered_facets={"invented": [identity]}),
        )
    with pytest.raises(ValueError):
        apply_requirement_coverage_assessment(
            requirement(), [recovery],
            RequirementCoverageAssessment(covered_facets={"expected_behavior": ["not-a-candidate"]}),
        )


def test_requirement_without_facet_contract_never_auto_resolves():
    recovery = fact(T.EXCEPTIONS, "recovery", "Retry once")
    record = reconcile_requirement_coverage_record(requirement(facets=False), [recovery])
    assert record.status == RequirementCoverageStatus.KNOWN_SHALLOW


@pytest.mark.parametrize("status,coverage_status", [
    (RequirementStatus.INACTIVE, RequirementCoverageStatus.INACTIVE),
    (RequirementStatus.DEFERRED, RequirementCoverageStatus.DEFERRED),
    (RequirementStatus.NOT_APPLICABLE, RequirementCoverageStatus.NOT_APPLICABLE),
])
def test_requirement_lifecycle_terminal_states_control_coverage(status, coverage_status):
    record = reconcile_requirement_coverage_record(requirement(status=status), [])
    assert record.status == coverage_status


def test_correction_removes_stale_facet_evidence_and_reopens_requirement():
    recovery = fact(T.EXCEPTIONS, "recovery", "When approval fails, retry once")
    identity = fact_id(recovery)
    req = requirement()
    assessed = apply_requirement_coverage_assessment(
        req, [recovery],
        RequirementCoverageAssessment(covered_facets={
            "failure_condition": [identity],
            "expected_behavior": [identity],
        }),
    )
    store = register_requirement({}, req.model_copy(update={"status": RequirementStatus.RESOLVED}))
    key = requirement_store_key(S.USER_APP, req.id)
    updated_store, coverage = reconcile_requirement_coverage(
        store, [], S.USER_APP, {key: assessed.model_dump(mode="json")}
    )
    assert coverage[key]["status"] == RequirementCoverageStatus.UNSEEN.value
    assert updated_store[key].status == RequirementStatus.ACTIVE


def test_resolved_coverage_synchronizes_requirement_status():
    recovery = fact(T.EXCEPTIONS, "recovery", "When approval fails, retry once")
    identity = fact_id(recovery)
    req = requirement()
    assessed = apply_requirement_coverage_assessment(
        req, [recovery],
        RequirementCoverageAssessment(covered_facets={
            "failure_condition": [identity],
            "expected_behavior": [identity],
        }),
    )
    store = register_requirement({}, req)
    key = requirement_store_key(S.USER_APP, req.id)
    updated_store, coverage = reconcile_requirement_coverage(
        store, [recovery], S.USER_APP, {key: assessed.model_dump(mode="json")}
    )
    assert coverage[key]["status"] == RequirementCoverageStatus.RESOLVED.value
    assert updated_store[key].status == RequirementStatus.RESOLVED


def test_other_scope_facts_do_not_count():
    recovery = fact(T.EXCEPTIONS, "recovery", "Retry once", scope=S.ADMIN_DASHBOARD)
    record = reconcile_requirement_coverage_record(requirement(scope=S.USER_APP), [recovery])
    assert record.status == RequirementCoverageStatus.UNSEEN


def test_resolved_schema_gap_does_not_resolve_specific_requirement():
    recovery = fact(T.EXCEPTIONS, "recovery", "Retry the external request once")
    req = requirement()
    key = requirement_store_key(S.USER_APP, req.id)
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [recovery],
        "active_requirements": {key: req},
        "requirement_coverage": {},
        "gap_coverage": {
            f"{S.USER_APP.value}|{T.EXCEPTIONS.value}|recovery": {"status": "RESOLVED"}
        },
    }
    from agents.requirement_coverage import requirement_coverage_node
    result = requirement_coverage_node(state)
    assert result["requirement_coverage"][key]["status"] == RequirementCoverageStatus.KNOWN_SHALLOW.value
    assert result["active_requirements"][key].status == RequirementStatus.ACTIVE


class _CoverageAssessor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, _messages):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _selected_requirement_state(req, response_fact, question="What happens next?"):
    key = requirement_store_key(S.USER_APP, req.id)
    response_fact = response_fact.model_copy(update={
        "source_turn": 5,
        "source_question": question,
    })
    return key, {
        "planner_source": "requirement",
        "selected_requirement_candidate": {
            "requirement_key": key,
            "target_facets": ["expected_behavior"],
        },
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=response_fact.evidence),
        ],
        "discovered_knowledge": [response_fact],
        "turn_count": 5,
    }


def test_invalid_requirement_coverage_assessment_gets_one_repair(monkeypatch):
    req = requirement()
    response_fact = fact(T.EXCEPTIONS, "recovery", "Retry the external request once")
    identity = fact_id(response_fact.model_copy(update={
        "source_turn": 5,
        "source_question": "What happens next?",
    }))
    key, state = _selected_requirement_state(req, response_fact)
    assessor = _CoverageAssessor([
        RequirementCoverageAssessment(
            covered_facets={"expected_behavior": ["invented-fact-id"]}
        ),
        RequirementCoverageAssessment(
            covered_facets={"expected_behavior": [identity]}
        ),
    ])
    monkeypatch.setattr(coverage_module, "requirement_coverage_assessor", lambda: assessor)

    updated_store, updated_coverage = assess_selected_requirement_answer(
        state, {key: req}, {}
    )

    assert assessor.calls == 2
    record = updated_coverage[key]
    assert record["facets"]["expected_behavior"]["state"] == RequirementFacetState.COVERED.value
    assert updated_store[key].status == RequirementStatus.ACTIVE


def test_requirement_coverage_repair_still_fails_closed(monkeypatch):
    req = requirement()
    response_fact = fact(T.EXCEPTIONS, "recovery", "Retry the external request once")
    key, state = _selected_requirement_state(req, response_fact)
    assessor = _CoverageAssessor([
        RequirementCoverageAssessment(
            covered_facets={"expected_behavior": ["invented-first"]}
        ),
        RequirementCoverageAssessment(
            covered_facets={"expected_behavior": ["invented-second"]}
        ),
    ])
    monkeypatch.setattr(coverage_module, "requirement_coverage_assessor", lambda: assessor)

    with pytest.raises(ExtractionFailed, match="after one repair attempt"):
        assess_selected_requirement_answer(state, {key: req}, {})

    assert assessor.calls == 2


def test_singleton_requirement_coverage_fact_id_is_normalized_to_list():
    assessment = RequirementCoverageAssessment.model_validate({
        "covered_facets": {"expected_behavior": "fact-123"},
        "not_applicable_facets": {},
    })
    assert assessment.covered_facets == {"expected_behavior": ["fact-123"]}


def test_singleton_not_applicable_fact_id_is_normalized_to_list():
    assessment = RequirementCoverageAssessment.model_validate({
        "covered_facets": {},
        "not_applicable_facets": {"expected_behavior": "fact-456"},
    })
    assert assessment.not_applicable_facets == {"expected_behavior": ["fact-456"]}
