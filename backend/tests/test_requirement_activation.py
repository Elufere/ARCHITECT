"""Focused tests for deterministic requirement activation and invalidation."""
from agents.requirement_activation import (
    FactCondition,
    RequirementActivationRule,
    RequirementTemplate,
    reconcile_active_requirements,
    requirement_activation_node,
)
from agents.requirements import RequirementStatus, requirement_store_key
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState


def fact(topic, key, value, *, scope=S.USER_APP, turn=1, absence=None):
    return KnowledgeItem(
        topic=topic,
        scope=scope,
        key=key,
        value=value,
        evidence=value,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
        absence=absence,
    )


def rule(rule_id="test.rule"):
    return RequirementActivationRule(
        id=rule_id,
        description="test",
        when=(FactCondition(topic=T.CORE_WORKFLOW, key="downstream_dependency"),),
        activates=(
            RequirementTemplate(
                id="workflow.external_dependency_failure",
                topic=T.EXCEPTIONS,
                parent_gap="recovery",
                label="External dependency failure",
                description="Determine failure behavior.",
            ),
        ),
    )


def test_confirmed_fact_activates_requirement_with_provenance():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank must approve settlement", turn=4)
    store = reconcile_active_requirements({}, [item], S.USER_APP, rules=(rule(),))
    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    requirement = store[key]
    assert requirement.status == RequirementStatus.ACTIVE
    assert len(requirement.activation_sources) == 1
    source = requirement.activation_sources[0]
    assert source.activation_rule_id == "test.rule"
    assert source.source_turn == 4
    assert source.source_key == "CORE_WORKFLOW.downstream_dependency"


def test_repeated_reconciliation_is_idempotent():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank must approve settlement")
    once = reconcile_active_requirements({}, [item], S.USER_APP, rules=(rule(),))
    twice = reconcile_active_requirements(once, [item], S.USER_APP, rules=(rule(),))
    assert twice == once


def test_trigger_removal_marks_requirement_inactive_without_deleting_history():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank must approve settlement")
    active = reconcile_active_requirements({}, [item], S.USER_APP, rules=(rule(),))
    inactive = reconcile_active_requirements(active, [], S.USER_APP, rules=(rule(),))
    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    assert key in inactive
    assert inactive[key].status == RequirementStatus.INACTIVE


def test_restored_trigger_reactivates_inactive_requirement():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank must approve settlement")
    active = reconcile_active_requirements({}, [item], S.USER_APP, rules=(rule(),))
    inactive = reconcile_active_requirements(active, [], S.USER_APP, rules=(rule(),))
    restored = reconcile_active_requirements(inactive, [item], S.USER_APP, rules=(rule(),))
    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    assert restored[key].status == RequirementStatus.ACTIVE
    assert restored[key].activation_sources


def test_substantive_rule_does_not_fire_for_explicit_absence():
    item = fact(
        T.CORE_WORKFLOW, "downstream_dependency", "none",
        absence="none",
    )
    store = reconcile_active_requirements({}, [item], S.USER_APP, rules=(rule(),))
    assert store == {}


def test_cross_scope_fact_does_not_activate_current_scope_requirement():
    item = fact(
        T.CORE_WORKFLOW, "downstream_dependency", "Admin approval required",
        scope=S.ADMIN_DASHBOARD,
    )
    store = reconcile_active_requirements({}, [item], S.USER_APP, rules=(rule(),))
    assert store == {}


def test_multiple_rules_keep_requirement_active_while_any_trigger_remains():
    first = rule("rule.one")
    second = RequirementActivationRule(
        id="rule.two",
        description="second",
        when=(FactCondition(topic=T.CONSTRAINTS, key="time_constraints"),),
        activates=first.activates,
    )
    dependency = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank approves")
    deadline = fact(T.CONSTRAINTS, "time_constraints", "Approval expires after 24 hours", turn=2)

    store = reconcile_active_requirements({}, [dependency, deadline], S.USER_APP, rules=(first, second))
    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    assert {s.activation_rule_id for s in store[key].activation_sources} == {"rule.one", "rule.two"}

    store = reconcile_active_requirements(store, [deadline], S.USER_APP, rules=(first, second))
    assert store[key].status == RequirementStatus.ACTIVE
    assert {s.activation_rule_id for s in store[key].activation_sources} == {"rule.two"}


def test_activation_never_writes_implication_to_confirmed_knowledge():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank must approve settlement")
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [item],
        "active_requirements": {},
    }
    result = requirement_activation_node(state)
    assert state["discovered_knowledge"] == [item]
    assert "discovered_knowledge" not in result
    assert result["active_requirements"]


def test_default_registry_activates_external_dependency_requirement():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A regulator approves release")
    result = requirement_activation_node({
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [item],
        "active_requirements": {},
    })
    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    assert key in result["active_requirements"]


def test_existing_requirement_refreshes_template_facets_on_reconciliation():
    item = fact(T.CORE_WORKFLOW, "downstream_dependency", "A bank must approve settlement")
    first = reconcile_active_requirements({}, [item], S.USER_APP)
    key = requirement_store_key(S.USER_APP, "workflow.external_dependency_failure")
    legacy = first[key].model_copy(update={"facets": []})
    refreshed = reconcile_active_requirements({key: legacy}, [item], S.USER_APP)
    assert {facet.id for facet in refreshed[key].facets} == {
        "failure_condition", "expected_behavior", "recovery_or_escalation"
    }
