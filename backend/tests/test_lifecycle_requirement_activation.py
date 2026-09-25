"""Focused tests for generic lifecycle-driven requirement activation."""
from agents.requirement_activation import (
    LIFECYCLE_ACTIVATION_RULES,
    FactCondition,
    reconcile_active_requirements,
    requirement_activation_node,
)
from agents.requirements import RequirementStatus, requirement_store_key
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem, KnowledgeState


def fact(topic, key, value, *, scope=S.USER_APP, turn=1, absence=None, role=None):
    return KnowledgeItem(
        topic=topic,
        scope=scope,
        key=key,
        value=value,
        evidence=value,
        role=role,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
        absence=absence,
    )


def lifecycle_only(knowledge, *, scope=S.USER_APP, store=None):
    return reconcile_active_requirements(
        store or {},
        knowledge,
        scope,
        rules=LIFECYCLE_ACTIVATION_RULES,
    )


def test_start_state_requires_trigger_and_workflow_steps():
    trigger = fact(T.CORE_WORKFLOW, "trigger", "A customer submits a request")
    steps = fact(T.CORE_WORKFLOW, "workflow_steps", "Customer submits, provider reviews, customer confirms")

    assert lifecycle_only([trigger]) == {}

    store = lifecycle_only([trigger, steps])
    key = requirement_store_key(S.USER_APP, "lifecycle.start_state_behavior")
    assert key in store
    assert store[key].status == RequirementStatus.ACTIVE
    assert store[key].metadata["requirement_family"] == "lifecycle"
    assert store[key].metadata["lifecycle_stages"] == ["creation", "active"]


def test_state_transition_requires_workflow_and_end_state():
    steps = fact(T.CORE_WORKFLOW, "workflow_steps", "Request is submitted, reviewed, then confirmed")
    end = fact(T.CORE_WORKFLOW, "end_state", "The request becomes completed")

    assert lifecycle_only([steps]) == {}

    store = lifecycle_only([steps, end])
    key = requirement_store_key(S.USER_APP, "lifecycle.state_transition_behavior")
    assert key in store
    assert {facet.id for facet in store[key].facets} == {
        "transition_conditions",
        "transition_result",
        "transition_reversibility",
    }


def test_post_completion_requires_completion_condition_and_end_state():
    completion = fact(
        T.CORE_WORKFLOW,
        "completion_condition",
        "The process is complete once both parties confirm handover",
    )
    end = fact(T.CORE_WORKFLOW, "end_state", "The transaction becomes completed")

    store = lifecycle_only([completion, end])
    key = requirement_store_key(S.USER_APP, "lifecycle.post_completion_behavior")
    assert key in store
    assert store[key].metadata["lifecycle_stages"] == ["completion", "post_completion"]


def test_explicit_cancellation_activates_transition_but_absence_does_not():
    cancellation = fact(
        T.EXCEPTIONS,
        "user_cancellations",
        "A customer can cancel before approval",
    )
    store = lifecycle_only([cancellation])
    key = requirement_store_key(S.USER_APP, "lifecycle.cancellation_transition")
    assert key in store

    no_cancellation = fact(
        T.EXCEPTIONS,
        "user_cancellations",
        "none",
        absence="none",
    )
    assert lifecycle_only([no_cancellation]) == {}


def test_explicit_role_transition_activates_role_lifecycle_but_absence_does_not():
    transition = fact(
        T.USER_ROLES,
        "role_transitions",
        "A buyer can later become a seller after verification",
    )
    store = lifecycle_only([transition])
    key = requirement_store_key(S.USER_APP, "lifecycle.role_transition_behavior")
    assert key in store

    none = fact(T.USER_ROLES, "role_transitions", "none", absence="none")
    assert lifecycle_only([none]) == {}


def test_time_and_limit_facts_activate_distinct_boundary_requirements():
    deadline = fact(
        T.CONSTRAINTS,
        "time_constraints",
        "The invitation expires after 48 hours",
    )
    limit = fact(
        T.BUSINESS_RULES,
        "limits",
        "A user may submit at most five active requests",
    )
    store = lifecycle_only([deadline, limit])
    assert requirement_store_key(S.USER_APP, "lifecycle.time_boundary_behavior") in store
    assert requirement_store_key(S.USER_APP, "lifecycle.limit_boundary_behavior") in store


def test_modification_signal_uses_word_stems_not_arbitrary_substrings():
    update = fact(
        T.USER_ROLES,
        "responsibilities",
        "Hosts can update event details",
        role="host",
    )
    store = lifecycle_only([update])
    key = requirement_store_key(S.USER_APP, "lifecycle.modification_behavior")
    assert key in store

    exchange = fact(
        T.CORE_WORKFLOW,
        "workflow_steps",
        "Users exchange messages before checkout",
    )
    store = lifecycle_only([exchange])
    assert requirement_store_key(S.USER_APP, "lifecycle.modification_behavior") not in store


def test_removal_signal_matches_inflected_explicit_actions():
    removal = fact(
        T.USER_ROLES,
        "permissions",
        "Administrators may archive or delete completed records",
        role="administrator",
    )
    store = lifecycle_only([removal])
    key = requirement_store_key(S.USER_APP, "lifecycle.removal_behavior")
    assert key in store
    assert store[key].priority_hints.business_risk == 0.85
    assert "historical_retention" in {facet.id for facet in store[key].facets}


def test_lifecycle_requirement_reactively_deactivates_when_trigger_is_corrected_away():
    trigger = fact(T.CORE_WORKFLOW, "trigger", "A customer submits a request")
    steps = fact(T.CORE_WORKFLOW, "workflow_steps", "Customer submits and provider reviews")
    active = lifecycle_only([trigger, steps])
    key = requirement_store_key(S.USER_APP, "lifecycle.start_state_behavior")
    assert active[key].status == RequirementStatus.ACTIVE

    corrected = lifecycle_only([steps], store=active)
    assert key in corrected
    assert corrected[key].status == RequirementStatus.INACTIVE
    assert corrected[key].activation_sources == []


def test_lifecycle_activation_is_scope_isolated():
    deadline = fact(
        T.CONSTRAINTS,
        "time_constraints",
        "Admin review expires after one day",
        scope=S.ADMIN_DASHBOARD,
    )
    assert lifecycle_only([deadline], scope=S.USER_APP) == {}

    store = lifecycle_only([deadline], scope=S.ADMIN_DASHBOARD)
    key = requirement_store_key(S.ADMIN_DASHBOARD, "lifecycle.time_boundary_behavior")
    assert key in store


def test_default_activation_node_includes_lifecycle_rules_without_writing_facts():
    trigger = fact(T.CORE_WORKFLOW, "trigger", "A guest opens a request")
    steps = fact(T.CORE_WORKFLOW, "workflow_steps", "Guest submits, host reviews")
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [trigger, steps],
        "active_requirements": {},
    }
    result = requirement_activation_node(state)
    key = requirement_store_key(S.USER_APP, "lifecycle.start_state_behavior")
    assert key in result["active_requirements"]
    assert state["discovered_knowledge"] == [trigger, steps]
    assert "discovered_knowledge" not in result


def test_word_prefix_condition_handles_inflection_and_rejects_exchange():
    condition = FactCondition(
        topic=T.CORE_WORKFLOW,
        key="workflow_steps",
        value_word_prefixes=("chang",),
    )
    changed = fact(T.CORE_WORKFLOW, "workflow_steps", "The user changed the delivery address")
    exchange = fact(T.CORE_WORKFLOW, "workflow_steps", "The users exchange messages")
    assert condition.matches(changed, S.USER_APP) is True
    assert condition.matches(exchange, S.USER_APP) is False
