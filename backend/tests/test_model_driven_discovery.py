from agents.discovery_coverage import fact_id
from agents.implications import infer_product_implications
from agents.inquiries import InquirySource, identify_open_inquiries
from agents.interview_planner import (
    all_discovery_resolved,
    all_required_gaps_resolved,
    interview_planner_node,
)
from agents.requirement_activation import reconcile_active_requirements
from agents.requirements import RequirementStatus, requirement_store_key
from agents.state import (
    DiscoveryScope as S,
    DiscoveryTopic as T,
    KnowledgeItem,
    KnowledgeState,
)


def fact(topic, key, value, *, role=None, roles=None, turn=1):
    return KnowledgeItem(
        topic=topic,
        scope=S.USER_APP,
        key=key,
        value=value,
        evidence=value,
        role=role,
        roles=roles,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
    )


def state_with(*facts):
    return {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": list(facts),
        "active_requirements": {},
        "requirement_coverage": {},
        "requirement_dependency_state": {},
        "eligible_requirement_keys": [],
        "validation_issues": [],
        "validation_blocking": False,
        "validation_candidate_blocking": False,
        "open_inquiries": [],
        "ranked_question_candidates": [],
        "question_candidate_priority": {},
        "gap_coverage": {},
        "topic_status": {},
        "topic_maturity": {},
        "planner_source": "model",
        "active_answer_result": None,
        "turn_count": 1,
    }


def test_model_frontier_starts_with_actor_not_schema_checklist():
    inquiries = identify_open_inquiries(state_with())

    assert len(inquiries) == 1
    assert inquiries[0].source == InquirySource.MODEL
    assert inquiries[0].anchor_gap == "primary_users"
    assert "secondary_users" not in {item.anchor_gap for item in inquiries}
    assert "role_transitions" not in {item.anchor_gap for item in inquiries}


def test_model_frontier_advances_by_product_coherence_not_topic_exhaustion():
    actor = fact(T.USER_ROLES, "primary_users", "customers", roles=["customer"])
    inquiries = identify_open_inquiries(state_with(actor))
    assert {item.anchor_gap for item in inquiries} == {"responsibilities::customer"}

    responsibility = fact(
        T.USER_ROLES,
        "responsibilities",
        "create and manage transactions",
        role="customer",
        turn=2,
    )
    inquiries = identify_open_inquiries(state_with(actor, responsibility))
    assert {item.anchor_gap for item in inquiries} == {"primary_user_goals::customer"}

    goal = fact(
        T.USER_GOALS,
        "primary_user_goals",
        "complete a transaction safely",
        role="customer",
        turn=3,
    )
    inquiries = identify_open_inquiries(state_with(actor, responsibility, goal))
    assert {item.anchor_gap for item in inquiries} == {"workflow_steps"}

    workflow = fact(
        T.CORE_WORKFLOW,
        "workflow_steps",
        "create transaction, fund it, fulfill it, confirm completion",
        turn=4,
    )
    inquiries = identify_open_inquiries(state_with(actor, responsibility, goal, workflow))
    assert {item.anchor_gap for item in inquiries} == {"completion_condition"}

    completion = fact(
        T.CORE_WORKFLOW,
        "completion_condition",
        "the agreement is fulfilled and completion is confirmed",
        turn=5,
    )
    inquiries = identify_open_inquiries(
        state_with(actor, responsibility, goal, workflow, completion)
    )
    assert inquiries == []


def test_incidental_cross_category_fact_does_not_skip_foundational_decision():
    actor = fact(T.USER_ROLES, "primary_users", "customers", roles=["customer"])
    responsibility = fact(
        T.USER_ROLES,
        "responsibilities",
        "create transactions",
        role="customer",
        turn=2,
    )
    polluted_goal = fact(
        T.USER_GOALS,
        "primary_user_goals",
        "Customers want to create transactions.",
        role="customer",
        turn=2,
    )
    state = state_with(actor, responsibility, polluted_goal)
    state["fact_acquisition"] = {
        fact_id(actor): {
            "acquisition": "DIRECT",
            "source_turn": 1,
            "active_topic": T.USER_ROLES.value,
            "active_gap": "primary_users",
        },
        fact_id(responsibility): {
            "acquisition": "DIRECT",
            "source_turn": 2,
            "active_topic": T.USER_ROLES.value,
            "active_gap": "responsibilities::customer",
        },
        fact_id(polluted_goal): {
            "acquisition": "INCIDENTAL",
            "source_turn": 2,
            "active_topic": T.USER_ROLES.value,
            "active_gap": "responsibilities::customer",
        },
    }

    inquiries = identify_open_inquiries(state)

    assert {item.anchor_gap for item in inquiries} == {"primary_user_goals::customer"}


def test_negative_role_transition_policy_does_not_activate_transition_depth():
    fixed = fact(
        T.USER_ROLES,
        "role_transitions",
        "A customer cannot switch roles within a transaction.",
    )

    store = reconcile_active_requirements({}, [fixed], S.USER_APP)
    key = requirement_store_key(S.USER_APP, "lifecycle.role_transition_behavior")

    assert key not in store
    assert not any(
        "lifecycle.role_transition_behavior" in implication.requirement_ids
        for implication in infer_product_implications([fixed], S.USER_APP)
    )


def test_positive_role_transition_policy_can_activate_transition_requirement():
    transition = fact(
        T.USER_ROLES,
        "role_transitions",
        "A customer may switch roles after approval.",
    )

    store = reconcile_active_requirements({}, [transition], S.USER_APP)
    key = requirement_store_key(S.USER_APP, "lifecycle.role_transition_behavior")

    assert key in store
    assert store[key].status == RequirementStatus.ACTIVE
    assert any(
        "lifecycle.role_transition_behavior" in implication.requirement_ids
        for implication in infer_product_implications([transition], S.USER_APP)
    )


def test_planner_can_finish_with_uncovered_schema_when_no_material_inquiry_remains():
    actor = fact(T.USER_ROLES, "primary_users", "customers", roles=["customer"])
    responsibility = fact(
        T.USER_ROLES,
        "responsibilities",
        "create transactions",
        role="customer",
    )
    goal = fact(
        T.USER_GOALS,
        "primary_user_goals",
        "complete transactions safely",
        role="customer",
    )
    workflow = fact(
        T.CORE_WORKFLOW,
        "workflow_steps",
        "create, fund, fulfill, confirm",
    )
    completion = fact(
        T.CORE_WORKFLOW,
        "completion_condition",
        "fulfillment is confirmed",
    )
    state = state_with(actor, responsibility, goal, workflow, completion)

    # The legacy schema is intentionally incomplete: secondary users,
    # permissions, motivations, limits, edge cases, etc. are not mandatory.
    assert not all_required_gaps_resolved(state)

    state["open_inquiries"] = [
        item.model_dump(mode="json")
        for item in identify_open_inquiries(state)
    ]
    assert state["open_inquiries"] == []

    update = interview_planner_node(state)
    merged = {**state, **update}

    assert update["awaiting_confirmation"] is True
    assert update["planner_source"] == "model"
    assert all_discovery_resolved(merged)
