"""Regressions from the short grocery-list end-to-end discovery run."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agents.conversation_manager as conversation
import agents.discovery_completion as completion
import agents.knowledge_tracker as tracker
import agents.discovery_threads as threads
from agents.graph import route_after_conversation_manager
from agents.discovery_threads import DiscoveryThreadPlan, ThreadFrontierInquiry
from agents.requirement_activation import reconcile_active_requirements, LIFECYCLE_ACTIVATION_RULES
from agents.requirements import requirement_store_key
from agents.state import KnowledgeItem
from agents.extraction_passes import NeutralClaim, canonical_role
from agents.question_generator import _core_actor_question
from agents.guardrails import _core_actor_question_matches_objective
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeState as K
from services.discovery_session import create_initial_discovery_state


def test_explicit_user_is_a_valid_canonical_actor_and_owns_same_turn_actions():
    text = (
        "I want a grocery list app for individual users. "
        "Users can create a list and mark items as bought."
    )
    state = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "turn_count": 0,
    }
    primary_roles: set[str] = set()
    secondary_roles: set[str] = set()

    actor = tracker._admit_claim_item(
        NeutralClaim(
            kind="primary_actor",
            role="Users",
            value="individual users",
            evidence="individual users",
            confidence=1,
            knowledge_state=K.CONFIRMED,
        ),
        state,
        S.USER_APP,
        primary_roles,
        secondary_roles,
    )

    assert actor is not None
    assert actor.key == "primary_users"
    assert actor.roles == ["user"]
    primary_roles.update(canonical_role(role) for role in actor.roles or [])

    action = tracker._admit_claim_item(
        NeutralClaim(
            kind="actor_action",
            role="user",
            value="create a list",
            evidence="Users can create a list",
            confidence=1,
            knowledge_state=K.CONFIRMED,
        ),
        {**state, "discovered_knowledge": [actor]},
        S.USER_APP,
        primary_roles,
        secondary_roles,
    )

    assert action is not None
    assert action.key == "responsibilities"
    assert action.role == "user"


def test_explicit_no_other_user_roles_becomes_secondary_actor_absence():
    text = "There are no other user roles, payments, integrations, or admin features."

    claim = tracker._explicit_additional_actor_absence_claim(text)

    assert claim is not None
    assert claim.kind == "secondary_actor"
    assert claim.absence == "none"
    assert claim.value == "none"
    assert claim.evidence == "There are no other user roles"


def test_access_answer_cannot_become_primary_user_absence(monkeypatch):
    state = {
        "messages": [
            AIMessage(
                content=(
                    "Should each user only be able to manage their own grocery lists, "
                    "or can they view other users' lists?"
                )
            ),
            HumanMessage(content="they can only manage their list"),
        ],
        "current_topic": T.USER_ROLES,
        "current_gap": "primary_users",
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "turn_count": 4,
    }

    monkeypatch.setattr(
        tracker,
        "semantic_decision",
        lambda *args, **kwargs: pytest.fail(
            "GAP_ANSWER must not run for a non-identity actor question"
        ),
    )

    assert tracker.extract_gap_absence(
        "they can only manage their list",
        state,
        S.USER_APP,
    ) is None


def test_core_actor_question_cannot_drift_into_access_or_capabilities():
    state = create_initial_discovery_state("A grocery list app.")

    question = _core_actor_question(state, S.USER_APP)

    assert question == "Who exactly will directly use or interact with this product?"
    assert _core_actor_question_matches_objective(question)
    assert not _core_actor_question_matches_objective(
        "Should each user manage only their own list or view other users' lists?"
    )


def test_bare_no_to_substantive_product_question_is_not_completion():
    state = create_initial_discovery_state("A grocery list app.")
    state["messages"].extend([
        AIMessage(content="Should users be able to share their lists?"),
        HumanMessage(content="no"),
    ])

    assert completion.should_review_completion_intent(state, "no") is False


def test_bare_no_to_wrap_up_question_can_be_completion():
    state = create_initial_discovery_state("A grocery list app.")
    state["messages"].extend([
        AIMessage(content="Is there anything else you want to cover?"),
        HumanMessage(content="no"),
    ])

    assert completion.should_review_completion_intent(state, "no") is True


def test_generate_prd_is_explicit_completion_without_model_call(monkeypatch):
    state = create_initial_discovery_state("A grocery list app.")

    monkeypatch.setattr(
        completion,
        "completion_intent_model",
        lambda: pytest.fail("Explicit generate PRD intent should not need an LLM call"),
    )

    review = completion.review_completion_intent(state, "generate prd")

    assert review.wants_to_finish_discovery is True


def test_completion_request_wins_over_free_text_deferral(monkeypatch):
    state = create_initial_discovery_state("A grocery list app.")
    state["messages"].append(
        HumanMessage(content="we have covered everything about the product, generate the prd")
    )
    state["turn_count"] = 8

    monkeypatch.setattr(
        conversation,
        "should_review_free_text_deferral",
        lambda current: pytest.fail(
            "Completion request must bypass free-text deferral classification"
        ),
    )

    update = conversation.conversation_manager_node(state)

    assert update["conversation_intent"] == "close_discovery"
    assert update["founder_requested_completion"] is True
    assert update["completion_arbitration_complete"] is False
    assert "messages" not in update



def test_advice_request_routes_directly_to_generator_not_extraction():
    state = {
        "conversation_intent": "advice_request",
        "current_objective": "Clarify whether removal has restrictions.",
        "selected_inquiry": {"id": "current-question"},
    }

    assert route_after_conversation_manager(state) == "generate"


def test_ordinary_remove_action_does_not_activate_archival_lifecycle():
    action = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        role="individual_user",
        value="add or remove items",
        evidence="Users can add or remove items",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )

    store = reconcile_active_requirements(
        {},
        [action],
        S.USER_APP,
        rules=LIFECYCLE_ACTIVATION_RULES,
    )

    assert requirement_store_key(
        S.USER_APP,
        "lifecycle.removal_behavior",
    ) not in store


def test_low_signal_crud_depth_is_rejected_before_semantic_assessment(monkeypatch):
    action = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="responsibilities",
        role="individual_user",
        value="add or remove items",
        evidence="Users can add or remove items",
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )
    plan = DiscoveryThreadPlan(
        thread_id="list_management",
        thread_label="List management",
        thread_objective="Understand list management.",
        frontier=ThreadFrontierInquiry(
            decision_key="item_removal_rules",
            topic=T.BUSINESS_RULES,
            objective="Clarify rules and conditions for removing list items.",
            question_hint="Can users remove items anytime or are there restrictions?",
            reason="Removal rules and limits are not specified.",
        ),
    )
    state = {
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [action],
        "captured_observations": [],
        "discovery_boundaries": [],
    }

    monkeypatch.setattr(
        threads,
        "inquiry_assessment_model",
        lambda: pytest.fail(
            "Low-signal CRUD depth should be rejected before an assessor call"
        ),
    )

    problem = threads._semantic_frontier_problem(
        plan,
        state,
        S.USER_APP,
    )

    assert problem is not None
    assert "LOW_MARGINAL_VALUE" in problem


def test_explicit_no_other_users_is_recovered_after_grounding_rejection(monkeypatch):
    text = (
        "I want a grocery list app for individual users. "
        "There are no other user roles."
    )
    primary = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="individual users",
        evidence="individual users",
        roles=["individual_user"],
        confidence=1,
        knowledge_state=K.CONFIRMED,
        source_turn=0,
    )

    monkeypatch.setattr(
        tracker,
        "extract_passes",
        lambda *_: type(
            "Batch",
            (list,),
            {"grounding_required": True, "concepts": [], "external_systems": [],
             "observations": [], "observation_candidates": {}},
        )([primary]),
    )
    monkeypatch.setattr(
        tracker,
        "ground_items",
        lambda items, *_: [item for item in items if item.key != "secondary_users"],
    )
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)

    state = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "current_topic": None,
        "current_gap": None,
        "discovered_knowledge": [],
        "superseded_knowledge": [],
        "fact_acquisition": {},
        "turn_count": 0,
    }

    result = tracker.knowledge_tracker_node(state)

    assert any(
        item.key == "secondary_users"
        and item.absence == "none"
        and item.evidence == "There are no other user roles"
        for item in result["discovered_knowledge"]
    )



def test_actor_value_none_without_absence_flag_is_normalized_before_alias_resolution():
    text = "There are no other user roles"
    state = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "turn_count": 0,
        "current_gap": None,
        "current_topic": None,
    }

    item = tracker._admit_claim_item(
        NeutralClaim(
            kind="secondary_actor",
            value="none",
            evidence=text,
            role=None,
            absence=None,
            confidence=1,
            knowledge_state=K.CONFIRMED,
        ),
        state,
        S.USER_APP,
        set(),
        set(),
    )

    assert item is not None
    assert item.key == "secondary_users"
    assert item.absence == "none"
    assert item.roles == []
    assert not item.aliases



def test_explicit_product_exclusions_recover_from_unclassified_claim():
    text = "There are no other user roles, payments, integrations, or admin features."
    claim = NeutralClaim(
        kind="unclassified",
        value="There are no payments, integrations, or admin features",
        evidence=text,
        confidence=1,
        knowledge_state=K.CONFIRMED,
    )

    recovered = tracker._recover_unclassified_scope_exclusion(claim)

    assert recovered.kind == "mvp_out_of_scope"
    state = {
        "messages": [HumanMessage(content=text)],
        "discovery_scope": S.USER_APP,
        "discovered_knowledge": [],
        "turn_count": 0,
        "current_gap": None,
        "current_topic": None,
    }
    item = tracker._admit_claim_item(
        recovered,
        state,
        S.USER_APP,
        set(),
        set(),
    )
    assert item is not None
    assert item.topic == T.MVP_SCOPE
    assert item.key == "out_of_scope"
    assert "payments" in item.value.lower()
    assert "integrations" in item.value.lower()
    assert "admin features" in item.value.lower()


def test_generic_negative_statement_is_not_promoted_to_scope_exclusion():
    claim = NeutralClaim(
        kind="unclassified",
        value="There are no side effects",
        evidence="There are no side effects",
        confidence=1,
        knowledge_state=K.CONFIRMED,
    )

    assert tracker._recover_unclassified_scope_exclusion(claim).kind == "unclassified"



def test_rewritten_coordinated_action_evidence_falls_back_to_literal_founder_turn():
    text = (
        "A user can create tasks, edit or delete them, and mark them as completed."
    )
    claim = NeutralClaim(
        kind="actor_action",
        role="user",
        value="edit tasks",
        evidence="A user can edit tasks",
        confidence=1,
        knowledge_state=K.CONFIRMED,
    )

    repaired = tracker._literalize_semantic_claim_evidence(claim, text)

    assert repaired.evidence == text
    assert repaired.value == "edit tasks"


def test_short_contextual_answer_uses_founder_answer_not_pm_question_as_evidence():
    answer = "delete immediately"
    claim = NeutralClaim(
        kind="actor_action",
        role="user",
        value="A user deletes a task immediately without confirmation",
        evidence="when a user deletes a task, should the app delete it immediately",
        confidence=1,
        knowledge_state=K.CONFIRMED,
    )

    repaired = tracker._literalize_semantic_claim_evidence(claim, answer)

    assert repaired.evidence == "delete immediately"
    assert "when a user" not in repaired.evidence


def test_explicit_scope_exclusion_is_recovered_even_when_capture_omits_it():
    text = (
        "There are no other user roles, payments, integrations, or admin features."
    )

    claim = tracker._explicit_product_scope_exclusion_claim(text)

    assert claim is not None
    assert claim.kind == "mvp_out_of_scope"
    assert claim.evidence == text
    assert "payments" in claim.value.lower()
    assert "integrations" in claim.value.lower()
    assert "admin features" in claim.value.lower()
    assert "user roles" not in claim.value.lower()



def test_product_concept_keeps_the_question_that_contextualizes_short_evidence():
    claim = NeutralClaim(
        kind="entity_attribute",
        subject="task",
        relation="has attribute",
        object="optional description",
        value="an optional description",
        evidence="an optional description",
        confidence=1,
        knowledge_state=K.CONFIRMED,
    )

    concept = tracker._concept_from_claim(
        claim,
        S.USER_APP,
        3,
        "What fields or information should each task have?",
    )

    assert concept.source_question == (
        "What fields or information should each task have?"
    )
    assert concept.evidence == "an optional description"
