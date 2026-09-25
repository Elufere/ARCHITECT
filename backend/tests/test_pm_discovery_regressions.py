from agents.conversation_manager import classify_turn
from agents.guardrails import questions_are_semantic_duplicates
from agents.interview_planner import (
    build_gap_info, get_roles_in_discovery_order, interview_planner_node,
    relevant_confirmed_context,
)
from agents.question_generator import permission_discovery_guidance
from agents.role_utils import split_role_labels
from agents.knowledge_tracker import item_directly_answers_gap, knowledge_tracker_node, validate_extraction
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeItem, KnowledgeState
from coverage_test_utils import coverage_for_facts


def fact(key: str, value: str) -> KnowledgeItem:
    return KnowledgeItem(
        topic=DiscoveryTopic.CORE_WORKFLOW,
        scope=DiscoveryScope.USER_APP,
        key=key,
        value=value,
        evidence=value,
        confidence=1.0,
    )


def test_workflow_gap_uses_a_real_schema_key():
    state = {"discovery_scope": DiscoveryScope.USER_APP, "discovered_knowledge": []}
    gap = build_gap_info(state, DiscoveryTopic.CORE_WORKFLOW)
    assert gap["current_gap"] == "trigger"


def test_workflow_advances_to_completion_condition_after_steps():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [fact("trigger", "A buyer or seller creates a deal."), fact("workflow_steps", "The creator sends an invite and the recipient joins.")],
    }
    state["gap_coverage"] = coverage_for_facts(state, DiscoveryTopic.CORE_WORKFLOW)
    gap = build_gap_info(state, DiscoveryTopic.CORE_WORKFLOW)
    assert gap["current_gap"] == "completion_condition"


def test_evidence_accepts_normalized_inflection_but_requires_source_quote():
    item = fact("workflow_steps", "The buyer receives the product and clicks join.")
    item.evidence = "buyer recieves the product and click join"
    valid, reason = validate_extraction(
        item,
        "The buyer recieves the product and click join.",
        "workflow_steps",
    )
    assert valid, reason


def test_source_span_validation_does_not_replace_semantic_grounding():
    item = fact("workflow_steps", "The app charges a platform fee.")
    item.evidence = "buyer creates a deal"
    valid, _ = validate_extraction(item, "The buyer creates a deal.", "workflow_steps")
    assert valid  # semantic entailment is enforced later by the grounding audit


def test_conversation_intents_cover_repair_turns():
    assert classify_turn("I told you already") == "objection"
    assert classify_turn("Your number 1 is wrong; sellers can also create a deal") == "correction"
    assert classify_turn("Yes, this is correct") == "confirmation"


def test_semantic_duplicate_workflow_questions_are_blocked():
    assert questions_are_semantic_duplicates(
        "Can you walk me through the transaction workflow from start to finish?",
        "Can you describe the transaction journey from beginning to end?",
    )


def test_distinct_role_goal_questions_are_not_duplicates():
    assert not questions_are_semantic_duplicates(
        "What goals do administrators have when using the escrow application?",
        "What goals do support staff have when using the escrow application?",
    )


def test_role_alias_is_accepted_when_user_says_administrators():
    item = KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=DiscoveryScope.USER_APP,
        key="secondary_users",
        value="administrators",
        evidence="they are called administrators",
        roles=["admin"],
        confidence=1.0,
    )
    valid, reason = validate_extraction(
        item, "Yes, they are called administrators", "secondary_users"
    )
    assert valid, reason


def test_schema_gaps_do_not_force_questions_once_actor_actions_are_known():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="primary_users", value="hosts and guests", evidence="hosts and guests",
                          roles=["hosts", "guests"], confidence=1.0),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="responsibilities", value="hosts create events", evidence="hosts create events",
                          role="hosts", confidence=1.0),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="responsibilities", value="guests buy packages", evidence="guests buy packages",
                          role="guests", confidence=1.0),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="secondary_users", value="none", evidence="There are no secondary users.",
                          roles=[], absence="none", confidence=1.0),
        ],
        "topic_status": {DiscoveryTopic.USER_ROLES: "PARTIAL"},
        "topic_maturity": {},
        "current_topic": DiscoveryTopic.USER_ROLES,
    }
    state["gap_coverage"] = coverage_for_facts(state, DiscoveryTopic.USER_ROLES)
    result = interview_planner_node(state)
    assert result["planner_source"] == "model"
    assert result["current_topic"] == DiscoveryTopic.USER_GOALS
    assert result["current_gap"] == "primary_user_goals::hosts"


def test_negative_phrase_marks_the_current_gap_known(monkeypatch):
    from agents import knowledge_tracker as tracker
    from agents.semantic_validation import GapAnswer, GroundingResult
    monkeypatch.setattr(tracker, "extract_passes", lambda *_: [])
    monkeypatch.setattr(tracker, "semantic_decision", lambda name, *_:
        GapAnswer(resolution="none", evidence="None I can think of", confidence=1)
        if name == "GAP_ANSWER" else GroundingResult(evidence_categories={"0": ["USER_ROLES.role_transitions"]}, supported_ids=[0], confirmed_absence_ids=[0]))
    state = {
        "messages": [],
        "current_gap": "role_transitions",
        "current_topic": DiscoveryTopic.USER_ROLES,
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [],
        "topic_status": {},
        "turn_count": 3,
    }
    from langchain_core.messages import HumanMessage
    state["messages"] = [HumanMessage(content="None I can think of")]
    result = knowledge_tracker_node(state)
    assert result["discovered_knowledge"][0].key == "role_transitions"
    assert result["discovered_knowledge"][0].value == "none"
    assert result["discovered_knowledge"][0].absence == "none"


def test_customer_roles_are_discovered_before_confirmed_admin_role():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="primary_users", value="buyers and sellers", evidence="buyers and sellers",
                          roles=["buyer", "seller"], confidence=1.0),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="secondary_users", value="administrators", evidence="administrators",
                          roles=["administrators"], confidence=1.0),
        ],
    }
    assert get_roles_in_discovery_order(state, DiscoveryTopic.USER_ROLES) == [
        "buyer", "seller", "administrators"
    ]
    state["gap_coverage"] = coverage_for_facts(state, DiscoveryTopic.USER_ROLES)
    # Remove role-specific coverage because the fixture only establishes actor identity.
    state["gap_coverage"] = {
        key: value for key, value in state["gap_coverage"].items()
        if "|responsibilities::" not in key and "|permissions::" not in key
    }
    assert build_gap_info(state, DiscoveryTopic.USER_ROLES)["current_gap"] == "responsibilities::buyer"


def role_fact(key: str, value: str, role: str) -> KnowledgeItem:
    return KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=DiscoveryScope.USER_APP,
        key=key,
        value=value,
        evidence=value,
        role=role,
        confidence=1.0,
    )


def test_permission_remains_a_gap_after_abstract_responsibility():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="primary_users", value="buyer", evidence="buyer",
                          roles=["buyer"], confidence=1.0),
            role_fact("responsibilities", "The buyer is responsible for completing the purchase.", "buyer"),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="secondary_users", value="none", evidence="There are no secondary users.",
                          roles=[], absence="none", confidence=1.0),
        ],
    }
    state["gap_coverage"] = coverage_for_facts(state, DiscoveryTopic.USER_ROLES)
    assert build_gap_info(state, DiscoveryTopic.USER_ROLES)["current_gap"] == "permissions::buyer"
    guidance = permission_discovery_guidance(state, "buyer")
    assert "high-level or vague" in guidance
    assert "The buyer is responsible for completing the purchase." in guidance


def test_permission_guidance_deepens_concrete_responsibility_evidence():
    state = {
        "discovered_knowledge": [
            role_fact(
                "responsibilities",
                "The buyer creates or joins a deal, funds escrow, confirms receipt, and raises disputes.",
                "buyer",
            ),
        ],
    }
    guidance = permission_discovery_guidance(state, "buyer")
    assert "do NOT ask the user to list" in guidance
    assert "authorization boundaries" in guidance
    assert "creates or joins a deal" in guidance


def test_compound_role_list_is_preserved_as_two_atomic_roles():
    assert split_role_labels(["administrators and support staff"]) == [
        "administrators", "support staff"
    ]


def test_compound_secondary_role_does_not_become_a_combined_planner_gap():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="secondary_users", value="administrators and support staff",
                          evidence="administrators and support staff",
                          roles=["administrators", "support staff"], confidence=1.0),
        ],
    }
    gap = build_gap_info(state, DiscoveryTopic.USER_ROLES)
    assert "responsibilities::administrators" in gap["missing_keys"]
    assert "responsibilities::support staff" in gap["missing_keys"]
    assert all("administrators and support staff" not in key for key in gap["missing_keys"])


def test_syntactic_validation_does_not_reclassify_a_misclassified_fact():
    item = KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=DiscoveryScope.USER_APP,
        key="secondary_users",
        value="Administrators manage users and transactions.",
        evidence="Administrators manage users and transactions.",
        roles=["admin"],
        confidence=1.0,
    )
    valid, reason = validate_extraction(
        item,
        "Administrators manage users and transactions.",
        "responsibilities::administrators and support staff",
    )
    assert valid, reason
    assert item.key == "secondary_users"
    assert item.roles == ["admin"]
    assert item.role is None


def test_hyphenated_secondary_role_is_grounded_and_accepted():
    item = KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=DiscoveryScope.USER_APP,
        key="secondary_users",
        value="super-admin, admin, and support staff",
        evidence="super-admin, admin, and support staff",
        roles=["super-admin", "admin", "support staff"],
        confidence=1.0,
    )
    valid, reason = validate_extraction(
        item, "Yes, super-admin, admin, and support staff", "secondary_users"
    )
    assert valid, reason


def test_incidental_fact_stays_inferred_when_it_is_not_the_active_gap():
    item = KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=DiscoveryScope.USER_APP,
        key="multiple_roles",
        value="A customer can be a buyer or seller depending on the deal.",
        evidence="buyer or seller depending on the deal",
        confidence=1.0,
        knowledge_state=KnowledgeState.INFERRED,
    )
    assert not item_directly_answers_gap(item, "primary_users")
    assert item.knowledge_state == KnowledgeState.INFERRED


def test_active_gap_cannot_supply_a_missing_goal_owner():
    item = KnowledgeItem(
        topic=DiscoveryTopic.USER_GOALS,
        scope=DiscoveryScope.USER_APP,
        key="primary_user_goals",
        value="Buyers want their money protected until delivery.",
        evidence="Buyers want their money protected until delivery.",
        confidence=1.0,
    )
    valid, reason = validate_extraction(
        item,
        "Buyers want their money protected until delivery.",
        "primary_user_goals::buyer",
    )
    assert not valid
    assert "owner" in reason.lower()
    assert item.role is None
    assert not item_directly_answers_gap(item, "primary_user_goals::buyer")


def test_inferred_schema_policy_does_not_force_confirmation_when_higher_value_inquiry_exists():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="primary_users", value="buyer", evidence="buyer",
                          roles=["buyer"], confidence=1.0),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="secondary_users", value="None specified", evidence="none",
                          confidence=1.0),
            role_fact("responsibilities", "Buyer completes the purchase.", "buyer"),
            role_fact("permissions", "Buyer can manage their purchase.", "buyer"),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="multiple_roles", value="A customer can be buyer or seller by deal.",
                          evidence="buyer or seller by deal", confidence=1.0,
                          knowledge_state=KnowledgeState.INFERRED),
        ],
        "topic_status": {DiscoveryTopic.USER_ROLES: "PARTIAL"},
        "topic_maturity": {},
        "current_topic": DiscoveryTopic.USER_ROLES,
    }
    state["gap_coverage"] = coverage_for_facts(state, DiscoveryTopic.USER_ROLES)
    # Inferred facts are never deliberate completion receipts.
    gap = build_gap_info(state, DiscoveryTopic.USER_ROLES)
    assert gap["current_gap"] == "multiple_roles"
    assert gap["inferred_gap_evidence"] == ["A customer can be buyer or seller by deal."]
    plan = interview_planner_node(state)
    assert plan["planner_source"] == "model"
    assert plan["current_gap"] == "primary_user_goals::buyer"
    assert plan["next_discovery_move"] == "resolve_model_uncertainty"


def goal_fact(key: str, value: str, role: str) -> KnowledgeItem:
    return KnowledgeItem(
        topic=DiscoveryTopic.USER_GOALS,
        scope=DiscoveryScope.USER_APP,
        key=key,
        value=value,
        evidence=value,
        role=role,
        confidence=1.0,
    )


def test_secondary_goal_gap_reuses_known_secondary_role_without_reasking_it():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="primary_users", value="buyer and seller", evidence="buyer and seller",
                          roles=["buyer", "seller"], confidence=1.0),
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="secondary_users", value="support staff, administrator, super-admin",
                          evidence="support staff, administrator, super-admin",
                          roles=["support staff", "administrator", "super-admin"], confidence=1.0),
            goal_fact("primary_user_goals", "Buyer completes a purchase.", "buyer"),
            goal_fact("primary_user_goals", "Seller completes a sale.", "seller"),
        ],
    }
    state["gap_coverage"] = coverage_for_facts(state, DiscoveryTopic.USER_GOALS)
    gap = build_gap_info(state, DiscoveryTopic.USER_GOALS)
    assert gap["current_gap"] == "secondary_user_goals::support staff"
    assert any("secondary_users" in fact and "support staff" in fact for fact in gap["relevant_context"])


def test_confirmed_roles_are_context_not_completion_for_later_goal_target():
    state = {
        "discovery_scope": DiscoveryScope.USER_APP,
        "discovered_knowledge": [
            KnowledgeItem(topic=DiscoveryTopic.USER_ROLES, scope=DiscoveryScope.USER_APP,
                          key="primary_users", value="buyer and seller", evidence="buyer and seller",
                          roles=["buyer", "seller"], confidence=1.0),
        ],
    }
    gap = build_gap_info(state, DiscoveryTopic.USER_GOALS)
    assert gap["current_gap"] == "primary_user_goals::buyer"
    context = relevant_confirmed_context(state, DiscoveryTopic.USER_GOALS, gap["current_gap"])
    assert any("primary_users" in fact for fact in context)
