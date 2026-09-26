"""Production claim capture extracts propositions once and admits them by meaning."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents import knowledge_tracker as tracker
from agents.state import DiscoveryScope as S, DiscoveryTopic as T, KnowledgeItem


def actor(role):
    text = f"{role.title()}s use the app."
    return KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value=role,
        evidence=text,
        roles=[role],
        confidence=1,
    )


def claim(kind, value, evidence, **extra):
    return dict(
        kind=kind,
        value=value,
        evidence=evidence,
        confidence=1,
        knowledge_state="CONFIRMED",
        **extra,
    )


def production_models(monkeypatch, items, calls):
    def invoke(messages):
        calls.append(messages)
        return {"parsed": {"items": items}}

    monkeypatch.setattr(
        tracker,
        "extraction_models",
        lambda: {"CLAIMS": SimpleNamespace(invoke=invoke)},
    )


def test_seller_goal_answer_does_not_become_action_permission_or_end_state(monkeypatch):
    text = (
        "The seller wants assurance that they’ll get paid once they fulfill what was agreed. "
        "The escrow should protect them from a buyer receiving the goods or service and then "
        "refusing to release the payment."
    )
    first = "The seller wants assurance that they’ll get paid once they fulfill what was agreed"
    second = (
        "The escrow should protect them from a buyer receiving the goods or service and then "
        "refusing to release the payment"
    )
    calls = []
    production_models(monkeypatch, [
        claim("desired_outcome", first, first, role="seller"),
        # Adversarial misclassifications must fail deterministic admission.
        claim("actor_action", "be protected from non-payment", second, role="seller"),
        claim("authorization_boundary", second, second, role="seller"),
        claim("workflow_end_state", "The seller gets paid once they fulfill what was agreed", first),
        claim("validation_rule", second, second),
        claim("eligibility_rule", first, first),
    ], calls)

    def unexpected(*_):
        pytest.fail("Admitted neutral claims must not enter cross-category grounding")

    monkeypatch.setattr(tracker, "ground_items", unexpected)
    monkeypatch.setattr(tracker, "extract_gap_absence", unexpected)
    monkeypatch.setattr(tracker, "confirms_existing", lambda *_: False)
    monkeypatch.setattr(tracker, "interpret_closed_answer", lambda *_: None)

    question = "What is the primary outcome or benefit the seller wants to achieve?"
    state = dict(
        messages=[AIMessage(content=question), HumanMessage(content=text)],
        discovery_scope=S.USER_APP,
        current_topic=T.USER_GOALS,
        current_gap="primary_user_goals::seller",
        discovered_knowledge=[actor("buyer"), actor("seller")],
        superseded_knowledge=[],
        fact_acquisition={},
        turn_count=2,
    )

    result = tracker.knowledge_tracker_node(state)
    added = [item for item in result["discovered_knowledge"] if item.source_turn == 2]

    assert len(calls) == 1
    assert len(added) == 1
    assert added[0].topic == T.USER_GOALS
    assert added[0].key == "primary_user_goals"
    assert added[0].role == "seller"
    assert not any(
        item.key in {"permissions", "end_state", "validation_rules", "eligibility_rules"}
        for item in added
    )
    assert not any(
        item.key == "responsibilities" and item.role == "seller"
        for item in added
    )


def test_initial_escrow_sequence_does_not_invent_trigger_or_success_definition(monkeypatch):
    text = (
        "A buyer pays into escrow, the seller fulfills what was agreed, and the money is "
        "released when the transaction is successfully completed."
    )
    calls = []
    production_models(monkeypatch, [
        claim("primary_actor", "buyer", "A buyer pays into escrow", role="buyer"),
        claim("primary_actor", "seller", "the seller fulfills what was agreed", role="seller"),
        claim("actor_action", "pay into escrow", "A buyer pays into escrow", role="buyer"),
        claim("actor_action", "fulfill what was agreed", "the seller fulfills what was agreed", role="seller"),
        claim("workflow_steps", text, text),
        # These are exactly the old over-classifications and must be rejected.
        claim("workflow_trigger", "A buyer pays into escrow", text),
        claim(
            "success_condition",
            "The money is released when the transaction is successfully completed",
            "the money is released when the transaction is successfully completed",
        ),
    ], calls)

    state = dict(
        messages=[HumanMessage(content=text)],
        discovery_scope=S.USER_APP,
        current_topic=None,
        current_gap=None,
        discovered_knowledge=[],
        turn_count=0,
    )
    batch = tracker.extract_passes(text, state, S.USER_APP)

    assert len(calls) == 1
    assert getattr(batch, "grounding_required") is False
    assert {item.key for item in batch} == {
        "primary_users",
        "responsibilities",
        "workflow_steps",
    }
    assert not any(item.key == "trigger" for item in batch)
    assert not any(item.key == "success_criteria" for item in batch)


def test_unclassified_claim_is_not_persisted(monkeypatch):
    text = "It should feel trustworthy."
    calls = []
    production_models(monkeypatch, [
        claim("unclassified", "It should feel trustworthy", text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=None,
            current_gap=None,
            turn_count=0,
        ),
        S.USER_APP,
    )

    assert calls
    assert batch == []
    assert getattr(batch, "grounding_required") is False


def test_capacity_alias_resolves_back_to_canonical_actor(monkeypatch):
    text = "The seller wants guaranteed payment after fulfilling the deal."
    calls = []
    production_models(monkeypatch, [
        claim("desired_outcome", text, text, role="seller"),
    ], calls)

    customer = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="Customers can act as buyers or sellers.",
        evidence="Customers can act as buyers or sellers.",
        roles=["customer"],
        aliases={"customer": ["buyer", "seller"]},
        confidence=1,
    )
    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[customer],
            current_topic=T.USER_GOALS,
            current_gap="primary_user_goals::customer",
            turn_count=4,
        ),
        S.USER_APP,
    )

    assert len(batch) == 1
    assert batch[0].key == "primary_user_goals"
    assert batch[0].role == "customer"


def test_explicit_external_participant_cannot_become_current_app_actor(monkeypatch):
    text = (
        "Payments in USER_APP remain pending until the fraud team approves them "
        "through a third-party platform, outside USER_APP."
    )
    calls = []
    production_models(monkeypatch, [
        claim("secondary_actor", "fraud team", text, role="fraud_team"),
        claim("workflow_dependency", text, text),
        claim("approval_rule", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=None,
            current_gap=None,
            turn_count=0,
        ),
        S.USER_APP,
    )

    assert not any(
        item.key in {"primary_users", "secondary_users"}
        and "fraud_team" in (item.roles or [])
        for item in batch
    )
    assert {item.key for item in batch} == {"downstream_dependency", "approval_rules"}


def test_process_participant_does_not_become_secondary_actor_without_app_membership(monkeypatch):
    text = "If there is a dispute, a mediator reviews the evidence."
    calls = []
    production_models(monkeypatch, [
        claim("secondary_actor", "mediator", text, role="mediator"),
        claim("workflow_steps", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[actor("customer")],
            current_topic=None,
            current_gap=None,
            turn_count=3,
        ),
        S.USER_APP,
    )

    assert not any(item.key == "secondary_users" for item in batch)
    assert any(item.key == "workflow_steps" for item in batch)


def test_direct_actor_answer_can_establish_membership_without_repeating_app_wording(monkeypatch):
    text = "Vendors."
    calls = []
    production_models(monkeypatch, [
        claim("secondary_actor", "vendor", text, role="vendor"),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[
                AIMessage(content="Besides customers, who else uses the app?"),
                HumanMessage(content=text),
            ],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[actor("customer")],
            current_topic=T.USER_ROLES,
            current_gap="secondary_users",
            turn_count=3,
        ),
        S.USER_APP,
    )

    assert len(batch) == 1
    assert batch[0].key == "secondary_users"
    assert batch[0].roles == ["vendor"]


def test_active_gap_absence_is_not_committed_by_claim_capture(monkeypatch):
    text = "No other users."
    calls = []
    production_models(monkeypatch, [
        claim(
            "secondary_actor",
            "none",
            text,
            absence="none",
        ),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[
                AIMessage(content="Besides customers, will anyone else use the app?"),
                HumanMessage(content=text),
            ],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[actor("customer")],
            current_topic=T.USER_ROLES,
            current_gap="secondary_users",
            turn_count=2,
        ),
        S.USER_APP,
    )

    assert batch == []
    assert getattr(batch, "grounding_required") is False


def test_missing_primary_actor_role_is_repaired_from_explicit_direct_answer(monkeypatch):
    text = "the main users are customers who can either be a buyer or seller in a transaction"
    calls = []
    production_models(monkeypatch, [
        claim("primary_actor", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[
                AIMessage(content="Who are the main users of the app?"),
                HumanMessage(content=text),
            ],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=T.USER_ROLES,
            current_gap="primary_users",
            turn_count=1,
        ),
        S.USER_APP,
    )

    assert len(batch) == 1
    assert batch[0].roles == ["customer"]
    assert batch[0].aliases == {"customer": ["buyer", "seller"]}


def test_missing_owned_role_resolves_from_canonical_actor_aliases(monkeypatch):
    customer = KnowledgeItem(
        topic=T.USER_ROLES,
        scope=S.USER_APP,
        key="primary_users",
        value="customers can be buyers or sellers",
        evidence="customers can be buyers or sellers",
        roles=["customer"],
        aliases={"customer": ["buyer", "seller"]},
        confidence=1,
    )
    text = "Either the buyer or seller is allowed to propose changes before funding."
    calls = []
    production_models(monkeypatch, [
        claim("authorization_boundary", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[customer],
            current_topic=T.BUSINESS_RULES,
            current_gap="ownership_rules",
            turn_count=5,
        ),
        S.USER_APP,
    )

    assert len(batch) == 1
    assert batch[0].key == "permissions"
    assert batch[0].role == "customer"


def test_intermediate_dispute_state_is_not_terminal_end_state(monkeypatch):
    text = "Once either party raises a dispute, the transaction moves into a disputed state."
    calls = []
    production_models(monkeypatch, [
        claim("workflow_end_state", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=T.USER_ROLES,
            current_gap="primary_users",
            turn_count=2,
        ),
        S.USER_APP,
    )

    assert batch == []


def test_history_statement_is_not_a_downstream_dependency(monkeypatch):
    text = "Any changes or state transitions should remain in the transaction history."
    calls = []
    production_models(monkeypatch, [
        claim("workflow_dependency", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=T.CORE_WORKFLOW,
            current_gap="workflow_steps",
            turn_count=6,
        ),
        S.USER_APP,
    )

    assert batch == []


def test_requirement_anchor_cannot_turn_history_rule_into_ownership(monkeypatch):
    text = "Changes affect the transaction going forward; they should never overwrite previous terms."
    calls = []
    production_models(monkeypatch, [
        claim("ownership_rule", text, text),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[
                AIMessage(content="Do modifications rewrite transaction history?"),
                HumanMessage(content=text),
            ],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=T.BUSINESS_RULES,
            current_gap="ownership_rules",
            planner_source="requirement",
            turn_count=9,
        ),
        S.USER_APP,
    )

    assert batch == []


def test_emergent_product_structure_is_preserved_outside_schema(monkeypatch):
    text = "There are groups, and packages are inside groups. A group can use NGN or USD."
    calls = []
    production_models(monkeypatch, [
        claim(
            "product_entity",
            "Groups are part of the product structure",
            "There are groups",
            subject="group",
        ),
        claim(
            "entity_relationship",
            "packages are inside groups",
            "packages are inside groups",
            subject="package",
            relation="contained_in",
            object="group",
        ),
        claim(
            "entity_attribute",
            "A group can use NGN or USD",
            "A group can use NGN or USD",
            subject="group",
            relation="currency",
            object="NGN or USD",
        ),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=T.CORE_WORKFLOW,
            current_gap="workflow_steps",
            turn_count=3,
        ),
        S.USER_APP,
    )

    assert batch == []
    assert len(batch.concepts) == 3
    assert {item.kind.value for item in batch.concepts} == {
        "ENTITY", "RELATIONSHIP", "ATTRIBUTE"
    }
    assert any(
        item.subject == "package"
        and item.relation == "contained_in"
        and item.object == "group"
        for item in batch.concepts
    )


def test_knowledge_tracker_adds_product_concepts_to_product_model(monkeypatch):
    text = "Packages are inside groups."
    calls = []
    production_models(monkeypatch, [
        claim(
            "entity_relationship",
            "Packages are inside groups",
            "Packages are inside groups",
            subject="package",
            relation="contained_in",
            object="group",
        ),
    ], calls)
    monkeypatch.setattr(tracker, "confirms_existing", lambda *_: False)
    monkeypatch.setattr(tracker, "interpret_closed_answer", lambda *_: None)
    monkeypatch.setattr(tracker, "extract_gap_absence", lambda *_: None)

    state = dict(
        messages=[HumanMessage(content=text)],
        discovery_scope=S.USER_APP,
        current_topic=None,
        current_gap=None,
        discovered_knowledge=[],
        product_concepts=[],
        superseded_knowledge=[],
        fact_acquisition={},
        turn_count=0,
    )
    result = tracker.knowledge_tracker_node(state)

    assert len(result["product_concepts"]) == 1
    assert "PRODUCT_STRUCTURE" in result["product_model"]
    assert "package -[contained_in]-> group" in result["product_model"]["PRODUCT_STRUCTURE"][0]


def test_product_entity_without_generic_value_preserves_evidence(monkeypatch):
    text = "My product is an ecommerce application for selling packages."
    calls = []
    production_models(monkeypatch, [{
        "kind": "product_entity",
        "evidence": "selling packages",
        "subject": "package",
        "confidence": 0.9,
        "knowledge_state": "CONFIRMED",
    }], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=None,
            current_gap=None,
            turn_count=0,
        ),
        S.USER_APP,
    )

    assert batch == []
    assert len(batch.concepts) == 1
    assert batch.concepts[0].subject == "package"
    assert batch.concepts[0].value == "selling packages"
    assert batch.concepts[0].relation is None
    assert batch.concepts[0].object is None


def test_invitation_before_visibility_is_an_explicit_workflow_dependency(monkeypatch):
    text = "Users are invited to a particular group before they can see the packages they are allowed to buy."
    calls = []
    production_models(monkeypatch, [
        claim(
            "workflow_dependency",
            "Users must be invited to a particular group before they can see the packages they are allowed to buy.",
            text,
        ),
    ], calls)

    batch = tracker.extract_passes(
        text,
        dict(
            messages=[HumanMessage(content=text)],
            discovery_scope=S.USER_APP,
            discovered_knowledge=[],
            current_topic=T.CORE_WORKFLOW,
            current_gap="workflow_steps",
            planner_source="model",
            turn_count=1,
        ),
        S.USER_APP,
    )

    assert len(batch) == 1
    assert batch[0].key == "downstream_dependency"
