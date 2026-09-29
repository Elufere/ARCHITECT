from agents.discovery_coverage import fact_id
from agents.product_concepts import ProductConcept, ProductConceptKind
from agents.requirement_coverage import (
    FacetCoverage,
    RequirementCoverageRecord,
    RequirementCoverageStatus,
    RequirementFacetState,
)
from agents.requirements import (
    ActiveRequirement,
    RequirementFacet,
    RequirementStatus,
    requirement_store_key,
)
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeItem, KnowledgeState
from agents.understanding_projection import build_understanding_projection


SCOPE = DiscoveryScope.USER_APP


def fact(topic, key, value, *, role=None, roles=None, state=KnowledgeState.CONFIRMED):
    return KnowledgeItem(
        topic=topic,
        scope=SCOPE,
        key=key,
        value=value,
        evidence=value,
        role=role,
        roles=roles,
        confidence=1.0,
        knowledge_state=state,
    )


def section_map(projection):
    return {section.id: section for section in projection.sections}


def test_projection_uses_only_confirmed_knowledge_and_does_not_invent_integrations():
    customer = fact(
        DiscoveryTopic.USER_ROLES,
        "primary_users",
        "two customers can create a transaction as buyer and seller",
        roles=["customer"],
    )
    workflow = fact(
        DiscoveryTopic.CORE_WORKFLOW,
        "workflow_steps",
        "buyer pays into escrow, seller delivers, buyer confirms completion",
    )
    inferred = fact(
        DiscoveryTopic.BUSINESS_RULES,
        "limits",
        "transactions are limited to 1000",
        state=KnowledgeState.INFERRED,
    )
    flutterwave = ProductConcept(
        kind=ProductConceptKind.ENTITY,
        scope=SCOPE,
        subject="flutterwave",
        value="flutterwave is a payment gateway thirdparty",
        evidence="flutterwave is a payment gateway thirdparty",
        confidence=1.0,
    )

    projection = build_understanding_projection({
        "discovery_scope": SCOPE,
        "discovered_knowledge": [customer, workflow, inferred],
        "product_concepts": [flutterwave.model_dump(mode="json")],
        "active_requirements": {},
        "requirement_coverage": {},
        "requirement_dependency_state": {},
        "open_inquiries": [],
        "validation_issues": [],
        "discovery_boundaries": [],
    })
    sections = section_map(projection)

    assert [item.label for item in sections["users"].items] == ["Customer"]
    assert sections["workflow"].items[0].label == workflow.value
    assert "business_rules" not in sections
    assert sections["product_structure"].items[0].label == "Flutterwave"
    assert "integrations" not in sections


def test_projection_maps_active_open_blocked_and_deferred_decisions():
    active = ActiveRequirement(
        id="dispute_resolution",
        scope=SCOPE,
        topic=DiscoveryTopic.BUSINESS_RULES,
        label="Dispute resolution",
        description="Decide how a disputed transaction reaches a resolution.",
        facets=[
            RequirementFacet(
                id="resolution_outcome",
                label="Resolution outcome",
                description="What outcomes can resolve a dispute.",
            ),
        ],
    )
    blocked = ActiveRequirement(
        id="seller_payout",
        scope=SCOPE,
        topic=DiscoveryTopic.CORE_WORKFLOW,
        label="Seller payout timing",
        description="Decide when payout becomes available.",
    )
    deferred = ActiveRequirement(
        id="delivery_ui",
        scope=SCOPE,
        topic=DiscoveryTopic.CORE_WORKFLOW,
        label="Delivery confirmation interaction",
        status=RequirementStatus.DEFERRED,
    )

    active_key = requirement_store_key(SCOPE, active.id)
    blocked_key = requirement_store_key(SCOPE, blocked.id)
    deferred_key = requirement_store_key(SCOPE, deferred.id)

    coverage = RequirementCoverageRecord(
        requirement_id=active.id,
        scope=SCOPE,
        status=RequirementCoverageStatus.NEEDS_EXPANSION,
        facets={
            "resolution_outcome": FacetCoverage(
                facet_id="resolution_outcome",
                state=RequirementFacetState.UNKNOWN,
            )
        },
        expansion_needed=True,
    )

    selected_id = f"{SCOPE.value}|requirement|{active.id}|resolution_outcome"
    projection = build_understanding_projection({
        "discovery_scope": SCOPE,
        "discovered_knowledge": [],
        "product_concepts": [],
        "active_requirements": {
            active_key: active,
            blocked_key: blocked,
            deferred_key: deferred,
        },
        "requirement_coverage": {
            active_key: coverage.model_dump(mode="json"),
        },
        "requirement_dependency_state": {
            active_key: {"eligible": True, "blocking_dependencies": {}},
            blocked_key: {
                "eligible": False,
                "blocking_dependencies": {"fund_release": "UNRESOLVED"},
            },
        },
        "open_inquiries": [
            {
                "id": selected_id,
                "source": "REQUIREMENT",
                "scope": SCOPE.value,
                "topic": DiscoveryTopic.BUSINESS_RULES.value,
                "objective": active.description,
                "question_hint": "Ask about the unresolved requirement.",
                "reason": "The dispute outcome is unresolved.",
                "requirement_id": active.id,
                "target_facets": ["resolution_outcome"],
            },
            {
                "id": f"{SCOPE.value}|model.completion",
                "source": "MODEL",
                "scope": SCOPE.value,
                "topic": DiscoveryTopic.CORE_WORKFLOW.value,
                "objective": "Understand what makes the transaction complete.",
                "question_hint": "Ask about completion.",
                "reason": "Completion is not established.",
            },
        ],
        "selected_inquiry": {
            "id": selected_id,
            "requirement_id": active.id,
            "target_facets": ["resolution_outcome"],
        },
        "validation_issues": [],
        "discovery_boundaries": [
            {
                "type": "implementation_deferred",
                "scope": SCOPE.value,
                "source_turn": 4,
                "evidence": "leave that to engineering",
                "objective": "Choose the internal payment confirmation mechanism.",
            }
        ],
    })

    sections = section_map(projection)
    open_states = {item.label: item.state.value for item in sections["open_decisions"].items}
    assert open_states["Dispute resolution"] == "active"
    assert open_states["Seller payout timing"] == "blocked"
    assert open_states["Understand what makes the transaction complete."] == "open"

    deferred_labels = {item.label for item in sections["deferred_decisions"].items}
    assert "Delivery confirmation interaction" in deferred_labels
    assert "Choose the internal payment confirmation mechanism." in deferred_labels


def test_projection_keeps_confirmed_workflow_as_one_fact_instead_of_splitting_it():
    workflow = fact(
        DiscoveryTopic.CORE_WORKFLOW,
        "workflow_steps",
        "buyer pays into escrow, seller delivers, buyer confirms completion",
    )
    projection = build_understanding_projection({
        "discovery_scope": SCOPE,
        "discovered_knowledge": [workflow],
        "product_concepts": [],
        "active_requirements": {},
        "requirement_coverage": {},
        "requirement_dependency_state": {},
        "open_inquiries": [],
        "validation_issues": [],
        "discovery_boundaries": [],
    })

    items = section_map(projection)["workflow"].items
    assert len(items) == 1
    assert items[0].label == workflow.value
    assert items[0].source_refs == [fact_id(workflow)]
