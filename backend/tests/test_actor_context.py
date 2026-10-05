from langchain_core.messages import AIMessage, HumanMessage

from agents.actor_context import (
    actor_registry,
    canonical_actor_for_label,
    resolve_owned_claim_role,
)
from agents.extraction_passes import NeutralClaim
from agents.knowledge_tracker import _admit_claim_item, _aliases_from_actor_claim
from agents.state import (
    DiscoveryScope,
    DiscoveryTopic,
    KnowledgeItem,
    KnowledgeState,
)


SCOPE = DiscoveryScope.USER_APP


def actor(role: str, *, aliases=None, key="primary_users", turn=1):
    return KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=SCOPE,
        key=key,
        value=f"{role} uses the app",
        evidence=f"{role} uses the app",
        roles=[role],
        aliases=aliases,
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
    )


def role_policy(*, turn=2):
    return KnowledgeItem(
        topic=DiscoveryTopic.USER_ROLES,
        scope=SCOPE,
        key="multiple_roles",
        value="A customer can be a buyer in one transaction and a seller in another.",
        evidence="A customer can be a buyer in one transaction and a seller in another.",
        roles=["customer"],
        aliases={"customer": ["buyer", "seller"]},
        confidence=1,
        knowledge_state=KnowledgeState.CONFIRMED,
        source_turn=turn,
    )


def state_with(*facts, messages=None, **updates):
    state = {
        "discovery_scope": SCOPE,
        "discovered_knowledge": list(facts),
        "messages": list(messages or []),
        "current_gap": None,
        "current_role": None,
        "selected_inquiry": None,
        "selected_requirement_candidate": None,
        "thread_frontier": None,
        "turn_count": 3,
    }
    state.update(updates)
    return state


def test_role_policy_aliases_extend_confirmed_actor_identity():
    state = state_with(actor("customer"), role_policy())

    registry = actor_registry(state, SCOPE)

    assert set(registry) == {"customer"}
    assert registry["customer"].labels == {"customer", "buyer", "seller"}
    assert canonical_actor_for_label("buyer", state, SCOPE) == "customer"
    assert canonical_actor_for_label("seller", state, SCOPE) == "customer"


def test_explicit_capacity_label_resolves_to_canonical_actor():
    state = state_with(actor("customer"), role_policy())

    assert (
        resolve_owned_claim_role(
            "seller",
            "The seller marks the item as shipped.",
            state,
            SCOPE,
        )
        == "customer"
    )


def test_pronoun_resolves_from_single_actor_alias_in_previous_question():
    state = state_with(
        actor("customer"),
        role_policy(),
        messages=[
            AIMessage(content="What should the seller do after shipping the item?"),
            HumanMessage(content="They should mark it as shipped."),
        ],
    )

    assert (
        resolve_owned_claim_role(
            None,
            "They should mark it as shipped.",
            state,
            SCOPE,
        )
        == "customer"
    )


def test_pronoun_resolves_from_explicit_active_role_focus():
    state = state_with(
        actor("customer"),
        actor("merchant", key="secondary_users"),
        messages=[
            AIMessage(content="What can they do?"),
            HumanMessage(content="They can update the order."),
        ],
        current_gap="responsibilities::merchant",
        current_role="merchant",
    )

    assert (
        resolve_owned_claim_role(
            None,
            "They can update the order.",
            state,
            SCOPE,
        )
        == "merchant"
    )


def test_ambiguous_pronoun_does_not_guess_between_distinct_actors():
    state = state_with(
        actor("buyer"),
        actor("seller", key="secondary_users"),
        messages=[
            AIMessage(content="What happens after both parties agree?"),
            HumanMessage(content="They confirm it."),
        ],
    )

    assert (
        resolve_owned_claim_role(
            None,
            "They confirm it.",
            state,
            SCOPE,
        )
        is None
    )


def test_unknown_role_label_does_not_create_actor_membership():
    state = state_with(actor("customer"), role_policy())

    assert (
        resolve_owned_claim_role(
            "recipient",
            "The recipient receives a notification.",
            state,
            SCOPE,
        )
        is None
    )


def test_transaction_capacity_phrase_recovery_handles_split_conditions():
    aliases = _aliases_from_actor_claim(
        "A customer can be a buyer in one transaction and a seller in another."
    )

    assert aliases == ["buyer", "seller"]


def test_multiple_roles_claim_persists_capacity_aliases():
    evidence = "A customer can be a buyer in one transaction and a seller in another."
    state = state_with(
        actor("customer"),
        messages=[HumanMessage(content=evidence)],
        current_gap="multiple_roles",
    )
    claim = NeutralClaim(
        kind="multiple_roles",
        value=evidence,
        evidence=evidence,
        role="customer",
        aliases=["buyer", "seller"],
        confidence=1,
    )

    item = _admit_claim_item(
        claim,
        state,
        SCOPE,
        primary_roles={"customer"},
        secondary_roles=set(),
    )

    assert item is not None
    assert item.key == "multiple_roles"
    assert item.roles == ["customer"]
    assert item.aliases == {"customer": ["buyer", "seller"]}


def test_owned_claim_with_capacity_alias_is_committed_to_canonical_actor():
    evidence = "The buyer pays into escrow."
    state = state_with(
        actor("customer"),
        role_policy(),
        messages=[HumanMessage(content=evidence)],
        current_gap="responsibilities::customer",
    )
    claim = NeutralClaim(
        kind="actor_action",
        value="The buyer pays into escrow.",
        evidence=evidence,
        role="buyer",
        confidence=1,
    )

    item = _admit_claim_item(
        claim,
        state,
        SCOPE,
        primary_roles={"customer"},
        secondary_roles=set(),
    )

    assert item is not None
    assert item.role == "customer"
    assert item.key == "responsibilities"



def test_descriptive_reference_uses_previous_alias_context_without_new_actor():
    state = state_with(
        actor("customer"),
        role_policy(),
        messages=[
            AIMessage(content="What should happen after the seller sends the item?"),
            HumanMessage(content="The person receiving it should confirm delivery."),
        ],
    )

    assert (
        resolve_owned_claim_role(
            None,
            "The person receiving it should confirm delivery.",
            state,
            SCOPE,
        )
        == "customer"
    )
