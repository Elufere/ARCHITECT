"""Focused tests for deterministic requirement dependency resolution."""
from agents.requirement_dependencies import (
    DependencyBlockReason,
    eligible_requirement_keys,
    requirement_dependency_node,
    resolve_requirement_dependencies,
)
from agents.requirements import (
    ActiveRequirement,
    RequirementStatus,
    register_requirement,
    requirement_store_key,
)
from agents.state import DiscoveryScope as S, DiscoveryTopic as T


def req(
    requirement_id,
    *,
    status=RequirementStatus.ACTIVE,
    dependencies=(),
    scope=S.USER_APP,
):
    return ActiveRequirement(
        id=requirement_id,
        scope=scope,
        topic=T.BUSINESS_RULES,
        label=requirement_id,
        status=status,
        dependencies=list(dependencies),
    )


def store_of(*requirements):
    store = {}
    for requirement in requirements:
        store = register_requirement(store, requirement)
    return store


def test_active_requirement_without_dependencies_is_eligible():
    store = store_of(req("payment.provider"))
    decisions = resolve_requirement_dependencies(store, S.USER_APP)
    assert decisions["payment.provider"].eligible is True
    assert eligible_requirement_keys(store, S.USER_APP, decisions) == [
        requirement_store_key(S.USER_APP, "payment.provider")
    ]


def test_active_dependency_blocks_until_resolved():
    store = store_of(
        req("payment.provider"),
        req("payment.settlement", dependencies=("payment.provider",)),
    )
    decisions = resolve_requirement_dependencies(store, S.USER_APP)
    assert decisions["payment.provider"].eligible is True
    settlement = decisions["payment.settlement"]
    assert settlement.eligible is False
    assert settlement.blocking_dependencies == {
        "payment.provider": DependencyBlockReason.UNRESOLVED
    }


def test_resolved_dependency_unlocks_requirement():
    store = store_of(
        req("payment.provider", status=RequirementStatus.RESOLVED),
        req("payment.settlement", dependencies=("payment.provider",)),
    )
    decisions = resolve_requirement_dependencies(store, S.USER_APP)
    assert decisions["payment.settlement"].eligible is True
    assert decisions["payment.provider"].eligible is False


def test_dependency_chain_exposes_only_current_frontier():
    store = store_of(
        req("payment.provider"),
        req("payment.settlement", dependencies=("payment.provider",)),
        req("payment.failure", dependencies=("payment.settlement",)),
    )
    decisions = resolve_requirement_dependencies(store, S.USER_APP)
    eligible = {
        requirement_id
        for requirement_id, decision in decisions.items()
        if decision.eligible
    }
    assert eligible == {"payment.provider"}
    assert decisions["payment.settlement"].blocking_dependencies["payment.provider"] == DependencyBlockReason.UNRESOLVED
    assert decisions["payment.failure"].blocking_dependencies["payment.settlement"] == DependencyBlockReason.UNRESOLVED


def test_missing_dependency_fails_closed():
    store = store_of(req("payment.failure", dependencies=("payment.settlement",)))
    decision = resolve_requirement_dependencies(store, S.USER_APP)["payment.failure"]
    assert decision.eligible is False
    assert decision.blocking_dependencies["payment.settlement"] == DependencyBlockReason.MISSING


def test_terminal_and_nonready_dependency_statuses_block_explicitly():
    cases = (
        (RequirementStatus.DEFERRED, DependencyBlockReason.DEFERRED),
        (RequirementStatus.INACTIVE, DependencyBlockReason.INACTIVE),
        (RequirementStatus.NOT_APPLICABLE, DependencyBlockReason.NOT_APPLICABLE),
        (RequirementStatus.ACTIVE, DependencyBlockReason.UNRESOLVED),
    )
    for status, reason in cases:
        store = store_of(
            req("upstream", status=status),
            req("downstream", dependencies=("upstream",)),
        )
        decision = resolve_requirement_dependencies(store, S.USER_APP)["downstream"]
        assert decision.eligible is False
        assert decision.blocking_dependencies["upstream"] == reason


def test_same_requirement_id_in_other_scope_does_not_satisfy_dependency():
    store = store_of(
        req("payment.provider", status=RequirementStatus.RESOLVED, scope=S.ADMIN_DASHBOARD),
        req("payment.settlement", dependencies=("payment.provider",), scope=S.USER_APP),
    )
    decision = resolve_requirement_dependencies(store, S.USER_APP)["payment.settlement"]
    assert decision.eligible is False
    assert decision.blocking_dependencies["payment.provider"] == DependencyBlockReason.MISSING


def test_cycle_nodes_are_blocked_without_crashing():
    store = store_of(
        req("a", dependencies=("b",)),
        req("b", dependencies=("a",)),
    )
    decisions = resolve_requirement_dependencies(store, S.USER_APP)
    assert decisions["a"].eligible is False
    assert decisions["b"].eligible is False
    assert DependencyBlockReason.CYCLE in decisions["a"].blocking_dependencies.values()
    assert DependencyBlockReason.CYCLE in decisions["b"].blocking_dependencies.values()


def test_self_dependency_is_a_cycle():
    store = store_of(req("a", dependencies=("a",)))
    decision = resolve_requirement_dependencies(store, S.USER_APP)["a"]
    assert decision.eligible is False
    assert decision.blocking_dependencies["a"] == DependencyBlockReason.CYCLE


def test_non_active_requirement_is_never_eligible_even_without_dependencies():
    for status in (
        RequirementStatus.RESOLVED,
        RequirementStatus.DEFERRED,
        RequirementStatus.NOT_APPLICABLE,
        RequirementStatus.INACTIVE,
    ):
        store = store_of(req("terminal", status=status))
        decision = resolve_requirement_dependencies(store, S.USER_APP)["terminal"]
        assert decision.eligible is False


def test_dependency_node_emits_scoped_keys_and_diagnostics_only():
    store = store_of(
        req("provider", status=RequirementStatus.RESOLVED),
        req("settlement", dependencies=("provider",)),
    )
    state = {
        "discovery_scope": S.USER_APP,
        "active_requirements": store,
    }
    result = requirement_dependency_node(state)
    settlement_key = requirement_store_key(S.USER_APP, "settlement")
    provider_key = requirement_store_key(S.USER_APP, "provider")
    assert result["eligible_requirement_keys"] == [settlement_key]
    assert settlement_key in result["requirement_dependency_state"]
    assert provider_key in result["requirement_dependency_state"]
    assert "active_requirements" not in result
