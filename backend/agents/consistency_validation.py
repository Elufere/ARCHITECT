"""Formal validation of confirmed product facts and requirement dependency state."""
from __future__ import annotations

from enum import Enum
from itertools import combinations
import json
from typing import Dict, Iterable, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agents.discovery_coverage import fact_id
from agents.extraction_passes import canonical_role
from agents.llm import get_structured_model
from agents.llm_errors import ExtractionFailed, raise_if_llm_failure
from agents.requirement_dependencies import DependencyBlockReason
from agents.requirements import RequirementStatus, requirement_store_key
from agents.state import AgentState, DiscoveryScope, DiscoveryTopic, KnowledgeItem, KnowledgeState


class ValidationIssueKind(str, Enum):
    FACT_CONTRADICTION = "FACT_CONTRADICTION"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    DEPENDENCY_CYCLE = "DEPENDENCY_CYCLE"
    RESOLVED_REQUIREMENT_BLOCKED = "RESOLVED_REQUIREMENT_BLOCKED"
    REQUIREMENT_COVERAGE_MISMATCH = "REQUIREMENT_COVERAGE_MISMATCH"


class ValidationIssueSeverity(str, Enum):
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"


class ValidationResolution(str, Enum):
    USER_CLARIFICATION = "USER_CLARIFICATION"
    SYSTEM_CONFIGURATION = "SYSTEM_CONFIGURATION"
    RECOMPUTE_STATE = "RECOMPUTE_STATE"


class DiscoveryValidationIssue(BaseModel):
    id: str
    kind: ValidationIssueKind
    severity: ValidationIssueSeverity
    resolution: ValidationResolution
    scope: DiscoveryScope
    message: str
    topic: DiscoveryTopic | None = None
    key: str | None = None
    fact_ids: List[str] = Field(default_factory=list)
    fact_values: List[str] = Field(default_factory=list)
    requirement_key: str | None = None
    related_requirements: List[str] = Field(default_factory=list)
    metadata: Dict[str, object] = Field(default_factory=dict)


class FactConflictVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str
    contradiction: bool
    confidence: float = Field(ge=0, le=1)
    explanation: str


class FactConflictBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: List[FactConflictVerdict]


FACT_CONFLICT_INSTRUCTION = """Review supplied pairs of confirmed product facts for contradiction.

A contradiction exists only when BOTH assertions cannot simultaneously be true
under their explicitly stated scope, actor, timing, conditions, quantities, and
exceptions.

Do NOT label as contradiction when:
- the facts apply under different conditions or lifecycle states,
- one is merely more specific,
- both are independent rules that can coexist,
- wording differs but the assertions are compatible,
- the pair is ambiguous.

Examples:
- "buyers may cancel before approval" and "buyers cannot cancel after approval"
  are compatible.
- "maximum 5 active requests" and "maximum 10 active requests" for the same actor
  and same conditions are contradictory unless one is explicitly historical or
  scoped differently.
- "payments are required in MVP" and "payments are out of scope for MVP" are
  contradictory.

Use only the supplied facts. Do not use outside domain assumptions.
Return exactly one verdict for every pair_id. When uncertain, contradiction=false.
"""


# Cross-field pairs where mutually exclusive product decisions can be expressed
# under different schema keys. The semantic reviewer still decides whether the
# actual assertions conflict.
ADDITIVE_FACT_FIELDS = {
    (DiscoveryTopic.USER_ROLES, "responsibilities"),
    (DiscoveryTopic.CORE_WORKFLOW, "workflow_steps"),
}

CROSS_FIELD_CONFLICT_PAIRS = {
    frozenset({
        (DiscoveryTopic.MVP_SCOPE, "must_have_features"),
        (DiscoveryTopic.MVP_SCOPE, "out_of_scope"),
    }),
    frozenset({
        (DiscoveryTopic.MVP_SCOPE, "nice_to_have_features"),
        (DiscoveryTopic.MVP_SCOPE, "out_of_scope"),
    }),
}


_conflict_model = None


def conflict_model():
    global _conflict_model
    if _conflict_model is None:
        _conflict_model = get_structured_model(
            call_name="consistency.fact_conflicts",
            schema=FactConflictBatch,
        )
    return _conflict_model


def _fact_owner(item: KnowledgeItem) -> tuple:
    return (
        canonical_role(item.role or ""),
        tuple(sorted(canonical_role(role) for role in item.roles or [])),
    )


def _same_semantic_bucket(first: KnowledgeItem, second: KnowledgeItem) -> bool:
    if not (
        first.scope == second.scope
        and first.topic == second.topic
        and first.key == second.key
        and first.knowledge_state == second.knowledge_state == KnowledgeState.CONFIRMED
    ):
        return False
    if (first.topic, first.key) in ADDITIVE_FACT_FIELDS:
        # Independent actions/steps are cumulative. Corrections are handled
        # during reconciliation; pairwise contradiction review here creates
        # quadratic model calls without useful signal.
        return False
    if first.topic == DiscoveryTopic.USER_ROLES and first.key in {"primary_users", "secondary_users"}:
        # Actor declarations are field-level membership assertions. Different
        # role lists may be compatible additions or incompatible exclusivity
        # claims; let the conservative semantic reviewer decide.
        return True
    return _fact_owner(first) == _fact_owner(second)


def _configured_cross_field_pair(first: KnowledgeItem, second: KnowledgeItem) -> bool:
    if first.scope != second.scope:
        return False
    if first.knowledge_state != KnowledgeState.CONFIRMED or second.knowledge_state != KnowledgeState.CONFIRMED:
        return False
    if first.role or second.role:
        if canonical_role(first.role or "") != canonical_role(second.role or ""):
            return False
    fields = frozenset({(first.topic, first.key), (second.topic, second.key)})
    return fields in CROSS_FIELD_CONFLICT_PAIRS


def _pair_signature(first: KnowledgeItem, second: KnowledgeItem) -> str:
    identities = sorted((fact_id(first), fact_id(second)))
    return "semantic-v1|" + "|".join(identities)


def _pair_id(first: KnowledgeItem, second: KnowledgeItem) -> str:
    identities = sorted((fact_id(first), fact_id(second)))
    return identities[0][:12] + ":" + identities[1][:12]


def _deterministic_absence_conflicts(
    knowledge: Sequence[KnowledgeItem],
    scope: DiscoveryScope,
) -> List[DiscoveryValidationIssue]:
    confirmed = [
        item for item in knowledge
        if item.scope == scope and item.knowledge_state == KnowledgeState.CONFIRMED
    ]
    issues = []
    for absence in confirmed:
        if not absence.absence:
            continue
        for positive in confirmed:
            if positive is absence or positive.absence:
                continue
            if (
                positive.topic == absence.topic
                and positive.key == absence.key
                and canonical_role(positive.role or "") == canonical_role(absence.role or "")
            ):
                ids = sorted((fact_id(absence), fact_id(positive)))
                issue_id = "fact-absence|" + "|".join(ids)
                issues.append(DiscoveryValidationIssue(
                    id=issue_id,
                    kind=ValidationIssueKind.FACT_CONTRADICTION,
                    severity=ValidationIssueSeverity.BLOCKING,
                    resolution=ValidationResolution.USER_CLARIFICATION,
                    scope=scope,
                    topic=absence.topic,
                    key=absence.key,
                    fact_ids=ids,
                    fact_values=[absence.value, positive.value],
                    message=(
                        f"Confirmed {absence.topic.value}.{absence.key} contains both "
                        "a whole-field absence and a positive assertion."
                    ),
                    metadata={"detector": "whole_field_absence"},
                ))
    unique = {}
    for issue in issues:
        unique[issue.id] = issue
    return list(unique.values())


def _semantic_pair_candidates(
    knowledge: Sequence[KnowledgeItem],
    scope: DiscoveryScope,
) -> List[tuple[KnowledgeItem, KnowledgeItem]]:
    confirmed = [
        item for item in knowledge
        if item.scope == scope
        and item.knowledge_state == KnowledgeState.CONFIRMED
        and not item.absence
    ]
    pairs = []
    for first, second in combinations(confirmed, 2):
        if first.value.strip().lower() == second.value.strip().lower():
            continue
        if _same_semantic_bucket(first, second) or _configured_cross_field_pair(first, second):
            pairs.append((first, second))
    return pairs


def _review_unknown_pairs(
    pairs: Sequence[tuple[KnowledgeItem, KnowledgeItem]],
    cache: Dict[str, dict],
) -> Dict[str, dict]:
    updated = dict(cache)
    unknown = [
        pair for pair in pairs
        if _pair_signature(*pair) not in updated
    ]
    # Keep model payloads bounded. Every unknown pair is still reviewed, in
    # deterministic batches, and cached by immutable fact IDs.
    for offset in range(0, len(unknown), 20):
        batch = unknown[offset:offset + 20]
        payload = [
            {
                "pair_id": _pair_id(first, second),
                "first": first.model_dump(mode="json"),
                "second": second.model_dump(mode="json"),
            }
            for first, second in batch
        ]
        try:
            result = conflict_model().invoke([
                SystemMessage(content=FACT_CONFLICT_INSTRUCTION),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])
            if not isinstance(result, FactConflictBatch):
                result = FactConflictBatch.model_validate(result)
        except Exception as exc:
            raise_if_llm_failure(exc)
            raise ExtractionFailed("Fact consistency review failed") from exc

        by_pair_id = {item.pair_id: item for item in result.verdicts}
        expected = {_pair_id(first, second) for first, second in batch}
        if set(by_pair_id) != expected:
            raise ExtractionFailed("Fact consistency review omitted or invented pair verdicts")

        for first, second in batch:
            verdict = by_pair_id[_pair_id(first, second)]
            updated[_pair_signature(first, second)] = verdict.model_dump(mode="json")
    return updated


def _semantic_fact_issues(
    knowledge: Sequence[KnowledgeItem],
    scope: DiscoveryScope,
    cache: Dict[str, dict],
) -> tuple[List[DiscoveryValidationIssue], Dict[str, dict]]:
    pairs = _semantic_pair_candidates(knowledge, scope)
    updated_cache = _review_unknown_pairs(pairs, cache)
    issues = []

    for first, second in pairs:
        verdict = FactConflictVerdict.model_validate(
            updated_cache[_pair_signature(first, second)]
        )
        if not verdict.contradiction or verdict.confidence < 0.95:
            continue
        ids = sorted((fact_id(first), fact_id(second)))
        same_field = first.topic == second.topic and first.key == second.key
        topic = first.topic if same_field else None
        key = first.key if same_field else None
        issues.append(DiscoveryValidationIssue(
            id="fact-semantic|" + "|".join(ids),
            kind=ValidationIssueKind.FACT_CONTRADICTION,
            severity=ValidationIssueSeverity.BLOCKING,
            resolution=ValidationResolution.USER_CLARIFICATION,
            scope=scope,
            topic=topic,
            key=key,
            fact_ids=ids,
            fact_values=[first.value, second.value],
            message=verdict.explanation or "Confirmed product facts are incompatible.",
            metadata={
                "detector": "semantic_pair",
                "first_field": f"{first.topic.value}.{first.key}",
                "second_field": f"{second.topic.value}.{second.key}",
            },
        ))
    return issues, updated_cache


def _dependency_issues(state: AgentState, scope: DiscoveryScope) -> List[DiscoveryValidationIssue]:
    store = state.get("active_requirements", {})
    dependency_state = state.get("requirement_dependency_state", {})
    issues = []

    for key, requirement in store.items():
        if requirement.scope != scope:
            continue
        decision = dependency_state.get(key) or {}
        blocking = decision.get("blocking_dependencies", {})

        for dependency_id, raw_reason in blocking.items():
            reason = DependencyBlockReason(raw_reason)
            if reason == DependencyBlockReason.MISSING:
                issues.append(DiscoveryValidationIssue(
                    id=f"dependency-missing|{key}|{dependency_id}",
                    kind=ValidationIssueKind.DEPENDENCY_MISSING,
                    severity=ValidationIssueSeverity.BLOCKING,
                    resolution=ValidationResolution.SYSTEM_CONFIGURATION,
                    scope=scope,
                    requirement_key=key,
                    related_requirements=[dependency_id],
                    message=(
                        f"Requirement '{requirement.id}' depends on missing requirement "
                        f"'{dependency_id}'."
                    ),
                ))
            elif reason == DependencyBlockReason.CYCLE:
                issues.append(DiscoveryValidationIssue(
                    id=f"dependency-cycle|{key}|{dependency_id}",
                    kind=ValidationIssueKind.DEPENDENCY_CYCLE,
                    severity=ValidationIssueSeverity.BLOCKING,
                    resolution=ValidationResolution.SYSTEM_CONFIGURATION,
                    scope=scope,
                    requirement_key=key,
                    related_requirements=[dependency_id],
                    message=(
                        f"Requirement '{requirement.id}' participates in a dependency cycle "
                        f"through '{dependency_id}'."
                    ),
                ))

        # The normal resolver short-circuits non-ACTIVE requirements. Validate
        # resolved requirements independently so an upstream regression cannot
        # leave a stale downstream result silently trusted.
        if requirement.status == RequirementStatus.RESOLVED:
            by_id = {
                item.id: item
                for item in store.values()
                if item.scope == scope
            }
            unresolved = []
            for dependency_id in requirement.dependencies:
                dependency = by_id.get(dependency_id)
                if dependency is None or dependency.status != RequirementStatus.RESOLVED:
                    unresolved.append(dependency_id)
            if unresolved:
                issues.append(DiscoveryValidationIssue(
                    id=f"resolved-blocked|{key}|" + ",".join(sorted(unresolved)),
                    kind=ValidationIssueKind.RESOLVED_REQUIREMENT_BLOCKED,
                    severity=ValidationIssueSeverity.BLOCKING,
                    resolution=ValidationResolution.RECOMPUTE_STATE,
                    scope=scope,
                    requirement_key=key,
                    related_requirements=sorted(unresolved),
                    message=(
                        f"Resolved requirement '{requirement.id}' has dependencies that are "
                        "no longer resolved."
                    ),
                ))
    return issues


def _coverage_consistency_issues(state: AgentState, scope: DiscoveryScope) -> List[DiscoveryValidationIssue]:
    store = state.get("active_requirements", {})
    coverage = state.get("requirement_coverage", {})
    issues = []
    for key, requirement in store.items():
        if requirement.scope != scope:
            continue
        payload = coverage.get(key)
        if not payload:
            continue
        coverage_status = payload.get("status")
        mismatch = (
            requirement.status == RequirementStatus.RESOLVED
            and coverage_status != "RESOLVED"
        ) or (
            requirement.status == RequirementStatus.ACTIVE
            and coverage_status == "RESOLVED"
        )
        if mismatch:
            issues.append(DiscoveryValidationIssue(
                id=f"coverage-mismatch|{key}|{requirement.status.value}|{coverage_status}",
                kind=ValidationIssueKind.REQUIREMENT_COVERAGE_MISMATCH,
                severity=ValidationIssueSeverity.BLOCKING,
                resolution=ValidationResolution.RECOMPUTE_STATE,
                scope=scope,
                requirement_key=key,
                message=(
                    f"Requirement '{requirement.id}' status '{requirement.status.value}' "
                    f"does not match coverage status '{coverage_status}'."
                ),
            ))
    return issues


def validate_discovery_consistency(
    state: AgentState,
) -> tuple[List[DiscoveryValidationIssue], Dict[str, dict]]:
    scope = state.get("discovery_scope", DiscoveryScope.USER_APP)
    knowledge = state.get("discovered_knowledge", [])
    cache = state.get("validation_pair_cache", {})

    deterministic = _deterministic_absence_conflicts(knowledge, scope)
    semantic, updated_cache = _semantic_fact_issues(knowledge, scope, cache)
    dependency = _dependency_issues(state, scope)
    coverage = _coverage_consistency_issues(state, scope)

    issues = deterministic + semantic + dependency + coverage
    unique = {issue.id: issue for issue in issues}
    ordered = sorted(
        unique.values(),
        key=lambda issue: (
            0 if issue.severity == ValidationIssueSeverity.BLOCKING else 1,
            issue.kind.value,
            issue.id,
        ),
    )
    return ordered, updated_cache


def consistency_validation_node(state: AgentState) -> dict:
    issues, cache = validate_discovery_consistency(state)
    compile_blocking = any(
        issue.severity == ValidationIssueSeverity.BLOCKING
        for issue in issues
    )
    candidate_blocking_kinds = {
        ValidationIssueKind.FACT_CONTRADICTION,
        ValidationIssueKind.DEPENDENCY_MISSING,
        ValidationIssueKind.DEPENDENCY_CYCLE,
        ValidationIssueKind.REQUIREMENT_COVERAGE_MISMATCH,
    }
    return {
        "validation_issues": [issue.model_dump(mode="json") for issue in issues],
        "validation_pair_cache": cache,
        "validation_blocking": compile_blocking,
        "validation_candidate_blocking": any(
            issue.severity == ValidationIssueSeverity.BLOCKING
            and issue.kind in candidate_blocking_kinds
            for issue in issues
        ),
    }
