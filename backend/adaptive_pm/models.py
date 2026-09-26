from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class KnowledgeStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DEFERRED = "DEFERRED"


class FactSource(str, Enum):
    USER = "USER"
    PM_INFERENCE = "PM_INFERENCE"
    PM_RECOMMENDATION = "PM_RECOMMENDATION"


class ConversationIntent(str, Enum):
    PRODUCT_INFORMATION = "PRODUCT_INFORMATION"
    CORRECTION = "CORRECTION"
    CLARIFICATION = "CLARIFICATION"
    RATIONALE_REQUEST = "RATIONALE_REQUEST"
    ADVICE_REQUEST = "ADVICE_REQUEST"
    UNCERTAINTY = "UNCERTAINTY"
    OBJECTION = "OBJECTION"
    DESIGN_DEFERRAL = "DESIGN_DEFERRAL"
    CONFIRMATION = "CONFIRMATION"


class BoundaryKind(str, Enum):
    REJECTED_INQUIRY = "REJECTED_INQUIRY"
    DESIGN_DEFERRAL = "DESIGN_DEFERRAL"
    DEFERRED_DECISION = "DEFERRED_DECISION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class CapturedFact(BaseModel):
    id: str = Field(description="Turn-local stable identifier, e.g. fact_1")
    statement: str
    evidence: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    entities: list[str] = Field(default_factory=list)
    negative: bool = False


class BoundaryProposal(BaseModel):
    kind: BoundaryKind
    evidence: str
    subject: str
    instruction: str


class TurnCapture(BaseModel):
    intent: ConversationIntent = ConversationIntent.PRODUCT_INFORMATION
    facts: list[CapturedFact] = Field(default_factory=list)
    boundary: BoundaryProposal | None = None
    confirms_previous_answer: bool = False
    correction_targets: list[str] = Field(default_factory=list)


class GroundingVerdict(BaseModel):
    fact_id: str
    supported: bool
    reason: str


class GroundingAudit(BaseModel):
    verdicts: list[GroundingVerdict] = Field(default_factory=list)


class Observation(BaseModel):
    id: str
    statement: str
    evidence: str
    source: FactSource = FactSource.USER
    source_turn: int
    confidence: float = Field(default=1.0, ge=0, le=1)
    entities: list[str] = Field(default_factory=list)
    negative: bool = False


class KnowledgeRecord(BaseModel):
    key: str
    statement: str
    category: str
    status: KnowledgeStatus = KnowledgeStatus.CONFIRMED
    evidence_observation_ids: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    affected_requirement_ids: list[str] = Field(default_factory=list)
    first_seen_turn: int = 0
    last_updated_turn: int = 0


class ProductConcept(BaseModel):
    id: str
    kind: Literal["ACTOR", "ENTITY", "RELATIONSHIP", "STATE", "WORKFLOW", "RULE", "INTEGRATION", "OTHER"]
    name: str
    description: str
    evidence_observation_ids: list[str] = Field(default_factory=list)


class RequirementRecord(BaseModel):
    id: str
    label: str
    category: str
    status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    coverage: float = Field(default=0.0, ge=0, le=1)
    depth: float = Field(default=0.0, ge=0, le=1)
    known: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    evidence_knowledge_keys: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    unlocks: list[str] = Field(default_factory=list)
    high_impact: bool = False
    defer_reason: str | None = None


class ImplicationRecord(BaseModel):
    id: str
    statement: str
    source_knowledge_keys: list[str]
    affected_requirement_ids: list[str] = Field(default_factory=list)
    status: KnowledgeStatus = KnowledgeStatus.PROPOSED


class ContradictionRecord(BaseModel):
    id: str
    knowledge_keys: list[str]
    description: str
    blocking: bool = True
    resolved: bool = False


class DiscoveryBoundary(BaseModel):
    id: str
    kind: BoundaryKind
    subject: str
    instruction: str
    evidence: str
    source_turn: int
    decision_key: str | None = None


class DecisionRecord(BaseModel):
    decision_key: str
    requirement_ids: list[str] = Field(default_factory=list)
    question: str
    asked_turn: int
    resolution: Literal["ANSWERED", "DEFERRED", "REJECTED", "OPEN"] = "OPEN"
    answer_summary: str | None = None


class KnowledgeMutation(BaseModel):
    action: Literal["ADD", "REFINE", "SUPERSEDE", "REJECT", "DEFER"]
    key: str
    statement: str
    category: str
    observation_ids: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    affected_requirement_ids: list[str] = Field(default_factory=list)
    replaces_key: str | None = None
    status: KnowledgeStatus = KnowledgeStatus.CONFIRMED


class RequirementMutation(BaseModel):
    id: str
    label: str
    category: str
    status: KnowledgeStatus
    coverage: float = Field(ge=0, le=1)
    depth: float = Field(ge=0, le=1)
    known: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    evidence_knowledge_keys: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    unlocks: list[str] = Field(default_factory=list)
    high_impact: bool = False
    defer_reason: str | None = None


class ReasoningUpdate(BaseModel):
    knowledge_mutations: list[KnowledgeMutation] = Field(default_factory=list)
    concepts: list[ProductConcept] = Field(default_factory=list)
    requirement_updates: list[RequirementMutation] = Field(default_factory=list)
    implications: list[ImplicationRecord] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)
    resolved_decision_keys: list[str] = Field(default_factory=list)


class QuestionCandidate(BaseModel):
    id: str
    decision_key: str
    question: str
    requirement_ids: list[str] = Field(default_factory=list)
    uncertainty: str
    why_now: str
    business_impact: float = Field(ge=0, le=1)
    architecture_impact: float = Field(ge=0, le=1)
    dependency_unlock: float = Field(ge=0, le=1)
    uncertainty_reduction: float = Field(ge=0, le=1)
    risk_reduction: float = Field(ge=0, le=1)
    contextual_relevance: float = Field(ge=0, le=1)
    repetition_penalty: float = Field(default=0.0, ge=0, le=1)
    premature_detail_penalty: float = Field(default=0.0, ge=0, le=1)
    fatigue_penalty: float = Field(default=0.0, ge=0, le=1)
    eligible: bool = True
    ineligibility_reason: str | None = None

    def score(self) -> float:
        positive = (
            self.business_impact
            + self.architecture_impact
            + self.dependency_unlock
            + self.uncertainty_reduction
            + self.risk_reduction
            + self.contextual_relevance
        )
        penalties = self.repetition_penalty + self.premature_detail_penalty + self.fatigue_penalty
        return positive - penalties


class CompletionAssessment(BaseModel):
    complete: bool
    core_product_model_coherent: bool
    major_entities_understood: bool
    primary_workflows_understood: bool
    high_impact_rules_understood: bool
    architecture_changing_unknowns_resolved_or_deferred: bool
    transactional_mechanics_understood: bool
    lifecycle_understood: bool
    major_failure_paths_addressed: bool
    mvp_boundaries_clear: bool
    reason: str


class PlanningResult(BaseModel):
    candidates: list[QuestionCandidate] = Field(default_factory=list)
    completion: CompletionAssessment
    advice_options: list[str] = Field(default_factory=list)


class QuestionAudit(BaseModel):
    passed: bool
    atomic: bool
    founder_friendly: bool
    relevant: bool
    non_repetitive: bool
    respects_boundaries: bool
    implementation_detail: bool
    reason: str
    revised_question: str | None = None
    reject_candidate: bool = False


class InterviewState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_idea: str = ""
    turn_count: int = 0
    last_question: str | None = None
    last_decision_key: str | None = None
    observations: list[Observation] = Field(default_factory=list)
    knowledge: dict[str, KnowledgeRecord] = Field(default_factory=dict)
    concepts: dict[str, ProductConcept] = Field(default_factory=dict)
    requirements: dict[str, RequirementRecord] = Field(default_factory=dict)
    implications: dict[str, ImplicationRecord] = Field(default_factory=dict)
    contradictions: dict[str, ContradictionRecord] = Field(default_factory=dict)
    boundaries: list[DiscoveryBoundary] = Field(default_factory=list)
    decisions: dict[str, DecisionRecord] = Field(default_factory=dict)
    recent_messages: list[dict] = Field(default_factory=list)
    complete: bool = False

    def compact_context(self) -> dict:
        return {
            "raw_idea": self.raw_idea,
            "knowledge": [item.model_dump(mode="json") for item in self.knowledge.values()],
            "concepts": [item.model_dump(mode="json") for item in self.concepts.values()],
            "requirements": [item.model_dump(mode="json") for item in self.requirements.values()],
            "implications": [item.model_dump(mode="json") for item in self.implications.values()],
            "contradictions": [item.model_dump(mode="json") for item in self.contradictions.values() if not item.resolved],
            "boundaries": [item.model_dump(mode="json") for item in self.boundaries],
            "decisions": [item.model_dump(mode="json") for item in self.decisions.values()],
            "recent_messages": self.recent_messages[-8:],
        }
