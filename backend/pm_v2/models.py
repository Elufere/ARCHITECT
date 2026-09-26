from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class KnowledgeStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DEFERRED = "DEFERRED"


class TurnIntent(str, Enum):
    PRODUCT_INFORMATION = "PRODUCT_INFORMATION"
    CORRECTION = "CORRECTION"
    CONFIRMATION = "CONFIRMATION"
    CLARIFICATION = "CLARIFICATION"
    RATIONALE_REQUEST = "RATIONALE_REQUEST"
    SUMMARY_REQUEST = "SUMMARY_REQUEST"
    ADVICE_REQUEST = "ADVICE_REQUEST"
    UNCERTAINTY = "UNCERTAINTY"
    DESIGN_DEFERRAL = "DESIGN_DEFERRAL"
    OBJECTION = "OBJECTION"


class FactRelation(str, Enum):
    NEW = "NEW"
    DUPLICATE = "DUPLICATE"
    REFINEMENT = "REFINEMENT"
    CORRECTION = "CORRECTION"
    CONTRADICTION = "CONTRADICTION"


class CoverageLevel(str, Enum):
    NONE = "NONE"
    PARTIAL = "PARTIAL"
    COVERED = "COVERED"


class DepthLevel(str, Enum):
    SHALLOW = "SHALLOW"
    ADEQUATE = "ADEQUATE"
    DEEP = "DEEP"


class SourceTurn(BaseModel):
    id: str = Field(default_factory=lambda: new_id("turn"))
    number: int
    text: str


class PendingFact(BaseModel):
    id: str = Field(default_factory=lambda: new_id("candidate"))
    statement: str
    evidence: str
    status: KnowledgeStatus = KnowledgeStatus.CONFIRMED
    confidence: float = Field(default=1.0, ge=0, le=1)
    negative: bool = False


class TurnInterpretation(BaseModel):
    intent: TurnIntent = TurnIntent.PRODUCT_INFORMATION
    answers_previous_question: bool = False
    facts: List[PendingFact] = Field(default_factory=list)
    control_reason: str = ""
    boundary_summary: Optional[str] = None


class GroundingVerdict(BaseModel):
    candidate_id: str
    supported: bool
    normalized_statement: str
    reason: str = ""


class GroundingBatch(BaseModel):
    verdicts: List[GroundingVerdict] = Field(default_factory=list)


class FactClassification(BaseModel):
    candidate_id: str
    canonical_key: str
    domains: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    summary: str


class ClassificationBatch(BaseModel):
    items: List[FactClassification] = Field(default_factory=list)


class FactRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fact"))
    canonical_key: str
    statement: str
    domains: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    status: KnowledgeStatus = KnowledgeStatus.CONFIRMED
    confidence: float = Field(default=1.0, ge=0, le=1)
    negative: bool = False
    source_turn_ids: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    active: bool = True
    superseded_by: Optional[str] = None


class ReconciliationDecision(BaseModel):
    candidate_id: str
    relation: FactRelation
    matched_fact_id: Optional[str] = None
    canonical_statement: str
    reason: str = ""


class ReconciliationBatch(BaseModel):
    decisions: List[ReconciliationDecision] = Field(default_factory=list)


class ImplicationProposal(BaseModel):
    statement: str
    based_on_fact_ids: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    needs_validation: bool = True
    importance: float = Field(default=0.5, ge=0, le=1)


class ImplicationBatch(BaseModel):
    items: List[ImplicationProposal] = Field(default_factory=list)


class ImplicationRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("imp"))
    statement: str
    based_on_fact_ids: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    status: KnowledgeStatus = KnowledgeStatus.PROPOSED
    needs_validation: bool = True
    importance: float = Field(default=0.5, ge=0, le=1)


class RequirementProposal(BaseModel):
    key: str
    label: str
    description: str
    basis_fact_ids: List[str] = Field(default_factory=list)
    basis_implication_ids: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    business_impact: float = Field(default=0.5, ge=0, le=1)
    architecture_impact: float = Field(default=0.5, ge=0, le=1)
    risk: float = Field(default=0.5, ge=0, le=1)
    reason: str = ""


class RequirementActivationBatch(BaseModel):
    items: List[RequirementProposal] = Field(default_factory=list)


class RequirementRecord(BaseModel):
    key: str
    label: str
    description: str
    status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    coverage: CoverageLevel = CoverageLevel.NONE
    depth: DepthLevel = DepthLevel.SHALLOW
    evidence_fact_ids: List[str] = Field(default_factory=list)
    implication_ids: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    missing_decisions: List[str] = Field(default_factory=list)
    business_impact: float = Field(default=0.5, ge=0, le=1)
    architecture_impact: float = Field(default=0.5, ge=0, le=1)
    risk: float = Field(default=0.5, ge=0, le=1)
    activation_reason: str = ""


class RequirementAssessment(BaseModel):
    key: str
    status: KnowledgeStatus
    coverage: CoverageLevel
    depth: DepthLevel
    supporting_fact_ids: List[str] = Field(default_factory=list)
    missing_decisions: List[str] = Field(default_factory=list)
    rationale: str = ""


class RequirementAssessmentBatch(BaseModel):
    items: List[RequirementAssessment] = Field(default_factory=list)


class ContradictionProposal(BaseModel):
    fact_ids: List[str] = Field(min_length=2)
    issue: str
    blocking: bool = True


class ContradictionBatch(BaseModel):
    items: List[ContradictionProposal] = Field(default_factory=list)


class ContradictionRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("conflict"))
    fact_ids: List[str]
    issue: str
    blocking: bool = True
    resolved: bool = False


class DiscoveryBoundary(BaseModel):
    id: str = Field(default_factory=lambda: new_id("boundary"))
    kind: Literal["DESIGN_DEFERRAL", "REJECTED_INQUIRY", "DEFERRED_DECISION"]
    source_turn_id: str
    evidence: str
    decision_key: Optional[str] = None
    question: Optional[str] = None
    instruction: str


class QuestionCandidate(BaseModel):
    id: str = Field(default_factory=lambda: new_id("question"))
    decision_key: str
    objective: str
    question: str
    requirement_keys: List[str] = Field(default_factory=list)
    contradiction_ids: List[str] = Field(default_factory=list)
    uncertainty: float = Field(default=0.5, ge=0, le=1)
    business_impact: float = Field(default=0.5, ge=0, le=1)
    architecture_impact: float = Field(default=0.5, ge=0, le=1)
    dependency_unlock: float = Field(default=0.5, ge=0, le=1)
    risk: float = Field(default=0.5, ge=0, le=1)
    contextual_relevance: float = Field(default=0.5, ge=0, le=1)
    repetition_penalty: float = Field(default=0.0, ge=0, le=1)
    premature_detail_penalty: float = Field(default=0.0, ge=0, le=1)
    user_fatigue_penalty: float = Field(default=0.0, ge=0, le=1)


class QuestionCandidateBatch(BaseModel):
    items: List[QuestionCandidate] = Field(default_factory=list)


class QuestionAudit(BaseModel):
    candidate_id: str
    eligible: bool
    already_answered: bool = False
    repeated: bool = False
    too_broad: bool = False
    too_technical: bool = False
    premature: bool = False
    low_value: bool = False
    boundary_violation: bool = False
    reason: str = ""


class QuestionAuditBatch(BaseModel):
    items: List[QuestionAudit] = Field(default_factory=list)


class QuestionHistoryItem(BaseModel):
    turn: int
    decision_key: str
    objective: str
    question: str
    requirement_keys: List[str] = Field(default_factory=list)
    status: Literal["ASKED", "ANSWERED", "DEFERRED", "REJECTED"] = "ASKED"


class CompletionAssessment(BaseModel):
    complete: bool
    blocking_requirement_keys: List[str] = Field(default_factory=list)
    unresolved_high_impact: List[str] = Field(default_factory=list)
    safely_deferred_or_low_value: List[str] = Field(default_factory=list)
    reason: str


class ControlReply(BaseModel):
    response: str
    suggestions: List[str] = Field(default_factory=list)


class RecommendationRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("recommendation"))
    statement: str
    source_turn_id: str
    status: KnowledgeStatus = KnowledgeStatus.PROPOSED


class DiscoveryState(BaseModel):
    session_id: str
    raw_idea: str = ""
    turn_count: int = 0
    source_turns: List[SourceTurn] = Field(default_factory=list)
    facts: List[FactRecord] = Field(default_factory=list)
    implications: List[ImplicationRecord] = Field(default_factory=list)
    recommendations: List[RecommendationRecord] = Field(default_factory=list)
    requirements: Dict[str, RequirementRecord] = Field(default_factory=dict)
    contradictions: List[ContradictionRecord] = Field(default_factory=list)
    boundaries: List[DiscoveryBoundary] = Field(default_factory=list)
    question_history: List[QuestionHistoryItem] = Field(default_factory=list)
    pending_question: Optional[str] = None
    pending_decision_key: Optional[str] = None
    pending_requirement_keys: List[str] = Field(default_factory=list)
    complete: bool = False
    completion_reason: str = ""
