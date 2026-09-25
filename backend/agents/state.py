from __future__ import annotations

from typing import Annotated, Literal, Optional, Dict, List, Union
from pydantic import BaseModel, model_validator
from typing_extensions import TypedDict
from enum import Enum
from langgraph.graph.message import add_messages
from agents.prd_schema import PRDContract
from agents.requirements import ActiveRequirement


class DiscoveryTopic(str, Enum):
    USER_ROLES = "USER_ROLES"
    USER_GOALS = "USER_GOALS"
    CORE_WORKFLOW = "CORE_WORKFLOW"
    BUSINESS_RULES = "BUSINESS_RULES"
    CONSTRAINTS = "CONSTRAINTS"
    MVP_SCOPE = "MVP_SCOPE"
    EXCEPTIONS = "EXCEPTIONS"
    EDGE_CASES = "EDGE_CASES"


class TopicStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"


class TopicMaturity(str, Enum):
    """Quality of understanding for a topic, independent of field storage."""

    UNSEEN = "UNSEEN"
    MENTIONED = "MENTIONED"
    SKETCHED = "SKETCHED"
    COHERENT = "COHERENT"
    DECISION_READY = "DECISION_READY"


class KnowledgeState(str, Enum):
    """How strongly a discovered fact has been established with the user."""
    INFERRED = "INFERRED"
    CONFIRMED = "CONFIRMED"

class DiscoveryScope(str, Enum):
    USER_APP = "USER_APP"
    ADMIN_DASHBOARD = "ADMIN_DASHBOARD"

# --- canonical keys, one Literal per topic ---
USER_ROLES_KEYS = Literal[
    "primary_users", "secondary_users", "responsibilities",
    "permissions", "multiple_roles", "role_transitions",
]
USER_GOALS_KEYS = Literal[
    "primary_user_goals", "secondary_user_goals",
    "success_criteria", "motivations",
]

CORE_WORKFLOW_KEYS = Literal[
    "trigger", "workflow_steps", "completion_condition", "downstream_dependency", "end_state",
]
BUSINESS_RULES_KEYS = Literal[
    "validation_rules", "approval_rules", "eligibility_rules", "limits",
    "ownership_rules", "visibility_rules",
]
CONSTRAINTS_KEYS = Literal[
    "legal_constraints", "business_constraints", "operational_constraints",
    "geographic_constraints", "time_constraints",
]
MVP_SCOPE_KEYS = Literal[
    "must_have_features", "nice_to_have_features", "out_of_scope", "success_metrics",
]
EXCEPTIONS_KEYS = Literal[
    "user_cancellations", "timeouts", "invalid_actions", "recovery",
]
EDGE_CASES_KEYS = Literal[
    "duplicate_actions", "boundary_conditions", "simultaneous_actions", "rare_scenarios",
]

TOPIC_KEY_MAP = {
    DiscoveryTopic.USER_ROLES: set(USER_ROLES_KEYS.__args__),
    DiscoveryTopic.USER_GOALS: set(USER_GOALS_KEYS.__args__),
    DiscoveryTopic.CORE_WORKFLOW: set(CORE_WORKFLOW_KEYS.__args__),
    DiscoveryTopic.BUSINESS_RULES: set(BUSINESS_RULES_KEYS.__args__),
    DiscoveryTopic.CONSTRAINTS: set(CONSTRAINTS_KEYS.__args__),
    DiscoveryTopic.MVP_SCOPE: set(MVP_SCOPE_KEYS.__args__),
    DiscoveryTopic.EXCEPTIONS: set(EXCEPTIONS_KEYS.__args__),
    DiscoveryTopic.EDGE_CASES: set(EDGE_CASES_KEYS.__args__),
}

class KnowledgeItem(BaseModel):
    topic: DiscoveryTopic
    scope: DiscoveryScope  
    key: Union[
        USER_ROLES_KEYS, USER_GOALS_KEYS, CORE_WORKFLOW_KEYS,
        BUSINESS_RULES_KEYS, CONSTRAINTS_KEYS, MVP_SCOPE_KEYS,
        EXCEPTIONS_KEYS, EDGE_CASES_KEYS,
    ]
    value: str
    evidence: str
    source_question: Optional[str] = None
    roles: Optional[List[str]] = None
    aliases: Optional[Dict[str, List[str]]] = None
    role: Optional[str] = None
    confidence: float
    knowledge_state: KnowledgeState = KnowledgeState.CONFIRMED
    source_turn: int = 0
    absence: Optional[Literal["none", "not_applicable"]] = None

    @model_validator(mode="after")
    def check_key_belongs_to_topic(self):
        allowed = TOPIC_KEY_MAP[self.topic]
        if self.key not in allowed:
            raise ValueError(f"key '{self.key}' not valid for topic '{self.topic}'")
        if self.absence:
            expected = "none" if self.absence == "none" else "not applicable"
            if self.value != expected or self.knowledge_state != KnowledgeState.CONFIRMED:
                raise ValueError("Absence requires a canonical value and confirmed knowledge")
            if self.roles or self.aliases:
                raise ValueError("Whole-gap absence cannot declare actors or aliases")
        if self.roles:
            for r in self.roles:
                if len(r.split()) > 3 or len(r) < 2:
                    raise ValueError(f"'{r}' does not look like a valid role name")
        return self


class AgentState(TypedDict):
    session_id: str
    checkpoint_cursor: str
    interview_status: str
    asked_gap: Optional[dict]
    gap_coverage: Dict[str, dict]
    fact_acquisition: Dict[str, dict]
    active_answer_result: Optional[dict]
    extraction_status: str
    known_gap_evidence: List[str]
    messages: Annotated[list, add_messages]
    raw_idea: str
    prd_contract: Optional[PRDContract]
    compilation_errors: List[str]
    pm_is_complete: bool

    discovered_knowledge: List[KnowledgeItem]
    superseded_knowledge: List[dict]
    active_requirements: Dict[str, "ActiveRequirement"]
    requirement_coverage: Dict[str, dict]
    requirement_dependency_state: Dict[str, dict]
    eligible_requirement_keys: List[str]
    question_candidates: List[dict]
    eligible_question_candidates: List[dict]
    question_candidate_eligibility: Dict[str, dict]
    ranked_question_candidates: List[dict]
    question_candidate_priority: Dict[str, dict]
    requirement_question_history: List[dict]
    planner_source: str
    selected_requirement_candidate: Optional[dict]
    selected_requirement_priority: Optional[dict]
    validation_issues: List[dict]
    validation_pair_cache: Dict[str, dict]
    validation_blocking: bool
    validation_candidate_blocking: bool
    selected_validation_issue: Optional[dict]
    topic_status: Dict[DiscoveryTopic, TopicStatus]
    current_topic: Optional[DiscoveryTopic]
    topic_dependencies: Dict[DiscoveryTopic, List[DiscoveryTopic]]
    topic_maturity: Dict[DiscoveryTopic, TopicMaturity]
    product_model: Dict[str, List[str]]

    current_gap: Optional[str]
    current_objective: Optional[str]
    question_hint: Optional[str]
    known_keys: List[str]
    missing_keys: List[str]
    next_discovery_move: Optional[str]
    inferred_gap_evidence: List[str]
    relevant_context: List[str]
    conversation_intent: Optional[str]
    is_correction: bool

    turn_count: int
    question_retry_count: int
    answer_followup: Optional[dict]
    awaiting_confirmation: bool
    current_role: Optional[str]
    discovery_scope: DiscoveryScope
