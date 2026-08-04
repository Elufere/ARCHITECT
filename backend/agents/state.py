from typing import Annotated, Literal, Optional, Dict, List, Union
from pydantic import BaseModel, model_validator
from typing_extensions import TypedDict
from enum import Enum
from langgraph.graph.message import add_messages
from agents.prd_schema import PRDContract


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
    "trigger", "workflow_steps", "completion_condition", "downstream_dependency",
]
BUSINESS_RULES_KEYS = Literal[
    "validations", "conditions", "policies",
]
CONSTRAINTS_KEYS = Literal[
    "legal", "technical", "operational", "cost", "performance",
]
MVP_SCOPE_KEYS = Literal[
    "required_mvp_functionality", "out_of_scope_functionality",
]
EXCEPTIONS_KEYS = Literal[
    "expected_error_scenarios", "failure_handling", "recovery_behavior",
]
EDGE_CASES_KEYS = Literal[
    "rare_scenarios", "unusual_inputs", "boundary_conditions",
    "duplicates", "empty_states",
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
    roles: Optional[List[str]] = None
    role: Optional[str] = None
    confidence: float
    source_turn: int = 0

    @model_validator(mode="after")
    def check_key_belongs_to_topic(self):
        allowed = TOPIC_KEY_MAP[self.topic]
        if self.key not in allowed:
            raise ValueError(f"key '{self.key}' not valid for topic '{self.topic}'")
        if self.roles:
            for r in self.roles:
                if len(r.split()) > 3 or len(r) < 2:
                    raise ValueError(f"'{r}' does not look like a valid role name")
        return self


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_idea: str
    prd_contract: Optional[PRDContract]
    pm_is_complete: bool

    discovered_knowledge: List[KnowledgeItem]
    topic_status: Dict[DiscoveryTopic, TopicStatus]
    current_topic: Optional[DiscoveryTopic]
    topic_dependencies: Dict[DiscoveryTopic, List[DiscoveryTopic]]

    current_gap: Optional[str]
    current_objective: Optional[str]
    question_hint: Optional[str]
    known_keys: List[str]
    missing_keys: List[str]

    turn_count: int
    awaiting_confirmation: bool
    current_role: Optional[str]
    discovery_scope: DiscoveryScope