from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from agents.prd_schema import PRDContract


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_idea: str
    prd_contract: Optional[PRDContract]
    pm_is_complete: bool
    
    # Graph-driven phase control
    turn_count: int             # How many times the user has answered
    awaiting_confirmation: bool  # True when user triggers the summary gate