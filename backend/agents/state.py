"""
LangGraph State Definition.
This dictionary is passed between all agents in the graph.
"""

from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages
from agents.prd_schema import PRDContract


class AgentState(TypedDict):
    """
    The central state object for the Architect multi-agent system.
    """
    # The chat history (user inputs and AI responses)
    messages: Annotated[list, add_messages]
    
    # The raw initial idea from the user
    raw_idea: str
    
    # The finalized PRD contract (None until the PM agent finishes)
    prd_contract: Optional[PRDContract]
    
    # Flags for graph routing
    pm_is_complete: bool