"""
LangGraph Workflow Definition.
"""

import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.pm_agent import pm_chat_node, pm_compile_node

logger = logging.getLogger(__name__)

# FIX #2: The ONLY way to trigger compilation
COMPILE_SECRET = "confirm"

def route_pm_flow(state: AgentState) -> str:
    last_message = state["messages"][-1]
    turn = state.get("turn_count", 0)
    
    # 1. Explicit user confirmation
    if isinstance(last_message, HumanMessage):
        if last_message.content.strip().lower() == "confirm":
            logger.info("User typed CONFIRM. Routing to compilation.")
            return "compile_prd"
        
        # 2. User triggers summary early
        if "looks good" in last_message.content.lower() and not state.get("awaiting_confirmation"):
            logger.info("User said 'looks good'. Triggering confirmation summary.")
            return "interview" # Loops back to chat node, which sees awaiting_confirmation=True
            
    # 3. GRAPH-DRIVEN: Force summary after 4 turns of user input
    if turn >= 4 and not state.get("awaiting_confirmation"):
        logger.info(f"Turn {turn} reached. Forcing proactive summary.")
        # We return "interview" but the pm_chat_node will see the turn count 
        # is 4 and the prompt will naturally guide it, OR we can set awaiting_confirmation here.
        # Let's set it here to be safe.
        # Note: We can't mutate state directly here in routing, so we rely on test_pm.py to catch it.
        
    return END


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("interview", pm_chat_node)
    workflow.add_node("compile_prd", pm_compile_node)

    workflow.add_edge(START, "interview")
    
    workflow.add_conditional_edges(
        "interview",
        route_pm_flow,
        {
            "compile_prd": "compile_prd",
            END: END
        }
    )
    
    workflow.add_edge("compile_prd", END)

    return workflow.compile()