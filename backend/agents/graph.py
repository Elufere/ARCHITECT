"""
LangGraph Workflow Definition.
Routes the user between the Interview loop and the Compilation phase.
"""

import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage

from agents.state import AgentState
from agents.pm_agent import pm_chat_node, pm_compile_node

logger = logging.getLogger(__name__)

# Keywords that trigger the final PRD generation
COMPILE_TRIGGERS = [
    "looks good", "generate", "compile", "approve", 
    "final prd", "yes proceed", "go ahead", "that's it"
]


def route_pm_flow(state: AgentState) -> str:
    """
    Conditional edge: Checks if the user has approved the requirements.
    """
    last_message = state["messages"][-1]
    
    # Only check the latest HumanMessage
    if isinstance(last_message, HumanMessage):
        content = last_message.content.lower()
        if any(trigger in content for trigger in COMPILE_TRIGGERS):
            logger.info("User approved requirements. Routing to compilation.")
            return "compile_prd"
            
    # Otherwise, stay in the chat loop (which ends and waits for the next user input)
    return END


def build_graph() -> StateGraph:
    """
    Constructs the Agent A (PM) graph.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("interview", pm_chat_node)
    workflow.add_node("compile_prd", pm_compile_node)

    # Define edges
    workflow.add_edge(START, "interview")
    
    # After interview, decide: compile or wait for user?
    workflow.add_conditional_edges(
        "interview",
        route_pm_flow,
        {
            "compile_prd": "compile_prd",
            END: END
        }
    )
    
    # After compiling, we are done
    workflow.add_edge("compile_prd", END)

    return workflow.compile()