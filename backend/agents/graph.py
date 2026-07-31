import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage

from agents.state import AgentState
# Import the new micro-graph nodes
from agents.knowledge_tracker import knowledge_tracker_node
from agents.interview_planner import interview_planner_node
from agents.question_generator import question_generator_node
# Keep the existing nodes
from agents.guardrails import guardrail_node
from agents.pm_agent import pm_compile_node

logger = logging.getLogger(__name__)


def route_after_plan(state: AgentState) -> str:
    """
    After planning, check if the interview planner decided we are done.
    If so, skip question generation and go straight to compilation.
    """
    if state.get("awaiting_confirmation"):
        logger.info("All discovery topics completed. Routing to compilation.")
        return "compile_prd"
    
    # Otherwise, generate questions for the planned topic
    return "generate"


def route_after_guardrail(state: AgentState) -> str:
    """
    Evaluates state directly after the guardrail checks the AI output.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If guardrail appended a SystemMessage (slap-on-the-wrist), loop back to the generator
    if isinstance(last_message, SystemMessage):
        logger.warning("Guardrail triggered. Forcing question regeneration.")
        return "generate"

    # If the message is clean (still an AIMessage), safe to return to user
    return END


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # 1. Register Nodes 
    workflow.add_node("extract", knowledge_tracker_node)
    workflow.add_node("plan", interview_planner_node)
    workflow.add_node("generate", question_generator_node)
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("compile_prd", pm_compile_node)

    # 2. Entry Point
    # Every invocation starts by extracting knowledge from the latest user input
    workflow.add_edge(START, "extract")

    # 3. Extract -> Plan (Always plan after extracting new info)
    workflow.add_edge("extract", "plan")

    # 4. Plan -> Compile OR Generate
    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "compile_prd": "compile_prd",
            "generate": "generate"
        }
    )

    # 5. Generate -> Guardrail (Always validate generated questions)
    workflow.add_edge("generate", "guardrail")

    # 6. Guardrail -> Regenerate OR End Turn
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "generate": "generate",  # Forces LLM to rewrite based on System Message feedback
            END: END                 # Returns control to the CLI for user input
        }
    )

    # 7. Compilation Exit
    workflow.add_edge("compile_prd", END)

    return workflow.compile()