import logging
from uuid import uuid4
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState, DiscoveryScope
from agents.llm_usage import usage_tracker
from agents.interview_checkpoint import durable_node
# Import the new micro-graph nodes
from agents.knowledge_tracker import knowledge_tracker_node
from agents.validation_resolution import validation_resolution_node
from agents.requirement_activation import requirement_activation_node
from agents.requirement_coverage import requirement_coverage_node
from agents.requirement_dependencies import requirement_dependency_node
from agents.consistency_validation import consistency_validation_node
from agents.question_candidates import question_candidate_builder_node, question_candidate_filter_node
from agents.question_priority import question_candidate_priority_node
from agents.conversation_manager import conversation_manager_node
from agents.interview_planner import interview_planner_node, all_discovery_resolved
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
    if state.get("awaiting_confirmation") and all_discovery_resolved(state):
        logger.info("All discovery topics completed. Routing to compilation.")
        return "compile_prd"
    
    # Otherwise, generate questions for the planned topic
    return "generate"


def route_after_conversation_manager(state: AgentState) -> str:
    """Only knowledge and corrections should flow into the extraction pipeline."""
    if state.get("conversation_intent") in {"product_information", "correction", "objection"}:
        return "extract"
    # A short confirmation can itself answer a discovery question. Never drop
    # it just because the intent classifier recognized the word "yes".
    if (
        state.get("conversation_intent") == "confirmation"
        and (state.get("current_gap") or state.get("next_discovery_move") == "confirm_inference")
    ):
        return "extract"
    if state.get("conversation_intent") == "confirmation":
        return "plan"
    return END


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
    workflow.add_node("conversation_manager", durable_node("conversation_manager", conversation_manager_node,
        lambda state: "waiting" if route_after_conversation_manager(state) == END else route_after_conversation_manager(state)))
    workflow.add_node("extract", durable_node("extract", knowledge_tracker_node, lambda _: "resolve_validation_answer"))
    workflow.add_node("resolve_validation_answer", durable_node("resolve_validation_answer", validation_resolution_node, lambda _: "activate_requirements"))
    workflow.add_node("activate_requirements", durable_node("activate_requirements", requirement_activation_node, lambda _: "cover_requirements"))
    workflow.add_node("cover_requirements", durable_node("cover_requirements", requirement_coverage_node, lambda _: "resolve_requirements"))
    workflow.add_node("resolve_requirements", durable_node("resolve_requirements", requirement_dependency_node, lambda _: "validate_consistency"))
    workflow.add_node("validate_consistency", durable_node("validate_consistency", consistency_validation_node, lambda _: "build_candidates"))
    workflow.add_node("build_candidates", durable_node("build_candidates", question_candidate_builder_node, lambda _: "filter_candidates"))
    workflow.add_node("filter_candidates", durable_node("filter_candidates", question_candidate_filter_node, lambda _: "prioritize_candidates"))
    workflow.add_node("prioritize_candidates", durable_node("prioritize_candidates", question_candidate_priority_node, lambda _: "plan"))
    workflow.add_node("plan", durable_node("plan", interview_planner_node, route_after_plan))
    workflow.add_node("generate", durable_node("generate", question_generator_node, lambda _: "guardrail"))
    workflow.add_node("guardrail", durable_node("guardrail", guardrail_node,
        lambda state: "waiting" if route_after_guardrail(state) == END else "generate"))
    def compile_when_covered(state):
        if not all_discovery_resolved(state):
            raise RuntimeError("PRD compilation blocked: schema or active-requirement coverage is incomplete")
        return pm_compile_node(state)
    workflow.add_node("compile_prd", durable_node("compile_prd", compile_when_covered,
        lambda state: ("phase_complete" if state["discovery_scope"] == DiscoveryScope.USER_APP else "completed")
        if state.get("pm_is_complete") else "compile_prd"))

    # 2. Entry Point
    def resume_at(state):
        cursor = state.get("checkpoint_cursor", "conversation_manager")
        if cursor == "waiting" and state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
            return "conversation_manager"
        return END if cursor in ("waiting", "phase_complete", "completed") else cursor
    workflow.add_conditional_edges(START, resume_at, {
        name: name for name in ("conversation_manager", "extract", "resolve_validation_answer", "activate_requirements", "cover_requirements", "resolve_requirements", "validate_consistency", "build_candidates", "filter_candidates", "prioritize_candidates", "plan", "generate", "guardrail", "compile_prd", END)})
    workflow.add_conditional_edges(
        "conversation_manager",
        route_after_conversation_manager,
        {"extract": "extract", "plan": "plan", END: END},
    )

    # 3. Extract -> Requirement activation -> Coverage -> Dependencies -> Candidate filtering/ranking -> Plan
    workflow.add_edge("extract", "resolve_validation_answer")
    workflow.add_edge("resolve_validation_answer", "activate_requirements")
    workflow.add_edge("activate_requirements", "cover_requirements")
    workflow.add_edge("cover_requirements", "resolve_requirements")
    workflow.add_edge("resolve_requirements", "validate_consistency")
    workflow.add_edge("validate_consistency", "build_candidates")
    workflow.add_edge("build_candidates", "filter_candidates")
    workflow.add_edge("filter_candidates", "prioritize_candidates")
    workflow.add_edge("prioritize_candidates", "plan")

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

    return workflow.compile().with_config(
        callbacks=[usage_tracker],
        metadata={"openai_usage_session": str(uuid4())},
    )
