"""
Agent A: The Product Manager
Handles intake, clarification, and PRD compilation.
"""

import json
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from agents.state import AgentState
from agents.prd_schema import PRDContract
from agents.kb_injection import load_static_rules, search_kb_patterns

logger = logging.getLogger(__name__)

# Standard LLM for chatting
chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
# Structured LLM for the final JSON output
structured_llm = chat_llm.with_structured_output(PRDContract)


def build_interview_prompt(context_rules: str, relevant_patterns: str) -> str:
    return f"""You are an elite Technical Product Manager. Your job is to take a raw product idea, identify gaps, ask clarifying questions, and ensure the idea adheres to mandatory engineering constraints.

### MANDATORY ENGINEERING RULES:
---
{context_rules}
---

### RELEVANT ENGINEERING PATTERNS TO CONSIDER:
---
{relevant_patterns}
---

### YOUR BEHAVIOR:
1. Analyze the user's input against the rules above.
2. Identify gaps (edge cases, auth flows, data constraints, scope creep).
3. Ask UP TO 3 highly specific questions at a time to fill those gaps. 
4. Do NOT generate the final PRD yet. Just interview the user.
"""


def build_compile_prompt(context_rules: str) -> str:
    return f"""You are an elite Technical Product Manager. You have just finished interviewing the user. 
Based on the ENTIRE conversation history below, compile the final Product Requirements Document (PRD).

### MANDATORY ENGINEERING RULES TO ENFORCE:
---
{context_rules}
---

### INSTRUCTIONS:
Extract all confirmed requirements from the chat and format them into the required JSON structure. 
If a requirement was discussed but not explicitly confirmed, use your best engineering judgment to formalize it.
Do not ask any more questions. Output ONLY the valid JSON object."""


def pm_chat_node(state: AgentState) -> dict:
    """Node for the interview/clarification loop."""
    messages = state["messages"]
    
    static_rules = load_static_rules(categories=["security", "standards"])
    relevant_patterns = ""
    if state.get("raw_idea"):
        relevant_patterns = search_kb_patterns(query=state["raw_idea"], category="standards", n_results=2)
        
    system_prompt = build_interview_prompt(static_rules, relevant_patterns)
    response = chat_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    return {"messages": [response]}


def pm_compile_node(state: AgentState) -> dict:
    """Node for compiling the final Pydantic JSON and saving to disk."""
    messages = state["messages"]
    
    # Only load strict rules for compilation (no need for patterns now)
    static_rules = load_static_rules(categories=["security", "standards"])
    system_prompt = build_compile_prompt(static_rules)
    
    # Invoke the structured LLM
    prd_contract: PRDContract = structured_llm.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    
    # Save to disk
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "requirements_mvp.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prd_contract.model_dump(), f, indent=2)
        
    logger.info(f"PRD successfully compiled and saved to {output_path}")
    
    # Return a conversational message + update the state flags
    success_msg = f"PRD compiled and saved to `output/requirements_mvp.json`. Handing off to the Architect Agent."
    return {
        "messages": [success_msg],
        "prd_contract": prd_contract,
        "pm_is_complete": True
    }