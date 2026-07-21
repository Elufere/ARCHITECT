"""
Agent A: The Product Manager
"""

import json
import logging
import re
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agents.state import AgentState
from agents.prd_schema import PRDContract
from agents.kb_injection import load_static_rules, search_kb_patterns

logger = logging.getLogger(__name__)

chat_llm = ChatOllama(model="llama3.1", temperature=0.2)
structured_llm = ChatOllama(model="llama3.1", temperature=0.0).bind_tools(
    [PRDContract], tool_choice="any"
)

REQUIRED_CATEGORIES = [
    "Auth & Identity", "Core User Flows", "Edge Cases", 
    "State Transitions", "Failure Scenarios"
]

# Simple keyword router to fix frozen RAG category
def _get_rag_category(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["password", "auth", "login", "jwt", "pii", "data retention"]): return "security"
    return "standards"

# Simple heuristic to update covered categories based on AI questions
def _extract_category_from_response(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["password", "log in", "sign up", "auth", "account", "role"]): return "Auth & Identity"
    if any(w in text for w in ["cancel", "leave", "no-show", "timeout", "close"]): return "State Transitions"
    if any(w in text for w in ["offline", "disconnect", "concurrency", "same time", "crash"]): return "Failure Scenarios"
    if any(w in text for w in ["capacity", "full", "duplicate", "waitlist"]): return "Edge Cases"
    return "Core User Flows"

def build_interview_prompt(focus_category: str, rag_context: str) -> str:
    return f"""You are an elite Technical Product Manager extracting MVP requirements.

### CURRENT FOCUS AREA:
{focus_category}

### RELEVANT ENGINEERING PATTERNS (Based on latest user input):
---
{rag_context if rag_context else "Standard web/mobile app patterns."}
---

### STRICT RULES:
1. Ask exactly UP TO 3 questions about the CURRENT FOCUS AREA.
2. Do NOT ask about future roadmaps, SDKs, or integrations.
3. Do NOT attempt to summarize or generate a PRD. Just ask the questions.
"""

def build_confirmation_prompt() -> str:
    return """You are a PM finalizing an interview. 
Output a brief summary of the MVP based on the chat, note any areas that might need more detail, and end with exactly this sentence:
"Type 'CONFIRM' to proceed with generating the PRD, or provide answers to fill any gaps."
"""

def build_compile_prompt(context_rules: str) -> str:
    return f"""You are compiling the final PRD.

### MANDATORY ENGINEERING RULES:
---
{context_rules}
---

### STRICT COMPILATION RULES (CRITICAL):
1. **NO HALLUCINATIONS:** ONLY include requirements explicitly confirmed by the user.
2. **UNANSWERED = DEFERRED:** If a question was asked but bypassed with "looks good", DO NOT invent an answer. Put it in the `deferred_items` or `open_questions` lists.
3. **VALIDATION IS REQUIRED:** Every FunctionalRequirement MUST have a `validation` string. If unknown, write "TBD".
"""

def pm_chat_node(state: AgentState) -> dict:
    messages = state["messages"]
    last_msg = messages[-1]
    
    # Dynamic RAG based on latest message
    query_text = last_msg.content if isinstance(last_msg, HumanMessage) else state.get("raw_idea", "")
    rag_category = _get_rag_category(query_text)
    rag_context = search_kb_patterns(query=query_text, category=rag_category, n_results=2)
    
    # Graph-driven phase control: Tell the LLM what to focus on based on turn count
    turn = state.get("turn_count", 0)
    if turn == 0: focus = "Auth & Identity and Core User Flows"
    elif turn == 1: focus = "Edge Cases (capacity, duplicates, waitlists)"
    elif turn == 2: focus = "State Transitions (cancel, leave, timeout, no-show)"
    elif turn == 3: focus = "Failure Scenarios (offline behavior, concurrency)"
    else: focus = "Any remaining gaps"
    
    # Check if we are in the confirmation flow
    if state.get("awaiting_confirmation"):
        system_prompt = build_confirmation_prompt()
    else:
        system_prompt = build_interview_prompt(focus, rag_context)
    
    response = chat_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # Stop tracking categories in state—the graph handles it now
    return {"messages": [response]}

def pm_compile_node(state: AgentState) -> dict:
    messages = state["messages"]
    static_rules = load_static_rules(categories=["security", "standards"])
    system_prompt = build_compile_prompt(static_rules)
    
    prd_contract: PRDContract = structured_llm.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "requirements_mvp.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prd_contract.model_dump(), f, indent=2)
        
    logger.info(f"PRD saved to {output_path}")
    
    success_msg = f"✅ PRD compiled and saved to `output/requirements_mvp.json`. Handing off to Architect Agent."
    return {
        "messages": [success_msg],
        "prd_contract": prd_contract,
        "pm_is_complete": True
    }