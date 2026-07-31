"""
Agent A: The Product Manager (PRD Compilation Only)
"""

import json
import logging
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

from agents.state import AgentState
from agents.prd_schema import PRDContract
from agents.kb_injection import load_static_rules

logger = logging.getLogger(__name__)

structured_llm = ChatOllama(model="llama3.1", temperature=0.0).bind_tools(
    [PRDContract], tool_choice="any"
)


def build_compile_prompt(context_rules: str) -> str:
    return f"""You are compiling the final PRD.

### MANDATORY ENGINEERING RULES:
---
{context_rules}
---

### STRICT COMPILATION RULES:
1. NO HALLUCINATIONS: ONLY include requirements explicitly confirmed by the user.
2. UNANSWERED = DEFERRED: Put unconfirmed items in `deferred_items` or `open_questions`.
3. VALIDATION IS REQUIRED: Every FunctionalRequirement MUST have a `validation` string. If unknown, write "TBD".
"""


def pm_compile_node(state: AgentState) -> dict:
    """Compiles the final PRD from the discovered knowledge and chat history."""
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