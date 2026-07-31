"""
Terminal test harness for Agent A (PM).
"""
import json
from agents.state import AgentState
from agents.graph import build_graph
from langchain_core.messages import HumanMessage

def run_cli():
    print("=" * 60)
    print("ARCHITECT - AGENT A: PRODUCT MANAGER")
    print("Type 'quit' to exit.")
    print("Use '.' on a blank line to send multi-line input.")
    print("=" * 60)
    
    print("\nWhat is your product idea?")
    lines = []
    while True:
        line = input()
        if line.strip() == '.': break
        lines.append(line)
    initial_idea = "\n".join(lines)
    
    # Initialize state
    state = AgentState(
        messages=[HumanMessage(content=initial_idea)],
        raw_idea=initial_idea,
        prd_contract=None,
        pm_is_complete=False,
        turn_count=0,
        awaiting_confirmation=False,
        discovered_knowledge=[],
        topic_status={},
        current_topic=None
    )
    
    graph = build_graph()
    
    while not state["pm_is_complete"]:
        # 1. Run the graph (Extract -> Plan -> Generate -> Validate)
        state = graph.invoke(state)
        
        # 2. If the planner determined all topics are complete, break
        if state["pm_is_complete"]:
            break
            
        # 3. Print the AI's generated questions
        last_msg = state["messages"][-1]
        print(f"\n🤖 PM Agent: {last_msg.content}")
        
        # 4. Get user input
        print("\n👤 You: ")
        user_lines = []
        while True:
            line = input()
            if line.strip() == '.': break
            if line.lower() == 'quit': return
            user_lines.append(line)
            
        user_input = "\n".join(user_lines)
        
        # 5. Append user input and increment turn count
        state["messages"].append(HumanMessage(content=user_input))
        state["turn_count"] = state.get("turn_count", 0) + 1
        
    if state["pm_is_complete"]:
        print("\n" + "=" * 60)
        print("PRD STATE OBJECT (First 500 chars):")
        print("=" * 60)
        if state.get("prd_contract"):
            print(state["prd_contract"].model_dump_json(indent=2)[:500] + "...")
        else:
            print("PM flow completed, but PRD contract is missing.")

if __name__ == "__main__":
    run_cli()