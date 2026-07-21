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
    print("Type 'quit' to exit. Type 'CONFIRM' to generate the PRD.")
    print("Use '.' on a blank line to send multi-line input.")
    print("=" * 60)
    
    print("\nWhat is your product idea?")
    lines = []
    while True:
        line = input()
        if line.strip() == '.': break
        lines.append(line)
    initial_idea = "\n".join(lines)
    
    state = AgentState(
        messages=[HumanMessage(content=initial_idea)],
        raw_idea=initial_idea,
        prd_contract=None,
        pm_is_complete=False,
        turn_count=0,
        awaiting_confirmation=False
    )
    
    graph = build_graph()
    
    while not state["pm_is_complete"]:
        state = graph.invoke(state)
        last_msg = state["messages"][-1]
        print(f"\n🤖 PM Agent: {last_msg.content}")
        
        if state["pm_is_complete"]: break

        # GRAPH-DRIVEN: If we hit turn 4, force the confirmation flow on next loop
        if state["turn_count"] >= 4 and not state["awaiting_confirmation"]:
            print("\n[System] Maximum discovery turns reached. Forcing summary...")
            state["awaiting_confirmation"] = True
            # Trick the graph into running the interview node one more time with the confirmation prompt
            state["messages"].append(HumanMessage(content="Summarize"))
            continue
            
        print("\n👤 You: ")
        user_lines = []
        while True:
            line = input()
            if line.strip() == '.': break
            if line.lower() == 'quit': return
            user_lines.append(line)
            
        user_input = "\n".join(user_lines)
        
        # Update confirmation state based on user input
        if "looks good" in user_input.lower():
            state["awaiting_confirmation"] = True
        elif user_input.strip().lower() == "confirm":
            pass # Graph router handles this
        else:
            state["awaiting_confirmation"] = False # Reset if they answer a question instead
            
        state["messages"].append(HumanMessage(content=user_input))
        
    if state["pm_is_complete"]:
        print("\n" + "=" * 60)
        print("PRD STATE OBJECT (First 500 chars):")
        print("=" * 60)
        print(state["prd_contract"].model_dump_json(indent=2)[:500] + "...")

if __name__ == "__main__":
    run_cli()