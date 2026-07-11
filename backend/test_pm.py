"""
Terminal test harness for Agent A (PM).
"""
import json
from agents.state import AgentState
from agents.graph import build_graph

def run_cli():
    print("=" * 60)
    print("ARCHITECT - AGENT A: PRODUCT MANAGER")
    print("Type 'quit' to exit. Type 'looks good' to generate the PRD.")
    print("=" * 60)
    
    initial_idea = input("\nWhat is your product idea? > ")
    
    # Initialize state
    state = AgentState(
        messages=[],
        raw_idea=initial_idea,
        prd_contract=None,
        pm_is_complete=False
    )
    
    # Inject the first human message into state
    from langchain_core.messages import HumanMessage
    state["messages"].append(HumanMessage(content=initial_idea))
    
    # Build the graph
    graph = build_graph()
    
    # Conversation loop
    while not state["pm_is_complete"]:
        # Run the graph up to the next interruption (END)
        state = graph.invoke(state)
        
        # Get the last AI message
        last_msg = state["messages"][-1]
        print(f"\n🤖 PM Agent: {last_msg.content}")
        
        if state["pm_is_complete"]:
            break
            
        # Get user input
        user_input = input("\n👤 You: ")
        if user_input.lower() == "quit":
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        
    if state["pm_is_complete"]:
        print("\n" + "=" * 60)
        print("PRD STATE OBJECT (First 500 chars):")
        print("=" * 60)
        # Print the Pydantic object nicely
        print(state["prd_contract"].model_dump_json(indent=2)[:500] + "...")

if __name__ == "__main__":
    run_cli()