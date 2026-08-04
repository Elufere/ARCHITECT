"""
Terminal test harness for Agent A (PM).
"""
import json
from agents.state import AgentState, DiscoveryScope
from agents.graph import build_graph
from langchain_core.messages import HumanMessage


def run_phase(graph, state: AgentState, phase_label: str) -> AgentState:
    print(f"\n{'=' * 60}\n{phase_label}\n{'=' * 60}")

    while not state["pm_is_complete"]:
        state = graph.invoke(state)

        if state["pm_is_complete"]:
            break

        last_msg = state["messages"][-1]
        print(f"\n🤖 PM Agent: {last_msg.content}")

        print("\n👤 You: ")
        user_lines = []
        while True:
            line = input()
            if line.strip() == '.':
                break
            if line.lower() == 'quit':
                return state
            user_lines.append(line)

        user_input = "\n".join(user_lines)
        state["messages"].append(HumanMessage(content=user_input))
        state["turn_count"] = state.get("turn_count", 0) + 1

    return state


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
        if line.strip() == '.':
            break
        lines.append(line)
    initial_idea = "\n".join(lines)

    state = AgentState(
        messages=[HumanMessage(content=initial_idea)],
        raw_idea=initial_idea,
        prd_contract=None,
        pm_is_complete=False,
        discovery_scope=DiscoveryScope.USER_APP,
        turn_count=0,
        awaiting_confirmation=False,
        discovered_knowledge=[],
        topic_status={},
        current_topic=None,
    )

    graph = build_graph()

    # ---- Phase 1: User App ----
    state = run_phase(graph, state, "PHASE 1: USER APP DISCOVERY")
    if not state.get("pm_is_complete"):
        return  # user quit mid-phase-1

    print("\n" + "=" * 60)
    print("USER APP DISCOVERY COMPLETE!")
    print("=" * 60)
    print("Run Phase 2: Admin Dashboard discovery? (y/n)")
    choice = input().strip().lower()

    if choice == 'y':
        state["discovery_scope"] = DiscoveryScope.ADMIN_DASHBOARD
        state["pm_is_complete"] = False
        state["topic_status"] = {}
        state["current_topic"] = None
        # discovered_knowledge is intentionally NOT cleared —
        # Phase 1 facts stay, but scope filtering keeps the
        # planner from treating them as answers to Phase 2 questions

        state = run_phase(graph, state, "PHASE 2: ADMIN DASHBOARD DISCOVERY")

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