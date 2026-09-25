"""Interactive PM CLI with durable input and graph-boundary recovery."""
from uuid import uuid4

from agents.state import AgentState, DiscoveryScope
from agents.graph import build_graph
from agents.interview_checkpoint import save_checkpoint, load_checkpoint, unfinished_sessions, session_lock
from agents.llm_errors import LLMCallFailed, ExtractionFailed
from agents.product_model import build_product_model
from langchain_core.messages import HumanMessage


def read_answer():
    lines = []
    while True:
        line = input()
        if line.strip() == ".":
            return "\n".join(lines)
        if line.strip().lower() == "quit":
            return None
        lines.append(line)


def run_phase(graph, state: AgentState, phase_label: str) -> AgentState:
    print(f"\n{'=' * 60}\n{phase_label}\n{'=' * 60}")
    while not state.get("pm_is_complete"):
        if state.get("checkpoint_cursor") != "waiting":
            state = graph.invoke(state, config={"metadata": {"openai_usage_session": state["session_id"]}})
        if state.get("pm_is_complete"):
            break
        if state.get("checkpoint_cursor") == "compile_prd":
            print(state["messages"][-1].content)
            print("Progress is saved. Run Architect again to retry compilation.")
            return state
        print(f"\nPM Agent: {state['messages'][-1].content}")
        print("\nYou:")
        answer = read_answer()
        if answer is None:
            print("Progress saved. Run Architect again to resume.")
            return state
        state["messages"].append(HumanMessage(content=answer, id=str(uuid4())))
        state["turn_count"] = state.get("turn_count", 0) + 1
        state.update(checkpoint_cursor="conversation_manager", interview_status="PROCESSING_ANSWER",
                     active_answer_result=None, extraction_status="PENDING")
        save_checkpoint(state)  # Before any graph/model work.
    return state


def choose_session():
    pending = unfinished_sessions()
    if not pending:
        return None
    print("\nUnfinished interviews:")
    for index, document in enumerate(pending, 1):
        saved = document["state"]
        print(f"{index}. {document['session_id']} | {saved.get('discovery_scope')} | "
              f"turn {saved.get('turn_count', 0)} | {saved.get('current_topic')} -> {saved.get('current_gap')} | "
              f"{document.get('interview_status')}")
    while True:
        choice = input("Enter to resume the latest, a session number, 'new', or 'quit': ").strip().lower()
        if choice in ("new", "quit"):
            return choice
        if not choice:
            return pending[0]["session_id"]
        if choice.isdigit() and 1 <= int(choice) <= len(pending):
            return pending[int(choice) - 1]["session_id"]
        print("Please choose a listed session or 'new'.")


def new_session():
    print("\nWhat is your product idea?")
    idea = read_answer()
    if idea is None:
        return None
    state = AgentState(session_id=str(uuid4()), messages=[HumanMessage(content=idea, id=str(uuid4()))],
        raw_idea=idea, prd_contract=None, pm_is_complete=False,
        discovery_scope=DiscoveryScope.USER_APP, turn_count=0, awaiting_confirmation=False,
        discovered_knowledge=[], superseded_knowledge=[], active_requirements={}, requirement_dependency_state={}, eligible_requirement_keys=[], topic_status={}, topic_maturity={}, product_model={},
        gap_coverage={}, fact_acquisition={}, asked_gap=None, active_answer_result=None,
        checkpoint_cursor="conversation_manager", interview_status="PROCESSING_ANSWER", extraction_status="PENDING",
        current_topic=None, current_gap=None, current_objective=None, question_hint=None,
        known_keys=[], missing_keys=[], inferred_gap_evidence=[], known_gap_evidence=[], relevant_context=[],
        next_discovery_move=None, conversation_intent=None, is_correction=False, current_role=None)
    save_checkpoint(state)
    return state


def run_session(state):
    graph = build_graph()
    state = run_phase(graph, state, f"{state['discovery_scope'].value} DISCOVERY")
    if not state.get("pm_is_complete"):
        return
    if state["discovery_scope"] == DiscoveryScope.USER_APP:
        print("\nUSER APP DISCOVERY COMPLETE!")
        choice = input("Run Phase 2: Admin Dashboard discovery? (y/n): ").strip().lower()
        if choice == "y":
            # Global knowledge/acquisition/coverage survive; coverage is scope-keyed.
            state.update(discovery_scope=DiscoveryScope.ADMIN_DASHBOARD, pm_is_complete=False,
                         prd_contract=None, topic_status={}, topic_maturity={}, current_topic=None,
                         current_gap=None, current_role=None, asked_gap=None, active_answer_result=None,
                         awaiting_confirmation=False, next_discovery_move=None, checkpoint_cursor="plan",
                         interview_status="ACTIVE", answer_followup=None,
                         requirement_dependency_state={}, eligible_requirement_keys=[])
            state["product_model"] = build_product_model(state["discovered_knowledge"], DiscoveryScope.ADMIN_DASHBOARD)
            save_checkpoint(state)
            state = run_phase(graph, state, "ADMIN_DASHBOARD DISCOVERY")
            if not state.get("pm_is_complete"):
                return
        else:
            state.update(checkpoint_cursor="completed", interview_status="COMPLETED")
            save_checkpoint(state)
    if state.get("prd_contract"):
        print("\nPRD STATE OBJECT (First 500 chars):")
        print(state["prd_contract"].model_dump_json(indent=2)[:500] + "...")
        print("Full verified PRD: backend/output/requirements_mvp.json (USER_APP) or requirements_admin_dashboard.json (ADMIN_DASHBOARD).")


def run_cli():
    print("ARCHITECT - PRODUCT MANAGER\nType 'quit' to save and exit. Use '.' on a blank line to send input.")
    session_id = None
    try:
        selected = choose_session()
        if selected == "quit":
            return
        if selected in (None, "new"):
            state = new_session()
            if state is None:
                return
            session_id = state["session_id"]
        else:
            session_id = selected
        with session_lock(session_id):
            state = load_checkpoint(session_id)
            print(f"Session: {session_id}")
            try:
                run_session(state)
            except (LLMCallFailed, ExtractionFailed) as exc:
                print(f"\n{exc}")
                saved = load_checkpoint(session_id)

                if saved.get("checkpoint_cursor") == "extract":
                    saved["extraction_status"] = "EXTRACTION_FAILED"
                    save_checkpoint(saved)

                print("Your interview progress and pending input have been saved.")

                topic = saved.get("current_topic")
                print(
                    f"Current position: "
                    f"{topic.value if topic else 'initial discovery'} "
                    f"-> {saved.get('current_gap')}"
                )

                if isinstance(exc, LLMCallFailed):
                    if exc.retryable:
                        print(
                            "OpenAI could not be reached after retries. "
                            "Run Architect again when connectivity/service availability returns."
                        )
                    else:
                        print(
                            "The OpenAI request could not be completed. "
                            "Check API configuration, credentials, quota, or request compatibility, "
                            "then run Architect again."
                        )
                else:
                    print(
                        "Architect could not safely process the model response. "
                        "The session has been preserved; run Architect again after resolving "
                        "the processing issue."
                    )
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. The last submitted answer and completed processing steps are checkpointed; run again to resume.")
    except Exception as exc:
        # Do not overwrite a newer node checkpoint with this stack's older state.
        print(f"Interview stopped ({type(exc).__name__}). Existing checkpoints have been retained; run again to resume.")


if __name__ == "__main__":
    run_cli()
