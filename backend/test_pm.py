"""Interactive PM CLI with durable input and graph-boundary recovery."""
from uuid import uuid4

from agents.state import AgentState, DiscoveryScope
from agents.graph import build_graph
from agents.interview_checkpoint import save_checkpoint, load_checkpoint, session_lock
from agents.llm_errors import LLMCallFailed, ExtractionFailed
from agents.product_model import build_product_model
from services.discovery_session import create_initial_discovery_state
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
        if state.get("prd_confirmation_pending"):
            print("\nType 'generate' to approve the PRD, or 'more' to continue discovery.")
        print("\nYou:")
        answer = read_answer()
        if answer is None:
            print("Progress saved. Run Architect again to resume.")
            return state

        message_kwargs = {}
        if state.get("prd_confirmation_pending"):
            decision = answer.strip().lower()
            if decision in {"generate", "yes", "y"}:
                answer = "Yes, generate the PRD."
                message_kwargs["architect_turn_type"] = "confirm_prd"
            elif decision in {"more", "no", "n"}:
                answer = "There is more I want to cover."
                message_kwargs["architect_turn_type"] = "continue_discovery"
            else:
                print("Please enter 'generate' or 'more'.")
                continue

        state["messages"].append(
            HumanMessage(
                content=answer,
                id=str(uuid4()),
                additional_kwargs=message_kwargs,
            )
        )
        state["turn_count"] = state.get("turn_count", 0) + 1
        state.update(checkpoint_cursor="conversation_manager", interview_status="PROCESSING_ANSWER",
                     active_answer_result=None, extraction_status="PENDING")
        save_checkpoint(state)  # Before any graph/model work.
    return state



def new_session():
    print("\nWhat is your product idea?")
    idea = read_answer()
    if idea is None:
        return None
    state = create_initial_discovery_state(idea)
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
                         prd_contract=None, current_topic=None,
                         current_gap=None, current_role=None, asked_gap=None, active_answer_result=None,
                         awaiting_confirmation=False, prd_confirmation_pending=False,
                         ready_to_compile=False, next_discovery_move=None, checkpoint_cursor="infer_implications",
                         interview_status="ACTIVE", answer_followup=None,
                         requirement_coverage={}, requirement_dependency_state={}, eligible_requirement_keys=[],
                         question_candidates=[], eligible_question_candidates=[], question_candidate_eligibility={},
                         ranked_question_candidates=[], question_candidate_priority={}, requirement_question_history=[],
                         model_implications=[],
                         discovery_threads={}, active_discovery_thread=None,
                         thread_frontier=None, thread_relevant_requirement_ids=[],
                         open_inquiries=[], selected_inquiry=None,
                         planner_source="model", selected_requirement_candidate=None, selected_requirement_priority=None,
                         validation_issues=[], validation_pair_cache={}, validation_blocking=False,
                         validation_candidate_blocking=False, selected_validation_issue=None,
                         question_retry_count=0, question_retry_exhausted=False)
            state["product_model"] = build_product_model(
                state["discovered_knowledge"],
                DiscoveryScope.ADMIN_DASHBOARD,
                state.get("product_concepts", []),
            )
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
        state = new_session()
        if state is None:
            return
        session_id = state["session_id"]
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
