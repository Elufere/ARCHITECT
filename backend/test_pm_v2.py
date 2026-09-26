"""Interactive CLI for the clean adaptive PM agent."""
from __future__ import annotations

from uuid import uuid4

from agents.llm_errors import ExtractionFailed, LLMCallFailed
from pm_v2.engine import PMDiscoveryEngine
from pm_v2.models import DiscoveryState
from pm_v2.store import load_state, save_state, unfinished_states


def read_answer() -> str | None:
    lines = []
    while True:
        line = input()
        if line.strip() == ".":
            return "\n".join(lines).strip()
        if line.strip().lower() == "quit":
            return None
        lines.append(line)


def choose_state() -> DiscoveryState | None | str:
    pending = unfinished_states()
    if not pending:
        return "new"

    print("\nUnfinished PM v2 interviews:")
    for index, state in enumerate(pending, 1):
        pending_label = state.pending_decision_key or "none"
        print(
            f"{index}. {state.session_id} | turn {state.turn_count} | "
            f"facts={len([f for f in state.facts if f.active])} | "
            f"requirements={len(state.requirements)} | pending={pending_label}"
        )

    while True:
        choice = input(
            "Enter to resume latest, a session number, 'new', or 'quit': "
        ).strip().lower()
        if choice == "quit":
            return None
        if choice == "new":
            return "new"
        if not choice:
            return pending[0]
        if choice.isdigit() and 1 <= int(choice) <= len(pending):
            return pending[int(choice) - 1]
        print("Choose a listed session, 'new', or 'quit'.")


def new_state() -> DiscoveryState | None:
    print("\nWhat is your product idea?")
    idea = read_answer()
    if idea is None:
        return None
    return DiscoveryState(session_id=str(uuid4()), raw_idea="")


def run() -> None:
    print(
        "ARCHITECT PM V2 - ADAPTIVE DISCOVERY\n"
        "Type 'quit' to save and exit. Use '.' on a blank line to send input."
    )
    selected = choose_state()
    if selected is None:
        return
    state = new_state() if selected == "new" else selected
    if state is None:
        return

    engine = PMDiscoveryEngine()

    try:
        if not state.source_turns:
            # The newly entered idea has not been processed yet.
            print("\nRe-enter your product idea to start discovery:")
            first = read_answer()
            if first is None:
                save_state(state)
                return
            result = engine.process(state, first)
            state = result.state
            save_state(state)
            print(f"\nPM Agent: {result.response}")

        while not state.complete:
            print("\nYou:")
            answer = read_answer()
            if answer is None:
                save_state(state)
                print("Progress saved.")
                return
            result = engine.process(state, answer)
            state = result.state
            save_state(state)
            print(f"\nPM Agent: {result.response}")

        print("\nDiscovery complete.")
        print(state.completion_reason)
        save_state(state)

    except (LLMCallFailed, ExtractionFailed) as exc:
        save_state(state)
        print(f"\n{exc}")
        print("The current PM v2 state has been saved; run again to resume.")
    except (KeyboardInterrupt, EOFError):
        save_state(state)
        print("\nStopped. Progress saved.")


if __name__ == "__main__":
    run()
