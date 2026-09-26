from __future__ import annotations

from .engine import AdaptivePMEngine
from .store import list_states, load_state


def read_message() -> str | None:
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == ".":
            return "\n".join(lines).strip()
        if line.strip().lower() == "quit":
            return None
        lines.append(line)


def choose_state():
    states = [item for item in list_states() if not item.complete]
    if not states:
        return None
    print("\nUnfinished Adaptive PM v2 interviews:")
    for index, state in enumerate(states, 1):
        print(f"{index}. {state.session_id} | turn {state.turn_count} | decisions {len(state.decisions)}")
    choice = input("Enter to resume latest, a session number, 'new', or 'quit': ").strip().lower()
    if choice == "quit":
        return "quit"
    if choice == "new":
        return None
    if not choice:
        return states[0]
    if choice.isdigit() and 1 <= int(choice) <= len(states):
        return states[int(choice) - 1]
    return None


def run():
    print("ARCHITECT — ADAPTIVE PM v2\nType 'quit' to exit. Type '.' on its own line to send.")
    engine = AdaptivePMEngine()
    selected = choose_state()
    if selected == "quit":
        return
    if selected is None:
        print("\nWhat is your product idea?")
        idea = read_message()
        if not idea:
            return
        state, response = engine.start(idea)
        print(f"\nSession: {state.session_id}")
        print(f"\nPM Agent: {response}")
    else:
        state = load_state(selected.session_id)
        print(f"\nSession: {state.session_id}")
        if state.last_question:
            print(f"\nPM Agent: {state.last_question}")

    while not state.complete:
        print("\nYou:")
        message = read_message()
        if message is None:
            print("Progress saved.")
            return
        if not message:
            continue
        response = engine.process_turn(state, message)
        print(f"\nPM Agent: {response}")


if __name__ == "__main__":
    run()
