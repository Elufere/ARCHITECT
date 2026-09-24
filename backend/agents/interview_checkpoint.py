"""Atomic JSON checkpoints at successful graph-node and CLI input boundaries."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from agents.prd_schema import PRDContract
from agents.state import DiscoveryScope, DiscoveryTopic, KnowledgeItem, TopicMaturity, TopicStatus


CURSORS = {"conversation_manager", "extract", "plan", "generate", "guardrail", "compile_prd",
           "waiting", "phase_complete", "completed"}


def checkpoint_directory():
    return Path(os.getenv("ARCHITECT_SESSION_DIR", str(Path(__file__).resolve().parents[1] / "sessions")))


def checkpoint_path(session_id):
    # Session identifiers cannot escape the designated directory.
    return checkpoint_directory() / f"{UUID(session_id)}.json"


def _json_value(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key.value if isinstance(key, Enum) else key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def save_checkpoint(state):
    if not state.get("session_id"):
        return  # Library callers can opt in by supplying a session identifier.
    path = checkpoint_path(state["session_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _json_value({key: value for key, value in state.items() if key != "messages"})
    snapshot["messages"] = messages_to_dict(state.get("messages", []))
    last = state.get("messages", [])[-1:]
    pending_answer = (last[0].content if last and isinstance(last[0], HumanMessage)
                      and state.get("checkpoint_cursor") in ("conversation_manager", "extract") else None)
    document = dict(version=1, session_id=state["session_id"], updated_at=datetime.now(timezone.utc).isoformat(),
                    interview_status=state.get("interview_status", "ACTIVE"),
                    pending_answer=pending_answer,
                    pending_question=last[0].content if last and isinstance(last[0], AIMessage) else None,
                    state=snapshot)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(session_id):
    document = json.loads(checkpoint_path(session_id).read_text(encoding="utf-8"))
    if document.get("version") != 1 or document.get("session_id") != str(UUID(session_id)):
        raise ValueError("Unsupported or mismatched interview checkpoint")
    state = document["state"]
    if state.get("checkpoint_cursor") not in CURSORS:
        raise ValueError("Invalid checkpoint processing cursor")
    state["messages"] = messages_from_dict(state.get("messages", []))
    state["discovery_scope"] = DiscoveryScope(state["discovery_scope"])
    if state.get("current_topic"):
        state["current_topic"] = DiscoveryTopic(state["current_topic"])
    state["topic_status"] = {DiscoveryTopic(key): TopicStatus(value) for key, value in state.get("topic_status", {}).items()}
    state["topic_maturity"] = {DiscoveryTopic(key): TopicMaturity(value) for key, value in state.get("topic_maturity", {}).items()}
    state["discovered_knowledge"] = [KnowledgeItem.model_validate(item) for item in state.get("discovered_knowledge", [])]
    if state.get("prd_contract"):
        state["prd_contract"] = PRDContract.model_validate(state["prd_contract"])
    if (state.get("answer_followup") or {}).get("scope"):
        state["answer_followup"]["scope"] = DiscoveryScope(state["answer_followup"]["scope"])
    return state


def unfinished_sessions():
    sessions = []
    for path in sorted(checkpoint_directory().glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("interview_status") != "COMPLETED":
                sessions.append(document)
        except (ValueError, OSError):
            print(f"Checkpoint could not be read: {path.name}; it has not been overwritten.")
    return sessions


@contextmanager
def session_lock(session_id):
    path = checkpoint_path(session_id).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("This interview is already open in another process") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def durable_node(name, node, next_node):
    def invoke(state):
        # An unsuccessful node cannot mutate the caller's last durable state.
        update = node(deepcopy(state))
        merged = {**state, **update}
        if "messages" in update:
            merged["messages"] = add_messages(deepcopy(state.get("messages", [])), update["messages"])
        cursor = next_node(merged)
        status = {"conversation_manager": "PROCESSING_ANSWER", "extract": "PROCESSING_ANSWER",
                  "plan": "ACTIVE", "generate": "GENERATING_QUESTION", "guardrail": "VALIDATING_QUESTION",
                  "compile_prd": "COMPILING_PRD", "waiting": "WAITING_FOR_USER",
                  "phase_complete": "AWAITING_PHASE_CHOICE", "completed": "COMPLETED"}[cursor]
        metadata = dict(checkpoint_cursor=cursor, interview_status=status)
        if name == "guardrail" and cursor == "waiting" and merged.get("current_gap"):
            metadata["asked_gap"] = dict(scope=merged["discovery_scope"].value,
                topic=merged["current_topic"].value, gap=merged["current_gap"],
                question=merged["messages"][-1].content)
        merged.update(metadata)
        save_checkpoint(merged)
        # Return the same message objects/IDs written to disk so reducers and
        # resumed runs see identical history, including guardrail feedback.
        return {**update, **metadata, **({"messages": merged["messages"]} if "messages" in update else {})}
    return invoke
