from __future__ import annotations

import hashlib
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage


def invoke_structured(model, system_prompt: str, payload: dict, schema):
    result = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ])
    return result if isinstance(result, schema) else schema.model_validate(result)


def recover_evidence(evidence: str, text: str) -> str | None:
    """Verify source provenance only; never decide semantic meaning."""
    if evidence in text:
        return evidence
    tokens = [re.escape(token) for token in evidence.split() if token]
    if not tokens:
        return None
    match = re.search(r"\s+".join(tokens), text)
    return match.group(0) if match else None


def observation_id(turn: int, statement: str, evidence: str) -> str:
    digest = hashlib.sha1(
        f"{turn}|{statement}|{evidence}".encode("utf-8")
    ).hexdigest()[:16]
    return f"obs_{digest}"


def previous_pm_response(state) -> str | None:
    return next(
        (
            item.get("text")
            for item in reversed(state.recent_messages[:-1])
            if item.get("speaker") == "pm"
        ),
        None,
    )


def record_message(state, speaker: str, text: str) -> None:
    state.recent_messages.append({
        "speaker": speaker,
        "text": text,
        "turn": state.turn_count,
    })
    state.recent_messages = state.recent_messages[-12:]
