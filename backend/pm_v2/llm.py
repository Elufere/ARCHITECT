from __future__ import annotations

import json
from typing import Type

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.llm import get_structured_model


def structured_call(
    *,
    call_name: str,
    schema: Type[BaseModel],
    instruction: str,
    payload: dict,
    max_tokens: int | None = None,
):
    """One batched structured call for one semantic responsibility."""
    model = get_structured_model(
        call_name=f"pm_v2.{call_name}",
        schema=schema,
        max_tokens=max_tokens,
    )
    result = model.invoke([
        SystemMessage(content=instruction),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ])
    return result if isinstance(result, schema) else schema.model_validate(result)
