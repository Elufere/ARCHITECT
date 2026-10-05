"""Central OpenAI configuration and lazy LangChain model construction."""
from functools import lru_cache
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_openai import ChatOpenAI
from openai import APIError, APIConnectionError, APITimeoutError, APIStatusError

from agents.llm_usage import usage_tracker
from agents.llm_errors import LLMCallFailed


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
DEFAULT_FAST_MODEL = "gpt-4.1-mini"
DEFAULT_REASONING_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_CALLS = (
    "discovery_threads.plan",
    "pm_compile.prose",
)


def _model_for_call(call_name: str) -> str:
    """Route expensive reasoning only to calls that benefit from it.

    OPENAI_FORCE_MODEL is an explicit emergency/testing override. The old
    OPENAI_MODEL setting is intentionally not used as a global override because
    that made every extraction/grounding/guardrail call inherit GPT-5.6 pricing.
    """
    forced = os.getenv("OPENAI_FORCE_MODEL", "").strip()
    if forced:
        return forced

    configured = os.getenv("OPENAI_REASONING_CALLS", "")
    reasoning_calls = tuple(
        item.strip()
        for item in configured.split(",")
        if item.strip()
    ) or DEFAULT_REASONING_CALLS

    if call_name in reasoning_calls:
        return os.getenv(
            "OPENAI_REASONING_MODEL",
            DEFAULT_REASONING_MODEL,
        ).strip() or DEFAULT_REASONING_MODEL

    return os.getenv(
        "OPENAI_FAST_MODEL",
        DEFAULT_FAST_MODEL,
    ).strip() or DEFAULT_FAST_MODEL


@lru_cache(maxsize=64)
def _client(call_name, max_tokens):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMCallFailed(call_name)
    model = _model_for_call(call_name)
    model_kwargs = {
        "api_key": key,
        "base_url": "https://api.openai.com/v1",
        "model": model,
        "timeout": float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        # Retries are centralized below; avoid multiplying SDK and node retries.
        "max_retries": 0,
        "max_tokens": max_tokens,
        "stream_usage": True,
        "callbacks": [usage_tracker],
        "metadata": {"openai_call_name": call_name},
    }

    if model.startswith("gpt-5.6"):
        # GPT-5.6 is a reasoning model. Medium is the API default and gives us
        # the cleanest first A/B comparison against the previous model.
        model_kwargs["reasoning_effort"] = os.getenv(
            "OPENAI_REASONING_EFFORT",
            "medium",
        )
    else:
        model_kwargs["temperature"] = 0.0

    return ChatOpenAI(**model_kwargs)


def _api_error_code(error) -> str | None:
    code = getattr(error, "code", None)
    if code:
        return str(code)
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict) and nested.get("code"):
            return str(nested["code"])
        if body.get("code"):
            return str(body["code"])
    return None


def _retryable(error):
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if not isinstance(error, APIStatusError):
        return False
    if _api_error_code(error) in {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "billing_not_active",
    }:
        return False
    return error.status_code in (408, 409, 429) or error.status_code >= 500


def _retry_after_seconds(error) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _retry_wait(error, attempt: int, base_delay: float) -> float:
    explicit = _retry_after_seconds(error)
    if explicit is not None:
        return min(60.0, explicit)
    if isinstance(error, APIStatusError) and error.status_code == 429:
        rate_base = max(
            1.0,
            float(os.getenv("OPENAI_RATE_LIMIT_RETRY_BASE_SECONDS", "10")),
        )
        return min(60.0, rate_base * (2 ** attempt))
    return min(10.0, base_delay * (2 ** attempt))


def _retry_settings(call_name):
    try:
        retries = max(0, min(5, int(os.getenv("OPENAI_MAX_RETRIES", "3"))))
        delay = max(.1, min(10., float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "1"))))
        return retries, delay
    except ValueError as exc:
        raise LLMCallFailed(call_name) from exc


def _with_retry(supplier, call_name):
    def invoke(value, config: RunnableConfig):
        retries, delay = _retry_settings(call_name)
        for attempt in range(retries + 1):
            try:
                return supplier().invoke(value, config=config)
            except APIError as exc:
                retryable = _retryable(exc)
                failure_kind = type(exc).__name__
                status_code = getattr(exc, "status_code", None)
                if not retryable or attempt == retries:
                    raise LLMCallFailed(
                        call_name,
                        retryable=retryable,
                        failure_kind=failure_kind,
                        status_code=status_code,
                    ) from exc
                wait = _retry_wait(exc, attempt, delay)
                diagnostic = failure_kind + (
                    f" status={status_code}" if status_code is not None else ""
                )
                print(
                    f"OpenAI {diagnostic}. Retrying ({attempt + 1}/{retries}) "
                    f"in {wait:g}s... [{call_name}]"
                )
                time.sleep(wait)

    async def ainvoke(value, config: RunnableConfig):
        retries, delay = _retry_settings(call_name)
        for attempt in range(retries + 1):
            try:
                return await supplier().ainvoke(value, config=config)
            except APIError as exc:
                retryable = _retryable(exc)
                failure_kind = type(exc).__name__
                status_code = getattr(exc, "status_code", None)
                if not retryable or attempt == retries:
                    raise LLMCallFailed(
                        call_name,
                        retryable=retryable,
                        failure_kind=failure_kind,
                        status_code=status_code,
                    ) from exc
                wait = _retry_wait(exc, attempt, delay)
                diagnostic = failure_kind + (
                    f" status={status_code}" if status_code is not None else ""
                )
                print(
                    f"OpenAI {diagnostic}. Retrying ({attempt + 1}/{retries}) "
                    f"in {wait:g}s... [{call_name}]"
                )
                await asyncio.sleep(wait)
    return RunnableLambda(invoke, afunc=ainvoke, name=call_name)


def get_chat_model(*, call_name, max_tokens=None):
    # Construction is lazy so imports/offline tooling never require credentials
    # or initialize a network client. Runnable config/callbacks propagate normally.
    return _with_retry(lambda: _client(call_name, max_tokens), call_name)


def get_structured_model(*, call_name, schema, include_raw=False, max_tokens=None):
    @lru_cache(maxsize=1)
    def model():
        # Tool calling with non-strict schemas preserves arbitrary dictionaries,
        # defaults and unions. Existing Pydantic validation remains authoritative.
        return _client(call_name, max_tokens).with_structured_output(
            schema, method="function_calling", strict=False, include_raw=include_raw)
    return _with_retry(model, call_name)
