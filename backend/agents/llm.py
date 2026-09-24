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
DEFAULT_MODEL = "gpt-4.1-mini"


@lru_cache(maxsize=64)
def _client(call_name, max_tokens):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMCallFailed(call_name)
    return ChatOpenAI(
        api_key=key,
        base_url="https://api.openai.com/v1",
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0.0,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        # Retries are centralized below; avoid multiplying SDK and node retries.
        max_retries=0,
        max_tokens=max_tokens,
        stream_usage=True,
        callbacks=[usage_tracker],
        metadata={"openai_call_name": call_name},
    )


def _retryable(error):
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    return (isinstance(error, APIStatusError)
            and (error.status_code in (408, 409, 429) or error.status_code >= 500)
            and getattr(error, "code", None) not in ("insufficient_quota", "billing_hard_limit_reached"))


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
                if not retryable or attempt == retries:
                    raise LLMCallFailed(call_name, retryable=retryable) from exc
                wait = min(10., delay * 2 ** attempt)
                print(f"OpenAI connection/service failure. Retrying ({attempt + 1}/{retries}) in {wait:g}s... [{call_name}]")
                time.sleep(wait)

    async def ainvoke(value, config: RunnableConfig):
        retries, delay = _retry_settings(call_name)
        for attempt in range(retries + 1):
            try:
                return await supplier().ainvoke(value, config=config)
            except APIError as exc:
                retryable = _retryable(exc)
                if not retryable or attempt == retries:
                    raise LLMCallFailed(call_name, retryable=retryable) from exc
                wait = min(10., delay * 2 ** attempt)
                print(f"OpenAI connection/service failure. Retrying ({attempt + 1}/{retries}) in {wait:g}s... [{call_name}]")
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
