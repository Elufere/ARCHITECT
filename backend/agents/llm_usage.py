"""OpenAI response-metadata accounting, separate from graph state and outputs."""
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
import json
import os
from threading import RLock

from langchain_core.callbacks import BaseCallbackHandler


# Standard text-token USD prices per million; verified against OpenAI's model
# page 2026-09-24. Unknown models are unpriced, never assigned another model's rate.
DEFAULT_PRICING = {
    name: {"input": "0.40", "cached_input": "0.10", "output": "1.60"}
    for name in ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14")
}


def estimated_cost(model, usage):
    prices = {**DEFAULT_PRICING, **json.loads(os.getenv("OPENAI_PRICING_JSON", "{}"))}
    rates = prices.get(model)
    if rates is None:
        return None
    cached = usage.get("input_token_details", {}).get("cache_read", 0)
    return (Decimal(usage["input_tokens"] - cached) * Decimal(str(rates["input"]))
             + Decimal(cached) * Decimal(str(rates["cached_input"]))
             + Decimal(usage["output_tokens"]) * Decimal(str(rates["output"]))) / Decimal(1_000_000)


@dataclass
class Totals:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: Decimal = Decimal(0)
    unpriced: int = 0
    missing_usage: int = 0

    def add(self, other):
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def lines(self):
        price = f"${self.cost:.8f}"
        if self.unpriced or self.missing_usage:
            price += f" (partial; {self.unpriced} unpriced, {self.missing_usage} calls without usage)"
        return (f"  input: {self.input_tokens:,}\n  output: {self.output_tokens:,}\n"
                f"  total: {self.total_tokens:,}\n  estimated cost: {price}")


@dataclass
class Turn:
    root_id: object
    session: str
    number: int
    calls: dict = field(default_factory=lambda: defaultdict(Totals))


class OpenAIUsageTracker(BaseCallbackHandler):
    """Shared handler on graph and models; callback run IDs isolate concurrent turns.

    LangChain deduplicates the identical handler when merging model and graph
    callbacks. Counts are taken before parsing so parse failures still cost tokens.
    """
    run_inline = True
    raise_error = False

    def __init__(self):
        self._lock = RLock()
        self._chains = {}
        self._calls = {}
        self._sessions = defaultdict(Totals)
        self._session_turns = defaultdict(int)

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        with self._lock:
            turn = self._chains.get(parent_run_id)
            session = (metadata or {}).get("openai_usage_session")
            if turn is None and session and isinstance(inputs, dict) and "messages" in inputs:
                turn = Turn(run_id, session, inputs.get("turn_count", 0))
            if turn is not None:
                self._chains[run_id] = turn

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            metadata=None, invocation_params=None, **kwargs):
        with self._lock:
            meta, params = metadata or {}, invocation_params or {}
            self._calls[run_id] = (self._chains.get(parent_run_id),
                                  meta.get("openai_call_name", "unlabelled"),
                                  params.get("model") or meta.get("ls_model_name", "unknown"))

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            call = self._calls.pop(run_id, None)
            if call is None:
                return
            turn, name, requested_model = call
            generation = response.generations[0][0] if response.generations and response.generations[0] else None
            message = getattr(generation, "message", None)
            meta = getattr(message, "response_metadata", {}) or {}
            output = response.llm_output or {}
            model = meta.get("model_name") or output.get("model_name") or requested_model
            usage = getattr(message, "usage_metadata", None)
            if not usage:
                tokens = meta.get("token_usage") or output.get("token_usage")
                if tokens and all(key in tokens for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                    usage = dict(input_tokens=tokens["prompt_tokens"], output_tokens=tokens["completion_tokens"],
                                 total_tokens=tokens["total_tokens"], input_token_details={
                                     "cache_read": (tokens.get("prompt_tokens_details") or {}).get("cached_tokens", 0)})
            if not usage:
                totals = Totals(missing_usage=1)
                detail = "  token usage: unavailable (not estimated)"
            else:
                try:
                    cost = estimated_cost(model, usage)
                except (ValueError, KeyError, TypeError, ArithmeticError):
                    cost = None
                totals = Totals(usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
                                cost or Decimal(0), int(cost is None))
                detail = totals.lines()
            print(f"===== OPENAI CALL {run_id} =====\ncall: {name}\nmodel: {model}\n{detail}")
            if turn:
                turn.calls[name].add(totals)

    def on_llm_error(self, error, *, run_id, **kwargs):
        # A transport error may not return usage. Do not pretend that it was free.
        with self._lock:
            call = self._calls.pop(run_id, None)
            if call:
                turn, name, model = call
                print(f"===== OPENAI CALL {run_id} =====\ncall: {name}\nmodel: {model}\n  failed; token usage unavailable")
                if turn:
                    turn.calls[name].add(Totals(missing_usage=1))

    def _finish(self, run_id, failed=False):
        with self._lock:
            turn = self._chains.pop(run_id, None)
            if turn is None or turn.root_id != run_id:
                return
            total = Totals()
            lines = [f"===== OPENAI USAGE — TURN {turn.number} =====", f"session: {turn.session}"]
            if failed:
                lines.append("graph outcome: failed")
            for name, usage in sorted(turn.calls.items()):
                lines.extend([f"{name}:", usage.lines()])
                total.add(usage)
            self._sessions[turn.session].add(total)
            self._session_turns[turn.session] += 1
            lines.extend(["--------------------------------", "TURN TOTAL", total.lines(),
                          "SESSION USAGE", f"turns: {self._session_turns[turn.session]}",
                          self._sessions[turn.session].lines(), "================================="])
            print("\n".join(lines))
            self._chains = {key: value for key, value in self._chains.items() if value is not turn}

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, failed=True)


usage_tracker = OpenAIUsageTracker()
