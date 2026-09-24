# OpenAI configuration and usage

All generative model calls use `agents/llm.py`, including extraction and its
repair/audit calls, question generation, guardrail evaluation, PRD compilation,
and independent PRD validation. The opt-in probe scripts use the same factory.
The model, response schemas, token logging and compilation context budget are
unchanged by the discovery-coverage/checkpoint update. See DISCOVERY_RECOVERY.md.

`OPENAI_API_KEY` is the only credential setting. Process environment variables
take precedence over `backend/.env`, then the repository-root `.env`. Neither
credentials nor prompts are included in the new usage logs. Client construction
is lazy; importing agent modules does not create a client or call the API.

| Setting | Default | Meaning |
| --- | --- | --- |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Shared model for every call; a snapshot ID is also accepted. |
| `OPENAI_TIMEOUT_SECONDS` | `60` | API request timeout. |
| `OPENAI_MAX_RETRIES` | `3` | Central retries after the initial attempt; capped at 5. SDK retries are disabled to avoid multiplication. |
| `OPENAI_RETRY_BASE_SECONDS` | `1` | Exponential retry delay, capped at 10 seconds per wait. |
| `OPENAI_PRICING_JSON` | Built-in rates below | Optional map from exact model IDs to input/cached_input/output USD prices per million tokens. |

Temperature is centrally fixed at zero to preserve the deterministic configuration.
The existing compilation output caps remain 4096 tokens for drafts and 1024 for
verification. Structured responses use non-strict OpenAI function calling with
unchanged Pydantic schemas and validation; no strict-schema rewrite is needed.

`agents/llm_usage.py` receives LangChain callbacks before structured parsing.
Each call logs its name, actual response model, actual input/output/total tokens,
and estimated cost. It prefers `AIMessage.usage_metadata`, falling back to raw
OpenAI token-usage metadata. It never estimates tokens from strings. Parse failures
still count; errors without returned usage are labelled unknown, not zero cost.

`build_graph()` attaches the same handler and a session ID through Runnable config.
One graph invocation is one user turn, labelled with the existing `turn_count`
(the CLI starts at 0). Callback run IDs isolate concurrent turns and include
retries, extraction audits, generation, evaluation, and compilation. Final summaries
print on graph success or failure without changing state or node outputs. A graph
instance defines an in-process usage session; create separate graphs or override
config metadata `openai_usage_session` for distinct conversations. Session totals
are in memory only and reset on process restart. Standalone model invocations log
per-call usage but have no graph-turn aggregate.
The resumable CLI uses its durable session ID for this metadata. A resumed attempt
can print another summary with the same turn number; usage totals themselves are
not persisted. Existing `OPENAI_MAX_RETRIES=0` settings explicitly disable retries.

Standard `gpt-4.1-mini` and `gpt-4.1-mini-2025-04-14` prices are $0.40 input,
$0.10 cached input, and $1.60 output per million tokens, verified 2026-09-24:
https://developers.openai.com/api/docs/models/gpt-4.1-mini

Cost = ((input - cached input) × input rate + cached input × cached rate
+ output × output rate) / 1,000,000. Unknown models require an explicit pricing
entry; summaries report partial/unavailable cost instead of silently applying the
default model's rates. These are standard text-token estimates, not billing totals;
discounts, taxes, service tiers, and unreported usage are not included.

Historical JSON replay outputs and historical regression reports retain their
original provider/model metadata. Chroma's existing local embedding function is
unchanged: it is a vector-retrieval component, not a generative LLM; changing it
would require rebuilding the persisted vector index and is outside this migration.

No tests, application execution, or API requests were performed during migration.
Live probes and opt-in live tests now consume OpenAI API credits when explicitly run.
