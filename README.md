# Architect

**Architect is an AI Product Manager that turns an early product idea into a grounded, implementation-ready product definition and PRD through a structured discovery conversation.**

**Live demo:** https://architect-y6hr.onrender.com

Instead of generating a PRD from a one-paragraph prompt, Architect interviews the founder, captures only supported product facts, follows meaningful product consequences, tracks unresolved decisions, and synthesizes the confirmed product model into a PRD.

## What Architect does

- Runs a product-discovery interview from an initial founder idea.
- Extracts grounded product facts with exact evidence provenance.
- Models users, capabilities, entities, attributes, states, rules, constraints, integrations, and lifecycle decisions.
- Chooses the next question based on the current product model rather than a fixed checklist.
- Distinguishes product decisions from UI/implementation detail.
- Supports durable sessions and retry/resume after model or network failures.
- Requires founder confirmation before PRD generation.
- Produces a structured PRD from confirmed discovery knowledge.

## Why I built it

Most AI PRD generators produce a document immediately from incomplete input. Architect treats product discovery as the primary problem: the system should know what is established, what is still materially uncertain, and what question has the highest product value next.

The core design principle is:

> **The model may decide what is worth asking next, but it cannot invent product knowledge or silently declare unresolved decisions complete.**

## Architecture

```text
Founder
   |
   v
Conversation / intent handling
   |
   v
Grounded claim extraction + semantic validation
   |
   v
Durable semantic product state
   |
   v
Discovery-thread planner
   |
   v
Question generation + guardrails
   |
   +------> Founder answers ------+
   |                             |
   +-----------------------------+
   |
   v
Founder PRD confirmation
   |
   v
Deterministic PRD projection
   |
   v
Grounded prose polish / validation
```

Architect separates **capture**, **semantic understanding**, and **interview planning** so that one model decision cannot both invent a fact and use that invented fact to close discovery.

## Cost-aware model routing

The production configuration routes model calls by responsibility:

| Work | Default model |
| --- | --- |
| Strategic discovery planning | GPT-5.6 Sol |
| Final PRD prose polish | GPT-5.6 Sol |
| Claim extraction / grounding | GPT-4.1 mini |
| Inquiry assessment / repair | GPT-4.1 mini |
| Question wording / guardrails | GPT-4.1 mini |
| PRD audit / classification | GPT-4.1 mini |

The planner receives a compact semantic snapshot instead of the full evidence ledger on every turn. Full evidence remains available to the grounding and validation layers.

## Tech stack

**Frontend:** React 19, TypeScript, Vite, React Router, TanStack Query, Tailwind CSS

**Backend:** Python, FastAPI, LangGraph, Pydantic, LangChain/OpenAI

**AI:** GPT-5.6 Sol for high-value planning, GPT-4.1 mini for lower-cost mechanical calls

**Persistence:** durable file-backed project/session checkpoints for the current prototype

**Deployment:** Docker + Render

## Run locally

### Backend

```bash
cd backend
python -m venv venv
# activate the virtual environment
pip install -r requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and add your API key:

```env
OPENAI_API_KEY=your_key_here
OPENAI_FAST_MODEL=gpt-4.1-mini
OPENAI_REASONING_MODEL=gpt-5.6-sol
OPENAI_REASONING_CALLS=discovery_threads.plan,pm_compile.prose
```

Start the API:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` requests to the local FastAPI service.

## Production deployment

The repository contains a multi-stage `Dockerfile` that:

1. builds the React frontend;
2. installs the Python backend;
3. copies the frontend build into the runtime image;
4. serves the SPA and API from the same FastAPI origin.

A Render Blueprint is included in `render.yaml`.

To deploy:

1. Connect this GitHub repository to Render.
2. Choose **New → Blueprint**.
3. Select `render.yaml`.
4. Provide `OPENAI_API_KEY` when Render prompts for the secret.
5. Deploy.

The public configuration enables a per-visitor request limit and disables public diagnostic-log downloads.

### Prototype persistence note

The included Render configuration uses the free plan, whose filesystem is ephemeral. That is suitable for a portfolio/demo deployment, but saved projects can disappear after an instance restart or redeploy. Production persistence should be moved to a managed database or a paid persistent disk.

## Environment controls

```env
OPENAI_FAST_MODEL=gpt-4.1-mini
OPENAI_REASONING_MODEL=gpt-5.6-sol
OPENAI_REASONING_CALLS=discovery_threads.plan,pm_compile.prose
OPENAI_REASONING_EFFORT=medium

# Public-demo protection; 0 disables the limit locally.
ARCHITECT_PUBLIC_DEMO_POSTS_PER_HOUR=30
ARCHITECT_ENABLE_DEBUG_LOG_DOWNLOAD=false
```

For a one-model debugging run, `OPENAI_FORCE_MODEL` can explicitly override routing.

## Status

Architect is an actively developed product prototype. The current focus is reliable model-driven discovery, grounded semantic product understanding, cost-efficient orchestration, and higher-quality PRD synthesis.
