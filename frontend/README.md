# Architect Frontend

React + TypeScript workspace for Architect.

## Run with the real backend

Start FastAPI from the repository's `backend` directory:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Then start React:

```bash
cd frontend
npm install
npm run dev
```

Live API mode is the default. During local development, Vite proxies `/api`
requests to `http://localhost:8000`, so `VITE_API_URL` does not need to be set.

Copy `.env.example` to `.env` only when you need to override the defaults:

```env
VITE_USE_MOCK_API=false
VITE_API_URL=
ARCHITECT_API_PROXY_TARGET=http://localhost:8000
```

For a separately hosted backend, set `VITE_API_URL` to its origin, for example:

```env
VITE_API_URL=https://api.example.com
```

The old in-browser mock workspace remains available only as an explicit UI-development
fallback:

```env
VITE_USE_MOCK_API=true
```

## Product routes

- `/` — projects
- `/projects/new` — create project
- `/projects/:projectId/discovery` — PM interview
- `/projects/:projectId/understanding` — founder-facing product understanding
- `/projects/:projectId/prd` — generated PRD

## API boundary

The React app talks only to product-facing FastAPI endpoints:

- `GET /api/projects`
- `POST /api/projects`
- `GET /api/projects/:projectId/workspace`
- `POST /api/projects/:projectId/discovery/turn`
- `POST /api/projects/:projectId/discovery/retry`
- `POST /api/projects/:projectId/prd/generate`

React does not know about LangGraph nodes, schema gaps, checkpoint cursors, or
knowledge-tracker internals. Durable recovery stays backend-owned; the workspace
snapshot tells the UI when saved work should be resumed.
