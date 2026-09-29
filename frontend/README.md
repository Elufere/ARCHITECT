# Architect Frontend

React + TypeScript workspace for Architect.

## Run

```bash
cd frontend
npm install
npm run dev
```

The frontend runs with a local mock service by default so the UI can be developed
before the Python API is exposed.

To use the live backend later:

```env
VITE_USE_MOCK_API=false
VITE_API_URL=http://localhost:8000
```

## Product routes

- `/` — projects
- `/projects/new` — create project
- `/projects/:projectId/discovery` — PM interview
- `/projects/:projectId/understanding` — founder-facing product understanding
- `/projects/:projectId/prd` — generated PRD

## API boundary

The React app only knows product-level concepts:

- projects
- workspace snapshots
- discovery turns
- PRD generation

It does not know about LangGraph nodes, schema gaps, checkpoint cursors, or
knowledge-tracker internals. Those remain backend implementation details.
