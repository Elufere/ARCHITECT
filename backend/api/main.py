"""FastAPI entry point for the Architect web application."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.diagnostic_log import install_diagnostic_streams
from api.public_demo import PublicDemoRateLimitMiddleware
from api.routes import router


def _frontend_origins() -> list[str]:
    raw = os.getenv("ARCHITECT_FRONTEND_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


install_diagnostic_streams()

app = FastAPI(
    title="Architect API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_frontend_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PublicDemoRateLimitMiddleware)

app.include_router(router)


def _frontend_dist() -> Path | None:
    raw = os.getenv("ARCHITECT_FRONTEND_DIST", "").strip()
    if not raw:
        return None
    path = Path(raw).resolve()
    return path if (path / "index.html").is_file() else None


_frontend = _frontend_dist()
if _frontend is not None:
    assets = _frontend / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_spa(path: str):
        """Serve the built React workspace and preserve client-side routes."""
        requested = (_frontend / path).resolve()
        try:
            requested.relative_to(_frontend)
        except ValueError:
            return FileResponse(_frontend / "index.html")

        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(_frontend / "index.html")
