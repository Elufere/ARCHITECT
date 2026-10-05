"""Small in-memory guardrail for public portfolio deployments.

This is intentionally infrastructure protection, not authentication. It limits
cost-bearing POST requests per client on a single app instance. Set the limit to
0 (the default) to disable it for local development.
"""
from __future__ import annotations

from collections import defaultdict, deque
import os
from threading import RLock
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def _hourly_post_limit() -> int:
    raw = os.getenv("ARCHITECT_PUBLIC_DEMO_POSTS_PER_HOUR", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


class PublicDemoRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    @staticmethod
    def _client_key(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        limit = _hourly_post_limit()
        if (
            limit <= 0
            or request.method != "POST"
            or not request.url.path.startswith("/api/")
        ):
            return await call_next(request)

        now = time.monotonic()
        cutoff = now - 3600
        key = self._client_key(request)

        with self._lock:
            bucket = self._requests[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": {
                            "message": (
                                "This public Architect demo has reached the hourly "
                                "request limit for this visitor. Please try again later."
                            ),
                            "retryable": True,
                        }
                    },
                )
            bucket.append(now)

        return await call_next(request)
