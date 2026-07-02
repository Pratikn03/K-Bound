"""Rate limiting for the UAIS API — Redis-backed when ``UAIS_REDIS_URL`` is set.

Gate P P5: multi-replica deployments require a shared counter; without Redis the
in-memory fixed-window limiter is used (single-instance only).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

REDIS_URL = os.getenv("UAIS_REDIS_URL", "").strip()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window limiter; uses Redis when ``UAIS_REDIS_URL`` is configured."""

    def __init__(self, app, limit: int, window_seconds: int):
        super().__init__(app)
        self.limit = limit
        self.window_seconds = window_seconds
        self._memory: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._redis = None
        if REDIS_URL:
            try:
                import redis
            except ImportError as err:  # pragma: no cover
                raise RuntimeError(
                    "UAIS_REDIS_URL is set but redis package is not installed"
                ) from err
            self._redis = redis.from_url(REDIS_URL, decode_responses=True)

    async def dispatch(self, request: Request, call_next):
        client = request.client.host if request.client else "unknown"
        path = request.url.path
        if self._redis is not None:
            key = f"uais:rate:{client}:{path}"
            try:
                count = int(self._redis.incr(key))
                if count == 1:
                    self._redis.expire(key, self.window_seconds)
                if count > self.limit:
                    ttl = self._redis.ttl(key)
                    retry = max(int(ttl), 1) if ttl and ttl > 0 else self.window_seconds
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "Rate limit exceeded"},
                        headers={"Retry-After": str(retry)},
                    )
            except Exception:  # pragma: no cover - fail open
                pass
            return await call_next(request)

        key = (client, path)
        now = time.monotonic()
        bucket = self._memory[key]
        while bucket and now - bucket[0] >= self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(self.window_seconds)},
            )
        bucket.append(now)
        return await call_next(request)
