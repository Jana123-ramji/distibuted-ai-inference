"""Platform metrics for GET /stats and the dashboard activity log.

Plain counters kept in memory - no Prometheus, no database. Every number here is
incremented by real request handling, never seeded with example values.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone

from .models import ActivityEntry

# How many finished jobs the dashboard activity log can show.
_ACTIVITY_LIMIT = 50


class Metrics:
    """Counters for requests, latency and recent activity."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._start = time.monotonic()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self._latency_sum_ms = 0.0
        self._latency_count = 0
        self._activity: deque[ActivityEntry] = deque(maxlen=_ACTIVITY_LIMIT)

    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._start, 1)

    async def record_request(self) -> None:
        """Called as soon as an inference request is accepted."""
        async with self._lock:
            self.total_requests += 1

    async def record_success(
        self, request_id: str, worker_id: str, latency_ms: float, prompt: str
    ) -> None:
        async with self._lock:
            self.successful_requests += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1
            self._activity.appendleft(
                ActivityEntry(
                    request_id=request_id,
                    worker_id=worker_id,
                    status="success",
                    latency_ms=round(latency_ms, 1),
                    prompt_preview=_preview(prompt),
                    timestamp=datetime.now(timezone.utc),
                )
            )

    async def record_failure(
        self, request_id: str, worker_id: str | None, latency_ms: float, prompt: str
    ) -> None:
        async with self._lock:
            self.failed_requests += 1
            self._activity.appendleft(
                ActivityEntry(
                    request_id=request_id,
                    worker_id=worker_id,
                    status="failed",
                    latency_ms=round(latency_ms, 1),
                    prompt_preview=_preview(prompt),
                    timestamp=datetime.now(timezone.utc),
                )
            )

    async def average_latency_ms(self) -> float:
        async with self._lock:
            if not self._latency_count:
                return 0.0
            return round(self._latency_sum_ms / self._latency_count, 1)

    async def recent_activity(self, limit: int = 20) -> list[ActivityEntry]:
        async with self._lock:
            return list(self._activity)[:limit]


def _preview(prompt: str, length: int = 70) -> str:
    """Shorten a prompt for the activity log."""
    single_line = " ".join(prompt.split())
    if len(single_line) <= length:
        return single_line
    return single_line[: length - 3] + "..."
