"""In-memory registry of inference workers.

Workers are *not* configured in a file: each worker calls
`POST /workers/register` when it starts and then sends `POST /workers/heartbeat`
every few seconds. The registry keeps what it was told plus what the API itself
observed (dispatched jobs, failures), and derives the effective status.

Why in-memory (and not Redis/Postgres)? Because the registry is short-lived,
self-healing state: if the API restarts, every worker re-registers on its next
heartbeat. This keeps the project simple, which is a project requirement.
Known limitation: with more than one API replica each replica has its own view,
so the API runs with `replicas: 1` in Kubernetes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import HeartbeatRequest, WorkerInfo, WorkerRegisterRequest, WorkerStatus

logger = logging.getLogger(__name__)


def _now() -> datetime:
    """Timezone-aware UTC timestamp (all times in the API are UTC)."""
    return datetime.now(timezone.utc)


@dataclass
class WorkerState:
    """Everything the API knows about one worker."""

    worker_id: str
    hostname: str
    url: str
    backend: str
    model_id: str
    registered_at: datetime
    last_heartbeat: datetime
    reported_status: WorkerStatus = "healthy"

    # Reported by the worker in its heartbeat.
    reported_current_jobs: int = 0
    reported_total_jobs: int = 0
    reported_average_latency_ms: float = 0.0

    # Measured by the API itself.
    inflight_jobs: int = 0  # jobs dispatched by this API and not finished yet
    dispatched_jobs: int = 0  # total jobs dispatched to this worker by the scheduler
    completed_jobs: int = 0  # jobs this API saw finish successfully
    failed_jobs: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    last_error: str | None = None
    consecutive_failures: int = 0

    def load(self) -> int:
        """Current job count used for scheduling.

        The worker's heartbeat can be a few seconds old, so we take the larger
        of "what the worker last reported" and "what this API has in flight".
        That prevents sending three fast requests to the same worker just
        because no heartbeat arrived in between.
        """
        return max(self.reported_current_jobs, self.inflight_jobs)

    def average_latency_ms(self) -> float:
        """Average latency measured by the API, falling back to the worker's."""
        if self.latency_count:
            return round(self.latency_sum_ms / self.latency_count, 1)
        return round(self.reported_average_latency_ms, 1)

    def total_jobs(self) -> int:
        """Finished jobs; prefer the worker's own counter when it is ahead."""
        return max(self.reported_total_jobs, self.completed_jobs + self.failed_jobs)


class WorkerRegistry:
    """Thread/task-safe (asyncio) collection of `WorkerState` objects."""

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self._workers: dict[str, WorkerState] = {}
        self._lock = asyncio.Lock()
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    # -- status helpers ---------------------------------------------------
    def _status_of(self, state: WorkerState, now: datetime) -> WorkerStatus:
        """Derive the status the rest of the system should trust."""
        age = (now - state.last_heartbeat).total_seconds()
        if age > self.heartbeat_timeout_seconds:
            return "unhealthy"  # stopped sending heartbeats (crashed/deleted pod)
        if state.reported_status == "unhealthy":
            return "unhealthy"  # e.g. the model failed to load
        if state.load() > 0:
            return "busy"  # still selectable, just working
        return "healthy"

    def _to_info(self, state: WorkerState, now: datetime) -> WorkerInfo:
        return WorkerInfo(
            worker_id=state.worker_id,
            hostname=state.hostname,
            url=state.url,
            status=self._status_of(state, now),
            current_jobs=state.load(),
            total_jobs=state.total_jobs(),
            average_latency_ms=state.average_latency_ms(),
            last_heartbeat=state.last_heartbeat,
            seconds_since_heartbeat=round((now - state.last_heartbeat).total_seconds(), 1),
            backend=state.backend,
            model_id=state.model_id,
            registered_at=state.registered_at,
            last_error=state.last_error,
        )

    # -- worker lifecycle -------------------------------------------------
    async def register(self, payload: WorkerRegisterRequest) -> WorkerState:
        """Add a worker, or refresh it if the same id registers again."""
        async with self._lock:
            now = _now()
            existing = self._workers.get(payload.worker_id)
            if existing is None:
                state = WorkerState(
                    worker_id=payload.worker_id,
                    hostname=payload.hostname,
                    url=payload.url.rstrip("/"),
                    backend=payload.backend,
                    model_id=payload.model_id,
                    registered_at=now,
                    last_heartbeat=now,
                    reported_status=payload.status,
                )
                self._workers[payload.worker_id] = state
                logger.info(
                    "[API] Worker registered: %s at %s (backend=%s, model=%s)",
                    state.worker_id,
                    state.url,
                    state.backend,
                    state.model_id,
                )
                return state

            # Re-registration: the worker restarted or the API restarted.
            existing.hostname = payload.hostname
            existing.url = payload.url.rstrip("/")
            existing.backend = payload.backend
            existing.model_id = payload.model_id
            existing.reported_status = payload.status
            existing.last_heartbeat = now
            existing.last_error = None
            existing.consecutive_failures = 0
            existing.inflight_jobs = 0
            logger.info("[API] Worker re-registered: %s at %s", existing.worker_id, existing.url)
            return existing

    async def heartbeat(self, payload: HeartbeatRequest) -> bool:
        """Update a worker from its heartbeat.

        Returns False when the worker is unknown, which tells the worker to
        register again (this is how recovery after an API restart works).
        """
        async with self._lock:
            state = self._workers.get(payload.worker_id)
            if state is None:
                return False
            state.last_heartbeat = _now()
            state.reported_status = payload.status
            state.reported_current_jobs = payload.current_jobs
            state.reported_total_jobs = payload.total_jobs
            state.reported_average_latency_ms = payload.average_latency_ms
            logger.debug(
                "[API] Heartbeat from %s (status=%s, current_jobs=%s, total_jobs=%s)",
                payload.worker_id,
                payload.status,
                payload.current_jobs,
                payload.total_jobs,
            )
            return True

    async def deregister(self, worker_id: str) -> bool:
        """Remove a worker (called by the worker on graceful shutdown)."""
        async with self._lock:
            if self._workers.pop(worker_id, None) is None:
                return False
        logger.info("[API] Worker deregistered: %s", worker_id)
        return True

    # -- reads ------------------------------------------------------------
    async def snapshot(self) -> list[WorkerInfo]:
        """All workers, newest status computed, sorted by worker id."""
        async with self._lock:
            now = _now()
            infos = [self._to_info(state, now) for state in self._workers.values()]
        return sorted(infos, key=lambda w: w.worker_id)

    async def available(self) -> list[WorkerState]:
        """Workers the scheduler may use (fresh heartbeat, model loaded)."""
        async with self._lock:
            now = _now()
            return [s for s in self._workers.values() if self._status_of(s, now) != "unhealthy"]

    async def counts(self) -> dict[str, int]:
        """Worker counters used by /health and /stats."""
        async with self._lock:
            now = _now()
            statuses = [self._status_of(s, now) for s in self._workers.values()]
            busy = sum(
                1
                for s in self._workers.values()
                if self._status_of(s, now) != "unhealthy" and s.load() > 0
            )
            return {
                "total": len(statuses),
                "active": sum(1 for st in statuses if st != "unhealthy"),
                "busy": busy,
                "unhealthy": sum(1 for st in statuses if st == "unhealthy"),
            }

    async def job_counts(self) -> dict[str, int]:
        """Finished jobs per worker id (real numbers, for /stats)."""
        async with self._lock:
            return {wid: s.total_jobs() for wid, s in sorted(self._workers.items())}

    # -- job bookkeeping (used by the scheduler in Phase 5) ---------------
    async def job_started(self, worker_id: str) -> None:
        async with self._lock:
            state = self._workers.get(worker_id)
            if state is not None:
                state.inflight_jobs += 1
                state.dispatched_jobs += 1

    async def job_finished(self, worker_id: str, latency_ms: float, success: bool) -> None:
        async with self._lock:
            state = self._workers.get(worker_id)
            if state is None:
                return
            state.inflight_jobs = max(0, state.inflight_jobs - 1)
            if success:
                state.completed_jobs += 1
                state.latency_sum_ms += latency_ms
                state.latency_count += 1
                state.consecutive_failures = 0
                state.last_error = None
            else:
                state.failed_jobs += 1
                state.consecutive_failures += 1

    async def mark_unhealthy(self, worker_id: str, reason: str) -> None:
        """Mark a worker unusable after a failed request (Phase 16 recovery)."""
        async with self._lock:
            state = self._workers.get(worker_id)
            if state is None:
                return
            state.reported_status = "unhealthy"
            state.last_error = reason
        logger.warning("[API] Worker %s marked unhealthy: %s", worker_id, reason)
