"""Scheduler - picks a worker and forwards the job to it.

Strategy: **least loaded**

    1. consider only workers whose status is not "unhealthy"
       (fresh heartbeat + model loaded)
    2. pick the one with the fewest current jobs
    3. tie-break on the lower average latency, then on worker id (stable)

Fault tolerance: if the selected worker is unreachable or returns an error, it is
marked unhealthy and the job is retried on the next best worker, up to
MAX_WORKER_RETRIES extra attempts. That is what keeps the platform working when
a pod is deleted.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from .config import Settings
from .metrics import Metrics
from .models import ErrorResponse, InferenceResponse
from .worker_registry import WorkerRegistry, WorkerState

logger = logging.getLogger(__name__)


class Scheduler:
    """Selects workers and dispatches inference jobs over HTTP."""

    def __init__(
        self,
        registry: WorkerRegistry,
        metrics: Metrics,
        settings: Settings,
        client: httpx.AsyncClient,
    ) -> None:
        self.registry = registry
        self.metrics = metrics
        self.settings = settings
        self.client = client
        self._rr_index = 0

    # ------------------------------------------------------------------
    def select(self, workers: list[WorkerState], skip: set[str]) -> WorkerState | None:
        """Worker selection that equally splits requests across healthy workers."""
        candidates = sorted(
            [w for w in workers if w.worker_id not in skip],
            key=lambda w: w.worker_id,
        )
        if not candidates:
            return None

        strategy = self.settings.scheduler_strategy.lower()
        if strategy == "round_robin":
            worker = candidates[self._rr_index % len(candidates)]
            self._rr_index = (self._rr_index + 1) % len(candidates)
            return worker

        # Default / balanced least_loaded:
        # Prioritize:
        # 1. Fewest active inflight jobs (w.load())
        # 2. Fewest total dispatched jobs (w.dispatched_jobs) so sequential/parallel requests split equally
        # 3. Lower average latency
        # 4. Worker ID for stable tie-breaking
        return min(candidates, key=lambda w: (w.load(), w.dispatched_jobs, w.average_latency_ms(), w.worker_id))

    # ------------------------------------------------------------------
    async def dispatch(
        self, request_id: str, prompt: str, max_new_tokens: int | None
    ) -> tuple[int, dict]:
        """Send the prompt to a worker and return (http_status, json_body)."""
        tried: set[str] = set()
        attempts = max(1, self.settings.max_worker_retries + 1)
        last_error = "No healthy inference workers available"
        overall_started = time.perf_counter()

        for attempt in range(1, attempts + 1):
            workers = await self.registry.available()
            if attempt == 1:
                logger.info("[SCHEDULER] Available workers: %d", len(workers))

            worker = self.select(workers, tried)
            if worker is None:
                break

            tried.add(worker.worker_id)
            logger.info(
                "[SCHEDULER] Selected %s (load=%d, avg_latency=%.0f ms) attempt %d/%d",
                worker.worker_id,
                worker.load(),
                worker.average_latency_ms(),
                attempt,
                attempts,
            )

            status_code, body, latency_ms = await self._call_worker(
                worker, request_id, prompt, max_new_tokens
            )

            if status_code == 200:
                total_ms = (time.perf_counter() - overall_started) * 1000
                await self.metrics.record_success(request_id, worker.worker_id, total_ms, prompt)
                response = InferenceResponse(
                    request_id=request_id,
                    worker_id=worker.worker_id,
                    result=body.get("result", ""),
                    latency_ms=round(total_ms, 1),
                    worker_latency_ms=body.get("latency_ms"),
                    backend=body.get("backend"),
                    model_id=body.get("model_id"),
                    mojo_used=body.get("mojo_used"),
                    timestamp=datetime.now(timezone.utc),
                )
                logger.info(
                    "[API] Returning result for %s from %s (%.0f ms end-to-end)",
                    request_id,
                    worker.worker_id,
                    total_ms,
                )
                return 200, response.model_dump(mode="json")

            last_error = str(body.get("error", "worker error"))
            logger.warning(
                "[SCHEDULER] Worker %s failed (%s) after %.0f ms; trying another worker",
                worker.worker_id,
                last_error,
                latency_ms,
            )

        # Nothing worked.
        total_ms = (time.perf_counter() - overall_started) * 1000
        await self.metrics.record_failure(request_id, None, total_ms, prompt)
        message = (
            "No healthy inference workers available"
            if not tried
            else f"All {len(tried)} attempted worker(s) failed. Last error: {last_error}"
        )
        logger.error("[API] Request %s failed: %s", request_id, message)
        return 503, ErrorResponse(error=message, request_id=request_id).model_dump(mode="json")

    # ------------------------------------------------------------------
    async def _call_worker(
        self, worker: WorkerState, request_id: str, prompt: str, max_new_tokens: int | None
    ) -> tuple[int, dict, float]:
        """One HTTP call to one worker, with bookkeeping and error handling."""
        payload: dict[str, object] = {"prompt": prompt, "request_id": request_id}
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens

        await self.registry.job_started(worker.worker_id)
        started = time.perf_counter()
        try:
            response = await self.client.post(
                f"{worker.url}/infer",
                json=payload,
                timeout=self.settings.inference_timeout_seconds,
            )
        except httpx.RequestError as exc:
            # Connection refused / DNS failure / timeout: the worker is gone
            # (for example its Kubernetes pod was deleted).
            latency_ms = (time.perf_counter() - started) * 1000
            await self.registry.job_finished(worker.worker_id, latency_ms, success=False)
            await self.registry.mark_unhealthy(
                worker.worker_id, f"unreachable at {worker.url} ({type(exc).__name__})"
            )
            return 503, {"error": f"worker unreachable ({type(exc).__name__})"}, latency_ms

        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 200:
            await self.registry.job_finished(worker.worker_id, latency_ms, success=True)
            return 200, response.json(), latency_ms

        # The worker answered, but with an error (e.g. model not loaded yet).
        await self.registry.job_finished(worker.worker_id, latency_ms, success=False)
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[:200]}
        if response.status_code >= 500:
            await self.registry.mark_unhealthy(
                worker.worker_id, f"HTTP {response.status_code}: {body.get('error', '')[:120]}"
            )
        return response.status_code, body, latency_ms
