"""Worker -> API registration and heartbeat.

A worker is never configured in the API. It announces itself:

    1. POST /workers/register   once at startup (and again after a 404)
    2. POST /workers/heartbeat  every HEARTBEAT_INTERVAL_SECONDS
    3. DELETE /workers/{id}     on graceful shutdown

If the worker dies, heartbeats stop and the API marks it unhealthy after
HEARTBEAT_TIMEOUT_SECONDS - that is what makes the failure demo work.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)


class HeartbeatClient:
    """Background task that keeps this worker visible to the API gateway."""

    def __init__(
        self,
        api_url: str,
        worker_id: str,
        hostname: str,
        advertise_url: str,
        interval_seconds: float,
        state_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.worker_id = worker_id
        self.hostname = hostname
        self.advertise_url = advertise_url
        self.interval_seconds = interval_seconds
        # Callback into the worker so the heartbeat always sends live numbers.
        self.state_provider = state_provider

        self._registered = False
        self._task: asyncio.Task[None] | None = None
        self._client = httpx.AsyncClient(timeout=5.0)

    # ------------------------------------------------------------------
    async def register(self) -> bool:
        """Announce this worker; returns True on success."""
        state = self.state_provider()
        payload = {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "url": self.advertise_url,
            "backend": state.get("backend", "unknown"),
            "model_id": state.get("model_id", "unknown"),
            "status": state.get("status", "unhealthy"),
        }
        try:
            response = await self._client.post(f"{self.api_url}/workers/register", json=payload)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - the API may not be up yet
            logger.warning(
                "[WORKER %s] Registration with %s failed (%s); retrying in %.0fs",
                self.worker_id,
                self.api_url,
                type(exc).__name__,
                self.interval_seconds,
            )
            self._registered = False
            return False

        self._registered = True
        logger.info(
            "[WORKER %s] Registered with API at %s (callback url %s)",
            self.worker_id,
            self.api_url,
            self.advertise_url,
        )
        return True

    async def _send_heartbeat(self) -> None:
        state = self.state_provider()
        payload = {
            "worker_id": self.worker_id,
            "status": state.get("status", "unhealthy"),
            "current_jobs": int(state.get("current_jobs", 0)),
            "total_jobs": int(state.get("total_jobs", 0)),
            "average_latency_ms": float(state.get("average_latency_ms", 0.0)),
        }
        response = await self._client.post(f"{self.api_url}/workers/heartbeat", json=payload)
        if response.status_code == 404:
            # The API forgot us (it restarted) - register again on the next tick.
            logger.info("[WORKER %s] API does not know this worker; re-registering", self.worker_id)
            self._registered = False
            return
        response.raise_for_status()
        logger.debug("[WORKER %s] Heartbeat sent: %s", self.worker_id, payload)

    async def _loop(self) -> None:
        """Register, then heartbeat forever (recovering from any error)."""
        while True:
            try:
                if not self._registered:
                    await self.register()
                else:
                    await self._send_heartbeat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never kill the worker
                logger.warning(
                    "[WORKER %s] Heartbeat failed (%s: %s)", self.worker_id, type(exc).__name__, exc
                )
                self._registered = False
            await asyncio.sleep(self.interval_seconds)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background heartbeat task."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name=f"heartbeat-{self.worker_id}")
            logger.info(
                "[WORKER %s] Heartbeat task started (every %.0fs)", self.worker_id, self.interval_seconds
            )

    async def stop(self) -> None:
        """Deregister and stop the background task (graceful shutdown)."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

        try:
            await self._client.delete(f"{self.api_url}/workers/{self.worker_id}")
            logger.info("[WORKER %s] Deregistered from API", self.worker_id)
        except Exception as exc:  # noqa: BLE001 - shutting down anyway
            logger.debug("[WORKER %s] Deregistration skipped (%s)", self.worker_id, type(exc).__name__)
        finally:
            await self._client.aclose()
