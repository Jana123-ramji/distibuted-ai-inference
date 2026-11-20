"""Inference worker service.

One process = one worker. It:

  * loads a model through the `InferenceEngine` abstraction (MAX or ONNX CPU),
  * registers itself with the API gateway and sends heartbeats,
  * runs real inference for prompts the scheduler sends it.

Endpoints:  GET /health   GET /info   POST /infer

Run locally (from the project root):

    python -m uvicorn worker.main:app --port 8001
"""

from __future__ import annotations

import asyncio
import logging
import platform
import socket
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import WorkerSettings, get_worker_settings
from .heartbeat import HeartbeatClient
from .inference_engine import InferenceEngine, create_engine
from .logging_config import setup_logging
from .models import (
    WorkerErrorResponse,
    WorkerHealthResponse,
    WorkerInferRequest,
    WorkerInferResponse,
    WorkerInfoResponse,
)
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)


class RuntimeState:
    """Live counters for this worker (the source of the heartbeat numbers)."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        self.current_jobs = 0
        self.total_jobs = 0
        self.failed_jobs = 0
        self.model_loaded = False
        self.error: str | None = "model is still loading"
        self._latency_sum_ms = 0.0
        self._latency_count = 0
        self._started = time.monotonic()

    # -- job bookkeeping --
    def job_started(self) -> None:
        self.current_jobs += 1

    def job_finished(self, latency_ms: float, success: bool) -> None:
        self.current_jobs = max(0, self.current_jobs - 1)
        self.total_jobs += 1
        if success:
            self._latency_sum_ms += latency_ms
            self._latency_count += 1
        else:
            self.failed_jobs += 1

    # -- derived values --
    def average_latency_ms(self) -> float:
        if not self._latency_count:
            return 0.0
        return round(self._latency_sum_ms / self._latency_count, 1)

    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._started, 1)

    def status(self) -> str:
        """Status reported to the API gateway.

        While the model is loading (or if loading failed) the worker is honest
        about being unusable, so the scheduler skips it.
        """
        if not self.model_loaded:
            return "unhealthy"
        return "busy" if self.current_jobs > 0 else "healthy"


async def _load_model(app: FastAPI) -> None:
    """Load the model in a worker thread so startup never blocks the event loop."""
    engine: InferenceEngine = app.state.engine
    state: RuntimeState = app.state.runtime
    try:
        await asyncio.to_thread(engine.load_model)
        state.model_loaded = True
        state.error = None
        logger.info(
            "[WORKER %s] Model loaded and ready (backend=%s, model=%s)",
            state.worker_id,
            engine.name,
            engine.model_id,
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash the process
        state.model_loaded = False
        state.error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[WORKER %s] Model loading FAILED (%s) -> reporting status=unhealthy",
            state.worker_id,
            state.error,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire everything together: engine, heartbeat, model loading."""
    settings: WorkerSettings = get_worker_settings()
    setup_logging(settings.log_level)

    worker_id = settings.resolved_worker_id()
    advertise_url = settings.resolved_advertise_url()

    app.state.settings = settings
    app.state.worker_id = worker_id
    app.state.advertise_url = advertise_url
    app.state.runtime = RuntimeState(worker_id)
    app.state.preprocessor = Preprocessor(settings.mojo_enabled, settings.mojo_binary)
    # This model runs on the CPU, so one inference at a time per worker. Extra
    # requests queue up and are visible as `current_jobs` (real load, not fake).
    app.state.infer_lock = asyncio.Semaphore(1)
    app.state.engine = create_engine(
        backend=settings.inference_backend,
        model_id=settings.model_id,
        model_dir=settings.model_dir,
        max_new_tokens=settings.max_new_tokens,
    )

    logger.info(
        "[WORKER %s] Starting v%s (backend=%s, model=%s, api=%s, mojo=%s)",
        worker_id,
        __version__,
        app.state.engine.name,
        settings.model_id,
        settings.api_url,
        app.state.preprocessor.mojo_available(),
    )

    def state_provider() -> dict[str, Any]:
        """Live numbers for each heartbeat."""
        state: RuntimeState = app.state.runtime
        return {
            "status": state.status(),
            "current_jobs": state.current_jobs,
            "total_jobs": state.total_jobs,
            "average_latency_ms": state.average_latency_ms(),
            "backend": app.state.engine.name,
            "model_id": app.state.engine.model_id,
        }

    heartbeat = HeartbeatClient(
        api_url=settings.api_url,
        worker_id=worker_id,
        hostname=socket.gethostname(),
        advertise_url=advertise_url,
        interval_seconds=settings.heartbeat_interval_seconds,
        state_provider=state_provider,
    )
    app.state.heartbeat = heartbeat

    # Register immediately (as "unhealthy" until the model is ready) and start
    # loading the model in the background.
    await heartbeat.register()
    heartbeat.start()
    app.state.load_task = asyncio.create_task(_load_model(app), name="load-model")

    yield

    logger.info("[WORKER %s] Shutting down", worker_id)
    app.state.load_task.cancel()
    await heartbeat.stop()


app = FastAPI(
    title="Distributed AI Inference Platform - Worker",
    description="Runs Mojo preprocessing plus real model inference for one prompt at a time.",
    version=__version__,
    lifespan=lifespan,
)


def _state(request: Request) -> RuntimeState:
    return request.app.state.runtime


# ---------------------------------------------------------------------------
@app.get("/health", response_model=WorkerHealthResponse, tags=["worker"])
async def health(request: Request) -> WorkerHealthResponse:
    """Liveness + the same numbers the heartbeat sends to the API."""
    state = _state(request)
    engine: InferenceEngine = request.app.state.engine
    return WorkerHealthResponse(
        worker_id=state.worker_id,
        status=state.status(),  # type: ignore[arg-type]
        model_loaded=state.model_loaded,
        backend=engine.name,
        model_id=engine.model_id,
        current_jobs=state.current_jobs,
        total_jobs=state.total_jobs,
        average_latency_ms=state.average_latency_ms(),
        uptime_seconds=state.uptime_seconds(),
        error=state.error,
    )


@app.get("/info", response_model=WorkerInfoResponse, tags=["worker"])
async def info(request: Request) -> WorkerInfoResponse:
    """Static description: who am I, what am I running, is Mojo active?"""
    settings: WorkerSettings = request.app.state.settings
    engine: InferenceEngine = request.app.state.engine
    preprocessor: Preprocessor = request.app.state.preprocessor
    return WorkerInfoResponse(
        worker_id=request.app.state.worker_id,
        hostname=socket.gethostname(),
        advertise_url=request.app.state.advertise_url,
        api_url=settings.api_url,
        backend=engine.name,
        model_id=engine.model_id,
        max_new_tokens=settings.max_new_tokens,
        engine=engine.describe(),
        mojo={
            "enabled": str(settings.mojo_enabled),
            "binary": settings.mojo_binary,
            "available": str(preprocessor.mojo_available()),
            "implementation": "mojo" if preprocessor.mojo_available() else "python-reference",
        },
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.machine()}",
    )


@app.post("/infer", response_model=WorkerInferResponse, tags=["worker"])
async def infer(payload: WorkerInferRequest, request: Request):
    """Preprocess (Mojo when available) then run real inference."""
    state = _state(request)
    engine: InferenceEngine = request.app.state.engine
    preprocessor: Preprocessor = request.app.state.preprocessor

    if not state.model_loaded or not engine.health_check():
        return JSONResponse(
            status_code=503,
            content=WorkerErrorResponse(
                error=f"Model not ready: {state.error or 'unknown reason'}",
                worker_id=state.worker_id,
            ).model_dump(),
        )

    request_id = payload.request_id or "-"
    state.job_started()
    started = time.perf_counter()
    success = False
    try:
        logger.info(
            "[WORKER %s] Starting inference (request_id=%s, prompt_chars=%d)",
            state.worker_id,
            request_id,
            len(payload.prompt),
        )
        # 1. Mojo (or Python reference) preprocessing
        pre = await asyncio.to_thread(preprocessor.run, payload.prompt)
        # 2. model inference - serialised because the CPU backend is single-threaded
        async with request.app.state.infer_lock:
            result = await asyncio.to_thread(engine.infer, pre.cleaned, payload.max_new_tokens)
        success = True

        logger.info(
            "[WORKER %s] Inference completed in %.0f ms (%d tokens, preprocess=%.1f ms, mojo=%s)",
            state.worker_id,
            result.latency_ms,
            result.output_tokens,
            pre.duration_ms,
            pre.mojo_used,
        )
        return WorkerInferResponse(
            worker_id=state.worker_id,
            result=result.text,
            latency_ms=result.latency_ms,
            backend=result.backend,
            model_id=result.model_id,
            mojo_used=pre.mojo_used,
            preprocess_ms=pre.duration_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - answer with an error, stay alive
        logger.exception("[WORKER %s] Inference failed: %s", state.worker_id, exc)
        return JSONResponse(
            status_code=500,
            content=WorkerErrorResponse(
                error=f"{type(exc).__name__}: {exc}", worker_id=state.worker_id
            ).model_dump(),
        )
    finally:
        state.job_finished((time.perf_counter() - started) * 1000, success)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    local_settings = get_worker_settings()
    uvicorn.run(
        "worker.main:app",
        host=local_settings.worker_host,
        port=local_settings.worker_port,
        log_level=local_settings.log_level.lower(),
    )
