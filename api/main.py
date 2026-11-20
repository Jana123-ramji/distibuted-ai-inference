"""API Gateway for the Distributed AI Inference Platform.

Responsibilities (Phase 2):
  * accept and validate inference requests            -> POST /infer
  * keep track of the workers that registered         -> POST /workers/register,
                                                         POST /workers/heartbeat,
                                                         DELETE /workers/{id}
  * expose worker status and platform statistics      -> GET /workers, /stats,
                                                         /activity, /health

The actual job dispatch (worker selection + forwarding) is added in Phase 5 by
`api/scheduler.py`; until then /infer answers with a clear error instead of a
fake result.

Run locally from the project root:

    python -m uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings, get_settings
from .logging_config import setup_logging
from .metrics import Metrics
from .models import (
    ActivityEntry,
    ErrorResponse,
    HealthResponse,
    HeartbeatRequest,
    InferenceRequest,
    InferenceResponse,
    RootResponse,
    SimpleResponse,
    StatsResponse,
    WorkerRegisterRequest,
    WorkersResponse,
)
from .scheduler import Scheduler
from .worker_registry import WorkerRegistry

logger = logging.getLogger(__name__)

# dashboard/ lives next to api/ in the repo and is copied next to it in the image.
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the shared state when the API starts, log a clean shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app.state.settings = settings
    app.state.registry = WorkerRegistry(heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds)
    app.state.metrics = Metrics()
    # One shared HTTP client for all worker calls (connection reuse).
    app.state.http_client = httpx.AsyncClient(timeout=settings.inference_timeout_seconds)
    app.state.scheduler = Scheduler(
        registry=app.state.registry,
        metrics=app.state.metrics,
        settings=settings,
        client=app.state.http_client,
    )

    # The real bind address is printed by uvicorn itself, so only log the
    # configuration this app actually decides.
    logger.info(
        "[API] Gateway v%s ready (strategy=%s, heartbeat_timeout=%ss, inference_timeout=%ss)",
        __version__,
        settings.scheduler_strategy,
        settings.heartbeat_timeout_seconds,
        settings.inference_timeout_seconds,
    )
    if DASHBOARD_DIR.is_dir():
        logger.info("[API] Dashboard available at /dashboard")
    yield
    await app.state.http_client.aclose()
    logger.info("[API] Gateway shutting down")


app = FastAPI(
    title="Distributed AI Inference Platform - API Gateway",
    description=(
        "Central entry point: receives inference requests, tracks inference "
        "workers via registration/heartbeat, and reports platform statistics."
    ),
    version=__version__,
    lifespan=lifespan,
)

# The dashboard is plain JavaScript served from this same API, but allowing any
# origin makes it possible to open dashboard/index.html directly from disk while
# developing. This platform is local-only and has no authentication by design.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# small accessors so the handlers stay readable
# ---------------------------------------------------------------------------
def _registry(request: Request) -> WorkerRegistry:
    return request.app.state.registry


def _metrics(request: Request) -> Metrics:
    return request.app.state.metrics


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _error(message: str, code: int, request_id: str | None = None) -> JSONResponse:
    """Return the platform's standard error envelope."""
    body = ErrorResponse(error=message, request_id=request_id)
    return JSONResponse(status_code=code, content=body.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# service information
# ---------------------------------------------------------------------------
@app.get("/", response_model=RootResponse, tags=["info"])
async def root() -> RootResponse:
    """Self-documenting index of the API."""
    return RootResponse(
        version=__version__,
        endpoints={
            "GET /health": "Liveness of the API plus worker counts",
            "GET /workers": "All registered workers with live status",
            "GET /stats": "Platform statistics (requests, latency, workers)",
            "GET /activity": "Most recent inference jobs",
            "POST /infer": "Run inference on the least-loaded healthy worker",
            "POST /workers/register": "Worker self-registration (called by workers)",
            "POST /workers/heartbeat": "Worker heartbeat (called by workers)",
            "DELETE /workers/{worker_id}": "Worker deregistration (called by workers)",
            "GET /docs": "Interactive OpenAPI documentation",
            "GET /dashboard": "Web dashboard (HTML/CSS/JS) served by this API",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["info"])
async def health(request: Request) -> HealthResponse:
    """Used by Docker/Kubernetes probes and by the dashboard."""
    counts = await _registry(request).counts()
    return HealthResponse(
        version=__version__,
        uptime_seconds=_metrics(request).uptime_seconds(),
        total_workers=counts["total"],
        healthy_workers=counts["active"],
        timestamp=_now(),
    )


# ---------------------------------------------------------------------------
# worker registration / heartbeat (called by the worker service)
# ---------------------------------------------------------------------------
@app.post("/workers/register", response_model=SimpleResponse, tags=["workers"])
async def register_worker(payload: WorkerRegisterRequest, request: Request) -> SimpleResponse:
    """A worker announces itself with the URL the API should call back on."""
    state = await _registry(request).register(payload)
    return SimpleResponse(message=f"Worker {state.worker_id} registered", worker_id=state.worker_id)


@app.post(
    "/workers/heartbeat",
    response_model=SimpleResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["workers"],
)
async def worker_heartbeat(payload: HeartbeatRequest, request: Request):
    """Periodic status update from a worker.

    A 404 means "I do not know you" - the worker reacts by registering again,
    which is how the platform recovers if the API restarted.
    """
    known = await _registry(request).heartbeat(payload)
    if not known:
        return _error(f"Unknown worker '{payload.worker_id}', please register first", 404)
    return SimpleResponse(message="Heartbeat accepted", worker_id=payload.worker_id)


@app.delete(
    "/workers/{worker_id}",
    response_model=SimpleResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["workers"],
)
async def deregister_worker(worker_id: str, request: Request):
    """Graceful shutdown path: a stopping worker removes itself immediately."""
    removed = await _registry(request).deregister(worker_id)
    if not removed:
        return _error(f"Unknown worker '{worker_id}'", 404)
    return SimpleResponse(message=f"Worker {worker_id} deregistered", worker_id=worker_id)


@app.get("/workers", response_model=WorkersResponse, tags=["workers"])
async def list_workers(request: Request) -> WorkersResponse:
    """Live view of every worker - this feeds the dashboard worker cards."""
    workers = await _registry(request).snapshot()
    healthy = sum(1 for w in workers if w.status != "unhealthy")
    return WorkersResponse(total_workers=len(workers), healthy_workers=healthy, workers=workers)


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@app.post(
    "/infer",
    response_model=InferenceResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["inference"],
)
async def infer(payload: InferenceRequest, request: Request):
    """Accept a prompt, let the scheduler pick a worker, return the result.

    The scheduler retries on another healthy worker when the selected one fails,
    and answers with the documented error envelope when nothing is available.
    """
    request_id = str(uuid4())
    scheduler: Scheduler = request.app.state.scheduler

    logger.info("[API] Received inference request %s (%d chars)", request_id, len(payload.prompt))
    await _metrics(request).record_request()

    status_code, body = await scheduler.dispatch(request_id, payload.prompt, payload.max_new_tokens)
    return JSONResponse(status_code=status_code, content=body)


# ---------------------------------------------------------------------------
# monitoring
# ---------------------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse, tags=["monitoring"])
async def stats(request: Request) -> StatsResponse:
    """Real platform counters (requests, latency, workers)."""
    metrics = _metrics(request)
    registry = _registry(request)
    counts = await registry.counts()
    return StatsResponse(
        total_requests=metrics.total_requests,
        successful_requests=metrics.successful_requests,
        failed_requests=metrics.failed_requests,
        average_latency_ms=await metrics.average_latency_ms(),
        total_workers=counts["total"],
        active_workers=counts["active"],
        busy_workers=counts["busy"],
        unhealthy_workers=counts["unhealthy"],
        worker_job_counts=await registry.job_counts(),
        uptime_seconds=metrics.uptime_seconds(),
        timestamp=_now(),
    )


@app.get("/activity", response_model=list[ActivityEntry], tags=["monitoring"])
async def activity(request: Request, limit: int = 20) -> list[ActivityEntry]:
    """Most recent finished jobs, newest first (dashboard activity log)."""
    limit = max(1, min(limit, 50))
    return await _metrics(request).recent_activity(limit=limit)


# ---------------------------------------------------------------------------
# dashboard: plain HTML/CSS/JS files served by this same API, so opening
# http://localhost:8000/dashboard is enough - no extra web server needed.
# ---------------------------------------------------------------------------
if DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


if __name__ == "__main__":  # pragma: no cover - convenience for `python api/main.py`
    import uvicorn

    local_settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=local_settings.api_host,
        port=local_settings.api_port,
        log_level=local_settings.log_level.lower(),
    )
