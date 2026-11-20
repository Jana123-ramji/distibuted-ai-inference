"""Pydantic models = the JSON contract of the platform.

Everything the API accepts or returns is declared here, so the dashboard, the
worker and the tests all agree on one shape. FastAPI validates requests against
these models automatically and publishes them at /docs.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A worker is either usable ("healthy"), usable but working ("busy"), or not
# usable ("unhealthy" - model failed to load or heartbeats stopped).
WorkerStatus = Literal["healthy", "busy", "unhealthy"]


# ---------------------------------------------------------------------------
# Worker -> API messages (registration and heartbeat)
# ---------------------------------------------------------------------------
class WorkerRegisterRequest(BaseModel):
    """Sent once by a worker when it starts up (POST /workers/register)."""

    # `model_id` would clash with pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())

    worker_id: str = Field(
        min_length=1,
        max_length=128,
        description="Unique worker id, derived from the Kubernetes pod name or hostname.",
    )
    hostname: str = Field(min_length=1, max_length=253, description="Container/pod hostname.")
    url: str = Field(
        min_length=1,
        description="Base URL the API must use to reach this worker, e.g. http://10.1.0.7:8001",
    )
    backend: str = Field(default="unknown", description="Active inference backend: max | fallback")
    model_id: str = Field(default="unknown", description="Model the worker loaded.")
    status: WorkerStatus = Field(default="healthy", description="Worker status at registration time.")


class HeartbeatRequest(BaseModel):
    """Sent every few seconds by a worker (POST /workers/heartbeat)."""

    worker_id: str = Field(min_length=1, max_length=128)
    status: WorkerStatus = "healthy"
    current_jobs: int = Field(default=0, ge=0, description="Jobs the worker is running right now.")
    total_jobs: int = Field(default=0, ge=0, description="Jobs the worker has finished since start.")
    average_latency_ms: float = Field(default=0.0, ge=0, description="Worker's own average latency.")


class SimpleResponse(BaseModel):
    """Generic acknowledgement for registration / heartbeat / deregistration."""

    success: bool = True
    message: str
    worker_id: str | None = None


# ---------------------------------------------------------------------------
# API -> client messages (worker views)
# ---------------------------------------------------------------------------
class WorkerInfo(BaseModel):
    """One worker as shown by GET /workers and on the dashboard cards."""

    model_config = ConfigDict(protected_namespaces=())

    worker_id: str
    hostname: str
    url: str
    status: WorkerStatus
    current_jobs: int
    total_jobs: int
    average_latency_ms: float
    last_heartbeat: datetime
    seconds_since_heartbeat: float
    backend: str
    model_id: str
    registered_at: datetime
    last_error: str | None = None


class WorkersResponse(BaseModel):
    """GET /workers"""

    success: bool = True
    total_workers: int
    healthy_workers: int
    workers: list[WorkerInfo]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class InferenceRequest(BaseModel):
    """POST /infer body."""

    prompt: str = Field(
        min_length=1,
        max_length=4000,
        description="The user prompt to run through the model.",
        examples=["Explain artificial intelligence in simple words"],
    )
    max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=512,
        description="Optional generation length override; the worker default is used when omitted.",
    )


class InferenceResponse(BaseModel):
    """Successful POST /infer reply."""

    model_config = ConfigDict(protected_namespaces=())

    success: bool = True
    request_id: str
    worker_id: str
    result: str
    latency_ms: float = Field(description="End-to-end latency measured by the API.")
    worker_latency_ms: float | None = Field(
        default=None, description="Latency measured inside the worker (inference only)."
    )
    backend: str | None = Field(default=None, description="Backend that produced the result: max | fallback")
    model_id: str | None = None
    mojo_used: bool | None = Field(
        default=None, description="True when the Mojo preprocessing binary handled the prompt."
    )
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Every failure uses this shape, e.g. no workers available."""

    success: bool = False
    error: str
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """GET /health"""

    success: bool = True
    status: Literal["ok"] = "ok"
    service: str = "api-gateway"
    version: str
    uptime_seconds: float
    total_workers: int
    healthy_workers: int
    timestamp: datetime


class RootResponse(BaseModel):
    """GET / - a short, self-documenting index of the API."""

    success: bool = True
    service: str = "Distributed AI Inference Platform - API Gateway"
    version: str
    endpoints: dict[str, str]


class ActivityEntry(BaseModel):
    """One finished job, shown in the dashboard activity log."""

    request_id: str
    worker_id: str | None
    status: Literal["success", "failed"]
    latency_ms: float
    prompt_preview: str
    timestamp: datetime


class StatsResponse(BaseModel):
    """GET /stats - real counters, never invented numbers."""

    success: bool = True
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    total_workers: int
    active_workers: int = Field(description="Workers currently healthy (fresh heartbeat).")
    busy_workers: int = Field(description="Healthy workers with at least one running job.")
    unhealthy_workers: int
    worker_job_counts: dict[str, int] = Field(description="Finished jobs per worker id.")
    uptime_seconds: float
    timestamp: datetime
