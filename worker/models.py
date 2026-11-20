"""Pydantic models for the worker's own HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkerInferRequest(BaseModel):
    """POST /infer body (sent by the API gateway's scheduler)."""

    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    request_id: str | None = Field(default=None, description="Correlation id from the gateway.")


class WorkerInferResponse(BaseModel):
    """Successful worker inference reply."""

    model_config = ConfigDict(protected_namespaces=())

    success: bool = True
    worker_id: str
    result: str
    latency_ms: float = Field(description="Inference time measured inside the worker.")
    backend: str
    model_id: str
    mojo_used: bool = Field(description="True when the compiled Mojo preprocessing binary was used.")
    preprocess_ms: float
    input_tokens: int
    output_tokens: int


class WorkerHealthResponse(BaseModel):
    """GET /health - also the source of the heartbeat numbers."""

    model_config = ConfigDict(protected_namespaces=())

    success: bool = True
    worker_id: str
    status: Literal["healthy", "busy", "unhealthy"]
    model_loaded: bool
    backend: str
    model_id: str
    current_jobs: int
    total_jobs: int
    average_latency_ms: float
    uptime_seconds: float
    error: str | None = None


class WorkerInfoResponse(BaseModel):
    """GET /info - static description of this worker."""

    model_config = ConfigDict(protected_namespaces=())

    success: bool = True
    worker_id: str
    hostname: str
    advertise_url: str
    api_url: str
    backend: str
    model_id: str
    max_new_tokens: int
    engine: dict[str, str]
    mojo: dict[str, str]
    python_version: str
    platform: str


class WorkerErrorResponse(BaseModel):
    """Consistent error envelope, same shape as the gateway's."""

    success: bool = False
    error: str
    worker_id: str | None = None
