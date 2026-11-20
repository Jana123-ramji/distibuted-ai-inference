"""Worker configuration - everything from environment variables.

Two values are computed instead of hardcoded, which is what makes the same
image work in Docker Compose and in Kubernetes:

* `worker_id`      -> POD_NAME (Kubernetes) or WORKER_ID or the hostname
* `advertise_url`  -> the URL the API must call back on: WORKER_ADVERTISE_HOST,
                      else POD_IP (Kubernetes), else the hostname (Compose DNS)
"""

from __future__ import annotations

import os
import socket
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Environment-driven settings for one worker process."""

    # `model_id` / `model_*` would collide with pydantic's protected namespace.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- HTTP server ---
    worker_host: str = "0.0.0.0"
    worker_port: int = 8001
    log_level: str = "INFO"

    # --- talking to the API gateway ---
    # Compose: http://api:8000   Kubernetes: http://api-service:8000
    api_url: str = "http://localhost:8000"
    heartbeat_interval_seconds: float = 5.0
    # Identity / callback address. Empty means "detect it at runtime".
    worker_id: str = ""
    worker_advertise_host: str = ""

    # --- inference ---
    # auto = MAX if usable, else the CPU fallback | max | fallback
    inference_backend: str = "auto"
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    # Folder holding the ONNX weights + tokenizer for the CPU fallback backend
    # (filled by `python scripts/download_model.py`).
    model_dir: str = "models/SmolLM2-135M-Instruct"
    max_new_tokens: int = 96

    # --- Mojo preprocessing ---
    mojo_enabled: bool = True
    mojo_binary: str = "mojo/preprocess"

    # -- derived values ---------------------------------------------------
    def resolved_worker_id(self) -> str:
        """Stable worker id, never hardcoded.

        In Kubernetes the pod name is injected as POD_NAME, so three replicas
        automatically become e.g. worker-7d9c5-abcde, worker-7d9c5-fghij, ...
        """
        pod_name = os.getenv("POD_NAME", "").strip()
        if pod_name:
            return pod_name
        if self.worker_id.strip():
            return self.worker_id.strip()
        return socket.gethostname()

    def resolved_advertise_url(self) -> str:
        """URL the API gateway uses to reach this worker."""
        host = self.worker_advertise_host.strip() or os.getenv("POD_IP", "").strip()
        if not host:
            # Docker Compose / Kubernetes both resolve the container hostname.
            host = socket.gethostname()
        return f"http://{host}:{self.worker_port}"


@lru_cache
def get_worker_settings() -> WorkerSettings:
    """Cached settings instance (environment read once)."""
    return WorkerSettings()
