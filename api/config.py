"""Configuration for the API gateway.

Every setting comes from an environment variable (or `.env` for local runs) so
nothing is hardcoded and no secret ever lives in the source code.
See `.env.example` for the documented list of variables.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the API gateway.

    Field name `api_host` is filled from the environment variable `API_HOST`,
    `heartbeat_timeout_seconds` from `HEARTBEAT_TIMEOUT_SECONDS`, and so on
    (matching is case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore worker-only variables that share the same .env
    )

    # --- HTTP server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # --- scheduling / worker bookkeeping ---
    # "round_robin" equally splits requests across workers in rotation; "least_loaded" picks fewest active jobs.
    scheduler_strategy: str = "round_robin"
    # A worker that has not sent a heartbeat within this many seconds is
    # considered unhealthy and is skipped by the scheduler.
    heartbeat_timeout_seconds: float = 15.0
    # How long the API waits for a worker to answer an inference request.
    inference_timeout_seconds: float = 120.0
    # How many different healthy workers to try before returning an error.
    max_worker_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment only once)."""
    return Settings()
