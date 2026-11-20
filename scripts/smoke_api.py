"""Smoke test for the API gateway - real HTTP calls, no mocking.

It exercises the public endpoints and, to prove the worker registry works, it
performs the same registration/heartbeat calls a real worker makes. The
registration is removed again at the end, so the API is left clean.

    python scripts/smoke_api.py [http://127.0.0.1:8000]
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def show(label: str, response: httpx.Response) -> None:
    print(f"===== {label} -> HTTP {response.status_code} =====")
    try:
        print(json.dumps(response.json(), indent=2)[:1200])
    except ValueError:
        print(response.text[:600])
    print()


def main() -> int:
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        show("GET /", client.get("/"))
        show("GET /health", client.get("/health"))
        show("GET /workers (before registration)", client.get("/workers"))
        show("POST /infer (expect 503, no workers yet)", client.post("/infer", json={"prompt": "hello"}))
        show("POST /infer (expect 422, empty prompt)", client.post("/infer", json={"prompt": ""}))

        # --- the two calls every worker makes on startup -------------------
        worker = {
            "worker_id": "smoke-worker",
            "hostname": "smoke-host",
            "url": "http://127.0.0.1:8001",
            "backend": "fallback",
            "model_id": "smoke-test-model",
            "status": "healthy",
        }
        show("POST /workers/register", client.post("/workers/register", json=worker))
        show(
            "POST /workers/heartbeat",
            client.post(
                "/workers/heartbeat",
                json={
                    "worker_id": "smoke-worker",
                    "status": "healthy",
                    "current_jobs": 1,
                    "total_jobs": 4,
                    "average_latency_ms": 137.5,
                },
            ),
        )
        show("GET /workers (after registration)", client.get("/workers"))
        show(
            "POST /workers/heartbeat (unknown worker -> 404)",
            client.post("/workers/heartbeat", json={"worker_id": "ghost", "status": "healthy"}),
        )
        show("GET /stats", client.get("/stats"))
        show("GET /activity", client.get("/activity"))

        # --- clean up so no leftover entry pollutes the registry -----------
        show("DELETE /workers/smoke-worker", client.delete("/workers/smoke-worker"))
        show("GET /workers (after cleanup)", client.get("/workers"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
