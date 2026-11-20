"""Start the complete cluster locally: API Gateway + 4 Workers.

Runs:
  * API Gateway on http://127.0.0.1:8000
  * 4 Worker instances on ports 8001, 8002, 8003, 8004

Press Ctrl+C to stop all services cleanly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

NUM_WORKERS = 4
BASE_WORKER_PORT = 8001
API_PORT = 8000


def main() -> int:
    procs: list[subprocess.Popen] = []

    def cleanup(*_args: object) -> None:
        print("\n[CLUSTER] Stopping all services...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("[CLUSTER] All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    env_base = os.environ.copy()
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base["SCHEDULER_STRATEGY"] = "round_robin"

    # 1. Start API Gateway
    print(f"[CLUSTER] Starting API Gateway on http://127.0.0.1:{API_PORT}...")
    api_proc = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(API_PORT),
        ],
        cwd=str(ROOT),
        env=env_base,
    )
    procs.append(api_proc)

    # Wait briefly for API Gateway to initialize
    time.sleep(2)

    # 2. Start Workers
    for i in range(1, NUM_WORKERS + 1):
        worker_id = f"worker-{i}"
        worker_port = BASE_WORKER_PORT + (i - 1)
        w_env = env_base.copy()
        w_env["WORKER_ID"] = worker_id
        w_env["WORKER_PORT"] = str(worker_port)
        w_env["WORKER_ADVERTISE_HOST"] = "127.0.0.1"
        w_env["API_URL"] = f"http://127.0.0.1:{API_PORT}"

        print(f"[CLUSTER] Starting {worker_id} on port {worker_port}...")
        w_proc = subprocess.Popen(
            [
                PYTHON,
                "-m",
                "uvicorn",
                "worker.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(worker_port),
            ],
            cwd=str(ROOT),
            env=w_env,
        )
        procs.append(w_proc)
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"[CLUSTER] All {NUM_WORKERS} workers and API Gateway are running!")
    print(f"[CLUSTER] Web Dashboard : http://localhost:{API_PORT}/dashboard")
    print(f"[CLUSTER] Interactive Docs: http://localhost:{API_PORT}/docs")
    print("=" * 60 + "\n")

    try:
        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"[CLUSTER] Warning: A process exited with code {p.returncode}")
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
