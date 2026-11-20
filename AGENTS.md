# AGENTS.md — Working rules for this project

This file is the contract for anyone (human or AI agent) working on this repository.
Read it before changing code.

## 1. What this project is

A **local distributed AI inference platform**. A user sends a prompt to a central
API; the API's scheduler picks a healthy worker; that worker runs a *real* AI model
and returns the result. Runs on one laptop with Docker Compose, then on Kubernetes
(Minikube) with 3 worker replicas.

Request path (must remain true in code, not just in docs):

```
Dashboard/curl -> API Gateway (FastAPI) -> Scheduler (least-loaded)
   -> Worker pod (HTTP) -> Mojo preprocessing -> InferenceEngine (MAX | CPU fallback)
   -> result -> API -> user
```

## 2. Hard rules (non-negotiable)

1. **No fake features.** No hardcoded worker lists, no canned inference strings, no
   fake stats, no static dashboard data. Every number shown must come from the API.
2. **Real inference only.** Both backends (MAX and the CPU fallback) must run an
   actual model. If a model cannot load, the worker reports `status=unhealthy`; it
   must never invent a response.
3. **Never invent MAX/Mojo APIs.** Before writing MAX code, check the *installed*
   version (`max --version`, `python -c "import max; print(max.__version__)"`) and
   use that version's documented API. Same for Mojo syntax (`mojo --version`).
4. **No secrets in code.** Configuration comes from environment variables only.
   `.env.example` is committed; `.env` is git-ignored and never created with secrets.
5. **No hardcoded hostnames.** Workers find the API via `API_URL`; the API reaches
   workers via the address each worker *registers*. Never hardcode `localhost` on a
   path that must work in Kubernetes.
6. **No hardcoded worker IDs.** `WORKER_ID` derives from the Kubernetes pod name
   (`POD_NAME`) or the container hostname.
7. **Keep it simple / student-explainable.** No Kafka, no Redis, no Postgres, no
   auth, no React, no cloud. In-memory state in the API is intentional and must be
   documented as a known limitation.
8. **Test before moving on.** Each phase must actually run (command + observed
   output) before the next phase starts. Never hide an error; read it, fix it, re-run.

## 3. Environment facts (verified 2026-08-31 on this machine)

| Tool | Status |
|---|---|
| Python | 3.12.2 (Windows) — OK |
| pip | 24.0, PyPI reachable — OK |
| git | 2.50.1 — OK (repo not yet initialised) |
| Docker / docker compose | **not installed** |
| kubectl / minikube | **not installed** |
| WSL2 distro | **not installed** |
| MAX / Mojo / pixi | **not installed** |

Consequences:

* Phases 2–6, 15–18 are developed and tested natively with Windows Python 3.12.
* **MAX and Mojo do not run on native Windows.** They run on Linux/macOS, so they
  are exercised inside the Linux worker Docker image (or WSL2). Latest upstream
  release at project start: **MAX 26.5 / Mojo 1.0.0**; install with
  `pixi add max-all` or `pip install max[all]`. The exact version actually used
  must be recorded in the README once installed.
* Docker Desktop (with the Kubernetes/WSL2 backend) and Minikube must be installed
  by the user before Phases 9–13 can be verified. Manifests/compose files are still
  written and reviewed, but they are only marked "done" after a real run.

## 4. Layout

```
api/        FastAPI gateway: routes, scheduler, worker registry, stats
worker/     Worker service: HTTP API, heartbeat, inference engines
  inference_engine.py  abstraction: load_model() / infer() / health_check()
  max_engine.py        MAX backend (primary) — all MAX code lives here only
  fallback_engine.py   CPU backend (development) — real model, no MAX
mojo/       preprocess.mojo — CPU-bound prompt preprocessing, called by worker
dashboard/  index.html / style.css / app.js (served by the API)
k8s/        namespace, configmap, api deployment+service, worker deployment+service
tests/      pytest: test_api.py, test_scheduler.py, test_worker.py
scripts/    small helper scripts for the demo
docs/       screenshots for the README
```

## 5. Conventions

* **Ports:** API `8000`, worker `8001` (compose maps workers to 8001/8002/8003).
* **Config:** read env vars in one place per service; document every var in
  `.env.example`. Key vars: `API_URL`, `WORKER_ID`, `POD_NAME`, `POD_IP`,
  `WORKER_PORT`, `HEARTBEAT_INTERVAL_SECONDS`, `HEARTBEAT_TIMEOUT_SECONDS`,
  `INFERENCE_BACKEND` (`auto|max|fallback`), `MODEL_ID`, `MOJO_ENABLED`,
  `SCHEDULER_STRATEGY`, `LOG_LEVEL`.
* **Code style:** type hints everywhere, Pydantic models for every request/response,
  `async` for I/O, small modules, comments only where the logic is non-obvious.
* **Logging:** `logging` module, never bare `print`. Format:
  `[API] ...`, `[SCHEDULER] ...`, `[WORKER <id>] ...`.
* **JSON shape:** every response has `success: bool`; inference adds
  `request_id`, `worker_id`, `result`, `latency_ms`, `timestamp`; errors add `error`.
* **Scheduling:** default `least_loaded` — fewest `current_jobs`, tie-break on lower
  `average_latency`. Unhealthy/stale workers are excluded. On worker failure, retry
  the next healthy worker.

## 6. Phase status (update as work completes)

| # | Phase | Status |
|---|---|---|
| 1 | Project structure + AGENTS.md | done |
| 2 | FastAPI API gateway | done — endpoints live; `/infer` dispatch waits for Phase 5 |
| 3 | Worker service | todo |
| 4 | Registration + heartbeat | todo |
| 5 | Scheduler | todo |
| 6 | Real inference engine (CPU fallback) | todo |
| 7 | MAX integration | todo |
| 8 | Mojo preprocessing | todo |
| 9 | Dockerfiles | todo |
| 10 | Docker Compose | todo |
| 11 | Compose test with 3 workers | todo |
| 12 | Kubernetes manifests | todo |
| 13 | Minikube deployment | todo |
| 14 | Dashboard | todo |
| 15 | Monitoring / statistics | todo |
| 16 | Failure recovery | todo |
| 17 | Tests (pytest) | todo |
| 18 | README | todo |
| 19 | End-to-end verification | todo |

## 7. Model choice (decided in Phase 6)

Default: a small instruct model that runs on a CPU-only student laptop —
`HuggingFaceTB/SmolLM2-135M-Instruct` (~270 MB) with
`Qwen/Qwen2.5-0.5B-Instruct` as the larger option. Chosen because the same model
family is supported by both MAX and the `transformers` CPU fallback, so the two
backends are comparable. Model id is configurable via `MODEL_ID`.

## 8. Definition of done for a phase

Implemented → runs locally → command and real output shown to the user →
files listed → remaining problems stated honestly.
