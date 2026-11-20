/*
 * Dashboard logic.
 *
 * Every number on this page comes from the API - there is no demo data:
 *   GET  /health    -> API status
 *   GET  /stats     -> counters
 *   GET  /workers   -> worker cards
 *   GET  /activity  -> activity log
 *   POST /infer     -> run inference
 *
 * The dashboard is served by the API itself, so the API base is simply the
 * origin this page was loaded from (works the same in Docker and Kubernetes).
 */

const API_BASE = window.location.origin;
const REFRESH_MS = 3000;

document.getElementById("api-base").textContent = API_BASE;

/* ---------------------------------------------------------------- helpers */
function $(id) {
  return document.getElementById(id);
}

function ms(value) {
  if (value === null || value === undefined) return "–";
  const n = Number(value);
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
}

function clockTime(isoString) {
  if (!isoString) return "–";
  const d = new Date(isoString);
  return Number.isNaN(d.getTime()) ? "–" : d.toLocaleTimeString();
}

async function getJSON(path) {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} -> HTTP ${response.status}`);
  return response.json();
}

/* ------------------------------------------------------------ status area */
function setApiState(online, text) {
  $("api-dot").className = `dot ${online ? "dot-ok" : "dot-bad"}`;
  $("api-text").textContent = text;
}

function renderStats(stats) {
  $("stat-total-workers").textContent = stats.total_workers;
  $("stat-healthy").textContent = stats.active_workers;
  $("stat-busy").textContent = stats.busy_workers;
  $("stat-unhealthy").textContent = stats.unhealthy_workers;
  $("stat-total-requests").textContent = stats.total_requests;
  $("stat-success").textContent = stats.successful_requests;
  $("stat-failed").textContent = stats.failed_requests;
  $("stat-latency").textContent = ms(stats.average_latency_ms);
}

function renderWorkers(payload) {
  const container = $("worker-cards");
  const workers = payload.workers || [];

  if (workers.length === 0) {
    container.innerHTML = '<p class="empty">No workers registered yet.</p>';
    return;
  }

  container.innerHTML = workers
    .map((w) => {
      const rows = [
        ["Status", w.status],
        ["Current jobs", w.current_jobs],
        ["Total jobs", w.total_jobs],
        ["Avg latency", ms(w.average_latency_ms)],
        ["Backend", w.backend],
        ["Model", w.model_id],
        ["Last heartbeat", `${w.seconds_since_heartbeat.toFixed(1)} s ago`],
        ["Address", w.url],
      ]
        .map(
          ([label, value]) =>
            `<div class="card-row"><span>${label}</span><span>${String(value)}</span></div>`
        )
        .join("");

      const error = w.last_error
        ? `<div class="card-error">last error: ${w.last_error}</div>`
        : "";

      return `
        <article class="card ${w.status}">
          <div class="card-head">
            <span class="card-id">${w.worker_id}</span>
            <span class="badge ${w.status}">${w.status}</span>
          </div>
          ${rows}
          ${error}
        </article>`;
    })
    .join("");
}

function renderActivity(entries) {
  const body = $("activity-body");
  if (!entries || entries.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="empty">No jobs yet.</td></tr>';
    return;
  }
  body.innerHTML = entries
    .map(
      (e) => `
      <tr>
        <td>${e.request_id.slice(0, 8)}</td>
        <td>${e.worker_id || "–"}</td>
        <td class="status-${e.status}">${e.status}</td>
        <td>${ms(e.latency_ms)}</td>
        <td>${e.prompt_preview}</td>
        <td>${clockTime(e.timestamp)}</td>
      </tr>`
    )
    .join("");
}

/* ------------------------------------------------------------ refresh loop */
async function refresh() {
  try {
    const [health, stats, workers, activity] = await Promise.all([
      getJSON("/health"),
      getJSON("/stats"),
      getJSON("/workers"),
      getJSON("/activity?limit=15"),
    ]);
    setApiState(
      true,
      `API online · v${health.version} · uptime ${Math.round(health.uptime_seconds)} s`
    );
    renderStats(stats);
    renderWorkers(workers);
    renderActivity(activity);
  } catch (error) {
    setApiState(false, `API unreachable (${error.message})`);
  }
}

/* --------------------------------------------------------------- inference */
function showResult(data) {
  $("result-box").classList.remove("hidden");
  $("result-worker").textContent = data.worker_id || "–";
  $("result-latency").textContent = `${ms(data.latency_ms)} (worker ${ms(data.worker_latency_ms)})`;
  $("result-backend").textContent = `${data.backend || "–"} · ${data.model_id || "–"}`;
  $("result-mojo").textContent = data.mojo_used ? "yes (Mojo binary)" : "no (Python reference)";
  $("result-time").textContent = clockTime(data.timestamp);
  $("result-text").textContent = data.result || "(empty response)";
}

function showError(message) {
  $("result-box").classList.remove("hidden");
  $("result-worker").textContent = "–";
  $("result-latency").textContent = "–";
  $("result-backend").textContent = "–";
  $("result-mojo").textContent = "–";
  $("result-time").textContent = clockTime(new Date().toISOString());
  $("result-text").textContent = `Error: ${message}`;
}

async function runInference() {
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    showError("Please enter a prompt.");
    return;
  }

  const button = $("run-btn");
  button.disabled = true;
  $("run-status").textContent = "running inference…";

  const body = { prompt, max_new_tokens: Number($("max-tokens").value) || 64 };

  try {
    const response = await fetch(`${API_BASE}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (response.ok && data.success) {
      showResult(data);
      $("run-status").textContent = `done in ${ms(data.latency_ms)}`;
    } else {
      showError(data.error || `HTTP ${response.status}`);
      $("run-status").textContent = "failed";
    }
  } catch (error) {
    showError(error.message);
    $("run-status").textContent = "failed";
  } finally {
    button.disabled = false;
    refresh();
  }
}

/* Fire several requests at once: the scheduler spreads them over the workers,
   which is visible in the worker cards and the activity log. */
async function runBurst() {
  const prompt = $("prompt").value.trim() || "Explain artificial intelligence in simple words.";
  const button = $("burst-btn");
  button.disabled = true;
  $("run-status").textContent = "sending 5 parallel requests…";

  const body = JSON.stringify({ prompt, max_new_tokens: Number($("max-tokens").value) || 64 });
  const calls = Array.from({ length: 5 }, () =>
    fetch(`${API_BASE}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    })
      .then((r) => r.json())
      .catch((e) => ({ success: false, error: e.message }))
  );

  const results = await Promise.all(calls);
  const used = {};
  results.forEach((r) => {
    const key = r.success ? r.worker_id : "failed";
    used[key] = (used[key] || 0) + 1;
  });
  const summary = Object.entries(used)
    .map(([worker, count]) => `${worker}: ${count}`)
    .join(", ");
  $("run-status").textContent = `5 requests distributed → ${summary}`;

  const firstOk = results.find((r) => r.success);
  if (firstOk) showResult(firstOk);

  button.disabled = false;
  refresh();
}

/* ------------------------------------------------------------------- init */
$("run-btn").addEventListener("click", runInference);
$("burst-btn").addEventListener("click", runBurst);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runInference();
});

refresh();
setInterval(refresh, REFRESH_MS);
