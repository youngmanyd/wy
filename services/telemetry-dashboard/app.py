"""Telemetry Dashboard — final DAG hop with Web UI, CSV output, and metrics aggregation."""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, print_env, create_app, SpanKind, logger,
)

DASHBOARD_MAX_RESULTS = 200
dashboard_results: list = []

DAG_EDGES = [
    ("gateway", "rgb-preprocessor"), ("gateway", "ir-preprocessor"),
    ("rgb-preprocessor", "rgb-detector"), ("ir-preprocessor", "ir-detector"),
    ("rgb-detector", "feature-fusion"), ("ir-detector", "feature-fusion"),
    ("feature-fusion", "object-tracker"), ("feature-fusion", "situation-awareness"),
    ("object-tracker", "decision-maker"), ("situation-awareness", "decision-maker"),
    ("decision-maker", "telemetry-dashboard"),
]

ALL_SERVICES = [
    "gateway", "rgb-preprocessor", "ir-preprocessor",
    "rgb-detector", "ir-detector", "feature-fusion",
    "object-tracker", "situation-awareness", "decision-maker",
    "telemetry-dashboard",
]


async def _handle_telemetry(body: bytes, t0: str, req_id: str, arrival: str, chain: str):
    with tracer.start_as_current_span("telemetry-aggregate", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time()
        e2e_latency_ms = (time.time() - float(t0)) * 1000

        node_timings = {}
        for segment in chain.replace(";", ",").split(","):
            segment = segment.strip()
            if not segment:
                continue
            parts = segment.split("|")
            if len(parts) == 3:
                node_name, node_arrival, node_compute_us = parts
                node_timings[node_name] = {
                    "arrival": float(node_arrival),
                    "compute_us": float(node_compute_us),
                }

        ms10_compute_us = (time.time() - compute_start) * 1e6
        node_timings["telemetry-dashboard"] = {"arrival": float(arrival), "compute_us": ms10_compute_us}

        node_compute = {name: t["compute_us"] / 1000.0 for name, t in node_timings.items()}

        net_latencies = {}
        for src, dst in DAG_EDGES:
            if src in node_timings and dst in node_timings:
                src_finish = node_timings[src]["arrival"] + node_timings[src]["compute_us"] / 1e6
                dst_arrival = node_timings[dst]["arrival"]
                net_latencies[f"{src}->{dst}"] = round((dst_arrival - src_finish) * 1000, 3)

        rgb_branch_ms = 0.0
        ir_branch_ms = 0.0
        if "rgb-preprocessor" in node_timings and "rgb-detector" in node_timings:
            rgb_start = node_timings["rgb-preprocessor"]["arrival"]
            rgb_end = node_timings["rgb-detector"]["arrival"] + node_timings["rgb-detector"]["compute_us"] / 1e6
            rgb_branch_ms = (rgb_end - rgb_start) * 1000
        if "ir-preprocessor" in node_timings and "ir-detector" in node_timings:
            ir_start = node_timings["ir-preprocessor"]["arrival"]
            ir_end = node_timings["ir-detector"]["arrival"] + node_timings["ir-detector"]["compute_us"] / 1e6
            ir_branch_ms = (ir_end - ir_start) * 1000

        csv_parts = [req_id, f"{e2e_latency_ms:.3f}", f"{rgb_branch_ms:.3f}", f"{ir_branch_ms:.3f}"]
        for svc in ALL_SERVICES:
            csv_parts.append(f"{node_compute.get(svc, 0.0):.3f}")
        for src, dst in DAG_EDGES:
            csv_parts.append(f"{net_latencies.get(f'{src}->{dst}', 0.0):.3f}")
        csv_line = ",".join(csv_parts)
        print(f"CSV_RESULT:{csv_line}", flush=True)

        result_entry = {
            "request_id": req_id,
            "timestamp": time.time(),
            "e2e_ms": round(e2e_latency_ms, 3),
            "rgb_branch_ms": round(rgb_branch_ms, 3),
            "ir_branch_ms": round(ir_branch_ms, 3),
            "node_compute_ms": {k: round(v, 3) for k, v in node_compute.items()},
            "network_latency_ms": net_latencies,
        }
        dashboard_results.append(result_entry)
        if len(dashboard_results) > DASHBOARD_MAX_RESULTS:
            dashboard_results.pop(0)

        span.set_attribute("e2e_latency_ms", e2e_latency_ms)
        return result_entry


@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    yield
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = create_app(lifespan_func=lifespan)


@app.get("/api/results")
async def api_results():
    return JSONResponse(content={"results": dashboard_results[-50:]})


@app.get("/api/latest")
async def api_latest():
    if dashboard_results:
        return JSONResponse(content=dashboard_results[-1])
    return JSONResponse(content={})


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UAV Edge Telemetry Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
.header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 20px; color: #58a6ff; }
.header .status { font-size: 13px; color: #8b949e; }
.container { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.card h3 { color: #58a6ff; margin-bottom: 12px; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
.card.full { grid-column: 1 / -1; }
#e2e-chart, #branch-chart, #compute-chart, #network-chart { width: 100%; height: 280px; }
.image-container { text-align: center; }
.image-container img { max-width: 100%; max-height: 400px; border-radius: 4px; border: 1px solid #30363d; }
.image-container .placeholder { color: #484f58; padding: 60px; font-size: 14px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; }
td { color: #c9d1d9; }
.metric-value { font-size: 28px; font-weight: bold; color: #58a6ff; }
.metric-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
.metrics-row { display: flex; gap: 24px; margin-bottom: 16px; }
.metric-box { flex: 1; text-align: center; }
@media (max-width: 768px) { .container { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="header">
  <h1>UAV Edge Telemetry Dashboard</h1>
  <div class="status" id="status">Waiting for data...</div>
</div>
<div class="container">
  <div class="card full">
    <div class="metrics-row">
      <div class="metric-box"><div class="metric-value" id="val-e2e">--</div><div class="metric-label">E2E Latency (ms)</div></div>
      <div class="metric-box"><div class="metric-value" id="val-rgb">--</div><div class="metric-label">RGB Branch (ms)</div></div>
      <div class="metric-box"><div class="metric-value" id="val-ir">--</div><div class="metric-label">IR Branch (ms)</div></div>
      <div class="metric-box"><div class="metric-value" id="val-count">0</div><div class="metric-label">Total Requests</div></div>
    </div>
  </div>
  <div class="card"><h3>E2E Latency Trend</h3><div id="e2e-chart"></div></div>
  <div class="card"><h3>Branch Latency Comparison</h3><div id="branch-chart"></div></div>
  <div class="card"><h3>Per-Service Compute Time</h3><div id="compute-chart"></div></div>
  <div class="card"><h3>Network Hop Latency</h3><div id="network-chart"></div></div>
  <div class="card"><h3>Detection Result Image</h3>
    <div class="image-container"><div class="placeholder" id="image-area">No image received yet</div></div>
  </div>
  <div class="card"><h3>Recent Requests</h3>
    <div style="max-height:400px;overflow-y:auto">
      <table><thead><tr><th>Request ID</th><th>E2E (ms)</th><th>RGB (ms)</th><th>IR (ms)</th></tr></thead>
      <tbody id="results-table"></tbody></table>
    </div>
  </div>
</div>
<script>
const e2eChart = echarts.init(document.getElementById('e2e-chart'));
const branchChart = echarts.init(document.getElementById('branch-chart'));
const computeChart = echarts.init(document.getElementById('compute-chart'));
const networkChart = echarts.init(document.getElementById('network-chart'));

const e2eData = [], rgbData = [], irData = [], labels = [];
const MAX_POINTS = 50;

function updateCharts(results) {
  results.forEach(r => {
    labels.push(r.request_id.substring(0, 8));
    e2eData.push(r.e2e_ms);
    rgbData.push(r.rgb_branch_ms);
    irData.push(r.ir_branch_ms);
  });
  while (labels.length > MAX_POINTS) { labels.shift(); e2eData.shift(); rgbData.shift(); irData.shift(); }

  e2eChart.setOption({
    tooltip: { trigger: 'axis' }, grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#8b949e', fontSize: 10 }, axisLine: { lineStyle: { color: '#30363d' } } },
    yAxis: { type: 'value', name: 'ms', axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
    series: [{ data: e2eData, type: 'line', smooth: true, lineStyle: { color: '#58a6ff' }, areaStyle: { color: 'rgba(88,166,255,0.1)' }, itemStyle: { color: '#58a6ff' } }]
  });
  branchChart.setOption({
    tooltip: { trigger: 'axis' }, grid: { left: 50, right: 20, top: 20, bottom: 30 }, legend: { data: ['RGB', 'IR'], textStyle: { color: '#8b949e' } },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#8b949e', fontSize: 10 }, axisLine: { lineStyle: { color: '#30363d' } } },
    yAxis: { type: 'value', name: 'ms', axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
    series: [
      { name: 'RGB', data: rgbData, type: 'bar', itemStyle: { color: '#3fb950' } },
      { name: 'IR', data: irData, type: 'bar', itemStyle: { color: '#f85149' } }
    ]
  });

  const latest = results[results.length - 1];
  if (latest && latest.node_compute_ms) {
    const svcNames = Object.keys(latest.node_compute_ms);
    const svcValues = Object.values(latest.node_compute_ms);
    computeChart.setOption({
      tooltip: { trigger: 'axis' }, grid: { left: 130, right: 20, top: 10, bottom: 10 },
      xAxis: { type: 'value', name: 'ms', axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
      yAxis: { type: 'category', data: svcNames, axisLabel: { color: '#c9d1d9', fontSize: 11 } },
      series: [{ data: svcValues, type: 'bar', itemStyle: { color: '#d2a8ff' } }]
    });
  }
  if (latest && latest.network_latency_ms) {
    const hopNames = Object.keys(latest.network_latency_ms);
    const hopValues = Object.values(latest.network_latency_ms);
    networkChart.setOption({
      tooltip: { trigger: 'axis' }, grid: { left: 180, right: 20, top: 10, bottom: 10 },
      xAxis: { type: 'value', name: 'ms', axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
      yAxis: { type: 'category', data: hopNames, axisLabel: { color: '#c9d1d9', fontSize: 10 } },
      series: [{ data: hopValues, type: 'bar', itemStyle: { color: '#f0883e' } }]
    });
  }
}

function updateMetrics(latest) {
  if (!latest || !latest.e2e_ms) return;
  document.getElementById('val-e2e').textContent = latest.e2e_ms.toFixed(1);
  document.getElementById('val-rgb').textContent = latest.rgb_branch_ms.toFixed(1);
  document.getElementById('val-ir').textContent = latest.ir_branch_ms.toFixed(1);
}

function updateTable(results) {
  const tbody = document.getElementById('results-table');
  tbody.innerHTML = results.slice(-20).reverse().map(r =>
    `<tr><td>${r.request_id.substring(0,8)}...</td><td>${r.e2e_ms.toFixed(1)}</td><td>${r.rgb_branch_ms.toFixed(1)}</td><td>${r.ir_branch_ms.toFixed(1)}</td></tr>`
  ).join('');
}

let prevCount = 0;
async function poll() {
  try {
    const resp = await fetch('/api/results');
    const data = await resp.json();
    const results = data.results || [];
    if (results.length > 0 && results.length !== prevCount) {
      prevCount = results.length;
      document.getElementById('val-count').textContent = results.length;
      document.getElementById('status').textContent = `Last update: ${new Date().toLocaleTimeString()} | ${results.length} requests`;
      updateCharts(results.slice(-MAX_POINTS));
      updateMetrics(results[results.length - 1]);
      updateTable(results);
      labels.length = 0; e2eData.length = 0; rgbData.length = 0; irData.length = 0;
      results.slice(-MAX_POINTS).forEach(r => { labels.push(r.request_id.substring(0,8)); e2eData.push(r.e2e_ms); rgbData.push(r.rgb_branch_ms); irData.push(r.ir_branch_ms); });
    }
  } catch(e) { document.getElementById('status').textContent = 'Connection error: ' + e.message; }
}
setInterval(poll, 2000);
poll();
window.addEventListener('resize', () => { e2eChart.resize(); branchChart.resize(); computeChart.resize(); networkChart.resize(); });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.post("/process")
async def process(request: Request):
    arrival_time = now_us()
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()

    x_start_time = request.headers.get("X-Start-Time", arrival_time)
    x_request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    x_timing_chain = request.headers.get("X-Timing-Chain", "")

    body = await request.body()
    result = await _handle_telemetry(body, x_start_time, x_request_id, arrival_time, x_timing_chain)

    compute_end = time.time()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id, "result": result})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
