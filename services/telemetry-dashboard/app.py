"""Telemetry Dashboard — final DAG node with CSV output, Web UI, and smoothing.

Fixes applied:
- #1: X-Original-Image received and decoded from header for Web UI display
- #2: Fix negative network latency (max(0, delta) + warning)
- #10: High-precision timestamps via time.time_ns()
- New: Sliding window smoothing (SMOOTHING_WINDOW_SIZE env var)
- New: Extended CSV with smoothed metrics
- New: Prometheus gauges for smoothed metrics
"""

import asyncio
import base64
import collections
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import Gauge

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, print_env, create_app, SpanKind, logger,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ---------------------------------------------------------------------------
# Sliding window configuration
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW_SIZE: int = int(os.environ.get("SMOOTHING_WINDOW_SIZE", "10"))

smoothing_window: collections.deque = collections.deque(maxlen=SMOOTHING_WINDOW_SIZE)

# Prometheus smoothed gauges
SMOOTHED_E2E = Gauge("uav_smoothed_e2e_latency_ms", "Smoothed E2E latency in ms")
SMOOTHED_COMPUTE = Gauge(
    "uav_smoothed_compute_latency_ms", "Smoothed compute latency per service", ["service_role"]
)
SMOOTHED_NETWORK = Gauge(
    "uav_smoothed_network_latency_ms", "Smoothed network latency per edge", ["edge"]
)

# ---------------------------------------------------------------------------
# In-memory store for recent results (for Web UI)
# ---------------------------------------------------------------------------
MAX_RESULTS = 100
recent_results: list[dict] = []
latest_image_b64: str = ""

# ---------------------------------------------------------------------------
# DAG topology
# ---------------------------------------------------------------------------
ORDERED_SERVICES = [
    "gateway", "rgb-preprocessor", "ir-preprocessor",
    "rgb-detector", "ir-detector", "feature-fusion",
    "object-tracker", "situation-awareness", "decision-maker",
    "telemetry-dashboard",
]

DAG_EDGES = [
    ("gateway", "rgb-preprocessor"),
    ("gateway", "ir-preprocessor"),
    ("rgb-preprocessor", "rgb-detector"),
    ("ir-preprocessor", "ir-detector"),
    ("rgb-detector", "feature-fusion"),
    ("ir-detector", "feature-fusion"),
    ("feature-fusion", "object-tracker"),
    ("feature-fusion", "situation-awareness"),
    ("object-tracker", "decision-maker"),
    ("situation-awareness", "decision-maker"),
    ("decision-maker", "telemetry-dashboard"),
]


def _parse_timing_chain(chain_str: str) -> dict:
    """Parse timing chain: 'service|arrival|compute_us,...;service|arrival|compute_us,...'"""
    entries = {}
    if not chain_str:
        return entries
    for segment in chain_str.replace(";", ",").split(","):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split("|")
        if len(parts) >= 3:
            svc = parts[0].strip()
            try:
                arrival = float(parts[1].strip())
                compute_us = float(parts[2].strip())
                entries[svc] = {"arrival": arrival, "compute_us": compute_us}
            except (ValueError, IndexError):
                continue
    return entries


def _compute_latencies(timing: dict, e2e_start: float, dashboard_arrival: float) -> dict:
    """Compute per-node compute, per-edge network, branch, and E2E latencies."""
    result = {
        "e2e_ms": max(0.0, (dashboard_arrival - e2e_start) * 1000.0),
        "node_compute_ms": {},
        "network_latency_ms": {},
        "rgb_branch_ms": 0.0,
        "ir_branch_ms": 0.0,
    }

    for svc in ORDERED_SERVICES:
        if svc in timing:
            result["node_compute_ms"][svc] = timing[svc]["compute_us"] / 1000.0

    for src, dst in DAG_EDGES:
        if src in timing and dst in timing:
            src_finish = timing[src]["arrival"] + timing[src]["compute_us"] / 1e6
            net_latency = (timing[dst]["arrival"] - src_finish) * 1000.0
            if net_latency < 0:
                logger.warning(
                    "[NEGATIVE_LATENCY] %s->%s: %.3f ms (src_finish=%.9f, dst_arrival=%.9f)",
                    src, dst, net_latency, src_finish, timing[dst]["arrival"],
                )
                net_latency = max(0.0, net_latency)
            result["network_latency_ms"][f"{src}->{dst}"] = net_latency

    if "gateway" in timing and "feature-fusion" in timing:
        gw_arrival = timing["gateway"]["arrival"]
        ff_arrival = timing["feature-fusion"]["arrival"]
        ff_compute = timing["feature-fusion"]["compute_us"] / 1e6
        result["rgb_branch_ms"] = max(0.0, (ff_arrival + ff_compute - gw_arrival) * 1000.0)
        result["ir_branch_ms"] = result["rgb_branch_ms"]

    return result


def _compute_smoothed(window: collections.deque) -> dict:
    """Compute averages over the sliding window."""
    if len(window) < 2:
        return {}

    n = len(window)
    smoothed = {"e2e_ms": 0.0, "compute": {}, "network": {}}
    compute_sums: dict[str, list[float]] = {}
    network_sums: dict[str, list[float]] = {}

    for entry in window:
        smoothed["e2e_ms"] += entry["e2e_ms"]
        for svc, val in entry.get("node_compute_ms", {}).items():
            compute_sums.setdefault(svc, []).append(val)
        for edge, val in entry.get("network_latency_ms", {}).items():
            network_sums.setdefault(edge, []).append(val)

    smoothed["e2e_ms"] /= n
    for svc, vals in compute_sums.items():
        smoothed["compute"][svc] = sum(vals) / len(vals)
    for edge, vals in network_sums.items():
        smoothed["network"][edge] = sum(vals) / len(vals)

    return smoothed


def _build_csv_line(request_id: str, latencies: dict, smoothed: dict) -> str:
    """Build CSV line: request_id, e2e_ms, rgb_branch_ms, ir_branch_ms,
    10x node_compute_ms, 11x network_latency_ms,
    smoothed_e2e_ms, 10x smoothed_compute_ms, 11x smoothed_network_ms
    """
    parts = [request_id]
    parts.append(f"{latencies['e2e_ms']:.3f}")
    parts.append(f"{latencies['rgb_branch_ms']:.3f}")
    parts.append(f"{latencies['ir_branch_ms']:.3f}")

    for svc in ORDERED_SERVICES:
        parts.append(f"{latencies['node_compute_ms'].get(svc, 0.0):.3f}")

    edge_labels = [f"{s}->{d}" for s, d in DAG_EDGES]
    for edge in edge_labels:
        parts.append(f"{latencies['network_latency_ms'].get(edge, 0.0):.3f}")

    if smoothed:
        parts.append(f"{smoothed.get('e2e_ms', 0.0):.3f}")
        for svc in ORDERED_SERVICES:
            parts.append(f"{smoothed.get('compute', {}).get(svc, 0.0):.3f}")
        for edge in edge_labels:
            parts.append(f"{smoothed.get('network', {}).get(edge, 0.0):.3f}")
    else:
        parts.extend([""] * (1 + len(ORDERED_SERVICES) + len(edge_labels)))

    return ",".join(parts)


async def _handle_telemetry(body: bytes, t0: str, req_id: str, arrival: str,
                            chain: str, original_image: str):
    global latest_image_b64

    timing = _parse_timing_chain(chain)
    e2e_start = float(t0)
    dashboard_arrival = float(arrival)

    timing["telemetry-dashboard"] = {
        "arrival": dashboard_arrival,
        "compute_us": 0,
    }

    latencies = _compute_latencies(timing, e2e_start, dashboard_arrival)

    compute_start = time.time_ns()
    try:
        upstream_data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        upstream_data = {}

    tracked_objects = upstream_data.get("tracked_objects", [])
    assessments = upstream_data.get("assessments", [])
    decision = upstream_data.get("decision", "unknown")

    if original_image:
        latest_image_b64 = original_image
    elif upstream_data.get("original_image_b64"):
        latest_image_b64 = upstream_data["original_image_b64"]

    compute_end = time.time_ns()
    compute_us = (compute_end - compute_start) / 1000.0
    timing["telemetry-dashboard"]["compute_us"] = compute_us

    entry = {
        "e2e_ms": latencies["e2e_ms"],
        "node_compute_ms": latencies["node_compute_ms"],
        "network_latency_ms": latencies["network_latency_ms"],
    }
    smoothing_window.append(entry)
    smoothed = _compute_smoothed(smoothing_window)

    SMOOTHED_E2E.set(smoothed.get("e2e_ms", 0.0))
    for svc, val in smoothed.get("compute", {}).items():
        SMOOTHED_COMPUTE.labels(service_role=svc).set(val)
    for edge, val in smoothed.get("network", {}).items():
        SMOOTHED_NETWORK.labels(edge=edge).set(val)

    csv_line = _build_csv_line(req_id, latencies, smoothed)
    print(f"CSV_RESULT:{csv_line}", flush=True)

    result_entry = {
        "request_id": req_id,
        "timestamp": arrival,
        "e2e_ms": latencies["e2e_ms"],
        "rgb_branch_ms": latencies["rgb_branch_ms"],
        "ir_branch_ms": latencies["ir_branch_ms"],
        "node_compute_ms": latencies["node_compute_ms"],
        "network_latency_ms": latencies["network_latency_ms"],
        "smoothed": smoothed,
        "decision": decision,
        "tracked_objects": len(tracked_objects),
        "assessments": len(assessments),
    }
    recent_results.append(result_entry)
    if len(recent_results) > MAX_RESULTS:
        recent_results.pop(0)

    return result_entry


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("  SMOOTHING_WINDOW_SIZE=%s", SMOOTHING_WINDOW_SIZE)
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    yield
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = create_app(lifespan_func=lifespan)


# ---------------------------------------------------------------------------
# Web Dashboard UI
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UAV Edge Telemetry Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
body{margin:0;padding:20px;font-family:system-ui,-apple-system,sans-serif;background:#0a0e17;color:#e0e0e0}
.header{text-align:center;padding:10px 0;border-bottom:1px solid #1e2a3a;margin-bottom:20px}
.header h1{margin:0;font-size:24px;color:#4fc3f7}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.card{background:#111827;border-radius:8px;padding:16px;border:1px solid #1e2a3a}
.card .label{font-size:12px;color:#9e9e9e;text-transform:uppercase}
.card .value{font-size:28px;font-weight:bold;color:#4fc3f7;margin-top:4px}
.card .sub{font-size:11px;color:#666;margin-top:2px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.chart-box{background:#111827;border-radius:8px;padding:16px;border:1px solid #1e2a3a;min-height:300px}
.image-box{background:#111827;border-radius:8px;padding:16px;border:1px solid #1e2a3a;text-align:center}
.image-box img{max-width:100%;max-height:400px;border-radius:4px}
table{width:100%;border-collapse:collapse;background:#111827;border-radius:8px;overflow:hidden}
th{background:#1a2332;padding:10px;text-align:left;font-size:12px;color:#9e9e9e;text-transform:uppercase}
td{padding:8px 10px;border-top:1px solid #1e2a3a;font-size:13px}
.status-ok{color:#66bb6a}.status-slow{color:#ffa726}.status-bad{color:#ef5350}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <h1>UAV Edge Telemetry Dashboard</h1>
  <p style="color:#666;margin:4px 0">Real-time 10-Node DAG Monitoring | Smoothing Window: """ + str(SMOOTHING_WINDOW_SIZE) + """</p>
</div>
<div class="cards" id="cards"></div>
<div class="grid">
  <div class="chart-box"><div id="latencyChart" style="height:300px"></div></div>
  <div class="chart-box"><div id="networkChart" style="height:300px"></div></div>
</div>
<div class="grid">
  <div class="image-box">
    <h3 style="margin:0 0 10px;color:#4fc3f7">Latest Processed Image</h3>
    <img id="latestImg" src="" alt="No image yet" style="display:none">
    <p id="noImg" style="color:#666">Waiting for image data...</p>
  </div>
  <div class="chart-box"><div id="computeChart" style="height:300px"></div></div>
</div>
<h3 style="color:#4fc3f7;margin:20px 0 10px">Recent Requests</h3>
<table><thead><tr>
<th>Request ID</th><th>E2E (ms)</th><th>RGB Branch</th><th>IR Branch</th>
<th>Decision</th><th>Tracked</th><th>Smoothed E2E</th>
</tr></thead><tbody id="tableBody"></tbody></table>
<script>
const latencyChart=echarts.init(document.getElementById('latencyChart'));
const networkChart=echarts.init(document.getElementById('networkChart'));
const computeChart=echarts.init(document.getElementById('computeChart'));
let e2eHistory=[],smoothedHistory=[],labels=[];
function statusClass(ms){return ms<100?'status-ok':ms<500?'status-slow':'status-bad'}
function updateCards(d){
  const c=document.getElementById('cards');
  const latest=d.results&&d.results.length?d.results[d.results.length-1]:{};
  const sm=latest.smoothed||{};
  c.innerHTML=`
    <div class="card"><div class="label">Total Requests</div><div class="value">${d.total||0}</div></div>
    <div class="card"><div class="label">E2E Latency</div><div class="value ${statusClass(latest.e2e_ms||0)}">${(latest.e2e_ms||0).toFixed(1)} ms</div>
      <div class="sub">Smoothed: ${(sm.e2e_ms||0).toFixed(1)} ms</div></div>
    <div class="card"><div class="label">RGB Branch</div><div class="value">${(latest.rgb_branch_ms||0).toFixed(1)} ms</div></div>
    <div class="card"><div class="label">IR Branch</div><div class="value">${(latest.ir_branch_ms||0).toFixed(1)} ms</div></div>
    <div class="card"><div class="label">Decision</div><div class="value" style="font-size:20px">${latest.decision||'N/A'}</div></div>
    <div class="card"><div class="label">Window Size</div><div class="value">""" + str(SMOOTHING_WINDOW_SIZE) + """</div></div>`;
}
function updateCharts(results){
  labels=results.map((_,i)=>'#'+(i+1));
  e2eHistory=results.map(r=>r.e2e_ms||0);
  smoothedHistory=results.map(r=>(r.smoothed&&r.smoothed.e2e_ms)||0);
  latencyChart.setOption({title:{text:'E2E Latency Trend',textStyle:{color:'#e0e0e0',fontSize:14}},
    tooltip:{trigger:'axis'},legend:{data:['Raw','Smoothed'],textStyle:{color:'#999'}},
    xAxis:{type:'category',data:labels,axisLabel:{color:'#666'}},
    yAxis:{type:'value',name:'ms',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
    series:[{name:'Raw',type:'line',data:e2eHistory,lineStyle:{color:'#4fc3f7'},itemStyle:{color:'#4fc3f7'}},
            {name:'Smoothed',type:'line',data:smoothedHistory,lineStyle:{color:'#66bb6a',type:'dashed'},itemStyle:{color:'#66bb6a'}}]});
  if(results.length){
    const last=results[results.length-1];
    const netData=Object.entries(last.network_latency_ms||{}).map(([k,v])=>({name:k.replace('->','\\n→\\n'),value:v}));
    networkChart.setOption({title:{text:'Network Latency (Last Request)',textStyle:{color:'#e0e0e0',fontSize:14}},
      tooltip:{trigger:'axis'},xAxis:{type:'category',data:netData.map(d=>d.name),axisLabel:{color:'#666',rotate:45,fontSize:9}},
      yAxis:{type:'value',name:'ms',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
      series:[{type:'bar',data:netData.map(d=>d.value),itemStyle:{color:'#ffa726'}}]});
    const compData=Object.entries(last.node_compute_ms||{}).map(([k,v])=>({name:k,value:v}));
    computeChart.setOption({title:{text:'Per-Node Compute Time',textStyle:{color:'#e0e0e0',fontSize:14}},
      tooltip:{trigger:'axis'},xAxis:{type:'category',data:compData.map(d=>d.name),axisLabel:{color:'#666',rotate:45,fontSize:9}},
      yAxis:{type:'value',name:'ms',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
      series:[{type:'bar',data:compData.map(d=>d.value),itemStyle:{color:'#ab47bc'}}]});
  }
}
function updateTable(results){
  const t=document.getElementById('tableBody');
  t.innerHTML=results.slice(-20).reverse().map(r=>`<tr>
    <td>${(r.request_id||'').substring(0,8)}</td>
    <td class="${statusClass(r.e2e_ms)}">${(r.e2e_ms||0).toFixed(1)}</td>
    <td>${(r.rgb_branch_ms||0).toFixed(1)}</td><td>${(r.ir_branch_ms||0).toFixed(1)}</td>
    <td>${r.decision||'N/A'}</td><td>${r.tracked_objects||0}</td>
    <td>${((r.smoothed&&r.smoothed.e2e_ms)||0).toFixed(1)}</td></tr>`).join('');
}
async function refresh(){
  try{
    const [dr,ir]=await Promise.all([fetch('/api/results').then(r=>r.json()),fetch('/api/image').then(r=>r.json())]);
    updateCards(dr);updateCharts(dr.results||[]);updateTable(dr.results||[]);
    if(ir.image){document.getElementById('latestImg').src='data:image/jpeg;base64,'+ir.image;
      document.getElementById('latestImg').style.display='block';document.getElementById('noImg').style.display='none';}
  }catch(e){console.error('Refresh error:',e)}
}
setInterval(refresh,2000);refresh();
window.addEventListener('resize',()=>{latencyChart.resize();networkChart.resize();computeChart.resize()});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/results")
async def api_results():
    return JSONResponse(content={"total": len(recent_results), "results": recent_results[-50:]})


@app.get("/api/image")
async def api_image():
    return JSONResponse(content={"image": latest_image_b64 if latest_image_b64 else ""})


@app.post("/process")
async def process(request: Request):
    arrival_time = now_us()
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()

    x_start_time = request.headers.get("X-Start-Time", arrival_time)
    x_request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    x_timing_chain = request.headers.get("X-Timing-Chain", "")
    x_original_image = request.headers.get("X-Original-Image", "")

    body = await request.body()

    with tracer.start_as_current_span("telemetry-process", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        result = await _handle_telemetry(body, x_start_time, x_request_id,
                                         arrival_time, x_timing_chain, x_original_image)

    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(result.get("e2e_ms", 0) / 1000.0)
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id, "result": result})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
