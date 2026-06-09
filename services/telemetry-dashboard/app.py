"""Telemetry Dashboard — Jaeger-based performance analytics engine.

OTel-centric architecture: ALL performance metrics derived solely from
Jaeger Trace data.  No local timing variables, no X-Timing-Chain.

Core capabilities:
1. Trace integrity validation with exponential backoff retry
2. Per-service compute time (INTERNAL span durations)
3. Per-edge communication time (Client - Server span durations)
4. Critical path extraction through the DAG
5. Top-N bottleneck ranking with Mbps throughput
6. CSV output + Web dashboard + Prometheus metrics
"""

import asyncio
import collections
import json
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, PAYLOAD_BYTES,
    parse_downstream, print_env,
    create_app, SpanKind, logger, trace,
    get_current_trace_id, JAEGER_QUERY_ENDPOINT,
)
from prometheus_client import Gauge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW_SIZE: int = int(os.environ.get("SMOOTHING_WINDOW_SIZE", "10"))

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

# Backoff retry delays (seconds)
BACKOFF_DELAYS = [0.5, 1.2, 2.4]

# Prometheus smoothed metrics
SMOOTHED_E2E = Gauge("uav_smoothed_e2e_latency_ms", "Smoothed E2E latency")
SMOOTHED_COMPUTE = Gauge("uav_smoothed_compute_latency_ms", "Smoothed compute latency", ["service_role"])
SMOOTHED_NETWORK = Gauge("uav_smoothed_network_latency_ms", "Smoothed network latency", ["edge"])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
recent_results: list = []
latest_image_b64: str = ""

# Sliding window for smoothing
_e2e_window = collections.deque(maxlen=SMOOTHING_WINDOW_SIZE)
_compute_windows: dict[str, collections.deque] = {
    svc: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE) for svc in ORDERED_SERVICES
}
_network_windows: dict[str, collections.deque] = {
    f"{s}->{d}": collections.deque(maxlen=SMOOTHING_WINDOW_SIZE) for s, d in DAG_EDGES
}


# ---------------------------------------------------------------------------
# Jaeger Query with Exponential Backoff
# ---------------------------------------------------------------------------
async def _query_jaeger_trace(trace_id: str) -> dict | None:
    """Query Jaeger for a trace by ID with exponential backoff retry."""
    if not trace_id:
        return None

    url = f"{JAEGER_QUERY_ENDPOINT}/api/traces/{trace_id}"

    for attempt, delay in enumerate(BACKOFF_DELAYS):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    traces = data.get("data", [])
                    if traces and traces[0].get("spans"):
                        trace_json = traces[0]
                        # Validate completeness
                        if _validate_trace_completeness(trace_json):
                            return trace_json
                        logger.warning(
                            "Trace %s incomplete (attempt %d/%d), retrying in %.1fs",
                            trace_id[:16], attempt + 1, len(BACKOFF_DELAYS), delay,
                        )
                    else:
                        logger.warning(
                            "Trace %s empty (attempt %d/%d), retrying in %.1fs",
                            trace_id[:16], attempt + 1, len(BACKOFF_DELAYS), delay,
                        )
                else:
                    logger.warning("Jaeger query returned %d for trace %s", resp.status_code, trace_id[:16])
        except Exception as e:
            logger.warning("Jaeger query error (attempt %d): %s", attempt + 1, e)

        await asyncio.sleep(delay)

    # Final attempt without delay
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                traces = data.get("data", [])
                if traces and traces[0].get("spans"):
                    trace_json = traces[0]
                    if not _validate_trace_completeness(trace_json):
                        logger.warning("Trace %s still incomplete after all retries", trace_id[:16])
                    return trace_json
    except Exception as e:
        logger.error("Final Jaeger query failed: %s", e)

    logger.error("Failed to retrieve trace %s after %d retries", trace_id[:16], len(BACKOFF_DELAYS) + 1)
    return None


# ---------------------------------------------------------------------------
# Trace Completeness Validation
# ---------------------------------------------------------------------------
def _validate_trace_completeness(trace_json: dict) -> bool:
    """Check that trace contains SERVER spans from all ORDERED_SERVICES."""
    processes = trace_json.get("processes", {})
    service_map = {}
    for pid, pinfo in processes.items():
        service_map[pid] = pinfo.get("serviceName", "")

    found_services = set()
    for span in trace_json.get("spans", []):
        pid = span.get("processID", "")
        svc = service_map.get(pid, "")
        kind = _get_tag(span, "span.kind")
        if kind == "server" and svc in ORDERED_SERVICES:
            found_services.add(svc)

    missing = set(ORDERED_SERVICES) - found_services
    if missing:
        logger.debug("Trace missing SERVER spans for: %s", sorted(missing))
        return False
    return True


# ---------------------------------------------------------------------------
# Span Helpers
# ---------------------------------------------------------------------------
def _get_tag(span: dict, key: str):
    """Get a tag value from a Jaeger span."""
    for tag in span.get("tags", []):
        if tag.get("key") == key:
            return tag.get("value")
    return None


def _get_parent_span_id(span: dict) -> str:
    """Get parent span ID from references."""
    for ref in span.get("references", []):
        if ref.get("refType") == "CHILD_OF":
            return ref.get("spanID", "")
    return ""


# ---------------------------------------------------------------------------
# Core Metrics Computation from Trace JSON
# ---------------------------------------------------------------------------
def _compute_metrics_from_trace(trace_json: dict) -> dict:
    """
    Extract all performance metrics from Jaeger trace data.

    Formulas:
    - E2E = duration of gateway's outermost SERVER span
    - Compute(i) = sum of INTERNAL span durations for service i
    - Communication(i->j) = duration(Client_i->j) - duration(Server_j)
    - NetEstimate(i->j) = Communication(i->j) / 2
    - Throughput(i->j) = payload_bytes * 8 / Communication_seconds / 1e6  (Mbps)
    """
    processes = trace_json.get("processes", {})
    service_map = {}
    for pid, pinfo in processes.items():
        service_map[pid] = pinfo.get("serviceName", "")

    spans = trace_json.get("spans", [])

    # Classify spans by service and kind
    server_spans: dict[str, list] = {svc: [] for svc in ORDERED_SERVICES}
    client_spans: dict[str, list] = {svc: [] for svc in ORDERED_SERVICES}
    internal_spans: dict[str, list] = {svc: [] for svc in ORDERED_SERVICES}
    span_by_id: dict[str, dict] = {}

    for span in spans:
        span_by_id[span.get("spanID", "")] = span
        svc = service_map.get(span.get("processID", ""), "")
        if svc not in ORDERED_SERVICES:
            continue
        kind = _get_tag(span, "span.kind") or ""
        if kind == "server":
            server_spans[svc].append(span)
        elif kind == "client":
            client_spans[svc].append(span)
        elif kind == "internal":
            internal_spans[svc].append(span)

    # 1. E2E Latency = gateway's outermost SERVER span duration
    e2e_ms = 0.0
    gateway_server = server_spans.get("gateway", [])
    if gateway_server:
        longest = max(gateway_server, key=lambda s: s.get("duration", 0))
        e2e_ms = longest.get("duration", 0) / 1000.0  # us -> ms

    # 2. Per-service compute time = sum of INTERNAL span durations
    compute_ms: dict[str, float] = {}
    for svc in ORDERED_SERVICES:
        total_us = sum(s.get("duration", 0) for s in internal_spans.get(svc, []))
        compute_ms[svc] = total_us / 1000.0

    # 3. Communication time per edge = Client duration - Server duration
    #    Match CLIENT(src) to SERVER(dst) via parent-child relationship
    comm_ms: dict[str, float] = {}
    payload_bytes: dict[str, int] = {}

    for src, dst in DAG_EDGES:
        edge_key = f"{src}->{dst}"
        comm_ms[edge_key] = 0.0
        payload_bytes[edge_key] = 0

        # Find CLIENT spans in src that called dst
        for cspan in client_spans.get(src, []):
            cspan_id = cspan.get("spanID", "")
            c_duration_us = cspan.get("duration", 0)

            # Check if this CLIENT span's URL targets dst
            http_url = _get_tag(cspan, "http.url") or _get_tag(cspan, "url.full") or ""
            if dst not in http_url and dst.replace("-", "") not in http_url:
                continue

            # Find matching SERVER span in dst (parent = this CLIENT span)
            matched_server = None
            for sspan in server_spans.get(dst, []):
                if _get_parent_span_id(sspan) == cspan_id:
                    matched_server = sspan
                    break

            if matched_server:
                s_duration_us = matched_server.get("duration", 0)
                comm_us = max(0, c_duration_us - s_duration_us)
                comm_ms[edge_key] = comm_us / 1000.0

                # Payload from CLIENT span attribute
                p_size = _get_tag(cspan, "messaging.payload_size_bytes")
                if p_size is not None:
                    payload_bytes[edge_key] = int(p_size)
                break

    # 4. Net estimate = communication / 2
    net_est_ms: dict[str, float] = {}
    for edge_key, comm in comm_ms.items():
        net_est_ms[edge_key] = comm / 2.0

    # 5. Throughput (Mbps) = payload * 8 / comm_seconds / 1e6
    throughput_mbps: dict[str, float] = {}
    for edge_key in comm_ms:
        comm_s = comm_ms[edge_key] / 1000.0  # ms -> s
        p_bytes = payload_bytes.get(edge_key, 0)
        if comm_s > 0 and p_bytes > 0:
            throughput_mbps[edge_key] = (p_bytes * 8) / comm_s / 1e6
        else:
            throughput_mbps[edge_key] = 0.0

    return {
        "e2e_ms": round(e2e_ms, 3),
        "compute_ms": {k: round(v, 3) for k, v in compute_ms.items()},
        "comm_ms": {k: round(v, 3) for k, v in comm_ms.items()},
        "net_est_ms": {k: round(v, 3) for k, v in net_est_ms.items()},
        "payload_bytes": payload_bytes,
        "throughput_mbps": {k: round(v, 3) for k, v in throughput_mbps.items()},
    }


# ---------------------------------------------------------------------------
# Critical Path Algorithm
# ---------------------------------------------------------------------------
def _find_critical_path(compute_ms: dict, comm_ms: dict) -> tuple[list[str], float]:
    """
    Find the critical (longest) path from gateway to telemetry-dashboard.

    Uses topological-order dynamic programming:
      dist[node] = max over predecessors of (dist[pred] + comm[pred->node]) + compute[node]
    """
    topo_order = [
        "gateway", "rgb-preprocessor", "ir-preprocessor",
        "rgb-detector", "ir-detector", "feature-fusion",
        "object-tracker", "situation-awareness", "decision-maker",
        "telemetry-dashboard",
    ]

    dist: dict[str, float] = {}
    pred: dict[str, str | None] = {}

    dist["gateway"] = compute_ms.get("gateway", 0.0)
    pred["gateway"] = None

    for node in topo_order[1:]:
        candidates = []
        for src, dst in DAG_EDGES:
            if dst == node:
                edge_key = f"{src}->{dst}"
                arrival = dist.get(src, 0.0) + comm_ms.get(edge_key, 0.0)
                total = arrival + compute_ms.get(node, 0.0)
                candidates.append((total, src))

        if candidates:
            best = max(candidates, key=lambda x: x[0])
            dist[node] = best[0]
            pred[node] = best[1]
        else:
            dist[node] = compute_ms.get(node, 0.0)
            pred[node] = None

    # Trace back
    path = []
    node = "telemetry-dashboard"
    while node is not None:
        path.append(node)
        node = pred.get(node)
    path.reverse()

    return path, round(dist.get("telemetry-dashboard", 0.0), 3)


# ---------------------------------------------------------------------------
# Top-N Bottleneck Ranking
# ---------------------------------------------------------------------------
def _rank_bottlenecks(compute_ms: dict, comm_ms: dict, payload_bytes: dict,
                      throughput_mbps: dict, top_n: int = 5) -> list[dict]:
    """
    Rank all compute times and communication times, return top-N bottlenecks.
    """
    items = []

    # Add compute items
    for svc, ms in compute_ms.items():
        items.append({
            "type": "compute",
            "name": svc,
            "latency_ms": ms,
        })

    # Add communication items
    for edge, ms in comm_ms.items():
        items.append({
            "type": "network",
            "name": edge,
            "latency_ms": ms,
            "payload_bytes": payload_bytes.get(edge, 0),
            "throughput_mbps": throughput_mbps.get(edge, 0.0),
        })

    items.sort(key=lambda x: x["latency_ms"], reverse=True)

    total = sum(x["latency_ms"] for x in items)
    for item in items:
        item["pct"] = round(item["latency_ms"] / total * 100, 1) if total > 0 else 0.0

    return items[:top_n]


# ---------------------------------------------------------------------------
# Console + CSV Output
# ---------------------------------------------------------------------------
def _print_analysis(metrics: dict, critical_path: list[str], cp_time: float,
                    bottlenecks: list[dict], request_id: str):
    """Formatted console output for debugging and log analysis."""
    logger.info("=" * 80)
    logger.info("TRACE ANALYSIS | request_id=%s", request_id)
    logger.info("-" * 80)
    logger.info("E2E Latency: %.3f ms", metrics["e2e_ms"])
    logger.info("Critical Path: %s (%.3f ms)", " -> ".join(critical_path), cp_time)
    logger.info("-" * 40)
    logger.info("Per-Service Compute Time:")
    for svc in ORDERED_SERVICES:
        logger.info("  %-25s %8.3f ms", svc, metrics["compute_ms"].get(svc, 0.0))
    logger.info("-" * 40)
    logger.info("Per-Edge Communication Time + Throughput:")
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        logger.info("  %-45s %8.3f ms | %8d B | %8.3f Mbps",
                     edge,
                     metrics["comm_ms"].get(edge, 0.0),
                     metrics["payload_bytes"].get(edge, 0),
                     metrics["throughput_mbps"].get(edge, 0.0))
    logger.info("-" * 40)
    logger.info("Top-%d Bottlenecks:", len(bottlenecks))
    for i, bn in enumerate(bottlenecks, 1):
        extra = ""
        if bn["type"] == "network":
            extra = f" ({bn.get('throughput_mbps', 0):.1f} Mbps)"
        logger.info("  #%d %-40s %8.3f ms (%4.1f%%){extra}",
                     i, bn["name"], bn["latency_ms"], bn["pct"])
    logger.info("=" * 80)


def _build_csv_line(request_id: str, metrics: dict, smoothed: dict) -> str:
    """Build CSV_RESULT line for log collection."""
    parts = [request_id, f"{metrics['e2e_ms']:.3f}"]

    # Per-service compute
    for svc in ORDERED_SERVICES:
        parts.append(f"{metrics['compute_ms'].get(svc, 0.0):.3f}")

    # Per-edge communication
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        parts.append(f"{metrics['comm_ms'].get(edge, 0.0):.3f}")

    # Per-edge payload bytes
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        parts.append(str(metrics["payload_bytes"].get(edge, 0)))

    # Per-edge throughput
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        parts.append(f"{metrics['throughput_mbps'].get(edge, 0.0):.3f}")

    # Smoothed values
    parts.append(f"{smoothed.get('e2e_ms', 0.0):.3f}")
    for svc in ORDERED_SERVICES:
        parts.append(f"{smoothed.get('compute_ms', {}).get(svc, 0.0):.3f}")
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        parts.append(f"{smoothed.get('comm_ms', {}).get(edge, 0.0):.3f}")

    return ",".join(parts)


def _build_csv_header() -> str:
    """Build CSV header line."""
    cols = ["request_id", "e2e_ms"]
    for svc in ORDERED_SERVICES:
        cols.append(f"compute_{svc}_ms")
    for src, dst in DAG_EDGES:
        cols.append(f"comm_{src}_to_{dst}_ms")
    for src, dst in DAG_EDGES:
        cols.append(f"payload_{src}_to_{dst}_bytes")
    for src, dst in DAG_EDGES:
        cols.append(f"throughput_{src}_to_{dst}_mbps")
    cols.append("smoothed_e2e_ms")
    for svc in ORDERED_SERVICES:
        cols.append(f"smoothed_compute_{svc}_ms")
    for src, dst in DAG_EDGES:
        cols.append(f"smoothed_comm_{src}_to_{dst}_ms")
    return ",".join(cols)


# ---------------------------------------------------------------------------
# Sliding Window Smoothing
# ---------------------------------------------------------------------------
def _update_smoothing(metrics: dict) -> dict:
    """Update sliding windows and return smoothed values."""
    _e2e_window.append(metrics["e2e_ms"])

    for svc in ORDERED_SERVICES:
        _compute_windows[svc].append(metrics["compute_ms"].get(svc, 0.0))

    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        _network_windows[edge].append(metrics["comm_ms"].get(edge, 0.0))

    smoothed = {
        "e2e_ms": round(sum(_e2e_window) / len(_e2e_window), 3) if _e2e_window else 0.0,
        "compute_ms": {},
        "comm_ms": {},
    }
    for svc in ORDERED_SERVICES:
        w = _compute_windows[svc]
        smoothed["compute_ms"][svc] = round(sum(w) / len(w), 3) if w else 0.0
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        w = _network_windows[edge]
        smoothed["comm_ms"][edge] = round(sum(w) / len(w), 3) if w else 0.0

    # Update Prometheus gauges
    SMOOTHED_E2E.set(smoothed["e2e_ms"])
    for svc in ORDERED_SERVICES:
        SMOOTHED_COMPUTE.labels(service_role=svc).set(smoothed["compute_ms"][svc])
    for src, dst in DAG_EDGES:
        edge = f"{src}->{dst}"
        SMOOTHED_NETWORK.labels(edge=edge).set(smoothed["comm_ms"][edge])

    return smoothed


# ---------------------------------------------------------------------------
# Main Telemetry Handler
# ---------------------------------------------------------------------------
async def _handle_telemetry(body: bytes, trace_id: str | None) -> dict:
    """Process incoming telemetry: query Jaeger, analyze trace, output results."""
    global latest_image_b64

    request_id = str(uuid.uuid4())[:8]

    # Parse body JSON from decision-maker
    decision_data = {}
    try:
        decision_data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract image for display
    img_b64 = decision_data.get("original_image_b64", "")
    if img_b64:
        latest_image_b64 = img_b64

    # Default empty metrics
    empty_metrics = {
        "e2e_ms": 0.0,
        "compute_ms": {svc: 0.0 for svc in ORDERED_SERVICES},
        "comm_ms": {f"{s}->{d}": 0.0 for s, d in DAG_EDGES},
        "net_est_ms": {f"{s}->{d}": 0.0 for s, d in DAG_EDGES},
        "payload_bytes": {f"{s}->{d}": 0 for s, d in DAG_EDGES},
        "throughput_mbps": {f"{s}->{d}": 0.0 for s, d in DAG_EDGES},
    }

    # Query Jaeger (with backoff + validation)
    metrics = empty_metrics
    trace_json = None
    if trace_id:
        trace_json = await _query_jaeger_trace(trace_id)
    if trace_json:
        try:
            metrics = _compute_metrics_from_trace(trace_json)
        except Exception:
            logger.exception("Failed to compute metrics from Jaeger trace")

    # Critical path
    critical_path, cp_time = _find_critical_path(
        metrics["compute_ms"], metrics["comm_ms"]
    )

    # Bottleneck ranking
    bottlenecks = _rank_bottlenecks(
        metrics["compute_ms"], metrics["comm_ms"],
        metrics["payload_bytes"], metrics["throughput_mbps"],
    )

    # Sliding window smoothing
    smoothed = _update_smoothing(metrics)

    # Console output
    _print_analysis(metrics, critical_path, cp_time, bottlenecks, request_id)

    # CSV output
    if not recent_results:
        logger.info("CSV_HEADER:%s", _build_csv_header())
    csv_line = _build_csv_line(request_id, metrics, smoothed)
    logger.info("CSV_RESULT:%s", csv_line)

    # Build result for storage and API
    result = {
        "request_id": request_id,
        "e2e_ms": metrics["e2e_ms"],
        "node_compute_ms": metrics["compute_ms"],
        "network_latency_ms": metrics["comm_ms"],
        "net_estimate_ms": metrics["net_est_ms"],
        "payload_bytes": metrics["payload_bytes"],
        "throughput_mbps": metrics["throughput_mbps"],
        "critical_path": critical_path,
        "critical_path_ms": cp_time,
        "bottlenecks": bottlenecks,
        "decision": decision_data.get("decision", "N/A"),
        "tracked_objects": decision_data.get("tracked_objects", 0),
        "risk_level": decision_data.get("risk_level", "N/A"),
        "rgb_detections": decision_data.get("rgb_detections", []),
        "ir_detections": decision_data.get("ir_detections", []),
        "smoothed": smoothed,
        "trace_id": trace_id or "",
    }

    recent_results.append(result)
    if len(recent_results) > 200:
        recent_results[:] = recent_results[-100:]

    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(metrics["e2e_ms"] / 1000.0)
    return result


# ---------------------------------------------------------------------------
# Web Dashboard HTML
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>UAV Edge Telemetry (OTel)</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;padding:16px}
h1{text-align:center;color:#4fc3f7;margin-bottom:10px;font-size:20px}
.subtitle{text-align:center;color:#888;font-size:12px;margin-bottom:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:16px}
.card{background:#16213e;border-radius:8px;padding:12px;text-align:center}
.card .value{font-size:24px;font-weight:bold;color:#4fc3f7}
.card .label{font-size:11px;color:#888;margin-top:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.chart-box{background:#16213e;border-radius:8px;padding:12px;min-height:250px}
.full-width{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 8px;text-align:center;border-bottom:1px solid #2a2a4a}
th{background:#16213e;color:#4fc3f7;position:sticky;top:0}
.status-ok{color:#66bb6a} .status-warn{color:#ffa726} .status-bad{color:#ef5350}
.img-container{background:#16213e;border-radius:8px;padding:12px;text-align:center;position:relative}
#latestImg{max-width:100%;max-height:300px;border-radius:4px}
.cp-box{background:#16213e;border-radius:8px;padding:12px;margin-bottom:16px}
.cp-path{color:#4fc3f7;font-family:monospace;font-size:13px}
.bn-table{margin-top:8px}
.bn-table td{text-align:left;padding:4px 8px}
#overlay{position:absolute;top:0;left:0;pointer-events:none}
</style></head><body>
<h1>UAV Edge Telemetry Dashboard (OTel-Only)</h1>
<p class="subtitle">Performance metrics derived solely from OpenTelemetry distributed traces</p>

<div class="cards">
  <div class="card"><div class="value" id="totalReqs">0</div><div class="label">Total Requests</div></div>
  <div class="card"><div class="value" id="e2eVal">-</div><div class="label">E2E Latency (ms)</div></div>
  <div class="card"><div class="value" id="smoothE2e">-</div><div class="label">Smoothed E2E (ms)</div></div>
  <div class="card"><div class="value" id="cpTime">-</div><div class="label">Critical Path (ms)</div></div>
  <div class="card"><div class="value" id="topBn">-</div><div class="label">Top Bottleneck</div></div>
  <div class="card"><div class="value" id="decision">-</div><div class="label">Decision</div></div>
</div>

<div class="cp-box" id="cpBox">
  <strong>Critical Path:</strong> <span class="cp-path" id="cpPath">-</span>
</div>

<div class="grid">
  <div class="chart-box"><div id="latencyChart" style="width:100%;height:230px"></div></div>
  <div class="chart-box"><div id="computeChart" style="width:100%;height:230px"></div></div>
  <div class="chart-box"><div id="networkChart" style="width:100%;height:230px"></div></div>
  <div class="chart-box"><div id="throughputChart" style="width:100%;height:230px"></div></div>
</div>

<div class="grid">
  <div class="img-container">
    <h3 style="margin-bottom:8px;font-size:14px">Latest Processed Image</h3>
    <div style="position:relative;display:inline-block">
      <img id="latestImg" style="display:none" />
      <canvas id="overlay"></canvas>
    </div>
    <p id="noImg" style="color:#666;padding:40px">No image received yet</p>
  </div>
  <div class="chart-box" style="overflow-y:auto;max-height:350px">
    <h3 style="margin-bottom:8px;font-size:14px">Recent Requests</h3>
    <table><thead><tr>
      <th>ID</th><th>E2E</th><th>Decision</th><th>Risk</th><th>Tracked</th><th>Trace</th>
    </tr></thead><tbody id="tableBody"></tbody></table>
  </div>
</div>

<div class="grid full-width" style="margin-top:12px">
  <div class="chart-box" style="overflow-y:auto;max-height:300px">
    <h3 style="margin-bottom:8px;font-size:14px">Top-5 Bottlenecks (Latest)</h3>
    <table class="bn-table"><thead><tr>
      <th>#</th><th>Type</th><th>Name</th><th>Latency</th><th>%</th><th>Throughput</th>
    </tr></thead><tbody id="bnBody"></tbody></table>
  </div>
</div>

<script>
const latencyChart=echarts.init(document.getElementById('latencyChart'));
const computeChart=echarts.init(document.getElementById('computeChart'));
const networkChart=echarts.init(document.getElementById('networkChart'));
const throughputChart=echarts.init(document.getElementById('throughputChart'));

function statusClass(ms){return ms<500?'status-ok':ms<1500?'status-warn':'status-bad'}

function resizeOverlay(){
  const img=document.getElementById('latestImg');
  const c=document.getElementById('overlay');
  if(img.naturalWidth){c.width=img.clientWidth;c.height=img.clientHeight;
    c.style.top=img.offsetTop+'px';c.style.left=img.offsetLeft+'px';}
}
function drawBoxes(dets){
  const c=document.getElementById('overlay');const ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  const img=document.getElementById('latestImg');
  if(!img.naturalWidth)return;
  const sx=c.width/640;const sy=c.height/640;
  (dets||[]).forEach(d=>{
    const b=d.bbox||[];if(b.length<4)return;
    ctx.strokeStyle=d.class_id===0?'#4fc3f7':'#ffa726';ctx.lineWidth=2;
    ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);
    ctx.fillStyle=ctx.strokeStyle;ctx.font='10px monospace';
    ctx.fillText(`${d.class_id} ${(d.score||0).toFixed(2)}`,b[0]*sx,b[1]*sy-2);
  });
}
function updateCards(dr){
  document.getElementById('totalReqs').textContent=dr.total||0;
  const r=dr.results;if(!r||!r.length)return;
  const last=r[r.length-1];
  document.getElementById('e2eVal').textContent=(last.e2e_ms||0).toFixed(1);
  document.getElementById('smoothE2e').textContent=((last.smoothed&&last.smoothed.e2e_ms)||0).toFixed(1);
  document.getElementById('cpTime').textContent=(last.critical_path_ms||0).toFixed(1);
  document.getElementById('decision').textContent=last.decision||'N/A';
  const bn=last.bottlenecks;
  document.getElementById('topBn').textContent=bn&&bn.length?bn[0].name:'N/A';
  document.getElementById('cpPath').textContent=(last.critical_path||[]).join(' -> ');
}
function updateCharts(results){
  const e2eHistory=results.map((r,i)=>[i,r.e2e_ms||0]);
  const smoothedHistory=results.map((r,i)=>[i,(r.smoothed&&r.smoothed.e2e_ms)||0]);
  latencyChart.setOption({title:{text:'E2E Latency Trend',textStyle:{color:'#e0e0e0',fontSize:14}},
    tooltip:{trigger:'axis'},xAxis:{type:'category',show:false},
    yAxis:{type:'value',name:'ms',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
    series:[{name:'Raw',type:'line',data:e2eHistory,lineStyle:{color:'#4fc3f7'},itemStyle:{color:'#4fc3f7'}},
            {name:'Smoothed',type:'line',data:smoothedHistory,lineStyle:{color:'#66bb6a',type:'dashed'},itemStyle:{color:'#66bb6a'}}]});
  if(!results.length)return;
  const last=results[results.length-1];
  const compData=Object.entries(last.node_compute_ms||{}).map(([k,v])=>({name:k,value:v}));
  computeChart.setOption({title:{text:'Per-Node Compute Time',textStyle:{color:'#e0e0e0',fontSize:14}},
    tooltip:{trigger:'axis'},xAxis:{type:'category',data:compData.map(d=>d.name),axisLabel:{color:'#666',rotate:45,fontSize:9}},
    yAxis:{type:'value',name:'ms',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
    series:[{type:'bar',data:compData.map(d=>d.value),itemStyle:{color:'#ab47bc'}}]});
  const netData=Object.entries(last.network_latency_ms||{}).map(([k,v])=>({name:k.replace('->','\\n->\\n'),value:v}));
  networkChart.setOption({title:{text:'Per-Edge Communication Time',textStyle:{color:'#e0e0e0',fontSize:14}},
    tooltip:{trigger:'axis'},xAxis:{type:'category',data:netData.map(d=>d.name),axisLabel:{color:'#666',rotate:45,fontSize:9}},
    yAxis:{type:'value',name:'ms',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
    series:[{type:'bar',data:netData.map(d=>d.value),itemStyle:{color:'#ffa726'}}]});
  const tpData=Object.entries(last.throughput_mbps||{}).map(([k,v])=>({name:k.replace('->','\\n->\\n'),value:v}));
  throughputChart.setOption({title:{text:'Per-Edge Throughput',textStyle:{color:'#e0e0e0',fontSize:14}},
    tooltip:{trigger:'axis'},xAxis:{type:'category',data:tpData.map(d=>d.name),axisLabel:{color:'#666',rotate:45,fontSize:9}},
    yAxis:{type:'value',name:'Mbps',axisLabel:{color:'#666'},nameTextStyle:{color:'#666'}},
    series:[{type:'bar',data:tpData.map(d=>d.value),itemStyle:{color:'#66bb6a'}}]});
}
function updateTable(results){
  const t=document.getElementById('tableBody');
  t.innerHTML=results.slice(-20).reverse().map(r=>`<tr>
    <td>${(r.request_id||'').substring(0,8)}</td>
    <td class="${statusClass(r.e2e_ms)}">${(r.e2e_ms||0).toFixed(1)}</td>
    <td>${r.decision||'N/A'}</td><td>${r.risk_level||'N/A'}</td>
    <td>${r.tracked_objects||0}</td>
    <td style="font-size:10px">${(r.trace_id||'').substring(0,12)}</td></tr>`).join('');
}
function updateBottlenecks(results){
  if(!results.length)return;
  const last=results[results.length-1];
  const bn=last.bottlenecks||[];
  const t=document.getElementById('bnBody');
  t.innerHTML=bn.map((b,i)=>`<tr>
    <td>${i+1}</td><td>${b.type}</td><td>${b.name}</td>
    <td>${b.latency_ms.toFixed(1)} ms</td><td>${b.pct.toFixed(1)}%</td>
    <td>${b.type==='network'?(b.throughput_mbps||0).toFixed(1)+' Mbps':'-'}</td></tr>`).join('');
}
async function refresh(){
  try{
    const [dr,ir]=await Promise.all([fetch('/api/results').then(r=>r.json()),fetch('/api/image').then(r=>r.json())]);
    updateCards(dr);updateCharts(dr.results||[]);updateTable(dr.results||[]);updateBottlenecks(dr.results||[]);
    const latest=dr&&dr.results&&dr.results.length?dr.results[dr.results.length-1]:null;
    const dets=[];
    if(latest){if(latest.rgb_detections)dets.push(...latest.rgb_detections);if(latest.ir_detections)dets.push(...latest.ir_detections);}
    const imgEl=document.getElementById('latestImg');
    if(ir.image){
      imgEl.onload=function(){resizeOverlay();drawBoxes(dets);};
      imgEl.src='data:image/jpeg;base64,'+ir.image;
      imgEl.style.display='block';document.getElementById('noImg').style.display='none';
    }else{imgEl.src='';imgEl.style.display='none';document.getElementById('noImg').style.display='block';}
  }catch(e){console.error('Refresh error:',e)}
}
setInterval(refresh,2000);refresh();
window.addEventListener('resize',()=>{latencyChart.resize();computeChart.resize();networkChart.resize();throughputChart.resize();resizeOverlay();});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("  SMOOTHING_WINDOW_SIZE=%d", SMOOTHING_WINDOW_SIZE)
    logger.info("  JAEGER_QUERY_ENDPOINT=%s", JAEGER_QUERY_ENDPOINT)
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    yield
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = create_app(lifespan_func=lifespan)


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
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()
    body = await request.body()

    # Record incoming payload on SERVER span
    server_span = trace.get_current_span()
    if server_span and server_span.is_recording():
        server_span.set_attribute("messaging.payload_size_bytes", len(body))

    # Extract trace-id from current span context (propagated via W3C traceparent)
    trace_id = get_current_trace_id()

    # Process telemetry data
    with tracer.start_as_current_span("compute:telemetry_analysis", kind=SpanKind.INTERNAL) as ispan:
        ispan.set_attribute("input_size_bytes", len(body))
        result = await _handle_telemetry(body, trace_id)
        ispan.set_attribute("e2e_ms", result.get("e2e_ms", 0.0))
        ispan.set_attribute("critical_path_ms", result.get("critical_path_ms", 0.0))

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "request_id": result.get("request_id", ""),
        "result": result,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
