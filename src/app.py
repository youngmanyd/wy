"""
Unified data-driven microservice for UAV Edge Computing DAG.
Each microservice role is determined by the SERVICE_ROLE env var.

Semantic Naming & DAG Topology:
  gateway -> rgb-preprocessor & ir-preprocessor
  rgb-preprocessor -> rgb-detector
  ir-preprocessor  -> ir-detector
  rgb-detector & ir-detector -> feature-fusion
  feature-fusion -> object-tracker & situation-awareness
  object-tracker & situation-awareness -> decision-maker
  decision-maker -> telemetry-dashboard
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ---------------------------------------------------------------------------
# OpenTelemetry imports
# ---------------------------------------------------------------------------
from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Link, SpanKind
from opentelemetry.propagate import extract, inject
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ---------------------------------------------------------------------------
# Service name mapping (semantic <-> legacy for internal use)
# ---------------------------------------------------------------------------
VALID_ROLES = {
    "gateway", "rgb-preprocessor", "ir-preprocessor",
    "rgb-detector", "ir-detector", "feature-fusion",
    "object-tracker", "situation-awareness", "decision-maker",
    "telemetry-dashboard",
}

# Roles that need ONNX
DETECTOR_ROLES = {"rgb-detector", "ir-detector"}
# Roles that are fusion (sync) nodes
FUSION_ROLES = {"feature-fusion", "decision-maker"}
# Roles that do matrix compute
COMPUTE_ROLES = {"object-tracker", "situation-awareness"}
# Preprocessor roles
PREPROC_ROLES = {"rgb-preprocessor", "ir-preprocessor"}

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------
SERVICE_ROLE: str = os.environ.get("SERVICE_ROLE", "gateway")
SERVICE_PORT: int = int(os.environ.get("SERVICE_PORT", "8000"))
DOWNSTREAM_URLS: str = os.environ.get("DOWNSTREAM_URLS", "")
ONNX_MODEL_PATH: str = os.environ.get("ONNX_MODEL_PATH", "/app/models/yolov8n.onnx")
JAEGER_ENDPOINT: str = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
OMP_NUM_THREADS: int = int(os.environ.get("OMP_NUM_THREADS", "1"))
FUSION_TIMEOUT: float = float(os.environ.get("FUSION_TIMEOUT", "5.0"))
TARGET_PREPROC_PAYLOAD: int = int(os.environ.get("TARGET_PREPROC_PAYLOAD", str(200 * 1024)))
TARGET_DETECT_PAYLOAD: int = int(os.environ.get("TARGET_DETECT_PAYLOAD", str(800 * 1024)))
HTTP_TIMEOUT: float = float(os.environ.get("HTTP_TIMEOUT", "10.0"))

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OMP_NUM_THREADS)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{SERVICE_ROLE}] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(SERVICE_ROLE)

# ---------------------------------------------------------------------------
# Thread pool for CPU-bound work (OpenCV / ONNX)
# ---------------------------------------------------------------------------
cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{SERVICE_ROLE}-cpu")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
onnx_session = None
fusion_buffers: dict = {}
fusion_events: dict = {}
fusion_contexts: dict = {}

# Dashboard state (ring buffer for last N results)
DASHBOARD_MAX_RESULTS = 200
dashboard_results: list = []

# Prometheus metrics
REQUEST_COUNT = Counter("ms_request_total", "Total requests processed", ["service_role"])
REQUEST_LATENCY = Histogram("ms_request_latency_seconds", "Request latency", ["service_role"])
COMPUTE_LATENCY = Histogram("ms_compute_latency_seconds", "Compute latency", ["service_role"])

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------
resource = Resource.create({"service.name": SERVICE_ROLE})
provider = TracerProvider(resource=resource)
try:
    otlp_exporter = OTLPSpanExporter(endpoint=JAEGER_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
except Exception as e:
    logger.warning("Failed to init OTLP exporter: %s", e)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(SERVICE_ROLE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_us() -> str:
    return f"{time.time():.6f}"


def _parse_downstream() -> list[str]:
    if not DOWNSTREAM_URLS.strip():
        return []
    return [u.strip() for u in DOWNSTREAM_URLS.split(",") if u.strip()]


def _print_env():
    logger.info("=" * 60)
    logger.info("Service Self-Check: %s", SERVICE_ROLE)
    logger.info("  SERVICE_PORT=%s", SERVICE_PORT)
    logger.info("  DOWNSTREAM_URLS=%s", DOWNSTREAM_URLS)
    logger.info("  ONNX_MODEL_PATH=%s", ONNX_MODEL_PATH)
    logger.info("  OTEL_EXPORTER_OTLP_ENDPOINT=%s", JAEGER_ENDPOINT)
    logger.info("  OMP_NUM_THREADS=%s", OMP_NUM_THREADS)
    logger.info("  FUSION_TIMEOUT=%s", FUSION_TIMEOUT)
    logger.info("  TARGET_PREPROC_PAYLOAD=%s", TARGET_PREPROC_PAYLOAD)
    logger.info("  TARGET_DETECT_PAYLOAD=%s", TARGET_DETECT_PAYLOAD)
    logger.info("  HTTP_TIMEOUT=%s", HTTP_TIMEOUT)
    logger.info("=" * 60)


def _load_onnx_session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = OMP_NUM_THREADS
    opts.inter_op_num_threads = OMP_NUM_THREADS
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    logger.info("Loading ONNX model from %s (threads=%d)", ONNX_MODEL_PATH, OMP_NUM_THREADS)
    session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    logger.info("ONNX model loaded. Input: %s", session.get_inputs()[0].shape)
    return session


# ---------------------------------------------------------------------------
# Compute functions (synchronous — dispatched via executor)
# ---------------------------------------------------------------------------
def _preprocess_image(img_bytes: bytes) -> bytes:
    """Real OpenCV resize + normalize. Output ~TARGET_PREPROC_PAYLOAD bytes."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_uint8 = (img_chw * 255).clip(0, 255).astype(np.uint8)
    tensor_bytes = img_uint8.tobytes()
    usable = TARGET_PREPROC_PAYLOAD - 4
    if len(tensor_bytes) > usable:
        tensor_bytes = tensor_bytes[:usable]
    header = len(tensor_bytes).to_bytes(4, "big")
    payload = header + tensor_bytes
    if len(payload) < TARGET_PREPROC_PAYLOAD:
        payload += b"\x00" * (TARGET_PREPROC_PAYLOAD - len(payload))
    return payload


def _run_yolo_detection(tensor_bytes: bytes) -> tuple[list[dict], bytes]:
    """Real YOLOv8 ONNX inference. Output ~TARGET_DETECT_PAYLOAD bytes."""
    global onnx_session
    if len(tensor_bytes) >= 4:
        tensor_len = int.from_bytes(tensor_bytes[:4], "big")
        raw_uint8 = tensor_bytes[4:4 + tensor_len]
    else:
        raw_uint8 = tensor_bytes

    total_elements = 3 * 640 * 640
    if len(raw_uint8) >= total_elements:
        img_uint8 = np.frombuffer(raw_uint8[:total_elements], dtype=np.uint8).reshape(3, 640, 640)
    else:
        padded = np.zeros(total_elements, dtype=np.uint8)
        padded[:len(raw_uint8)] = np.frombuffer(raw_uint8, dtype=np.uint8)
        img_uint8 = padded.reshape(3, 640, 640)
    img_tensor = (img_uint8.astype(np.float32) / 255.0).reshape(1, 3, 640, 640)

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: img_tensor})

    raw_output = outputs[0]
    detections = []
    if raw_output.ndim == 3 and raw_output.shape[1] >= 5:
        preds = raw_output[0].T
        class_scores = preds[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        top_indices = np.argsort(max_scores)[-10:]
        for idx in top_indices:
            cx, cy, w, h = preds[idx, :4].tolist()
            class_id = int(np.argmax(preds[idx, 4:]))
            score = float(max_scores[idx])
            if score > 0.1:
                detections.append({
                    "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    "class_id": class_id,
                    "score": round(score, 4),
                })

    det_json = json.dumps(detections).encode()
    output_slice = raw_output[0, :, :].tobytes()
    header = len(det_json).to_bytes(4, "big")
    payload = header + det_json + output_slice
    if len(payload) > TARGET_DETECT_PAYLOAD:
        payload = payload[:TARGET_DETECT_PAYLOAD]
    elif len(payload) < TARGET_DETECT_PAYLOAD:
        payload += b"\x00" * (TARGET_DETECT_PAYLOAD - len(payload))
    return detections, payload


def _draw_detections_on_image(img_bytes: bytes, detections: list[dict]) -> bytes:
    """Draw bounding boxes on image, return JPEG bytes for dashboard display."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    sx, sy = w / 640.0, h / 640.0
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"cls{det['class_id']}:{det['score']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes()


def _matrix_computation() -> dict:
    """Real matrix math (~20-50ms) for tracker/situation-awareness."""
    size = 256
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    C = A @ B
    eigenvalues = np.linalg.eigvalsh(C[:64, :64])
    svd_u, svd_s, _ = np.linalg.svd(A[:128, :128], full_matrices=False)
    return {
        "trace": float(np.trace(C)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "top_singular_values": svd_s[:5].tolist(),
        "determinant_sign": float(np.sign(np.linalg.det(A[:32, :32]))),
    }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    global onnx_session
    _print_env()
    if SERVICE_ROLE in DETECTOR_ROLES:
        loop = asyncio.get_running_loop()
        onnx_session = await loop.run_in_executor(cpu_executor, _load_onnx_session)
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    yield
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = FastAPI(title=f"UAV-Edge-{SERVICE_ROLE}", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


# ---------------------------------------------------------------------------
# Common endpoints
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok", "role": SERVICE_ROLE}


# ---------------------------------------------------------------------------
# Async HTTP forward
# ---------------------------------------------------------------------------
async def _forward(url: str, payload: bytes, headers: dict,
                   content_type: str = "application/octet-stream") -> Optional[dict]:
    fwd_headers = {k: v for k, v in headers.items()}
    fwd_headers["Content-Type"] = content_type
    inject(fwd_headers)
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, content=payload, headers=fwd_headers)
            return resp.json() if resp.status_code == 200 else {"error": resp.status_code}
    except Exception as e:
        logger.error("Forward to %s failed: %s", url, e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------
async def _wait_for_fusion(request_id: str, source: str, data: dict, otel_ctx) -> Optional[dict]:
    if request_id not in fusion_buffers:
        fusion_buffers[request_id] = {}
        fusion_events[request_id] = asyncio.Event()
        fusion_contexts[request_id] = []

    fusion_buffers[request_id][source] = data
    fusion_contexts[request_id].append(otel_ctx)

    if len(fusion_buffers[request_id]) >= 2:
        fusion_events[request_id].set()

    try:
        await asyncio.wait_for(fusion_events[request_id].wait(), timeout=FUSION_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Fusion timeout for request %s (source=%s)", request_id, source)
        return None

    if request_id in fusion_buffers:
        fused = fusion_buffers.pop(request_id)
        fusion_events.pop(request_id, None)
        ctx_list = fusion_contexts.pop(request_id, [])
        return {"fused": fused, "contexts": ctx_list}
    return None


# ---------------------------------------------------------------------------
# Main processing route
# ---------------------------------------------------------------------------
@app.post("/process")
async def process(request: Request):
    arrival_time = _now_us()
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()

    x_start_time = request.headers.get("X-Start-Time", arrival_time)
    x_request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    x_source = request.headers.get("X-Source", "unknown")
    x_timing_chain = request.headers.get("X-Timing-Chain", "")

    body = await request.body()
    incoming_ctx = extract(dict(request.headers))
    loop = asyncio.get_running_loop()

    if SERVICE_ROLE == "gateway":
        result = await _handle_gateway(body, x_start_time, x_request_id, arrival_time, x_timing_chain)
    elif SERVICE_ROLE in PREPROC_ROLES:
        result = await _handle_preprocessor(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif SERVICE_ROLE in DETECTOR_ROLES:
        result = await _handle_detector(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif SERVICE_ROLE in FUSION_ROLES:
        result = await _handle_fusion(body, x_start_time, x_request_id, x_source, arrival_time, x_timing_chain, incoming_ctx, loop)
    elif SERVICE_ROLE in COMPUTE_ROLES:
        result = await _handle_compute(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif SERVICE_ROLE == "telemetry-dashboard":
        result = await _handle_telemetry(body, x_start_time, x_request_id, arrival_time, x_timing_chain)
    else:
        result = {"error": f"Unknown role: {SERVICE_ROLE}"}

    compute_end = time.time()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id, "result": result})


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------
async def _handle_gateway(body: bytes, t0: str, req_id: str, arrival: str, chain: str):
    with tracer.start_as_current_span("gateway-dispatch", kind=SpanKind.SERVER):
        compute_start = time.time()
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"gateway|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry
        downstream = _parse_downstream()
        if not downstream:
            return {"status": "no_downstream"}
        headers = {
            "X-Start-Time": t0, "X-Request-ID": req_id,
            "X-Source": "gateway", "X-Timing-Chain": new_chain,
            "X-Original-Image": base64.b64encode(body).decode() if len(body) < 500_000 else "",
        }
        tasks = [_forward(url + "/process", body, headers) for url in downstream]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"forwarded_to": downstream, "results_count": len(results)}


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
async def _handle_preprocessor(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("preprocess", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time()
        processed = await loop.run_in_executor(cpu_executor, _preprocess_image, body)
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry
        downstream = _parse_downstream()
        # Pass original image in header for later visualization
        orig_b64 = base64.b64encode(body).decode() if len(body) < 500_000 else ""
        headers = {
            "X-Start-Time": t0, "X-Request-ID": req_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
            "X-Original-Image": orig_b64,
        }
        tasks = [_forward(url + "/process", processed, headers) for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"preprocessed_size": len(processed), "forwarded": len(downstream)}


# ---------------------------------------------------------------------------
# Detector (ONNX YOLOv8)
# ---------------------------------------------------------------------------
async def _handle_detector(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("yolo-detect", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time()
        detections, payload = await loop.run_in_executor(cpu_executor, _run_yolo_detection, body)
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        span.set_attribute("detections.count", len(detections))
        timing_entry = f"{SERVICE_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry
        downstream = _parse_downstream()
        headers = {
            "X-Start-Time": t0, "X-Request-ID": req_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
            "X-Detections": json.dumps(detections),
        }
        tasks = [_forward(url + "/process", payload, headers) for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"detections": len(detections), "payload_size": len(payload)}


# ---------------------------------------------------------------------------
# Fusion (with OTel Span Links)
# ---------------------------------------------------------------------------
async def _handle_fusion(body: bytes, t0: str, req_id: str, source: str,
                         arrival: str, chain: str, incoming_ctx, loop):
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {"raw_size": len(body), "source": source}
    data["_arrival"] = arrival
    data["_chain"] = chain
    data["_source"] = source

    fused = await _wait_for_fusion(req_id, source, data, incoming_ctx)
    if fused is None:
        return {"status": "waiting_or_timeout"}

    links = []
    for ctx in fused.get("contexts", []):
        upstream_span = trace.get_current_span(ctx)
        if upstream_span and upstream_span.get_span_context().is_valid:
            links.append(Link(upstream_span.get_span_context()))

    with tracer.start_as_current_span(f"fusion-{SERVICE_ROLE}", kind=SpanKind.SERVER, links=links) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        span.set_attribute("fusion.sources", str(list(fused["fused"].keys())))
        compute_start = time.time()
        chains = []
        for src_data in fused["fused"].values():
            if isinstance(src_data, dict) and "_chain" in src_data:
                chains.append(src_data["_chain"])
        merged_chain = ";".join(chains)
        fusion_result = {
            "sources": list(fused["fused"].keys()),
            "merged_keys": sum(len(v) if isinstance(v, dict) else 0 for v in fused["fused"].values()),
        }
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{merged_chain},{timing_entry}" if merged_chain else timing_entry
        downstream = _parse_downstream()
        if not downstream:
            return {"fused": fusion_result, "chain": new_chain}
        fused_payload = json.dumps(fusion_result).encode()
        headers = {
            "X-Start-Time": t0, "X-Request-ID": req_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
        }
        tasks = [_forward(url + "/process", fused_payload, headers, "application/json") for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"fused": fusion_result, "forwarded": len(downstream)}


# ---------------------------------------------------------------------------
# Compute (matrix math)
# ---------------------------------------------------------------------------
async def _handle_compute(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("matrix-compute", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time()
        result = await loop.run_in_executor(cpu_executor, _matrix_computation)
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry
        downstream = _parse_downstream()
        payload = json.dumps(result).encode()
        headers = {
            "X-Start-Time": t0, "X-Request-ID": req_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
        }
        tasks = [_forward(url + "/process", payload, headers, "application/json") for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)
        return {"compute_result": result, "forwarded": len(downstream)}


# ---------------------------------------------------------------------------
# Telemetry & Dashboard
# ---------------------------------------------------------------------------
# DAG edges using semantic names
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

        # CSV output
        csv_parts = [req_id, f"{e2e_latency_ms:.3f}", f"{rgb_branch_ms:.3f}", f"{ir_branch_ms:.3f}"]
        for svc in ALL_SERVICES:
            csv_parts.append(f"{node_compute.get(svc, 0.0):.3f}")
        for src, dst in DAG_EDGES:
            csv_parts.append(f"{net_latencies.get(f'{src}->{dst}', 0.0):.3f}")
        csv_line = ",".join(csv_parts)
        print(f"CSV_RESULT:{csv_line}", flush=True)

        # Store for dashboard
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


# ---------------------------------------------------------------------------
# Dashboard API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/results")
async def api_results():
    return JSONResponse(content={"results": dashboard_results[-50:]})


@app.get("/api/latest")
async def api_latest():
    if dashboard_results:
        return JSONResponse(content=dashboard_results[-1])
    return JSONResponse(content={})


# ---------------------------------------------------------------------------
# Dashboard HTML (served at /)
# ---------------------------------------------------------------------------
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
      // Reset chart data to avoid duplication
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
    if SERVICE_ROLE == "telemetry-dashboard":
        return HTMLResponse(content=DASHBOARD_HTML)
    return HTMLResponse(content=f"<h1>UAV Edge Service: {SERVICE_ROLE}</h1><p><a href='/health'>Health</a> | <a href='/metrics'>Metrics</a></p>")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
