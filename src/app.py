"""
Unified data-driven microservice for UAV Edge Computing DAG.
Each microservice role (MS-1 through MS-10) is determined by the MS_ROLE env var.

Architecture:
  MS-1 (Gateway) -> MS-2 (RGB Pre) & MS-3 (IR Pre)
  MS-2 -> MS-4 (RGB Detector)   MS-3 -> MS-5 (IR Detector)
  MS-4 & MS-5 -> MS-6 (Feature Fusion)
  MS-6 -> MS-7 (Object Tracker) & MS-8 (Situation Awareness)
  MS-7 & MS-8 -> MS-9 (Decision Maker)
  MS-9 -> MS-10 (Telemetry & Aggregator)
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
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
# Environment & configuration
# ---------------------------------------------------------------------------
MS_ROLE: str = os.environ.get("MS_ROLE", "ms-1")
MS_PORT: int = int(os.environ.get("MS_PORT", "8000"))
DOWNSTREAM_URLS: str = os.environ.get("DOWNSTREAM_URLS", "")  # comma-separated
ONNX_MODEL_PATH: str = os.environ.get("ONNX_MODEL_PATH", "/app/models/yolov8n.onnx")
JAEGER_ENDPOINT: str = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
OMP_NUM_THREADS: int = int(os.environ.get("OMP_NUM_THREADS", "1"))
FUSION_TIMEOUT: float = float(os.environ.get("FUSION_TIMEOUT", "5.0"))
TARGET_PREPROC_PAYLOAD: int = int(os.environ.get("TARGET_PREPROC_PAYLOAD", str(200 * 1024)))
TARGET_DETECT_PAYLOAD: int = int(os.environ.get("TARGET_DETECT_PAYLOAD", str(800 * 1024)))
HTTP_TIMEOUT: float = float(os.environ.get("HTTP_TIMEOUT", "10.0"))

# Force ONNX/OpenMP thread control
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OMP_NUM_THREADS)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{MS_ROLE}] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(MS_ROLE)

# ---------------------------------------------------------------------------
# Thread pool for CPU-bound work (OpenCV / ONNX)
# ---------------------------------------------------------------------------
cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{MS_ROLE}-cpu")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
onnx_session = None  # lazily loaded for detector roles
fusion_buffers: dict = {}  # request_id -> {partial results}
fusion_events: dict = {}  # request_id -> asyncio.Event
fusion_contexts: dict = {}  # request_id -> list of OTel contexts

# Prometheus metrics
REQUEST_COUNT = Counter(
    "ms_request_total", "Total requests processed", ["ms_role"]
)
REQUEST_LATENCY = Histogram(
    "ms_request_latency_seconds", "Request latency in seconds", ["ms_role"]
)
COMPUTE_LATENCY = Histogram(
    "ms_compute_latency_seconds", "Compute latency in seconds", ["ms_role"]
)

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------
resource = Resource.create({"service.name": MS_ROLE})
provider = TracerProvider(resource=resource)
try:
    otlp_exporter = OTLPSpanExporter(endpoint=JAEGER_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
except Exception as e:
    logger.warning("Failed to init OTLP exporter (Jaeger may be unavailable): %s", e)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(MS_ROLE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_us() -> str:
    """Return current time as microsecond-precision string."""
    return f"{time.time():.6f}"


def _parse_downstream() -> list[str]:
    """Parse comma-separated downstream URLs."""
    if not DOWNSTREAM_URLS.strip():
        return []
    return [u.strip() for u in DOWNSTREAM_URLS.split(",") if u.strip()]


def _print_env():
    """Print environment variables at startup for self-check."""
    logger.info("=" * 60)
    logger.info("Microservice Self-Check: %s", MS_ROLE)
    logger.info("  MS_PORT=%s", MS_PORT)
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
    """Load ONNX model with strict thread control."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = OMP_NUM_THREADS
    opts.inter_op_num_threads = OMP_NUM_THREADS
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    logger.info("Loading ONNX model from %s (threads=%d)", ONNX_MODEL_PATH, OMP_NUM_THREADS)
    session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    logger.info("ONNX model loaded successfully. Input: %s", session.get_inputs()[0].shape)
    return session


# ---------------------------------------------------------------------------
# Role-specific compute functions (all synchronous — run via executor)
# ---------------------------------------------------------------------------

def _preprocess_image(img_bytes: bytes) -> bytes:
    """
    MS-2 / MS-3: Real OpenCV resize + normalize.
    Returns a payload of ~TARGET_PREPROC_PAYLOAD bytes.
    Format: 4-byte header (tensor length) + float32 tensor (quantized) + padding.
    The tensor is stored as uint8 (0-255 rescaled from float32) to compress to ~200KB,
    with a header so the detector can reconstruct the float32 version.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    # CHW layout for ONNX
    img_chw = np.transpose(img_norm, (2, 0, 1))  # (3, 640, 640)
    # Quantize to uint8 to keep payload small (~1.2MB -> ~400KB)
    img_uint8 = (img_chw * 255).clip(0, 255).astype(np.uint8)  # (3, 640, 640)
    tensor_bytes = img_uint8.tobytes()  # 3*640*640 = 1,228,800 bytes
    # Truncate to fit target payload (keep first TARGET_PREPROC_PAYLOAD - 4 bytes)
    usable = TARGET_PREPROC_PAYLOAD - 4
    if len(tensor_bytes) > usable:
        tensor_bytes = tensor_bytes[:usable]
    header = len(tensor_bytes).to_bytes(4, "big")
    payload = header + tensor_bytes
    # Pad to exact target
    if len(payload) < TARGET_PREPROC_PAYLOAD:
        payload += b"\x00" * (TARGET_PREPROC_PAYLOAD - len(payload))
    return payload


def _run_yolo_detection(tensor_bytes: bytes) -> tuple[list[dict], bytes]:
    """
    MS-4 / MS-5: Real YOLOv8 ONNX inference.
    Input: preprocessed payload from MS-2/MS-3 (4-byte header + uint8 tensor + padding).
    Returns (detections_list, padded_payload_bytes ~800KB).
    """
    global onnx_session
    # Decode the preprocessor payload
    if len(tensor_bytes) >= 4:
        tensor_len = int.from_bytes(tensor_bytes[:4], "big")
        raw_uint8 = tensor_bytes[4:4 + tensor_len]
    else:
        raw_uint8 = tensor_bytes

    # Reconstruct float32 tensor for ONNX
    # The preprocessor sent uint8 CHW data; reconstruct to (1,3,640,640) float32
    total_elements = 3 * 640 * 640
    if len(raw_uint8) >= total_elements:
        img_uint8 = np.frombuffer(raw_uint8[:total_elements], dtype=np.uint8).reshape(3, 640, 640)
    else:
        # Partial data — pad with zeros
        padded = np.zeros(total_elements, dtype=np.uint8)
        padded[:len(raw_uint8)] = np.frombuffer(raw_uint8, dtype=np.uint8)
        img_uint8 = padded.reshape(3, 640, 640)
    img_tensor = (img_uint8.astype(np.float32) / 255.0).reshape(1, 3, 640, 640)

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: img_tensor})

    # Parse YOLOv8 output: shape (1, 84, 8400) — 84 = 4 bbox + 80 classes
    raw_output = outputs[0]  # (1, 84, 8400)
    detections = []
    if raw_output.ndim == 3 and raw_output.shape[1] >= 5:
        preds = raw_output[0].T  # (8400, 84)
        class_scores = preds[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        top_indices = np.argsort(max_scores)[-10:]  # top 10
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

    # Build ~800KB payload: JSON detections + raw output tensor slice
    det_json = json.dumps(detections).encode()
    # Include ONNX output tensor (8400*84*4 = ~2.8MB → take a slice) + detections
    output_slice = raw_output[0, :, :].tobytes()  # full ONNX output
    header = len(det_json).to_bytes(4, "big")
    payload = header + det_json + output_slice
    # Trim or pad to ~TARGET_DETECT_PAYLOAD
    if len(payload) > TARGET_DETECT_PAYLOAD:
        payload = payload[:TARGET_DETECT_PAYLOAD]
    elif len(payload) < TARGET_DETECT_PAYLOAD:
        payload += b"\x00" * (TARGET_DETECT_PAYLOAD - len(payload))
    return detections, payload


def _matrix_computation() -> dict:
    """
    MS-7 / MS-8: Real matrix math (~20-50ms).
    """
    # Simulate tracking / situation-awareness with matrix ops
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
# Lifespan: init resources
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global onnx_session
    _print_env()
    if MS_ROLE in ("ms-4", "ms-5"):
        loop = asyncio.get_running_loop()
        onnx_session = await loop.run_in_executor(cpu_executor, _load_onnx_session)
    logger.info("Microservice %s is ready on port %d", MS_ROLE, MS_PORT)
    yield
    cpu_executor.shutdown(wait=False)
    logger.info("Microservice %s shutting down", MS_ROLE)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title=f"UAV-Edge-{MS_ROLE}", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok", "role": MS_ROLE}


# ---------------------------------------------------------------------------
# Async HTTP client (shared)
# ---------------------------------------------------------------------------

async def _forward(
    url: str,
    payload: bytes,
    headers: dict,
    content_type: str = "application/octet-stream",
) -> Optional[dict]:
    """POST payload to downstream URL with tracing headers."""
    fwd_headers = {k: v for k, v in headers.items()}
    fwd_headers["Content-Type"] = content_type
    # Inject OTel context into outgoing headers
    inject(fwd_headers)
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, content=payload, headers=fwd_headers)
            return resp.json() if resp.status_code == 200 else {"error": resp.status_code}
    except Exception as e:
        logger.error("Forward to %s failed: %s", url, e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Fusion helpers (MS-6, MS-9)
# ---------------------------------------------------------------------------

def _get_fusion_key(request_id: str, source: str) -> str:
    return f"{request_id}"


async def _wait_for_fusion(request_id: str, source: str, data: dict, otel_ctx) -> Optional[dict]:
    """
    Wait for both branches to arrive. Returns fused result when complete.
    """
    if request_id not in fusion_buffers:
        fusion_buffers[request_id] = {}
        fusion_events[request_id] = asyncio.Event()
        fusion_contexts[request_id] = []

    fusion_buffers[request_id][source] = data
    fusion_contexts[request_id].append(otel_ctx)

    if len(fusion_buffers[request_id]) >= 2:
        fusion_events[request_id].set()

    try:
        await asyncio.wait_for(
            fusion_events[request_id].wait(),
            timeout=FUSION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Fusion timeout for request %s (source=%s)", request_id, source)
        return None

    # Only the second arrival does cleanup and returns fused result
    result = fusion_buffers.get(request_id)
    contexts = fusion_contexts.get(request_id, [])

    # Cleanup (first caller to pop wins)
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
    REQUEST_COUNT.labels(ms_role=MS_ROLE).inc()

    # Extract headers
    x_start_time = request.headers.get("X-Start-Time", arrival_time)
    x_request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    x_source = request.headers.get("X-Source", "unknown")

    # Upstream timing headers (accumulated CSV for MS-10)
    x_timing_chain = request.headers.get("X-Timing-Chain", "")

    body = await request.body()

    # Extract incoming OTel context
    incoming_ctx = extract(dict(request.headers))

    loop = asyncio.get_running_loop()
    compute_start = time.time()

    # ----- Role dispatch -----
    if MS_ROLE == "ms-1":
        result = await _handle_gateway(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif MS_ROLE in ("ms-2", "ms-3"):
        result = await _handle_preprocessor(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif MS_ROLE in ("ms-4", "ms-5"):
        result = await _handle_detector(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif MS_ROLE in ("ms-6", "ms-9"):
        result = await _handle_fusion(
            body, x_start_time, x_request_id, x_source, arrival_time,
            x_timing_chain, incoming_ctx, loop,
        )
    elif MS_ROLE in ("ms-7", "ms-8"):
        result = await _handle_compute(body, x_start_time, x_request_id, arrival_time, x_timing_chain, loop)
    elif MS_ROLE == "ms-10":
        result = await _handle_telemetry(body, x_start_time, x_request_id, arrival_time, x_timing_chain)
    else:
        result = {"error": f"Unknown role: {MS_ROLE}"}

    compute_end = time.time()
    compute_ms = (compute_end - compute_start) * 1000
    COMPUTE_LATENCY.labels(ms_role=MS_ROLE).observe(compute_end - compute_start)
    REQUEST_LATENCY.labels(ms_role=MS_ROLE).observe(compute_end - float(x_start_time))

    return JSONResponse(content={"role": MS_ROLE, "request_id": x_request_id, "result": result})


# ---------------------------------------------------------------------------
# MS-1: Gateway
# ---------------------------------------------------------------------------

async def _handle_gateway(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("gateway-dispatch", kind=SpanKind.SERVER):
        compute_start = time.time()
        # Gateway does minimal work — just timestamps
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"

        timing_entry = f"{MS_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry

        downstream = _parse_downstream()
        if not downstream:
            return {"status": "no_downstream"}

        headers = {
            "X-Start-Time": t0,
            "X-Request-ID": req_id,
            "X-Source": MS_ROLE,
            "X-Timing-Chain": new_chain,
        }

        # Concurrent fan-out to MS-2 and MS-3
        tasks = [_forward(url + "/process", body, headers) for url in downstream]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"forwarded_to": downstream, "results_count": len(results)}


# ---------------------------------------------------------------------------
# MS-2/MS-3: Preprocessor
# ---------------------------------------------------------------------------

async def _handle_preprocessor(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("preprocess", kind=SpanKind.SERVER) as span:
        span.set_attribute("ms.role", MS_ROLE)
        compute_start = time.time()

        # Real OpenCV preprocessing in thread pool
        processed = await loop.run_in_executor(cpu_executor, _preprocess_image, body)

        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{MS_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry

        downstream = _parse_downstream()
        headers = {
            "X-Start-Time": t0,
            "X-Request-ID": req_id,
            "X-Source": MS_ROLE,
            "X-Timing-Chain": new_chain,
        }
        tasks = [_forward(url + "/process", processed, headers) for url in downstream]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"preprocessed_size": len(processed), "forwarded": len(downstream)}


# ---------------------------------------------------------------------------
# MS-4/MS-5: Detector (ONNX YOLOv8)
# ---------------------------------------------------------------------------

async def _handle_detector(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("yolo-detect", kind=SpanKind.SERVER) as span:
        span.set_attribute("ms.role", MS_ROLE)
        compute_start = time.time()

        detections, payload = await loop.run_in_executor(
            cpu_executor, _run_yolo_detection, body
        )

        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        span.set_attribute("detections.count", len(detections))

        timing_entry = f"{MS_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry

        downstream = _parse_downstream()
        headers = {
            "X-Start-Time": t0,
            "X-Request-ID": req_id,
            "X-Source": MS_ROLE,
            "X-Timing-Chain": new_chain,
        }
        tasks = [_forward(url + "/process", payload, headers) for url in downstream]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"detections": len(detections), "payload_size": len(payload)}


# ---------------------------------------------------------------------------
# MS-6/MS-9: Fusion (with OTel Span Links)
# ---------------------------------------------------------------------------

async def _handle_fusion(
    body: bytes, t0: str, req_id: str, source: str, arrival: str,
    chain: str, incoming_ctx, loop,
):
    # Parse incoming JSON (for fusion nodes, body may be JSON or binary)
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Binary payload from detector — wrap it
        data = {"raw_size": len(body), "source": source}

    data["_arrival"] = arrival
    data["_chain"] = chain
    data["_source"] = source

    fused = await _wait_for_fusion(req_id, source, data, incoming_ctx)

    if fused is None:
        return {"status": "waiting_or_timeout"}

    # Create span with Links to both upstream spans
    links = []
    for ctx in fused.get("contexts", []):
        upstream_span = trace.get_current_span(ctx)
        if upstream_span and upstream_span.get_span_context().is_valid:
            links.append(Link(upstream_span.get_span_context()))

    with tracer.start_as_current_span(
        f"fusion-{MS_ROLE}",
        kind=SpanKind.SERVER,
        links=links,
    ) as span:
        span.set_attribute("ms.role", MS_ROLE)
        span.set_attribute("fusion.sources", str(list(fused["fused"].keys())))
        compute_start = time.time()

        # Merge timing chains from both branches
        chains = []
        for src_key, src_data in fused["fused"].items():
            if isinstance(src_data, dict) and "_chain" in src_data:
                chains.append(src_data["_chain"])

        merged_chain = ";".join(chains)

        # Fusion logic: merge detections/results
        fusion_result = {
            "sources": list(fused["fused"].keys()),
            "merged_keys": sum(len(v) if isinstance(v, dict) else 0 for v in fused["fused"].values()),
        }

        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{MS_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{merged_chain},{timing_entry}" if merged_chain else timing_entry

        downstream = _parse_downstream()
        if not downstream:
            return {"fused": fusion_result, "chain": new_chain}

        fused_payload = json.dumps(fusion_result).encode()
        headers = {
            "X-Start-Time": t0,
            "X-Request-ID": req_id,
            "X-Source": MS_ROLE,
            "X-Timing-Chain": new_chain,
        }

        tasks = [
            _forward(url + "/process", fused_payload, headers, "application/json")
            for url in downstream
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"fused": fusion_result, "forwarded": len(downstream)}


# ---------------------------------------------------------------------------
# MS-7/MS-8: Compute (matrix math)
# ---------------------------------------------------------------------------

async def _handle_compute(body: bytes, t0: str, req_id: str, arrival: str, chain: str, loop):
    with tracer.start_as_current_span("matrix-compute", kind=SpanKind.SERVER) as span:
        span.set_attribute("ms.role", MS_ROLE)
        compute_start = time.time()

        result = await loop.run_in_executor(cpu_executor, _matrix_computation)

        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{MS_ROLE}|{arrival}|{compute_time}"
        new_chain = f"{chain},{timing_entry}" if chain else timing_entry

        downstream = _parse_downstream()
        payload = json.dumps(result).encode()
        headers = {
            "X-Start-Time": t0,
            "X-Request-ID": req_id,
            "X-Source": MS_ROLE,
            "X-Timing-Chain": new_chain,
        }
        tasks = [
            _forward(url + "/process", payload, headers, "application/json")
            for url in downstream
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"compute_result": result, "forwarded": len(downstream)}


# ---------------------------------------------------------------------------
# MS-10: Telemetry & Aggregator (CSV output)
# ---------------------------------------------------------------------------

async def _handle_telemetry(body: bytes, t0: str, req_id: str, arrival: str, chain: str):
    with tracer.start_as_current_span("telemetry-aggregate", kind=SpanKind.SERVER) as span:
        span.set_attribute("ms.role", MS_ROLE)
        compute_start = time.time()

        e2e_latency_ms = (time.time() - float(t0)) * 1000

        # Parse timing chain: "ms-1|arrival|compute;ms-2|arrival|compute,..."
        # The chain may have ; separators from fusion merges
        node_timings = {}
        all_entries = []
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
                all_entries.append(node_name)

        # Add MS-10 self
        ms10_compute_us = (time.time() - compute_start) * 1e6
        node_timings["ms-10"] = {
            "arrival": float(arrival),
            "compute_us": ms10_compute_us,
        }

        # Compute per-node compute latency (ms)
        node_compute = {}
        for name, t in node_timings.items():
            node_compute[name] = t["compute_us"] / 1000.0

        # Compute inter-hop network latency (ms)
        # DAG edges:
        edges = [
            ("ms-1", "ms-2"), ("ms-1", "ms-3"),
            ("ms-2", "ms-4"), ("ms-3", "ms-5"),
            ("ms-4", "ms-6"), ("ms-5", "ms-6"),
            ("ms-6", "ms-7"), ("ms-6", "ms-8"),
            ("ms-7", "ms-9"), ("ms-8", "ms-9"),
            ("ms-9", "ms-10"),
        ]
        net_latencies = {}
        for src, dst in edges:
            if src in node_timings and dst in node_timings:
                src_finish = node_timings[src]["arrival"] + node_timings[src]["compute_us"] / 1e6
                dst_arrival = node_timings[dst]["arrival"]
                net_ms = (dst_arrival - src_finish) * 1000
                net_latencies[f"{src}->{dst}"] = round(net_ms, 3)

        # Branch latencies
        rgb_branch_ms = 0.0
        ir_branch_ms = 0.0
        if "ms-2" in node_timings and "ms-4" in node_timings:
            rgb_start = node_timings["ms-2"]["arrival"]
            rgb_end = node_timings["ms-4"]["arrival"] + node_timings["ms-4"]["compute_us"] / 1e6
            rgb_branch_ms = (rgb_end - rgb_start) * 1000
        if "ms-3" in node_timings and "ms-5" in node_timings:
            ir_start = node_timings["ms-3"]["arrival"]
            ir_end = node_timings["ms-5"]["arrival"] + node_timings["ms-5"]["compute_us"] / 1e6
            ir_branch_ms = (ir_end - ir_start) * 1000

        # Build CSV output line
        # Format: request_id, e2e_ms, rgb_branch_ms, ir_branch_ms,
        #         ms1_compute, ms2_compute, ..., ms10_compute,
        #         net_ms1->ms2, net_ms1->ms3, ...
        csv_parts = [
            req_id,
            f"{e2e_latency_ms:.3f}",
            f"{rgb_branch_ms:.3f}",
            f"{ir_branch_ms:.3f}",
        ]
        for i in range(1, 11):
            name = f"ms-{i}"
            csv_parts.append(f"{node_compute.get(name, 0.0):.3f}")

        for src, dst in edges:
            key = f"{src}->{dst}"
            csv_parts.append(f"{net_latencies.get(key, 0.0):.3f}")

        csv_line = ",".join(csv_parts)
        # Print CSV with prefix for grep collection
        print(f"CSV_RESULT:{csv_line}", flush=True)

        span.set_attribute("e2e_latency_ms", e2e_latency_ms)

        return {
            "request_id": req_id,
            "e2e_latency_ms": round(e2e_latency_ms, 3),
            "rgb_branch_ms": round(rgb_branch_ms, 3),
            "ir_branch_ms": round(ir_branch_ms, 3),
            "node_compute_ms": {k: round(v, 3) for k, v in node_compute.items()},
            "network_latency_ms": net_latencies,
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=MS_PORT, log_level="info")
