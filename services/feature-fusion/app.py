"""Feature Fusion — synchronization point for RGB & IR detector branches.

Uses **trace-id** (from W3C traceparent, auto-propagated) as the correlation
key for the two branches.  No custom X-* headers are used.

OTel architecture:
- SERVER span: auto (FastAPIInstrumentor) — one per incoming HTTP request
- INTERNAL span: wraps fusion logic, created with **SpanLinks** to both
  upstream SERVER spans for Jaeger diamond-DAG visualization
- CLIENT spans: auto (HTTPXInstrumentor) for downstream forwards
- Forward calls OUTSIDE the INTERNAL span
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, PAYLOAD_BYTES, FUSION_TIMEOUT, FUSION_BUFFER_TTL,
    parse_downstream, print_env, forward,
    create_app, SpanKind, Link, logger, trace,
    get_current_trace_id,
)

# Fusion synchronization state
fusion_buffers: dict = {}   # trace_id -> {source_key: parsed_data, ...}
fusion_events: dict = {}    # trace_id -> asyncio.Event
fusion_contexts: dict = {}  # trace_id -> [span_context, ...]
fusion_timestamps: dict = {}  # trace_id -> creation time

_buffer_lock = asyncio.Lock()


async def _cleanup_stale_buffers():
    """Remove fusion buffers older than TTL to prevent memory leaks."""
    while True:
        await asyncio.sleep(FUSION_BUFFER_TTL)
        async with _buffer_lock:
            now = time.monotonic()
            stale = [tid for tid, ts in fusion_timestamps.items()
                     if now - ts > FUSION_BUFFER_TTL]
            for tid in stale:
                fusion_buffers.pop(tid, None)
                fusion_events.pop(tid, None)
                fusion_contexts.pop(tid, None)
                fusion_timestamps.pop(tid, None)
            if stale:
                logger.info("Cleaned %d stale fusion buffers", len(stale))


def _parse_detector_payload(body: bytes) -> dict:
    """Parse detector binary: 4B_json_len + json + raw_tensor."""
    if len(body) < 4:
        return {"detections": [], "original_image_b64": "", "payload_size": len(body)}
    json_len = int.from_bytes(body[:4], "big")
    if 0 < json_len < len(body):
        try:
            det_data = json.loads(body[4:4 + json_len])
            return {
                "detections": det_data.get("detections", []),
                "original_image_b64": det_data.get("original_image_b64", ""),
                "payload_size": len(body),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"detections": [], "original_image_b64": "", "payload_size": len(body)}


def _fuse_data(fused: dict) -> tuple[dict, bytes]:
    """Fuse data from multiple sources (CPU-bound)."""
    all_detections = {"rgb": [], "ir": []}
    original_image_b64 = ""

    for src_key, src_data in fused.items():
        if isinstance(src_data, dict):
            dets = src_data.get("detections", [])
            if "rgb" in src_key:
                all_detections["rgb"] = dets
            elif "ir" in src_key:
                all_detections["ir"] = dets
            if not original_image_b64 and src_data.get("original_image_b64"):
                original_image_b64 = src_data["original_image_b64"]

    fusion_result = {
        "sources": list(fused.keys()),
        "rgb_detections": all_detections["rgb"],
        "ir_detections": all_detections["ir"],
        "total_detections": len(all_detections["rgb"]) + len(all_detections["ir"]),
        "original_image_b64": original_image_b64,
    }
    payload = json.dumps(fusion_result).encode()
    return fusion_result, payload


@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    cleanup_task = asyncio.create_task(_cleanup_stale_buffers())
    yield
    cleanup_task.cancel()
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = create_app(lifespan_func=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        content=f"<h1>UAV Edge Service: {SERVICE_ROLE}</h1>"
        f"<p>Fusion buffers: {len(fusion_buffers)}</p>"
        f"<p><a href='/health'>Health</a> | <a href='/metrics'>Metrics</a></p>"
    )


@app.post("/process")
async def process(request: Request):
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()
    body = await request.body()

    # Record incoming payload on SERVER span
    server_span = trace.get_current_span()
    if server_span and server_span.is_recording():
        server_span.set_attribute("messaging.payload_size_bytes", len(body))

    # Use trace-id as correlation key (auto-propagated via W3C traceparent)
    trace_id = get_current_trace_id()
    if not trace_id:
        return JSONResponse(content={"role": SERVICE_ROLE, "error": "no_trace_id"})

    # Capture current span context for SpanLinks
    current_span_ctx = trace.get_current_span().get_span_context()

    # Parse detector binary payload
    parsed = _parse_detector_payload(body)

    # Determine source key from span attributes or content
    source_key = "rgb" if "rgb" in SERVICE_ROLE else "ir"
    if parsed.get("detections"):
        if any("ir" in str(d.get("class_id", "")) for d in parsed["detections"]):
            source_key = "ir"

    # Determine source from the content detection patterns
    content_type = request.headers.get("content-type", "")
    source_key_from_url = "unknown"
    referer = request.headers.get("referer", "")
    # Auto-instrumentation propagates trace context; we identify source by
    # checking which fields are present. As both detectors send via forward(),
    # we use buffer count to determine ordering.
    async with _buffer_lock:
        if trace_id not in fusion_buffers:
            fusion_buffers[trace_id] = {}
            fusion_events[trace_id] = asyncio.Event()
            fusion_contexts[trace_id] = []
            fusion_timestamps[trace_id] = time.monotonic()
            source_key = "rgb"
        else:
            source_key = "ir"

        fusion_buffers[trace_id][source_key] = parsed
        if current_span_ctx and current_span_ctx.is_valid:
            fusion_contexts[trace_id].append(current_span_ctx)

        if len(fusion_buffers[trace_id]) >= 2:
            fusion_events[trace_id].set()

    # Wait for both branches
    try:
        await asyncio.wait_for(fusion_events[trace_id].wait(), timeout=FUSION_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Fusion timeout for trace_id=%s (got %d sources)",
                       trace_id[:16], len(fusion_buffers.get(trace_id, {})))

    async with _buffer_lock:
        fused = fusion_buffers.pop(trace_id, {})
        links_ctx = fusion_contexts.pop(trace_id, [])
        fusion_events.pop(trace_id, None)
        fusion_timestamps.pop(trace_id, None)

    if len(fused) < 2:
        return JSONResponse(content={"role": SERVICE_ROLE, "error": "incomplete_fusion"})

    # Build SpanLinks
    links = [Link(ctx) for ctx in links_ctx if ctx.is_valid]

    loop = asyncio.get_running_loop()

    # --- INTERNAL span: pure fusion computation (with SpanLinks) ---
    with tracer.start_as_current_span("compute:feature_fusion", kind=SpanKind.INTERNAL, links=links) as ispan:
        ispan.set_attribute("fusion.sources", str(list(fused.keys())))
        fusion_result, payload = await loop.run_in_executor(cpu_executor, _fuse_data, fused)
        ispan.set_attribute("fusion.total_detections", fusion_result["total_detections"])
        ispan.set_attribute("output_size_bytes", len(payload))

    # --- Forward OUTSIDE INTERNAL span ---
    downstream = parse_downstream()
    tasks = [forward(url + "/process", payload, "application/json") for url in downstream]
    await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "result": {"sources": fusion_result["sources"],
                    "total_detections": fusion_result["total_detections"]},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
