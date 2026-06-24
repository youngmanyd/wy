"""Decision Maker — second synchronization point (tracker + awareness).

Uses **trace-id** as the correlation key. SpanLinks connect both upstream paths.

OTel architecture:
- SERVER span: auto (FastAPIInstrumentor)
- INTERNAL span: wraps decision logic (with SpanLinks)
- CLIENT span: auto (HTTPXInstrumentor)
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

# Decision synchronization state
fusion_buffers: dict = {}
fusion_events: dict = {}
fusion_contexts: dict = {}
fusion_timestamps: dict = {}

_buffer_lock = asyncio.Lock()


async def _cleanup_stale_buffers():
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
                logger.info("Cleaned %d stale decision buffers", len(stale))


def _make_decision(merged: dict) -> dict:
    """Decision logic (CPU-bound)."""
    tracker_data = merged.get("tracker", {})
    awareness_data = merged.get("awareness", {})

    tracked = tracker_data.get("total_tracked", 0)
    risk_level = awareness_data.get("risk_level", "LOW")
    mean_threat = awareness_data.get("mean_threat_score", 0.0)

    if risk_level == "HIGH" and tracked > 3:
        decision = "EVADE"
    elif risk_level == "MEDIUM" or tracked > 5:
        decision = "ALERT"
    elif tracked > 0:
        decision = "MONITOR"
    else:
        decision = "CLEAR"

    original_image_b64 = (tracker_data.get("original_image_b64", "")
                          or awareness_data.get("original_image_b64", ""))

    return {
        "decision": decision,
        "tracked_objects": tracked,
        "risk_level": risk_level,
        "mean_threat_score": round(mean_threat, 4),
        "rgb_detections": tracker_data.get("rgb_detections", [])
                          or awareness_data.get("rgb_detections", []),
        "ir_detections": tracker_data.get("ir_detections", [])
                         or awareness_data.get("ir_detections", []),
        "original_image_b64": original_image_b64,
    }


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
        f"<p>Decision buffers: {len(fusion_buffers)}</p>"
        f"<p><a href='/health'>Health</a> | <a href='/metrics'>Metrics</a></p>"
    )


@app.post("/process")
async def process(request: Request):
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()
    body = await request.body()

    server_span = trace.get_current_span()
    if server_span and server_span.is_recording():
        server_span.set_attribute("messaging.payload_size_bytes", len(body))

    trace_id = get_current_trace_id()
    if not trace_id:
        return JSONResponse(content={"role": SERVICE_ROLE, "error": "no_trace_id"})

    current_span_ctx = trace.get_current_span().get_span_context()

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        data = {}

    # Determine source: tracker vs awareness
    async with _buffer_lock:
        if trace_id not in fusion_buffers:
            fusion_buffers[trace_id] = {}
            fusion_events[trace_id] = asyncio.Event()
            fusion_contexts[trace_id] = []
            fusion_timestamps[trace_id] = time.monotonic()
            source_key = "tracker"
        else:
            source_key = "awareness"

        fusion_buffers[trace_id][source_key] = data
        if current_span_ctx and current_span_ctx.is_valid:
            fusion_contexts[trace_id].append(current_span_ctx)

        if len(fusion_buffers[trace_id]) >= 2:
            fusion_events[trace_id].set()

    try:
        await asyncio.wait_for(fusion_events[trace_id].wait(), timeout=FUSION_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Decision timeout for trace_id=%s", trace_id[:16])

    async with _buffer_lock:
        merged = fusion_buffers.pop(trace_id, {})
        links_ctx = fusion_contexts.pop(trace_id, [])
        fusion_events.pop(trace_id, None)
        fusion_timestamps.pop(trace_id, None)

    if len(merged) < 2:
        return JSONResponse(content={"role": SERVICE_ROLE, "error": "incomplete_decision"})

    links = [Link(ctx) for ctx in links_ctx if ctx.is_valid]
    loop = asyncio.get_running_loop()

    # --- INTERNAL span: pure decision computation (with SpanLinks) ---
    with tracer.start_as_current_span("compute:decision_logic", kind=SpanKind.INTERNAL, links=links) as ispan:
        result = await loop.run_in_executor(cpu_executor, _make_decision, merged)
        ispan.set_attribute("decision", result["decision"])
        ispan.set_attribute("tracked_objects", result["tracked_objects"])
        ispan.set_attribute("risk_level", result["risk_level"])

    # --- Forward OUTSIDE INTERNAL span ---
    payload = json.dumps(result).encode()
    downstream = parse_downstream()
    tasks = [forward(url + "/process", payload, "application/json") for url in downstream]
    await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "result": {"decision": result["decision"],
                    "tracked": result["tracked_objects"]},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
