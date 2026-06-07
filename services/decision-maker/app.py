"""Decision Maker — fusion node merging object-tracker and situation-awareness.

Fixes applied:
- #1: X-Original-Image passthrough
- #2: Fix negative network latency by using max(input arrivals) as fusion arrival time
- #3: Parse upstream JSON correctly (tracker_result, awareness_result)
- #4: TTL-based cleanup for fusion_buffers to prevent memory leaks
- #10: High-precision timestamps via time.time_ns()
- #11: asyncio.Lock for all shared buffer access
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, FUSION_TIMEOUT, FUSION_BUFFER_TTL,
    cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, parse_downstream, print_env, forward,
    create_app, SpanKind, Link, logger, trace, extract,
)

fusion_buffers: dict = {}
fusion_events: dict = {}
fusion_contexts: dict = {}
fusion_timestamps: dict = {}
_buffer_lock = asyncio.Lock()


async def _wait_for_fusion(request_id: str, source: str, data: dict, otel_ctx, arrival_time: str):
    """Wait for both branches to arrive. Returns merged data or None on timeout."""
    async with _buffer_lock:
        if request_id not in fusion_buffers:
            fusion_buffers[request_id] = {}
            fusion_events[request_id] = asyncio.Event()
            fusion_contexts[request_id] = []
            fusion_timestamps[request_id] = {"created": time.time(), "arrivals": []}

        fusion_buffers[request_id][source] = data
        fusion_contexts[request_id].append(otel_ctx)
        fusion_timestamps[request_id]["arrivals"].append(float(arrival_time))

        if len(fusion_buffers[request_id]) >= 2:
            fusion_events[request_id].set()

    try:
        await asyncio.wait_for(fusion_events[request_id].wait(), timeout=FUSION_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Fusion timeout for request %s (source=%s)", request_id, source)
        return None, arrival_time

    async with _buffer_lock:
        if request_id in fusion_buffers:
            fused = fusion_buffers.pop(request_id)
            fusion_events.pop(request_id, None)
            ctx_list = fusion_contexts.pop(request_id, [])
            ts_info = fusion_timestamps.pop(request_id, {"arrivals": [float(arrival_time)]})
            max_arrival = max(ts_info["arrivals"])
            max_arrival_str = f"{int(max_arrival)}.{int((max_arrival % 1) * 1e9):09d}"
            return {"fused": fused, "contexts": ctx_list}, max_arrival_str
        return None, arrival_time


async def _cleanup_stale_buffers():
    """Periodically clean up stale fusion buffers older than FUSION_BUFFER_TTL."""
    while True:
        await asyncio.sleep(FUSION_BUFFER_TTL)
        now = time.time()
        async with _buffer_lock:
            stale_keys = [
                k for k, v in fusion_timestamps.items()
                if now - v["created"] > FUSION_BUFFER_TTL
            ]
            for k in stale_keys:
                fusion_buffers.pop(k, None)
                fusion_events.pop(k, None)
                fusion_contexts.pop(k, None)
                fusion_timestamps.pop(k, None)
            if stale_keys:
                logger.info("Cleaned up %d stale fusion buffers", len(stale_keys))


@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    cleanup_task = asyncio.create_task(_cleanup_stale_buffers())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = create_app(lifespan_func=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        content=f"<h1>UAV Edge Service: {SERVICE_ROLE}</h1>"
        f"<p><a href='/health'>Health</a> | <a href='/metrics'>Metrics</a></p>"
    )


@app.post("/process")
async def process(request: Request):
    arrival_time = now_us()
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()

    x_start_time = request.headers.get("X-Start-Time", arrival_time)
    x_request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    x_source = request.headers.get("X-Source", "unknown")
    x_timing_chain = request.headers.get("X-Timing-Chain", "")
    x_original_image = request.headers.get("X-Original-Image", "")
    incoming_ctx = extract(dict(request.headers))

    body = await request.body()

    try:
        upstream_data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        upstream_data = {"raw_size": len(body), "source": x_source}

    data = {
        "upstream": upstream_data,
        "_arrival": arrival_time,
        "_chain": x_timing_chain,
        "_source": x_source,
        "_original_image": upstream_data.get("original_image_b64", "") or x_original_image,
    }

    fused, effective_arrival = await _wait_for_fusion(
        x_request_id, x_source, data, incoming_ctx, arrival_time
    )
    if fused is None:
        return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                     "result": {"status": "waiting_or_timeout"}})

    links = []
    for ctx in fused.get("contexts", []):
        upstream_span = trace.get_current_span(ctx)
        if upstream_span and upstream_span.get_span_context().is_valid:
            links.append(Link(upstream_span.get_span_context()))

    with tracer.start_as_current_span(f"fusion-{SERVICE_ROLE}", kind=SpanKind.SERVER, links=links) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        span.set_attribute("fusion.sources", str(list(fused["fused"].keys())))
        compute_start = time.time_ns()

        chains = []
        original_image_b64 = ""
        tracker_result = {}
        awareness_result = {}
        for src_key, src_data in fused["fused"].items():
            if isinstance(src_data, dict):
                if "_chain" in src_data:
                    chains.append(src_data["_chain"])
                upstream = src_data.get("upstream", {})
                if "tracker_result" in upstream:
                    tracker_result = upstream["tracker_result"]
                if "awareness_result" in upstream:
                    awareness_result = upstream["awareness_result"]
                if src_data.get("_original_image") and not original_image_b64:
                    original_image_b64 = src_data["_original_image"]

        merged_chain = ";".join(chains)
        decision_result = {
            "sources": list(fused["fused"].keys()),
            "n_tracked": tracker_result.get("n_tracked", 0),
            "n_assessed": awareness_result.get("n_assessed", 0),
            "tracked_objects": tracker_result.get("tracked_objects", []),
            "assessments": awareness_result.get("assessments", []),
            "decision": "proceed" if tracker_result.get("n_tracked", 0) > 0 else "hold",
            "original_image_b64": original_image_b64,
        }

        compute_end_ns = time.time_ns()
        compute_us = (compute_end_ns - compute_start) / 1000.0
        compute_time = f"{compute_us:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{effective_arrival}|{compute_time}"
        new_chain = f"{merged_chain},{timing_entry}" if merged_chain else timing_entry

        downstream = parse_downstream()
        if not downstream:
            COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe((compute_end_ns - compute_start) / 1e9)
            return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                         "result": decision_result})

        fused_payload = json.dumps(decision_result).encode()
        headers = {
            "X-Start-Time": x_start_time,
            "X-Request-ID": x_request_id,
            "X-Source": SERVICE_ROLE,
            "X-Timing-Chain": new_chain,
            "X-Original-Image": original_image_b64,
        }
        tasks = [forward(url + "/process", fused_payload, headers, "application/json") for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)

    total_end = time.time_ns()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe((compute_end_ns - compute_start) / 1e9)
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe((total_end / 1e9) - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"decision": decision_result["decision"],
                                            "forwarded": len(downstream)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
