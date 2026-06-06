"""Decision Maker — fusion node merging object-tracker and situation-awareness."""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, FUSION_TIMEOUT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, parse_downstream, print_env, forward,
    create_app, SpanKind, Link, logger, trace, extract,
)

fusion_buffers: dict = {}
fusion_events: dict = {}
fusion_contexts: dict = {}


async def _wait_for_fusion(request_id: str, source: str, data: dict, otel_ctx):
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


@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    yield
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

    body = await request.body()
    incoming_ctx = extract(dict(request.headers))

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {"raw_size": len(body), "source": x_source}
    data["_arrival"] = arrival_time
    data["_chain"] = x_timing_chain
    data["_source"] = x_source

    fused = await _wait_for_fusion(x_request_id, x_source, data, incoming_ctx)
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
        timing_entry = f"{SERVICE_ROLE}|{arrival_time}|{compute_time}"
        new_chain = f"{merged_chain},{timing_entry}" if merged_chain else timing_entry

        downstream = parse_downstream()
        if not downstream:
            compute_end = time.time()
            COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
            return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                         "result": {"fused": fusion_result, "chain": new_chain}})
        fused_payload = json.dumps(fusion_result).encode()
        headers = {
            "X-Start-Time": x_start_time, "X-Request-ID": x_request_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
        }
        tasks = [forward(url + "/process", fused_payload, headers, "application/json") for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)

    compute_end = time.time()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"fused": fusion_result, "forwarded": len(downstream)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
