"""Gateway microservice — entry point for the UAV Edge DAG.

Fixes applied:
- #1: X-Original-Image passthrough (base64 of raw input image)
- #5: Measure real compute time (header construction, base64 encoding, dispatch scheduling)
- #10: High-precision timestamps via time.time_ns()
- #11: No shared mutable state in gateway (stateless)
"""

import asyncio
import base64
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, parse_downstream, print_env, forward,
    create_app, SpanKind, logger,
)


def _encode_image_b64(body: bytes) -> str:
    """Encode raw image body to base64 for X-Original-Image passthrough.
    Offloaded to thread pool to avoid blocking event loop (base64 is CPU-bound for large images).
    """
    return base64.b64encode(body).decode("ascii")


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
    x_timing_chain = request.headers.get("X-Timing-Chain", "")

    body = await request.body()
    loop = asyncio.get_running_loop()

    with tracer.start_as_current_span("gateway-dispatch", kind=SpanKind.SERVER):
        compute_start = time.time_ns()

        orig_b64 = await loop.run_in_executor(cpu_executor, _encode_image_b64, body)

        downstream = parse_downstream()
        headers = {
            "X-Start-Time": x_start_time,
            "X-Request-ID": x_request_id,
            "X-Source": "gateway",
            "X-Original-Image": orig_b64,
        }

        compute_end_ns = time.time_ns()
        compute_us = (compute_end_ns - compute_start) / 1000.0
        compute_time = f"{compute_us:.0f}"
        timing_entry = f"gateway|{arrival_time}|{compute_time}"
        new_chain = f"{x_timing_chain},{timing_entry}" if x_timing_chain else timing_entry
        headers["X-Timing-Chain"] = new_chain

        if not downstream:
            return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                         "result": {"status": "no_downstream"}})

        tasks = [forward(url + "/process", body, headers) for url in downstream]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_end = time.time_ns()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe((total_end - compute_start) / 1e9)
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(
        (total_end / 1e9) - float(x_start_time)
    )
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"forwarded_to": downstream, "results_count": len(results)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
