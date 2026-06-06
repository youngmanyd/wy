"""Object Tracker — matrix computation for tracking objects."""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, parse_downstream, print_env, forward,
    create_app, SpanKind, logger,
)


def _matrix_computation() -> dict:
    """Real matrix math (~20-50ms) for tracker."""
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

    with tracer.start_as_current_span("matrix-compute", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time()
        result = await loop.run_in_executor(cpu_executor, _matrix_computation)
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{arrival_time}|{compute_time}"
        new_chain = f"{x_timing_chain},{timing_entry}" if x_timing_chain else timing_entry

        downstream = parse_downstream()
        payload = json.dumps(result).encode()
        headers = {
            "X-Start-Time": x_start_time, "X-Request-ID": x_request_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
        }
        tasks = [forward(url + "/process", payload, headers, "application/json") for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)

    compute_end = time.time()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"compute_result": result, "forwarded": len(downstream)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
