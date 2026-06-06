"""Gateway microservice — entry point for the UAV Edge DAG."""

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

    with tracer.start_as_current_span("gateway-dispatch", kind=SpanKind.SERVER):
        compute_start = time.time()
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        timing_entry = f"gateway|{arrival_time}|{compute_time}"
        new_chain = f"{x_timing_chain},{timing_entry}" if x_timing_chain else timing_entry

        downstream = parse_downstream()
        if not downstream:
            return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                         "result": {"status": "no_downstream"}})

        headers = {
            "X-Start-Time": x_start_time, "X-Request-ID": x_request_id,
            "X-Source": "gateway", "X-Timing-Chain": new_chain,
            "X-Original-Image": base64.b64encode(body).decode() if len(body) < 500_000 else "",
        }
        tasks = [forward(url + "/process", body, headers) for url in downstream]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    compute_end = time.time()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"forwarded_to": downstream, "results_count": len(results)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
