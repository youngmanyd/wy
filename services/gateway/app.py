"""Gateway — entry point for the UAV Edge DAG.

OTel architecture:
- SERVER span: auto-created by FastAPIInstrumentor
- INTERNAL span: wraps body read + image encoding (pure computation)
- CLIENT spans: auto-created by HTTPXInstrumentor for each downstream call
- Forward calls are OUTSIDE the INTERNAL span
"""

import asyncio
import base64
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, PAYLOAD_BYTES,
    parse_downstream, print_env, forward,
    create_app, SpanKind, logger, trace,
)


def _encode_image_b64(body: bytes) -> str:
    """Encode raw image bytes to base64 (CPU-bound, offloaded to executor)."""
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
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()
    body = await request.body()

    # Record incoming payload on auto-created SERVER span
    server_span = trace.get_current_span()
    if server_span and server_span.is_recording():
        server_span.set_attribute("messaging.payload_size_bytes", len(body))

    loop = asyncio.get_running_loop()

    # --- INTERNAL span: pure computation (base64 encoding) ---
    with tracer.start_as_current_span("compute:encode_image_b64", kind=SpanKind.INTERNAL) as ispan:
        ispan.set_attribute("input_size_bytes", len(body))
        orig_b64 = await loop.run_in_executor(cpu_executor, _encode_image_b64, body)
        ispan.set_attribute("b64_length", len(orig_b64))

    # --- Forward to downstream (OUTSIDE INTERNAL span) ---
    # CLIENT spans are auto-created by HTTPXInstrumentor
    downstream = parse_downstream()
    if not downstream:
        return JSONResponse(content={"role": SERVICE_ROLE, "result": "no_downstream"})

    # Gateway sends raw image bytes as body to preprocessors.
    # The orig_b64 is NOT sent in headers (forbidden). Preprocessors will
    # base64-encode the raw bytes themselves and carry it through the body chain.
    tasks = [forward(url + "/process", body) for url in downstream]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "result": {"forwarded_to": downstream, "results_count": len(results)},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
