"""Object Tracker — Kalman filter-based multi-object tracking.

Consumes fused detection JSON from feature-fusion, performs tracking
computation (matrix operations), forwards results to decision-maker.

OTel architecture:
- SERVER span: auto (FastAPIInstrumentor)
- INTERNAL span: wraps tracking computation
- CLIENT span: auto (HTTPXInstrumentor)
- Forward calls OUTSIDE the INTERNAL span
"""

import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, PAYLOAD_BYTES,
    parse_downstream, print_env, forward,
    create_app, SpanKind, logger, trace,
)


def _tracking_computation(data: dict) -> dict:
    """Kalman filter + Hungarian matching (CPU-bound matrix operations)."""
    all_dets = data.get("rgb_detections", []) + data.get("ir_detections", [])
    n = max(len(all_dets), 4)

    state_matrix = np.eye(4 * n, dtype=np.float32)
    for i, det in enumerate(all_dets):
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            state_matrix[i * 4:(i + 1) * 4, 0] = bbox[:4]

    # Kalman prediction
    F = np.eye(4 * n, dtype=np.float32)
    for i in range(n):
        if i * 4 + 3 < 4 * n:
            F[i * 4, i * 4 + 1] = 0.1
            F[i * 4 + 2, i * 4 + 3] = 0.1
    predicted = F @ state_matrix

    # Cost matrix for Hungarian-like assignment
    cost_matrix = np.random.rand(n, n).astype(np.float32)
    for _ in range(5):
        cost_matrix = cost_matrix @ cost_matrix.T + np.eye(n, dtype=np.float32) * 0.1

    tracked_objects = [
        {
            "track_id": i,
            "bbox": all_dets[i]["bbox"] if i < len(all_dets) else [0, 0, 0, 0],
            "score": all_dets[i].get("score", 0.0) if i < len(all_dets) else 0.0,
            "state": predicted[i * 4:(i + 1) * 4, 0].tolist() if i * 4 + 4 <= predicted.shape[0] else [0, 0, 0, 0],
        }
        for i in range(min(n, len(all_dets)))
    ]

    return {
        "tracked_objects": tracked_objects,
        "total_tracked": len(tracked_objects),
        "original_image_b64": data.get("original_image_b64", ""),
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
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()
    body = await request.body()

    server_span = trace.get_current_span()
    if server_span and server_span.is_recording():
        server_span.set_attribute("messaging.payload_size_bytes", len(body))

    # Parse JSON from fusion
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        data = {"rgb_detections": [], "ir_detections": []}

    loop = asyncio.get_running_loop()

    # --- INTERNAL span: pure tracking computation ---
    with tracer.start_as_current_span("compute:kalman_tracking", kind=SpanKind.INTERNAL) as ispan:
        all_dets = data.get("rgb_detections", []) + data.get("ir_detections", [])
        ispan.set_attribute("input.detections", len(all_dets))
        result = await loop.run_in_executor(cpu_executor, _tracking_computation, data)
        ispan.set_attribute("tracked_objects", result["total_tracked"])

    # --- Forward OUTSIDE INTERNAL span ---
    output_data = {
        "tracked_objects": result["tracked_objects"],
        "total_tracked": result["total_tracked"],
        "rgb_detections": data.get("rgb_detections", []),
        "ir_detections": data.get("ir_detections", []),
        "original_image_b64": result.get("original_image_b64", ""),
    }
    payload = json.dumps(output_data).encode()

    downstream = parse_downstream()
    tasks = [forward(url + "/process", payload, "application/json") for url in downstream]
    await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "result": {"tracked": result["total_tracked"]},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
