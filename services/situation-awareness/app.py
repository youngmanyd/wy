"""Situation Awareness — threat assessment and risk evaluation.

Consumes fused detection JSON from feature-fusion, performs threat
evaluation (matrix operations), forwards results to decision-maker.

OTel architecture:
- SERVER span: auto (FastAPIInstrumentor)
- INTERNAL span: wraps awareness computation
- CLIENT span: auto (HTTPXInstrumentor)
- Forward calls OUTSIDE the INTERNAL span
"""

import asyncio
import json
import math
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


def _awareness_computation(data: dict) -> dict:
    """Threat assessment + risk evaluation (CPU-bound matrix operations)."""
    all_dets = data.get("rgb_detections", []) + data.get("ir_detections", [])
    n = max(len(all_dets), 4)

    # Distance matrix
    positions = np.zeros((n, 2), dtype=np.float32)
    for i, det in enumerate(all_dets):
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            positions[i, 0] = (bbox[0] + bbox[2]) / 2
            positions[i, 1] = (bbox[1] + bbox[3]) / 2

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Threat scoring
    threat_scores = np.zeros(n, dtype=np.float32)
    for i, det in enumerate(all_dets):
        score = det.get("score", 0.5)
        area = 1.0
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        proximity = np.sum(1.0 / (dist_matrix[i] + 1.0))
        threat_scores[i] = score * math.log1p(area) * proximity

    # Risk level
    mean_threat = float(np.mean(threat_scores[:len(all_dets)])) if all_dets else 0.0
    if mean_threat > 5.0:
        risk_level = "HIGH"
    elif mean_threat > 2.0:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    assessment = {
        "risk_level": risk_level,
        "mean_threat_score": round(mean_threat, 4),
        "total_objects": len(all_dets),
        "threat_breakdown": [
            {"index": i, "score": round(float(threat_scores[i]), 4)}
            for i in range(min(len(all_dets), 10))
        ],
        "original_image_b64": data.get("original_image_b64", ""),
    }
    return assessment


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

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        data = {"rgb_detections": [], "ir_detections": []}

    loop = asyncio.get_running_loop()

    # --- INTERNAL span: pure awareness computation ---
    with tracer.start_as_current_span("compute:situation_assessment", kind=SpanKind.INTERNAL) as ispan:
        all_dets = data.get("rgb_detections", []) + data.get("ir_detections", [])
        ispan.set_attribute("input.detections", len(all_dets))
        result = await loop.run_in_executor(cpu_executor, _awareness_computation, data)
        ispan.set_attribute("risk_level", result["risk_level"])
        ispan.set_attribute("mean_threat_score", result["mean_threat_score"])

    # --- Forward OUTSIDE INTERNAL span ---
    output_data = {
        "risk_level": result["risk_level"],
        "mean_threat_score": result["mean_threat_score"],
        "threat_breakdown": result["threat_breakdown"],
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
        "result": {"risk_level": result["risk_level"],
                    "mean_threat": result["mean_threat_score"]},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
