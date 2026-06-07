"""Situation Awareness — detection-driven matrix computation for situational analysis.

Fixes applied:
- #1: X-Original-Image passthrough
- #6: Consume real upstream detection data (bounding boxes as threat matrix input)
- #10: High-precision timestamps via time.time_ns()
"""

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


def _awareness_computation(detections: list[dict]) -> dict:
    """Detection-driven threat assessment (~20-50ms).
    Uses upstream bounding boxes to compute spatial distribution,
    threat priority matrix, and cluster analysis.
    """
    n_dets = max(len(detections), 4)
    size = min(n_dets * 4, 256)

    if detections:
        bbox_matrix = np.zeros((n_dets, 4), dtype=np.float32)
        for i, det in enumerate(detections[:n_dets]):
            bbox = det.get("bbox", [0, 0, 0, 0])
            bbox_matrix[i] = bbox[:4] if len(bbox) >= 4 else [0, 0, 0, 0]
        centers = np.column_stack([
            (bbox_matrix[:, 0] + bbox_matrix[:, 2]) / 2,
            (bbox_matrix[:, 1] + bbox_matrix[:, 3]) / 2,
        ])
        areas = (bbox_matrix[:, 2] - bbox_matrix[:, 0]) * (bbox_matrix[:, 3] - bbox_matrix[:, 1])
        seed_value = float(np.sum(areas))
    else:
        centers = np.zeros((4, 2), dtype=np.float32)
        areas = np.ones(4, dtype=np.float32)
        seed_value = 42.0

    rng = np.random.RandomState(int(abs(seed_value)) % (2**31))

    if len(centers) >= 2:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
    else:
        dist_matrix = np.zeros((1, 1), dtype=np.float32)

    threat_scores = np.abs(areas) / (np.max(np.abs(areas)) + 1e-6)
    threat_matrix = rng.randn(size, size).astype(np.float32)
    threat_matrix = threat_matrix @ threat_matrix.T
    eigenvalues = np.linalg.eigvalsh(threat_matrix[:min(64, size), :min(64, size)])
    svd_u, svd_s, _ = np.linalg.svd(threat_matrix[:min(128, size), :min(128, size)], full_matrices=False)

    assessments = []
    for i, det in enumerate(detections):
        assessments.append({
            "bbox": det.get("bbox", []),
            "class_id": det.get("class_id", -1),
            "threat_score": float(threat_scores[i]) if i < len(threat_scores) else 0.0,
            "awareness_level": "high" if (i < len(threat_scores) and threat_scores[i] > 0.5) else "low",
        })

    return {
        "assessments": assessments,
        "n_assessed": len(assessments),
        "max_threat_eigenvalue": float(np.max(eigenvalues)),
        "top_singular_values": svd_s[:5].tolist(),
        "spatial_spread": float(np.max(dist_matrix)) if dist_matrix.size > 1 else 0.0,
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
    x_original_image = request.headers.get("X-Original-Image", "")

    body = await request.body()
    loop = asyncio.get_running_loop()

    try:
        upstream_data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        upstream_data = {}

    all_detections = upstream_data.get("rgb_detections", []) + upstream_data.get("ir_detections", [])
    original_image_b64 = upstream_data.get("original_image_b64", "") or x_original_image

    with tracer.start_as_current_span("awareness-compute", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        span.set_attribute("input.detections", len(all_detections))
        compute_start = time.time_ns()
        result = await loop.run_in_executor(cpu_executor, _awareness_computation, all_detections)
        compute_end_ns = time.time_ns()
        compute_us = (compute_end_ns - compute_start) / 1000.0
        compute_time = f"{compute_us:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{arrival_time}|{compute_time}"
        new_chain = f"{x_timing_chain},{timing_entry}" if x_timing_chain else timing_entry

        downstream = parse_downstream()
        output_data = {
            "awareness_result": result,
            "original_image_b64": original_image_b64,
        }
        payload = json.dumps(output_data).encode()
        headers = {
            "X-Start-Time": x_start_time,
            "X-Request-ID": x_request_id,
            "X-Source": SERVICE_ROLE,
            "X-Timing-Chain": new_chain,
            "X-Original-Image": original_image_b64,
        }
        tasks = [forward(url + "/process", payload, headers, "application/json") for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)

    total_end = time.time_ns()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe((compute_end_ns - compute_start) / 1e9)
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe((total_end / 1e9) - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"n_assessed": result["n_assessed"], "forwarded": len(downstream)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
