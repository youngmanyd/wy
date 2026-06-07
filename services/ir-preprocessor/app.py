"""IR Pre-processor — OpenCV resize + normalize for infrared images.

Fixes applied:
- #1: X-Original-Image passthrough from upstream
- #8: Truncation warning logging when image data is clipped
- #10: High-precision timestamps via time.time_ns()
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY,
    now_us, parse_downstream, print_env, forward,
    create_app, SpanKind, logger,
)

import os
TARGET_PREPROC_PAYLOAD: int = int(os.environ.get("TARGET_PREPROC_PAYLOAD", str(200 * 1024)))


def _preprocess_image(img_bytes: bytes) -> bytes:
    """Real OpenCV resize + normalize. Output ~TARGET_PREPROC_PAYLOAD bytes."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("[WARNING] Failed to decode input image (size=%d bytes), using synthetic fallback", len(img_bytes))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_uint8 = (img_chw * 255).clip(0, 255).astype(np.uint8)
    tensor_bytes = img_uint8.tobytes()
    usable = TARGET_PREPROC_PAYLOAD - 4
    if len(tensor_bytes) > usable:
        logger.warning("[WARNING] Preprocessor data truncation: tensor %d bytes > target %d bytes, clipping", len(tensor_bytes), usable)
        tensor_bytes = tensor_bytes[:usable]
    header = len(tensor_bytes).to_bytes(4, "big")
    payload = header + tensor_bytes
    if len(payload) < TARGET_PREPROC_PAYLOAD:
        payload += b"\x00" * (TARGET_PREPROC_PAYLOAD - len(payload))
    return payload


@asynccontextmanager
async def lifespan(application):
    print_env()
    logger.info("  TARGET_PREPROC_PAYLOAD=%s", TARGET_PREPROC_PAYLOAD)
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

    with tracer.start_as_current_span("preprocess", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time_ns()
        processed = await loop.run_in_executor(cpu_executor, _preprocess_image, body)
        compute_end_ns = time.time_ns()
        compute_us = (compute_end_ns - compute_start) / 1000.0
        compute_time = f"{compute_us:.0f}"
        timing_entry = f"{SERVICE_ROLE}|{arrival_time}|{compute_time}"
        new_chain = f"{x_timing_chain},{timing_entry}" if x_timing_chain else timing_entry

        downstream = parse_downstream()
        headers = {
            "X-Start-Time": x_start_time,
            "X-Request-ID": x_request_id,
            "X-Source": SERVICE_ROLE,
            "X-Timing-Chain": new_chain,
            "X-Original-Image": x_original_image,
        }
        tasks = [forward(url + "/process", processed, headers) for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)

    total_end = time.time_ns()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe((compute_end_ns - compute_start) / 1e9)
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe((total_end / 1e9) - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"preprocessed_size": len(processed), "forwarded": len(downstream)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
