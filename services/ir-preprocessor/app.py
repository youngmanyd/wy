"""IR Pre-processor — OpenCV resize + normalize for infrared images.

Identical logic to RGB preprocessor; separated for per-service image builds.

OTel architecture:
- SERVER span: auto (FastAPIInstrumentor)
- INTERNAL span: wraps OpenCV resize + normalize + base64 encode
- CLIENT span: auto (HTTPXInstrumentor)
- Forward calls OUTSIDE the INTERNAL span
"""

import asyncio
import base64
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer,
    REQUEST_COUNT, PAYLOAD_BYTES,
    parse_downstream, print_env, forward,
    create_app, SpanKind, logger, trace,
)

TARGET_PREPROC_PAYLOAD: int = int(os.environ.get("TARGET_PREPROC_PAYLOAD", str(1228804)))


def _preprocess_image(img_bytes: bytes) -> bytes:
    """Real OpenCV resize + normalize + carry original image b64 in body."""
    orig_b64 = base64.b64encode(img_bytes)

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Failed to decode input image (%d bytes), using synthetic", len(img_bytes))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_uint8 = (img_chw * 255).clip(0, 255).astype(np.uint8)
    tensor_bytes = img_uint8.tobytes()

    tensor_header = len(tensor_bytes).to_bytes(4, "big")
    orig_header = len(orig_b64).to_bytes(4, "big")
    payload = tensor_header + tensor_bytes + orig_header + orig_b64

    if len(payload) < TARGET_PREPROC_PAYLOAD:
        payload += b"\x00" * (TARGET_PREPROC_PAYLOAD - len(payload))
    elif len(payload) > TARGET_PREPROC_PAYLOAD:
        logger.warning("Preprocessor payload %d > target %d (not truncating)",
                        len(payload), TARGET_PREPROC_PAYLOAD)

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
    REQUEST_COUNT.labels(service_role=SERVICE_ROLE).inc()
    body = await request.body()

    server_span = trace.get_current_span()
    if server_span and server_span.is_recording():
        server_span.set_attribute("messaging.payload_size_bytes", len(body))

    loop = asyncio.get_running_loop()

    with tracer.start_as_current_span("compute:preprocess_resize_normalize", kind=SpanKind.INTERNAL) as ispan:
        ispan.set_attribute("input_size_bytes", len(body))
        processed = await loop.run_in_executor(cpu_executor, _preprocess_image, body)
        ispan.set_attribute("output_size_bytes", len(processed))

    downstream = parse_downstream()
    tasks = [forward(url + "/process", processed) for url in downstream]
    await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "result": {"preprocessed_size": len(processed), "forwarded": len(downstream)},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
