"""RGB Detector — YOLOv8 ONNX inference on RGB image tensors.

OTel architecture:
- SERVER span: auto (FastAPIInstrumentor)
- INTERNAL span: wraps ONNX inference (pure computation)
- CLIENT span: auto (HTTPXInstrumentor)
- Forward calls OUTSIDE the INTERNAL span

Binary input  (from preprocessor): 4B_tensor_len + tensor + 4B_orig_b64_len + orig_b64 [+ pad]
Binary output (to fusion):         4B_json_len + json + raw_tensor [+ pad]
  where json = {"detections": [...], "original_image_b64": "..."}
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from common import (
    SERVICE_ROLE, SERVICE_PORT, cpu_executor, tracer, OMP_NUM_THREADS,
    REQUEST_COUNT, PAYLOAD_BYTES,
    parse_downstream, print_env, forward,
    create_app, SpanKind, logger, trace,
)

ONNX_MODEL_PATH: str = os.environ.get("ONNX_MODEL_PATH", "/app/models/yolov8n.onnx")
TARGET_DETECT_PAYLOAD: int = int(os.environ.get("TARGET_DETECT_PAYLOAD", str(800 * 1024)))

onnx_session = None
inference_mode = "real"


def _load_onnx_session():
    global inference_mode
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = OMP_NUM_THREADS
        opts.inter_op_num_threads = OMP_NUM_THREADS
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        logger.info("Loading ONNX model from %s (threads=%d)", ONNX_MODEL_PATH, OMP_NUM_THREADS)
        session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
        logger.info("ONNX model loaded. Input: %s", session.get_inputs()[0].shape)
        inference_mode = "real"
        return session
    except Exception as e:
        logger.error("[FALLBACK] Failed to load ONNX model: %s — switching to mock", e)
        inference_mode = "mock"
        return None


def _parse_preprocessor_payload(body: bytes) -> tuple[bytes, str]:
    """Parse preprocessor binary: 4B_tensor_len + tensor + 4B_orig_b64_len + orig_b64."""
    if len(body) < 4:
        return body, ""
    tensor_len = int.from_bytes(body[:4], "big")
    tensor_bytes = body[4:4 + tensor_len]
    offset = 4 + tensor_len
    orig_b64 = ""
    if offset + 4 <= len(body):
        orig_b64_len = int.from_bytes(body[offset:offset + 4], "big")
        if 0 < orig_b64_len <= len(body) - offset - 4:
            try:
                orig_b64 = body[offset + 4:offset + 4 + orig_b64_len].decode("ascii")
            except (UnicodeDecodeError, ValueError):
                pass
    return tensor_bytes, orig_b64


def _mock_detection(orig_b64: str) -> tuple[list[dict], bytes]:
    detections = [
        {"bbox": [100.0, 100.0, 200.0, 200.0], "class_id": 0, "score": 0.85},
        {"bbox": [300.0, 150.0, 450.0, 350.0], "class_id": 1, "score": 0.72},
    ]
    det_dict = {"detections": detections, "original_image_b64": orig_b64}
    det_json = json.dumps(det_dict).encode()
    mock_tensor = np.random.randint(0, 255, (84, 8400), dtype=np.uint8).tobytes()
    header = len(det_json).to_bytes(4, "big")
    payload = header + det_json + mock_tensor
    if len(payload) > TARGET_DETECT_PAYLOAD:
        payload = payload[:TARGET_DETECT_PAYLOAD]
    elif len(payload) < TARGET_DETECT_PAYLOAD:
        payload += b"\x00" * (TARGET_DETECT_PAYLOAD - len(payload))
    return detections, payload


def _run_yolo_detection(body: bytes) -> tuple[list[dict], bytes]:
    """Parse preprocessor payload, run ONNX inference, produce detection output."""
    tensor_bytes, orig_b64 = _parse_preprocessor_payload(body)

    global onnx_session
    if onnx_session is None:
        return _mock_detection(orig_b64)

    total_elements = 3 * 640 * 640
    if len(tensor_bytes) >= total_elements:
        img_uint8 = np.frombuffer(tensor_bytes[:total_elements], dtype=np.uint8).reshape(3, 640, 640)
    else:
        padded = np.zeros(total_elements, dtype=np.uint8)
        padded[:len(tensor_bytes)] = np.frombuffer(tensor_bytes, dtype=np.uint8)
        img_uint8 = padded.reshape(3, 640, 640)
    img_tensor = (img_uint8.astype(np.float32) / 255.0).reshape(1, 3, 640, 640)

    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: img_tensor})

    raw_output = outputs[0]
    detections = []
    if raw_output.ndim == 3 and raw_output.shape[1] >= 5:
        preds = raw_output[0].T
        class_scores = preds[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        top_indices = np.argsort(max_scores)[-10:]
        for idx in top_indices:
            cx, cy, w, h = preds[idx, :4].tolist()
            class_id = int(np.argmax(preds[idx, 4:]))
            score = float(max_scores[idx])
            if score > 0.1:
                detections.append({
                    "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    "class_id": class_id,
                    "score": round(score, 4),
                })

    # Build binary output: 4B_json_len + json (with orig_b64) + raw_tensor
    det_dict = {"detections": detections, "original_image_b64": orig_b64}
    det_json = json.dumps(det_dict).encode()
    output_slice = raw_output[0, :, :].tobytes()
    header = len(det_json).to_bytes(4, "big")
    payload = header + det_json + output_slice
    if len(payload) > TARGET_DETECT_PAYLOAD:
        payload = payload[:TARGET_DETECT_PAYLOAD]
    elif len(payload) < TARGET_DETECT_PAYLOAD:
        payload += b"\x00" * (TARGET_DETECT_PAYLOAD - len(payload))
    return detections, payload


@asynccontextmanager
async def lifespan(application):
    global onnx_session
    print_env()
    logger.info("  ONNX_MODEL_PATH=%s", ONNX_MODEL_PATH)
    logger.info("  TARGET_DETECT_PAYLOAD=%s", TARGET_DETECT_PAYLOAD)
    loop = asyncio.get_running_loop()
    onnx_session = await loop.run_in_executor(cpu_executor, _load_onnx_session)
    logger.info("  inference_mode=%s", inference_mode)
    logger.info("Service %s ready on port %d", SERVICE_ROLE, SERVICE_PORT)
    yield
    cpu_executor.shutdown(wait=False)
    logger.info("Service %s shutting down", SERVICE_ROLE)


app = create_app(lifespan_func=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        content=f"<h1>UAV Edge Service: {SERVICE_ROLE}</h1>"
        f"<p>Inference mode: {inference_mode}</p>"
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

    # --- INTERNAL span: pure ONNX inference ---
    with tracer.start_as_current_span("compute:yolo_inference", kind=SpanKind.INTERNAL) as ispan:
        ispan.set_attribute("inference.mode", inference_mode)
        ispan.set_attribute("input_size_bytes", len(body))
        detections, payload = await loop.run_in_executor(cpu_executor, _run_yolo_detection, body)
        ispan.set_attribute("detections.count", len(detections))
        ispan.set_attribute("output_size_bytes", len(payload))

    # --- Forward OUTSIDE INTERNAL span ---
    downstream = parse_downstream()
    tasks = [forward(url + "/process", payload) for url in downstream]
    await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(content={
        "role": SERVICE_ROLE,
        "result": {"detections": len(detections), "payload_size": len(payload),
                    "inference_mode": inference_mode},
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
