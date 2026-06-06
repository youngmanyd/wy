"""IR Detector — YOLOv8 ONNX inference on infrared image tensors."""

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
    REQUEST_COUNT, COMPUTE_LATENCY, REQUEST_LATENCY, OMP_NUM_THREADS,
    now_us, parse_downstream, print_env, forward,
    create_app, SpanKind, logger,
)

import os
ONNX_MODEL_PATH: str = os.environ.get("ONNX_MODEL_PATH", "/app/models/yolov8n.onnx")
TARGET_DETECT_PAYLOAD: int = int(os.environ.get("TARGET_DETECT_PAYLOAD", str(800 * 1024)))

onnx_session = None


def _load_onnx_session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = OMP_NUM_THREADS
    opts.inter_op_num_threads = OMP_NUM_THREADS
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    logger.info("Loading ONNX model from %s (threads=%d)", ONNX_MODEL_PATH, OMP_NUM_THREADS)
    session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    logger.info("ONNX model loaded. Input: %s", session.get_inputs()[0].shape)
    return session


def _run_yolo_detection(tensor_bytes: bytes) -> tuple[list[dict], bytes]:
    """Real YOLOv8 ONNX inference. Output ~TARGET_DETECT_PAYLOAD bytes."""
    global onnx_session
    if len(tensor_bytes) >= 4:
        tensor_len = int.from_bytes(tensor_bytes[:4], "big")
        raw_uint8 = tensor_bytes[4:4 + tensor_len]
    else:
        raw_uint8 = tensor_bytes

    total_elements = 3 * 640 * 640
    if len(raw_uint8) >= total_elements:
        img_uint8 = np.frombuffer(raw_uint8[:total_elements], dtype=np.uint8).reshape(3, 640, 640)
    else:
        padded = np.zeros(total_elements, dtype=np.uint8)
        padded[:len(raw_uint8)] = np.frombuffer(raw_uint8, dtype=np.uint8)
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

    det_json = json.dumps(detections).encode()
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

    with tracer.start_as_current_span("yolo-detect", kind=SpanKind.SERVER) as span:
        span.set_attribute("service.role", SERVICE_ROLE)
        compute_start = time.time()
        detections, payload = await loop.run_in_executor(cpu_executor, _run_yolo_detection, body)
        compute_time = f"{(time.time() - compute_start) * 1e6:.0f}"
        span.set_attribute("detections.count", len(detections))
        timing_entry = f"{SERVICE_ROLE}|{arrival_time}|{compute_time}"
        new_chain = f"{x_timing_chain},{timing_entry}" if x_timing_chain else timing_entry

        downstream = parse_downstream()
        headers = {
            "X-Start-Time": x_start_time, "X-Request-ID": x_request_id,
            "X-Source": SERVICE_ROLE, "X-Timing-Chain": new_chain,
            "X-Detections": json.dumps(detections),
        }
        tasks = [forward(url + "/process", payload, headers) for url in downstream]
        await asyncio.gather(*tasks, return_exceptions=True)

    compute_end = time.time()
    COMPUTE_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(arrival_time))
    REQUEST_LATENCY.labels(service_role=SERVICE_ROLE).observe(compute_end - float(x_start_time))
    return JSONResponse(content={"role": SERVICE_ROLE, "request_id": x_request_id,
                                 "result": {"detections": len(detections), "payload_size": len(payload)}})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT, log_level="info")
