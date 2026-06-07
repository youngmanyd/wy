"""
Shared utilities for all UAV Edge microservices.
Provides OTel tracing, Prometheus metrics, FastAPI app factory, and HTTP forwarding.
"""

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from opentelemetry import trace, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Link, SpanKind
from opentelemetry.propagate import extract, inject
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------
SERVICE_ROLE: str = os.environ.get("SERVICE_ROLE", "unknown")
SERVICE_PORT: int = int(os.environ.get("SERVICE_PORT", "8000"))
DOWNSTREAM_URLS: str = os.environ.get("DOWNSTREAM_URLS", "")
JAEGER_ENDPOINT: str = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
OMP_NUM_THREADS: int = int(os.environ.get("OMP_NUM_THREADS", "1"))
FUSION_TIMEOUT: float = float(os.environ.get("FUSION_TIMEOUT", "5.0"))
HTTP_TIMEOUT: float = float(os.environ.get("HTTP_TIMEOUT", "10.0"))
FUSION_BUFFER_TTL: float = float(os.environ.get("FUSION_BUFFER_TTL", "10.0"))

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OMP_NUM_THREADS)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{SERVICE_ROLE}] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(SERVICE_ROLE)

# ---------------------------------------------------------------------------
# Thread pool for CPU-bound work (OpenCV / ONNX / matrix)
# ---------------------------------------------------------------------------
cpu_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{SERVICE_ROLE}-cpu")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("ms_request_total", "Total requests processed", ["service_role"])
REQUEST_LATENCY = Histogram("ms_request_latency_seconds", "Request latency", ["service_role"])
COMPUTE_LATENCY = Histogram("ms_compute_latency_seconds", "Compute latency", ["service_role"])

# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------
resource = Resource.create({"service.name": SERVICE_ROLE})
provider = TracerProvider(resource=resource)
try:
    otlp_exporter = OTLPSpanExporter(endpoint=JAEGER_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
except Exception as e:
    logger.warning("Failed to init OTLP exporter: %s", e)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(SERVICE_ROLE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_ns() -> str:
    """High-precision timestamp in seconds with nanosecond resolution (Issue #10)."""
    t = time.time_ns()
    sec = t // 1_000_000_000
    nsec = t % 1_000_000_000
    return f"{sec}.{nsec:09d}"


def now_us() -> str:
    """Alias kept for backward compatibility — delegates to now_ns()."""
    return now_ns()


def parse_downstream() -> list[str]:
    if not DOWNSTREAM_URLS.strip():
        return []
    return [u.strip() for u in DOWNSTREAM_URLS.split(",") if u.strip()]


def print_env():
    logger.info("=" * 60)
    logger.info("Service Self-Check: %s", SERVICE_ROLE)
    logger.info("  SERVICE_PORT=%s", SERVICE_PORT)
    logger.info("  DOWNSTREAM_URLS=%s", DOWNSTREAM_URLS)
    logger.info("  OTEL_EXPORTER_OTLP_ENDPOINT=%s", JAEGER_ENDPOINT)
    logger.info("  OMP_NUM_THREADS=%s", OMP_NUM_THREADS)
    logger.info("  FUSION_TIMEOUT=%s", FUSION_TIMEOUT)
    logger.info("  HTTP_TIMEOUT=%s", HTTP_TIMEOUT)
    logger.info("  FUSION_BUFFER_TTL=%s", FUSION_BUFFER_TTL)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Async HTTP forward with large-header support
# ---------------------------------------------------------------------------
async def forward(url: str, payload: bytes, headers: dict,
                  content_type: str = "application/octet-stream") -> Optional[dict]:
    fwd_headers = {k: v for k, v in headers.items() if v}
    fwd_headers["Content-Type"] = content_type
    inject(fwd_headers)
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:
            resp = await client.post(url, content=payload, headers=fwd_headers)
            return resp.json() if resp.status_code == 200 else {"error": resp.status_code}
    except Exception as e:
        logger.error("Forward to %s failed: %s", url, e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------
def create_app(lifespan_func=None) -> FastAPI:
    application = FastAPI(title=f"UAV-Edge-{SERVICE_ROLE}", lifespan=lifespan_func)
    FastAPIInstrumentor.instrument_app(application)

    @application.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/health")
    async def health():
        return {"status": "ok", "role": SERVICE_ROLE}

    return application
