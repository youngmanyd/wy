"""
Shared utilities for all UAV Edge microservices.
OTel-centric architecture: auto-instrumented FastAPI (SERVER) / HTTPX (CLIENT)
with manual INTERNAL spans for pure computation isolation.

All local timing variables and X-* custom headers are eliminated.
Performance is diagnosed solely via OpenTelemetry Spans.
"""

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Link, SpanKind
from opentelemetry.propagate import extract
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------
SERVICE_ROLE: str = os.environ.get("SERVICE_ROLE", "unknown")
SERVICE_PORT: int = int(os.environ.get("SERVICE_PORT", "8000"))
DOWNSTREAM_URLS: str = os.environ.get("DOWNSTREAM_URLS", "")
JAEGER_ENDPOINT: str = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")
JAEGER_QUERY_ENDPOINT: str = os.environ.get("JAEGER_QUERY_ENDPOINT", "http://jaeger:16686")
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
COMPUTE_LATENCY = Histogram("ms_compute_latency_seconds", "Compute latency", ["service_role"])
PAYLOAD_BYTES = Gauge(
    "uav_network_payload_bytes",
    "Network payload size in bytes",
    ["source", "destination"],
)

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
# HTTPX auto-instrumentation with payload size hooks
# ---------------------------------------------------------------------------
def _httpx_request_hook(span, request):
    """Capture outgoing payload size on auto-created CLIENT spans."""
    try:
        size = len(request.content)
        span.set_attribute("messaging.payload_size_bytes", size)
    except Exception:
        pass


def _httpx_response_hook(span, request, response):
    """Response hook placeholder."""
    pass


HTTPXClientInstrumentor().instrument(
    request_hook=_httpx_request_hook,
    response_hook=_httpx_response_hook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_downstream() -> list[str]:
    """Parse comma-separated DOWNSTREAM_URLS into list."""
    if not DOWNSTREAM_URLS.strip():
        return []
    return [u.strip() for u in DOWNSTREAM_URLS.split(",") if u.strip()]


def _extract_service_from_url(url: str) -> str:
    """Extract service name from URL for Prometheus labeling."""
    try:
        host = url.split("//")[1].split(":")[0].split("/")[0]
        return host
    except Exception:
        return "unknown"


def print_env():
    """Print all loaded environment variables at service startup."""
    logger.info("=" * 60)
    logger.info("Service Self-Check: %s", SERVICE_ROLE)
    logger.info("  SERVICE_PORT=%s", SERVICE_PORT)
    logger.info("  DOWNSTREAM_URLS=%s", DOWNSTREAM_URLS)
    logger.info("  OTEL_EXPORTER_OTLP_ENDPOINT=%s", JAEGER_ENDPOINT)
    logger.info("  JAEGER_QUERY_ENDPOINT=%s", JAEGER_QUERY_ENDPOINT)
    logger.info("  OMP_NUM_THREADS=%s", OMP_NUM_THREADS)
    logger.info("  FUSION_TIMEOUT=%s", FUSION_TIMEOUT)
    logger.info("  HTTP_TIMEOUT=%s", HTTP_TIMEOUT)
    logger.info("  FUSION_BUFFER_TTL=%s", FUSION_BUFFER_TTL)
    logger.info("=" * 60)


def get_current_trace_id() -> str:
    """Get trace-id from the current span context (hex, 32-char)."""
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx and span_ctx.is_valid:
        return format(span_ctx.trace_id, "032x")
    return ""


# ---------------------------------------------------------------------------
# Async HTTP forward (auto-instrumented by HTTPXClientInstrumentor)
# ---------------------------------------------------------------------------
async def forward(url: str, payload: bytes,
                  content_type: str = "application/octet-stream") -> Optional[dict]:
    """Forward payload to downstream service.
    CLIENT span is auto-created by HTTPXInstrumentor (with traceparent injection).
    No custom headers — only Content-Type is set explicitly.
    """
    headers = {"Content-Type": content_type}

    # Record payload size in Prometheus
    dest = _extract_service_from_url(url)
    PAYLOAD_BYTES.labels(source=SERVICE_ROLE, destination=dest).set(len(payload))

    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        ) as client:
            resp = await client.post(url, content=payload, headers=headers)
            return resp.json() if resp.status_code == 200 else {"error": resp.status_code}
    except Exception as e:
        logger.error("Forward to %s failed: %s", url, e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------
def create_app(lifespan_func=None) -> FastAPI:
    """Create FastAPI app with auto-instrumented SERVER spans."""
    application = FastAPI(title=f"UAV-Edge-{SERVICE_ROLE}", lifespan=lifespan_func)
    FastAPIInstrumentor.instrument_app(application)

    @application.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/health")
    async def health():
        return {"status": "ok", "role": SERVICE_ROLE}

    return application
