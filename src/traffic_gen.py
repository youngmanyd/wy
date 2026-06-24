#!/usr/bin/env python3
"""
UAV Edge Computing Traffic Generator
Reads real images from a folder and sends them at configurable FPS to MS-1 (Gateway).
Injects microsecond-precision X-Start-Time into each request header.

Usage:
    python traffic_gen.py --gateway http://GATEWAY_IP:8001 --images ./test_images --fps 5 --total 100

Environment variables (override CLI args):
    GATEWAY_URL, IMAGE_DIR, FPS, TOTAL_FRAMES, CONCURRENCY
"""

import argparse
import asyncio
import glob
import logging
import os
import sys
import time
import uuid

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TrafficGen] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("TrafficGen")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8001")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "./test_images")
FPS = int(os.environ.get("FPS", "5"))
TOTAL_FRAMES = int(os.environ.get("TOTAL_FRAMES", "100"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "30.0"))


def discover_images(directory: str) -> list[str]:
    """Find all image files in the directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    files.sort()
    if not files:
        logger.warning("No images found in %s — will generate synthetic images", directory)
    return files


def generate_synthetic_image() -> bytes:
    """Generate a synthetic test image (~100KB) when no real images are available."""
    import numpy as np
    import cv2

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add some structure: gradient + shapes
    for i in range(480):
        img[i, :, 0] = int(i / 480 * 255)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 3)
    cv2.circle(img, (400, 200), 80, (0, 0, 255), -1)
    cv2.putText(img, "UAV-TEST", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()


async def send_frame(
    client: httpx.AsyncClient,
    gateway_url: str,
    image_data: bytes,
    frame_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send one frame to the gateway."""
    async with semaphore:
        request_id = str(uuid.uuid4())
        start_time = f"{time.time():.6f}"

        headers = {
            "X-Start-Time": start_time,
            "X-Request-ID": request_id,
            "X-Source": "traffic-gen",
            "Content-Type": "application/octet-stream",
        }

        try:
            t0 = time.time()
            resp = await client.post(
                f"{gateway_url}/process",
                content=image_data,
                headers=headers,
            )
            elapsed = (time.time() - t0) * 1000
            if resp.status_code == 200:
                logger.info(
                    "Frame %04d | req=%s | status=%d | e2e=%.1fms | size=%dB",
                    frame_id, request_id[:8], resp.status_code, elapsed, len(image_data),
                )
                return {"frame": frame_id, "status": "ok", "elapsed_ms": elapsed}
            else:
                logger.warning(
                    "Frame %04d | req=%s | status=%d | body=%s",
                    frame_id, request_id[:8], resp.status_code, resp.text[:200],
                )
                return {"frame": frame_id, "status": "error", "code": resp.status_code}
        except Exception as e:
            logger.error("Frame %04d | req=%s | error=%s", frame_id, request_id[:8], e)
            return {"frame": frame_id, "status": "exception", "error": str(e)}


async def run_traffic(
    gateway_url: str,
    image_dir: str,
    fps: int,
    total_frames: int,
    concurrency: int,
):
    """Main traffic generation loop."""
    logger.info("=" * 60)
    logger.info("UAV Edge Traffic Generator")
    logger.info("  Gateway:    %s", gateway_url)
    logger.info("  Image Dir:  %s", image_dir)
    logger.info("  FPS:        %d", fps)
    logger.info("  Total:      %d frames", total_frames)
    logger.info("  Concurrency: %d", concurrency)
    logger.info("=" * 60)

    # Load images
    image_files = discover_images(image_dir)
    image_cache: list[bytes] = []

    if image_files:
        for f in image_files:
            with open(f, "rb") as fh:
                image_cache.append(fh.read())
        logger.info("Loaded %d real images from %s", len(image_cache), image_dir)
    else:
        logger.info("Generating %d synthetic test images...", min(10, total_frames))
        for _ in range(min(10, total_frames)):
            image_cache.append(generate_synthetic_image())

    semaphore = asyncio.Semaphore(concurrency)
    interval = 1.0 / fps
    results = []

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        tasks = []
        gen_start = time.time()

        for i in range(total_frames):
            img_data = image_cache[i % len(image_cache)]
            task = asyncio.create_task(
                send_frame(client, gateway_url, img_data, i, semaphore)
            )
            tasks.append(task)

            # Rate limiting: wait to maintain target FPS
            elapsed = time.time() - gen_start
            expected = (i + 1) * interval
            if expected > elapsed:
                await asyncio.sleep(expected - elapsed)

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    total_time = time.time() - gen_start
    ok_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "ok")
    err_count = len(results) - ok_count
    avg_latency = 0.0
    if ok_count > 0:
        latencies = [r["elapsed_ms"] for r in results if isinstance(r, dict) and r.get("elapsed_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    logger.info("=" * 60)
    logger.info("Traffic Generation Complete")
    logger.info("  Total time:     %.2fs", total_time)
    logger.info("  Actual FPS:     %.1f", total_frames / total_time if total_time > 0 else 0)
    logger.info("  Success:        %d / %d", ok_count, total_frames)
    logger.info("  Errors:         %d", err_count)
    logger.info("  Avg latency:    %.1fms", avg_latency)
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="UAV Edge Traffic Generator")
    parser.add_argument("--gateway", default=GATEWAY_URL, help="Gateway URL")
    parser.add_argument("--images", default=IMAGE_DIR, help="Image directory")
    parser.add_argument("--fps", type=int, default=FPS, help="Target FPS")
    parser.add_argument("--total", type=int, default=TOTAL_FRAMES, help="Total frames to send")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(run_traffic(args.gateway, args.images, args.fps, args.total, args.concurrency))


if __name__ == "__main__":
    main()
