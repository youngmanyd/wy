"""Image Analyzer - evaluates captured image quality using OpenCV.

Provides blur detection (Laplacian variance), exposure analysis,
and overall quality scoring for inspection decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ImageQualityResult:
    """Result of image quality evaluation."""

    blur_score: float
    exposure_score: float
    contrast_score: float
    overall_quality: float
    laplacian_variance: float
    mean_brightness: float
    std_brightness: float
    is_acceptable: bool
    details: dict


class ImageAnalyzer:
    """Evaluates image quality for UAV inspection tasks.

    Quality metrics:
      - Blur: Laplacian variance (higher = sharper)
      - Exposure: deviation from ideal mid-range brightness
      - Contrast: standard deviation of pixel intensities
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_contrast: float = 30.0,
        quality_threshold: float = 0.6,
    ) -> None:
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is required for ImageAnalyzer")

        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.quality_threshold = quality_threshold

    def evaluate(self, image: np.ndarray) -> ImageQualityResult:
        """Evaluate image quality and return structured result."""
        if image is None or image.size == 0:
            return ImageQualityResult(
                blur_score=0.0, exposure_score=0.0, contrast_score=0.0,
                overall_quality=0.0, laplacian_variance=0.0,
                mean_brightness=0.0, std_brightness=0.0,
                is_acceptable=False, details={"error": "empty_image"},
            )

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blur_score, lap_var = self._compute_blur_score(gray)
        exposure_score, mean_bright = self._compute_exposure_score(gray)
        contrast_score, std_bright = self._compute_contrast_score(gray)

        overall = 0.50 * blur_score + 0.25 * exposure_score + 0.25 * contrast_score
        is_acceptable = overall >= self.quality_threshold

        return ImageQualityResult(
            blur_score=blur_score,
            exposure_score=exposure_score,
            contrast_score=contrast_score,
            overall_quality=overall,
            laplacian_variance=lap_var,
            mean_brightness=mean_bright,
            std_brightness=std_bright,
            is_acceptable=is_acceptable,
            details={
                "blur_threshold": self.blur_threshold,
                "quality_threshold": self.quality_threshold,
                "image_shape": list(image.shape),
            },
        )

    def _compute_blur_score(self, gray: np.ndarray) -> tuple[float, float]:
        """Blur detection via Laplacian variance. Higher = sharper."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        score = min(1.0, variance / self.blur_threshold)
        return score, variance

    def _compute_exposure_score(self, gray: np.ndarray) -> tuple[float, float]:
        """Exposure quality based on mean brightness deviation from ideal."""
        mean_brightness = float(np.mean(gray))
        ideal = 127.5
        deviation = abs(mean_brightness - ideal) / ideal

        if mean_brightness < self.min_brightness:
            score = mean_brightness / self.min_brightness * 0.3
        elif mean_brightness > self.max_brightness:
            score = (255.0 - mean_brightness) / (255.0 - self.max_brightness) * 0.3
        else:
            score = 1.0 - deviation * 0.5

        return max(0.0, min(1.0, score)), mean_brightness

    def _compute_contrast_score(self, gray: np.ndarray) -> tuple[float, float]:
        """Contrast measured by standard deviation of pixel intensities."""
        std_dev = float(np.std(gray))
        score = min(1.0, std_dev / self.min_contrast) if self.min_contrast > 0 else 1.0
        return score, std_dev

    def save_image(self, image: np.ndarray, filepath: str | Path) -> Path:
        """Save an image to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(filepath), image)
        return filepath
