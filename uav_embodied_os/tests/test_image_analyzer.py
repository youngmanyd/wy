"""Tests for ImageAnalyzer - OpenCV image quality evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from uav_eios.image_analyzer import ImageAnalyzer, ImageQualityResult


class TestImageAnalyzer:
    """Test image quality evaluation."""

    def setup_method(self) -> None:
        self.analyzer = ImageAnalyzer()

    def test_empty_image(self) -> None:
        result = self.analyzer.evaluate(np.array([]))
        assert result.overall_quality == 0.0
        assert not result.is_acceptable
        assert result.details.get("error") == "empty_image"

    def test_none_image(self) -> None:
        result = self.analyzer.evaluate(None)
        assert result.overall_quality == 0.0
        assert not result.is_acceptable

    def test_sharp_well_exposed_image(self) -> None:
        """A sharp image with good exposure and contrast should score high."""
        rng = np.random.default_rng(42)
        image = rng.integers(60, 200, size=(480, 640, 3), dtype=np.uint8)
        result = self.analyzer.evaluate(image)

        assert result.blur_score > 0.5
        assert result.exposure_score > 0.5
        assert result.contrast_score > 0.5
        assert result.overall_quality > 0.5

    def test_blurry_image(self) -> None:
        """A uniform gray image has zero Laplacian variance -> low blur score."""
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = self.analyzer.evaluate(image)

        assert result.blur_score < 0.1
        assert result.laplacian_variance < 1.0
        assert not result.is_acceptable

    def test_dark_image(self) -> None:
        """Very dark image should have low exposure score."""
        image = np.full((480, 640, 3), 10, dtype=np.uint8)
        result = self.analyzer.evaluate(image)

        assert result.exposure_score < 0.3
        assert result.mean_brightness < 20.0

    def test_bright_image(self) -> None:
        """Very bright/overexposed image should have low exposure score."""
        image = np.full((480, 640, 3), 250, dtype=np.uint8)
        result = self.analyzer.evaluate(image)

        assert result.exposure_score < 0.5
        assert result.mean_brightness > 240.0

    def test_low_contrast_image(self) -> None:
        """Image with very low contrast should score low on contrast."""
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
        noise = np.random.default_rng(0).integers(-2, 3, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = self.analyzer.evaluate(image)

        assert result.contrast_score < 0.3
        assert result.std_brightness < 10.0

    def test_grayscale_image(self) -> None:
        """Analyzer should handle grayscale (2D) images."""
        rng = np.random.default_rng(42)
        gray = rng.integers(40, 210, size=(480, 640), dtype=np.uint8)
        result = self.analyzer.evaluate(gray)

        assert result.overall_quality > 0.0
        assert isinstance(result, ImageQualityResult)

    def test_quality_threshold(self) -> None:
        """Custom quality threshold should affect is_acceptable."""
        strict_analyzer = ImageAnalyzer(quality_threshold=0.95)
        rng = np.random.default_rng(42)
        image = rng.integers(60, 200, size=(480, 640, 3), dtype=np.uint8)
        result = strict_analyzer.evaluate(image)

        if result.overall_quality < 0.95:
            assert not result.is_acceptable

    def test_result_fields(self) -> None:
        """Verify all expected fields are present in result."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        result = self.analyzer.evaluate(image)

        assert isinstance(result.blur_score, float)
        assert isinstance(result.exposure_score, float)
        assert isinstance(result.contrast_score, float)
        assert isinstance(result.overall_quality, float)
        assert isinstance(result.laplacian_variance, float)
        assert isinstance(result.mean_brightness, float)
        assert isinstance(result.std_brightness, float)
        assert isinstance(result.is_acceptable, bool)
        assert isinstance(result.details, dict)

    def test_scores_bounded_0_1(self) -> None:
        """All scores should be in [0, 1]."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            image = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
            result = self.analyzer.evaluate(image)

            assert 0.0 <= result.blur_score <= 1.0
            assert 0.0 <= result.exposure_score <= 1.0
            assert 0.0 <= result.contrast_score <= 1.0
            assert 0.0 <= result.overall_quality <= 1.0

    def test_weighted_overall_score(self) -> None:
        """Overall = 0.5*blur + 0.25*exposure + 0.25*contrast."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        result = self.analyzer.evaluate(image)

        expected = (
            0.50 * result.blur_score
            + 0.25 * result.exposure_score
            + 0.25 * result.contrast_score
        )
        assert abs(result.overall_quality - expected) < 1e-6

    def test_save_image(self, tmp_path) -> None:
        """Test saving image to disk."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        path = self.analyzer.save_image(image, tmp_path / "test.jpg")
        assert path.exists()
        assert path.stat().st_size > 0
