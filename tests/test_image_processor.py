"""
Unit tests for tome_core.processors.image_processor.ImageProcessor.

Covers: process_image, process_image_to_base64, process_batch,
        validate_and_get_info, process_base64_image.
"""

import pytest
from PIL import Image

from tome_core.processors import ImageProcessor


class TestImageProcessorInit:
    """Tests for ImageProcessor construction."""

    def test_default_params(self):
        proc = ImageProcessor()
        assert proc.max_dimension == 2048
        assert proc.target_format == "RGB"

    def test_custom_params(self):
        proc = ImageProcessor(max_dimension=512, target_format="RGBA")
        assert proc.max_dimension == 512
        assert proc.target_format == "RGBA"


class TestProcessImage:
    """Tests for ImageProcessor.process_image."""

    def test_returns_pil_image(self, image_processor, test_image):
        result = image_processor.process_image(test_image)
        assert isinstance(result, Image.Image)

    def test_rgb_output(self, image_processor, test_image):
        result = image_processor.process_image(test_image)
        assert result.mode == "RGB"

    def test_resizes_large_image(self, image_processor, test_image_large):
        result = image_processor.process_image(test_image_large)
        assert max(result.size) <= image_processor.max_dimension

    def test_preserves_small_image(self, image_processor, test_image):
        result = image_processor.process_image(test_image)
        assert result.size == test_image.size

    def test_converts_rgba_to_rgb(self, image_processor, test_image_rgba):
        result = image_processor.process_image(test_image_rgba)
        assert result.mode == "RGB"

    def test_raises_on_unsupported_format(self, image_processor):
        cmyk = Image.new("CMYK", (100, 100))
        with pytest.raises(ValueError, match="Unsupported image format"):
            image_processor.process_image(cmyk)


class TestProcessImageToBase64:
    """Tests for ImageProcessor.process_image_to_base64."""

    def test_returns_string(self, image_processor, test_image):
        result = image_processor.process_image_to_base64(test_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resizes_first(self, image_processor, test_image_large):
        result = image_processor.process_image_to_base64(test_image_large)
        assert isinstance(result, str)

    def test_format_parameter(self, image_processor, test_image):
        png = image_processor.process_image_to_base64(test_image, format="PNG")
        jpeg = image_processor.process_image_to_base64(test_image, format="JPEG")
        assert png != jpeg  # Different encodings


class TestValidateAndGetInfo:
    """Tests for ImageProcessor.validate_and_get_info."""

    def test_returns_dict(self, image_processor, test_image):
        info = image_processor.validate_and_get_info(test_image)
        assert isinstance(info, dict)

    def test_has_expected_keys(self, image_processor, test_image):
        info = image_processor.validate_and_get_info(test_image)
        for key in ("valid", "info", "needs_resize", "needs_format_conversion"):
            assert key in info

    def test_valid_image(self, image_processor, test_image):
        info = image_processor.validate_and_get_info(test_image)
        assert info["valid"] is True
        assert info["needs_resize"] is False

    def test_needs_resize_large(self, image_processor, test_image_large):
        info = image_processor.validate_and_get_info(test_image_large)
        assert info["needs_resize"] is True

    def test_needs_format_conversion(self, image_processor, test_image_rgba):
        info = image_processor.validate_and_get_info(test_image_rgba)
        assert info["needs_format_conversion"] is True


class TestProcessBatch:
    """Tests for batch processing."""

    def test_returns_list(self, image_processor, test_images_batch):
        results = image_processor.process_batch(test_images_batch)
        assert isinstance(results, list)
        assert len(results) == len(test_images_batch)

    def test_all_are_images(self, image_processor, test_images_batch):
        results = image_processor.process_batch(test_images_batch)
        for r in results:
            assert isinstance(r, Image.Image)

    def test_all_rgb(self, image_processor, test_images_batch):
        results = image_processor.process_batch(test_images_batch)
        for r in results:
            assert r.mode == "RGB"

    def test_empty_batch(self, image_processor):
        results = image_processor.process_batch([])
        assert results == []

    def test_batch_to_base64(self, image_processor, test_images_batch):
        results = image_processor.process_batch_to_base64(test_images_batch)
        assert len(results) == len(test_images_batch)
        for r in results:
            assert isinstance(r, str)


class TestProcessBase64Image:
    """Tests for base64 round-trip processing."""

    def test_roundtrip(self, image_processor, test_image):
        from tome_core.utils.image_utils import PILimage_to_base64
        b64_in = PILimage_to_base64(test_image)
        b64_out = image_processor.process_base64_image(b64_in)
        assert isinstance(b64_out, str)
        assert len(b64_out) > 0
