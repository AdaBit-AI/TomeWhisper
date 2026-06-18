"""
Unit tests for tome_core.utils.image_utils.

Covers: PILimage_to_base64, base64_to_PILimage, resize_image,
        validate_image_format, convert_to_rgb, get_image_info.
"""

import base64
import io

import pytest
from PIL import Image

from tome_core.utils.image_utils import (
    PILimage_to_base64,
    base64_to_PILimage,
    resize_image,
    validate_image_format,
    convert_to_rgb,
    get_image_info,
)


class TestPILimageToBase64:
    """Tests for PILimage_to_base64."""

    def test_returns_string(self, test_image):
        result = PILimage_to_base64(test_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_base64(self, test_image):
        result = PILimage_to_base64(test_image)
        # Should decode without error
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_roundtrip(self, test_image):
        """base64 encode → decode → same image dimensions."""
        b64 = PILimage_to_base64(test_image)
        restored = base64_to_PILimage(b64)
        assert restored.size == test_image.size

    def test_jpeg_format(self, test_image):
        result = PILimage_to_base64(test_image, format="JPEG")
        assert isinstance(result, str)

    def test_rgba_image(self, test_image_rgba):
        b64 = PILimage_to_base64(test_image_rgba)
        restored = base64_to_PILimage(b64)
        assert restored.size == test_image_rgba.size


class TestBase64ToPILimage:
    """Tests for base64_to_PILimage."""

    def test_returns_pil_image(self, test_image):
        b64 = PILimage_to_base64(test_image)
        restored = base64_to_PILimage(b64)
        assert isinstance(restored, Image.Image)

    def test_preserves_size(self, test_image):
        b64 = PILimage_to_base64(test_image)
        restored = base64_to_PILimage(b64)
        assert restored.size == test_image.size

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            base64_to_PILimage("not-valid-base64!!!")


class TestResizeImage:
    """Tests for resize_image."""

    def test_no_resize_when_small(self, test_image):
        """Small images should not be resized."""
        resized = resize_image(test_image, max_dimension=1024)
        assert resized.size == test_image.size

    def test_downscale_wide(self):
        """Wide images scale to max_dimension on width."""
        img = Image.new("RGB", (3000, 1000))
        resized = resize_image(img, max_dimension=1024)
        assert resized.size == (1024, 341)  # 1024 * 1000/3000 ≈ 341

    def test_downscale_tall(self):
        """Tall images scale to max_dimension on height."""
        img = Image.new("RGB", (1000, 3000))
        resized = resize_image(img, max_dimension=1024)
        assert resized.size == (341, 1024)

    def test_aspect_ratio_preserved(self, test_image_large):
        """Aspect ratio must be maintained."""
        resized = resize_image(test_image_large, max_dimension=1024)
        orig_ratio = test_image_large.size[0] / test_image_large.size[1]
        new_ratio = resized.size[0] / resized.size[1]
        assert abs(orig_ratio - new_ratio) < 0.02


class TestValidateImageFormat:
    """Tests for validate_image_format."""

    def test_rgb_valid(self, test_image):
        assert validate_image_format(test_image) is True

    def test_rgba_valid(self, test_image_rgba):
        assert validate_image_format(test_image_rgba) is True

    def test_grayscale_valid(self, test_image_grayscale):
        assert validate_image_format(test_image_grayscale) is True

    def test_cmyk_invalid(self):
        img = Image.new("CMYK", (100, 100))
        assert validate_image_format(img) is False


class TestConvertToRGB:
    """Tests for convert_to_rgb."""

    def test_rgb_unchanged(self, test_image):
        result = convert_to_rgb(test_image)
        assert result.mode == "RGB"
        assert result == test_image  # same object when already RGB

    def test_rgba_to_rgb(self, test_image_rgba):
        result = convert_to_rgb(test_image_rgba)
        assert result.mode == "RGB"

    def test_grayscale_to_rgb(self, test_image_grayscale):
        result = convert_to_rgb(test_image_grayscale)
        assert result.mode == "RGB"


class TestGetImageInfo:
    """Tests for get_image_info."""

    def test_returns_dict(self, test_image):
        info = get_image_info(test_image)
        assert isinstance(info, dict)

    def test_has_expected_keys(self, test_image):
        info = get_image_info(test_image)
        for key in ("size", "mode", "format", "info"):
            assert key in info

    def test_size_correct(self, test_image):
        info = get_image_info(test_image)
        assert info["size"] == (100, 100)

    def test_mode_correct(self, test_image_rgba):
        info = get_image_info(test_image_rgba)
        assert info["mode"] == "RGBA"
