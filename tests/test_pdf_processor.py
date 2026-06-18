"""
Integration tests for tome_core.processors.pdf_processor.PDFProcessor.

Requires: olmocr + poppler (pdfinfo, pdftoppm).
Skip with: pytest -m "not requires_olmocr"
"""

import pytest
from PIL import Image

from tome_core.processors import PDFProcessor


pytestmark = pytest.mark.requires_olmocr


class TestOlmocrAvailability:
    """Tests for olmocr availability detection."""

    def test_is_olmocr_available_returns_bool(self):
        assert isinstance(PDFProcessor.is_olmocr_available(), bool)


class TestRenderPDFPage:
    """Tests for PDF page rendering — requires sample PDF + olmocr."""

    def test_render_page_to_base64(self, sample_pdf_path):
        """Render page 1 to base64."""
        result = PDFProcessor.render_pdf_page_to_base64(
            str(sample_pdf_path), page_number=1, target_longest_image_dim=1024
        )
        assert isinstance(result, str)
        assert len(result) > 100

    def test_render_page_to_image(self, sample_pdf_path):
        """Render page 1 to PIL Image."""
        image = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=1, target_longest_image_dim=1024
        )
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert max(image.size) <= 1024

    def test_different_pages_produce_different_images(self, sample_pdf_path):
        """Pages 1 and 2 should be visually different."""
        img1 = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=1, target_longest_image_dim=512
        )
        img2 = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=2, target_longest_image_dim=512
        )
        # At the very least dimensions or pixel content should differ
        import numpy as np
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert not np.array_equal(arr1, arr2) or img1.size != img2.size

    def test_render_page_invalid_page(self, sample_pdf_path):
        """Rendering a non-existent page should raise ValueError."""
        with pytest.raises(ValueError, match="Failed to render PDF page"):
            PDFProcessor.render_pdf_page_to_base64(
                str(sample_pdf_path), page_number=99999, target_longest_image_dim=512
            )

    def test_render_page_invalid_pdf(self):
        """Rendering a nonexistent PDF should raise."""
        with pytest.raises(ValueError, match="Failed to render PDF page"):
            PDFProcessor.render_pdf_page_to_base64(
                "/nonexistent/file.pdf", page_number=1
            )

    def test_target_dimension_respected(self, sample_pdf_path):
        """The output image should respect target_longest_image_dim."""
        for dim in [256, 512, 1024]:
            img = PDFProcessor.render_pdf_page_to_image(
                str(sample_pdf_path), page_number=1, target_longest_image_dim=dim
            )
            assert max(img.size) <= dim, f"Expected max dim <= {dim}, got {img.size}"


class TestGetPageCount:
    """Tests for PDF page count."""

    def test_positive_page_count(self, sample_pdf_path):
        count = PDFProcessor.get_pdf_page_count(str(sample_pdf_path))
        assert isinstance(count, int)
        assert count > 0

    def test_known_page_count(self, sample_pdf_path):
        """paper.pdf is known to have 33 pages."""
        count = PDFProcessor.get_pdf_page_count(str(sample_pdf_path))
        assert count == 33


class TestRenderAllPages:
    """Tests for render_pdf_to_images."""

    def test_renders_all_pages(self, sample_pdf_path):
        images = PDFProcessor.render_pdf_to_images(
            str(sample_pdf_path), target_longest_image_dim=512
        )
        page_count = PDFProcessor.get_pdf_page_count(str(sample_pdf_path))
        assert len(images) == page_count
        for img in images:
            assert isinstance(img, Image.Image)

    def test_respects_dimension(self, sample_pdf_path):
        images = PDFProcessor.render_pdf_to_images(
            str(sample_pdf_path), target_longest_image_dim=256
        )
        for img in images:
            assert max(img.size) <= 256


class TestPDFProcessorEdgeCases:
    """Edge case handling."""

    def test_olmocr_not_available_mocked(self, monkeypatch):
        """When olmocr is not importable, methods should raise ImportError."""
        # Simulate olmocr not being available
        import tome_core.processors.pdf_processor as pmod
        monkeypatch.setattr(pmod, "OLMOC_PDF_AVAILABLE", False)

        with pytest.raises(ImportError, match="olmocr is not available"):
            PDFProcessor.render_pdf_page_to_base64("dummy.pdf", 1)

        with pytest.raises(ImportError, match="olmocr is not available"):
            PDFProcessor.get_pdf_page_count("dummy.pdf")
