"""
End-to-end pipeline test: PDF → image processing → OCR (remote or local).

This is the slowest test — marks: slow, remote, requires_olmocr.
Run selectively: pytest tests/test_pipeline.py -v
"""

import base64
import io
import time

import pytest
import requests
from PIL import Image

from tome_core.processors import PDFProcessor, ImageProcessor
from tome_core.utils.prompt_utils import get_prompt_by_mode


pytestmark = [pytest.mark.slow, pytest.mark.remote, pytest.mark.requires_olmocr]


class TestLocalPDFPipeline:
    """PDF → image processing pipeline (no model needed)."""

    def test_render_all_pages(self, sample_pdf_path):
        """All pages in the sample PDF should render without error."""
        images = PDFProcessor.render_pdf_to_images(
            str(sample_pdf_path), target_longest_image_dim=1024,
        )
        assert len(images) == 33
        for i, img in enumerate(images, 1):
            assert isinstance(img, Image.Image)
            assert max(img.size) <= 1024, f"Page {i} too large: {img.size}"

    def test_process_all_rendered_pages(self, sample_pdf_path):
        """All rendered pages should pass through ImageProcessor."""
        proc = ImageProcessor(max_dimension=1024)
        images = PDFProcessor.render_pdf_to_images(
            str(sample_pdf_path), target_longest_image_dim=2048,
        )
        for i, img in enumerate(images, 1):
            info = proc.validate_and_get_info(img)
            assert info["valid"], f"Page {i} invalid"
            processed = proc.process_image(img)
            assert max(processed.size) <= 1024, f"Page {i} not resized"

    def test_produces_valid_base64(self, sample_pdf_path):
        """Rendered pages encode to valid base64 PNG."""
        proc = ImageProcessor(max_dimension=512)
        img = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=1, target_longest_image_dim=512,
        )
        processed = proc.process_image(img)
        b64 = proc.process_image_to_base64(processed)

        assert isinstance(b64, str)
        assert len(b64) > 1000
        decoded = base64.b64decode(b64)
        assert len(decoded) > 500


class TestRemoteFullPipeline:
    """Full pipeline: PDF → process → remote OCR → receive text."""

    def test_full_pipeline_page1(self, remote_server_url, sample_pdf_path):
        """Complete end-to-end: PDF page 1 through remote OCR."""
        t0 = time.time()

        # 1. Render & process
        img = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=1, target_longest_image_dim=1024,
        )
        proc = ImageProcessor(max_dimension=1024)
        processed = proc.process_image(img)

        # 2. Encode
        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        buf.seek(0)
        setup_time = time.time() - t0

        # 3. Remote OCR
        prompt = get_prompt_by_mode("prompt_no_anchoring_v4_yaml")
        t_ocr = time.time()
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("page1.png", buf, "image/png")},
            data={"prompt": prompt, "max_new_tokens": 1024},
            timeout=300,
        )
        ocr_time = time.time() - t_ocr
        total_time = time.time() - t0

        # 4. Validate
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        text = data["text"]
        assert len(text) > 200, f"OCR produced too little text: {len(text)} chars"
        assert "olmOCR" in text or "language model" in text.lower()

        # Timing sanity
        assert setup_time < 30, f"Setup too slow: {setup_time:.1f}s"
        assert ocr_time < 120, f"OCR too slow: {ocr_time:.1f}s"

        print(f"\n  Pipeline timing: setup={setup_time:.1f}s, "
              f"ocr={ocr_time:.1f}s, total={total_time:.1f}s")
        print(f"  Text extracted: {len(text)} chars")

    def test_full_pipeline_page2(self, remote_server_url, sample_pdf_path):
        """Complete end-to-end: PDF page 2 through remote OCR."""
        proc = ImageProcessor(max_dimension=1024)
        img = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=2, target_longest_image_dim=1024,
        )
        processed = proc.process_image(img)

        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        buf.seek(0)

        prompt = get_prompt_by_mode("prompt_no_anchoring_v4_yaml")
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("page2.png", buf, "image/png")},
            data={"prompt": prompt, "max_new_tokens": 1024},
            timeout=300,
        )

        assert resp.status_code == 200
        data = resp.json()
        text = data["text"]
        assert len(text) > 200
        assert any(kw in text.lower() for kw in ["introduction", "figure 1", "benchmark"])


class TestPipelineEdgeCases:
    """Edge cases in the pipeline."""

    def test_empty_image_fails_gracefully(self, remote_server_url):
        """An all-white image should still get processed (not crash)."""
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("white.png", buf, "image/png")},
            data={"max_new_tokens": 64},
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
