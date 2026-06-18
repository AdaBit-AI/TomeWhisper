"""
Integration tests for the remote OCR server.

Requires: running remote server (see remote_server.py).
Skip with: pytest -m "not remote"
"""

import io

import pytest
import requests
from PIL import Image

from tome_core.processors import PDFProcessor, ImageProcessor
from tome_core.utils.prompt_utils import get_prompt_by_mode


pytestmark = pytest.mark.remote


@pytest.mark.usefixtures("remote_healthy")
class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_200(self, remote_server_url):
        resp = requests.get(f"{remote_server_url}/health", timeout=10)
        assert resp.status_code == 200

    def test_json_response(self, remote_server_url):
        resp = requests.get(f"{remote_server_url}/health", timeout=10)
        data = resp.json()
        assert data["status"] == "healthy"

    def test_model_loaded(self, remote_server_url):
        resp = requests.get(f"{remote_server_url}/health", timeout=10)
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["device"] == "cuda"


@pytest.mark.usefixtures("remote_healthy")
class TestRootEndpoint:
    """Tests for GET /."""

    def test_returns_service_info(self, remote_server_url):
        resp = requests.get(f"{remote_server_url}/", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "endpoints" in data


@pytest.mark.usefixtures("remote_healthy")
class TestOCREndpoint:
    """Tests for POST /ocr."""

    @pytest.fixture
    def png_bytes(self):
        """A minimal PNG image as bytes."""
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    def test_ocr_returns_200(self, remote_server_url, png_bytes):
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("test.png", png_bytes, "image/png")},
            data={"max_new_tokens": 128},
            timeout=120,
        )
        assert resp.status_code == 200

    def test_ocr_response_has_text(self, remote_server_url, png_bytes):
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("test.png", png_bytes, "image/png")},
            data={"max_new_tokens": 128},
            timeout=120,
        )
        data = resp.json()
        assert data["status"] == "success"
        assert "text" in data
        assert isinstance(data["text"], str)

    def test_ocr_response_has_elapsed(self, remote_server_url, png_bytes):
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("test.png", png_bytes, "image/png")},
            data={"max_new_tokens": 128},
            timeout=120,
        )
        data = resp.json()
        assert "elapsed_ms" in data
        assert data["elapsed_ms"] > 0

    def test_ocr_with_custom_prompt(self, remote_server_url, png_bytes):
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("test.png", png_bytes, "image/png")},
            data={
                "prompt": "What color is this image? Reply in one word.",
                "max_new_tokens": 64,
            },
            timeout=120,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text"]) > 0

    def test_ocr_default_tokens_limit(self, remote_server_url, png_bytes):
        """With a low max_new_tokens, the output should be short."""
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("test.png", png_bytes, "image/png")},
            data={"max_new_tokens": 16},
            timeout=120,
        )
        data = resp.json()
        assert isinstance(data["text"], str)

    def test_ocr_no_image_returns_error(self, remote_server_url):
        resp = requests.post(
            f"{remote_server_url}/ocr",
            data={"max_new_tokens": 32},
            timeout=30,
        )
        assert resp.status_code == 422


@pytest.mark.usefixtures("remote_healthy")
class TestRemoteOCRWithPDFPage:
    """End-to-end: render a PDF page locally, OCR it remotely."""

    def test_ocr_on_pdf_page(self, remote_server_url, sample_pdf_path):
        """Render page 1 of the sample PDF → remote OCR."""
        img = PDFProcessor.render_pdf_page_to_image(
            str(sample_pdf_path), page_number=1, target_longest_image_dim=1024,
        )
        proc = ImageProcessor(max_dimension=1024)
        processed = proc.process_image(img)

        buf = io.BytesIO()
        processed.save(buf, format="PNG")
        buf.seek(0)

        prompt = get_prompt_by_mode("prompt_no_anchoring_v4_yaml")
        resp = requests.post(
            f"{remote_server_url}/ocr",
            files={"image": ("page1.png", buf, "image/png")},
            data={"prompt": prompt, "max_new_tokens": 512},
            timeout=180,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["text"]) > 100
        assert "olmOCR" in data["text"] or "PDF" in data["text"]
