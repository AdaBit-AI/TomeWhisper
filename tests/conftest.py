"""
Shared fixtures and configuration for the TomeWhisper test suite.

Usage:
    pytest tests/                          # run all tests
    pytest tests/ -m "not remote"          # skip tests needing remote server
    pytest tests/ -m "not requires_olmocr" # skip tests needing olmocr/poppler
    pytest tests/ -v -k "image"            # run image-related tests only

API address: read from server_config.yaml, override with env vars.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

# Ensure the repo root is on sys.path so tome_core imports work
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── Remote server config ────────────────────────────────────────────
# 1. env vars take highest priority
# 2. then server_config.yaml at repo root
# 3. then hardcoded fallback

def _load_server_config():
    """Load server config from server_config.yaml, falling back to env / defaults."""
    host = os.environ.get("TOMEWHISPER_HOST")
    port = os.environ.get("TOMEWHISPER_PORT")
    password = os.environ.get("TOMEWHISPER_PASSWORD")

    config_path = REPO_ROOT.parent / "server_config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            hostname = cfg.get("hostname", "")
            if not host and hostname:
                # hostname format: "user@host" — extract host
                host = hostname.split("@")[-1] if "@" in hostname else hostname
            if not password:
                password = cfg.get("password")
        except Exception:
            pass

    host = host or "192.168.31.156"
    port = int(port or "8000")
    return host, port, password


REMOTE_HOST, REMOTE_PORT, REMOTE_PASSWORD = _load_server_config()
REMOTE_URL = f"http://{REMOTE_HOST}:{REMOTE_PORT}"


# ── Markers ─────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "remote: test requires a running remote OCR server"
    )
    config.addinivalue_line(
        "markers", "requires_olmocr: test requires olmocr + poppler installed"
    )
    config.addinivalue_line(
        "markers", "slow: test is slow (model inference or full pipeline)"
    )
    config.addinivalue_line(
        "markers", "requires_model: test requires local model + torch (CUDA/MPS)"
    )


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def repo_root():
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_pdf_path(repo_root):
    """Path to the sample PDF included in the repo."""
    path = repo_root / "paper.pdf"
    if not path.exists():
        pytest.skip("sample PDF (paper.pdf) not found in repo root")
    return path


@pytest.fixture(scope="session")
def remote_server_url():
    """Base URL of the remote OCR server (from server_config.yaml or env)."""
    return REMOTE_URL


@pytest.fixture
def test_image():
    """A simple 100x100 white RGB PIL Image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def test_image_rgba():
    """A 50x50 RGBA PIL Image with transparency."""
    return Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))


@pytest.fixture
def test_image_grayscale():
    """A grayscale PIL Image."""
    return Image.new("L", (200, 150), color=128)


@pytest.fixture
def test_image_large():
    """A 4000x3000 RGB image — needs resizing for default ImageProcessor."""
    return Image.new("RGB", (4000, 3000), color="blue")


@pytest.fixture
def test_images_batch():
    """A batch of 3 small test images."""
    return [
        Image.new("RGB", (100, 80), color="red"),
        Image.new("RGB", (200, 150), color="green"),
        Image.new("RGB", (300, 200), color="blue"),
    ]


@pytest.fixture
def image_processor():
    """Default ImageProcessor instance (max 1024px)."""
    from tome_core.processors import ImageProcessor  # noqa: E402
    return ImageProcessor(max_dimension=1024)


@pytest.fixture
def image_processor_fullres():
    """ImageProcessor with no downscaling."""
    from tome_core.processors import ImageProcessor  # noqa: E402
    return ImageProcessor(max_dimension=8192)


@pytest.fixture(scope="session")
def olmocr_available():
    """Check once whether olmocr + poppler are functional."""
    from tome_core.processors import PDFProcessor  # noqa: E402
    return PDFProcessor.is_olmocr_available()


@pytest.fixture(scope="session")
def require_olmocr(olmocr_available):
    """Skip the test if olmocr is not available."""
    if not olmocr_available:
        pytest.skip("olmocr / poppler not available")


@pytest.fixture(scope="session")
def remote_healthy(remote_server_url):
    """Skip the test if the remote OCR server is not reachable."""
    import requests  # noqa: E402
    try:
        resp = requests.get(f"{remote_server_url}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            return True
    except Exception:
        pass
    pytest.skip(f"Remote OCR server not healthy at {remote_server_url}")
