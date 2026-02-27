"""
tome_ray: Ray Serve deployment for TomeWhisper OCR processing.

This package contains Ray Serve deployments and FastAPI integration
for scalable OCR processing using the tome_core functionality.
"""

from .deployments import VLLMOCRDeployment, TransformersOCRDeployment
from .api import create_ocr_app

__version__ = "0.1.0"
__all__ = [
    "VLLMOCRDeployment",
    "TransformersOCRDeployment", 
    "create_ocr_app"
]