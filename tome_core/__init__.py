"""
tome_core: Core functionality for TomeWhisper OCR processing.

Subpackages:
    models      — OCR model interfaces (VLLM, Transformers)
    processors  — PDF, image, and figure extraction processors
    serving     — FastAPI server + model manager
    pipeline    — Batch OCR pipeline with optional figure extraction
    utils       — Image utilities, prompts, config
"""

from .config import config
from .models import BaseOCRModel, TransformersOCRModel
try:
    from .models import VLLMOCRModel
except ImportError:
    VLLMOCRModel = None  # type: ignore
from .processors import ImageProcessor, PDFProcessor, FigureExtractor
from .utils import image_utils, prompt_utils

__version__ = "0.3.0"
__all__ = [
    "BaseOCRModel",
    "VLLMOCRModel",
    "TransformersOCRModel",
    "ImageProcessor",
    "PDFProcessor",
    "FigureExtractor",
    "config",
    "image_utils",
    "prompt_utils",
]
