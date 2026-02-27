"""
tome_core: Core functionality for TomeWhisper OCR processing.

This package contains the core OCR processing logic, model interfaces,
and utility functions that are independent of the deployment framework.
"""

from .models import BaseOCRModel, VLLMOCRModel, TransformersOCRModel
from .processors import ImageProcessor, PDFProcessor
from .utils import image_utils, prompt_utils

__version__ = "0.1.0"
__all__ = [
    "BaseOCRModel",
    "VLLMOCRModel", 
    "TransformersOCRModel",
    "ImageProcessor",
    "PDFProcessor",
    "image_utils",
    "prompt_utils"
]