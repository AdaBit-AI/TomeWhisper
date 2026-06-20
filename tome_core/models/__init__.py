"""
Model interfaces and implementations for OCR processing.
"""

from .base import BaseOCRModel
from .transformers_model import TransformersOCRModel

try:
    from .vllm_model import VLLMOCRModel
except ImportError:
    VLLMOCRModel = None  # type: ignore

__all__ = ["BaseOCRModel", "VLLMOCRModel", "TransformersOCRModel"]