"""
Model interfaces and implementations for OCR processing.
"""

from .base import BaseOCRModel
from .vllm_model import VLLMOCRModel
from .transformers_model import TransformersOCRModel

__all__ = ["BaseOCRModel", "VLLMOCRModel", "TransformersOCRModel"]