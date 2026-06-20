"""
Model serving: FastAPI server + model manager for OCR inference.
"""

from .model_manager import ModelManager
from .server import create_app

__all__ = ["ModelManager", "create_app"]
