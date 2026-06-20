"""
Processor modules for tome_core.
"""

from .image_processor import ImageProcessor
from .pdf_processor import PDFProcessor
from .figure_extractor import FigureExtractor

__all__ = ["ImageProcessor", "PDFProcessor", "FigureExtractor"]