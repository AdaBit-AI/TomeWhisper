"""
Batch OCR pipeline: PDF → images → remote OCR → markdown + figures.
"""

from .batch_pipeline import BatchPipeline

__all__ = ["BatchPipeline"]
