"""
Base interface for OCR models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from PIL import Image


class BaseOCRModel(ABC):
    """Abstract base class for OCR models."""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the OCR model.
        
        Args:
            model_path: Path to the model
            **kwargs: Additional model-specific parameters
        """
        self.model_path = model_path
        self.model = None
        self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """Load the model implementation."""
        pass
    
    @abstractmethod
    async def generate_async(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generate OCR text from an image asynchronously.
        
        Args:
            image: PIL Image object
            prompt: Prompt text for OCR
            **kwargs: Additional generation parameters
            
        Returns:
            Generated OCR text
        """
        pass
    
    @abstractmethod
    def generate_sync(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generate OCR text from an image synchronously.
        
        Args:
            image: PIL Image object
            prompt: Prompt text for OCR
            **kwargs: Additional generation parameters
            
        Returns:
            Generated OCR text
        """
        pass
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image before OCR (can be overridden by subclasses).
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        return image
    
    def validate_inputs(self, image: Image.Image, prompt: str) -> None:
        """
        Validate input parameters.
        
        Args:
            image: PIL Image object
            prompt: Prompt text
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image object")
        
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")