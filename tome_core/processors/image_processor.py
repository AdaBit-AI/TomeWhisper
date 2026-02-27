"""
Image processing utilities for OCR.
"""

from typing import Optional, Union, List
from PIL import Image

from ..utils.image_utils import (
    PILimage_to_base64, 
    base64_to_PILimage, 
    resize_image, 
    validate_image_format, 
    convert_to_rgb,
    get_image_info
)


class ImageProcessor:
    """Image processing utilities for OCR workflows."""
    
    def __init__(self, max_dimension: int = 2048, target_format: str = "RGB"):
        """
        Initialize the image processor.
        
        Args:
            max_dimension: Maximum dimension for image resizing
            target_format: Target image format (RGB, RGBA, L, etc.)
        """
        self.max_dimension = max_dimension
        self.target_format = target_format
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """
        Process an image for OCR - validate, resize, and convert format.
        
        Args:
            image: PIL Image object
            
        Returns:
            Processed PIL Image object
            
        Raises:
            ValueError: If image format is not supported
        """
        # Validate image format
        if not validate_image_format(image):
            raise ValueError(f"Unsupported image format: {image.mode}")
        
        # Convert to target format if needed
        if self.target_format and image.mode != self.target_format:
            image = convert_to_rgb(image) if self.target_format == "RGB" else image.convert(self.target_format)
        
        # Resize if needed
        if max(image.size) > self.max_dimension:
            image = resize_image(image, self.max_dimension)
        
        return image
    
    def process_image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Process an image and convert to base64.
        
        Args:
            image: PIL Image object
            format: Image format for encoding
            
        Returns:
            Base64 encoded image string
        """
        processed_image = self.process_image(image)
        return PILimage_to_base64(processed_image, format)
    
    def process_base64_image(self, base64_string: str) -> str:
        """
        Process a base64 encoded image and return processed base64.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Processed base64 encoded image string
        """
        image = base64_to_PILimage(base64_string)
        processed_image = self.process_image(image)
        return PILimage_to_base64(processed_image)
    
    def validate_and_get_info(self, image: Image.Image) -> dict:
        """
        Validate an image and get its information.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with validation result and image information
        """
        is_valid = validate_image_format(image)
        info = get_image_info(image)
        
        return {
            "valid": is_valid,
            "info": info,
            "needs_resize": max(image.size) > self.max_dimension,
            "needs_format_conversion": image.mode != self.target_format
        }
    
    def process_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of processed PIL Image objects
        """
        return [self.process_image(image) for image in images]
    
    def process_batch_to_base64(self, images: List[Image.Image], format: str = "PNG") -> List[str]:
        """
        Process a batch of images and convert to base64.
        
        Args:
            images: List of PIL Image objects
            format: Image format for encoding
            
        Returns:
            List of base64 encoded image strings
        """
        processed_images = self.process_batch(images)
        return [PILimage_to_base64(image, format) for image in processed_images]