"""
Image processing utilities for OCR.
"""

import base64
import io
from typing import Union
from PIL import Image


def PILimage_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format for encoding (default: PNG)
        
    Returns:
        Base64 encoded image string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_to_PILimage(base64_string: str) -> Image.Image:
    """
    Convert a base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def resize_image(image: Image.Image, max_dimension: int = 2048) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_dimension: Maximum dimension for the longest side
        
    Returns:
        Resized PIL Image object
    """
    width, height = image.size
    
    if max(width, height) <= max_dimension:
        return image
    
    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def validate_image_format(image: Image.Image) -> bool:
    """
    Validate if the image format is supported.
    
    Args:
        image: PIL Image object
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_formats = {'RGB', 'RGBA', 'L', 'P'}
    return image.mode in supported_formats


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB format if needed.
    
    Args:
        image: PIL Image object
        
    Returns:
        RGB PIL Image object
    """
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def get_image_info(image: Image.Image) -> dict:
    """
    Get basic information about the image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'info': image.info
    }