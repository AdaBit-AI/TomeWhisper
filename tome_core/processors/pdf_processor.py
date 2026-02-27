"""
PDF processing utilities using olmocr.
"""

import base64
from typing import Optional, Union
from PIL import Image

try:
    from olmocr.data.renderpdf import render_pdf_to_base64png
    OLMOC_PDF_AVAILABLE = True
except ImportError:
    OLMOC_PDF_AVAILABLE = False
    render_pdf_to_base64png = None

from ..utils.image_utils import base64_to_PILimage, PILimage_to_base64


class PDFProcessor:
    """PDF processing utilities using olmocr functions."""
    
    @staticmethod
    def render_pdf_page_to_base64(pdf_path: str, page_number: int, 
                                 target_longest_image_dim: int = 2048) -> str:
        """
        Render a PDF page to base64 encoded PNG image.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to render (1-indexed)
            target_longest_image_dim: Target dimension for the longest side of the image
            
        Returns:
            Base64 encoded PNG image string
            
        Raises:
            ImportError: If olmocr is not available
            ValueError: If PDF processing fails
        """
        if not OLMOC_PDF_AVAILABLE:
            raise ImportError("olmocr is not available. Please install olmocr to use PDF processing features.")
        
        if render_pdf_to_base64png is None:
            raise ImportError("render_pdf_to_base64png function is not available from olmocr.")
        
        try:
            return render_pdf_to_base64png(pdf_path, page_number, target_longest_image_dim)
        except Exception as e:
            raise ValueError(f"Failed to render PDF page {page_number}: {str(e)}")
    
    @staticmethod
    def render_pdf_page_to_image(pdf_path: str, page_number: int, 
                                target_longest_image_dim: int = 2048) -> Image.Image:
        """
        Render a PDF page to PIL Image.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to render (1-indexed)
            target_longest_image_dim: Target dimension for the longest side of the image
            
        Returns:
            PIL Image object
            
        Raises:
            ImportError: If olmocr is not available
            ValueError: If PDF processing fails
        """
        base64_image = PDFProcessor.render_pdf_page_to_base64(
            pdf_path, page_number, target_longest_image_dim
        )
        return base64_to_PILimage(base64_image)
    
    @staticmethod
    def get_pdf_page_count(pdf_path: str) -> int:
        """
        Get the number of pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages in the PDF
            
        Raises:
            ImportError: If olmocr is not available
            ValueError: If PDF processing fails
        """
        if not OLMOC_PDF_AVAILABLE:
            raise ImportError("olmocr is not available. Please install olmocr to use PDF processing features.")
        
        # For now, we'll use a simple approach by trying to render the first few pages
        # In a real implementation, you might want to use a proper PDF library
        page_count = 0
        max_attempts = 1000  # Reasonable upper limit
        
        for page_num in range(1, max_attempts + 1):
            try:
                PDFProcessor.render_pdf_page_to_base64(pdf_path, page_num, 512)  # Small preview size
                page_count = page_num
            except Exception:
                break
        
        return page_count
    
    @staticmethod
    def render_pdf_to_images(pdf_path: str, target_longest_image_dim: int = 2048) -> list:
        """
        Render all pages of a PDF to PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            target_longest_image_dim: Target dimension for the longest side of the images
            
        Returns:
            List of PIL Image objects
            
        Raises:
            ImportError: If olmocr is not available
            ValueError: If PDF processing fails
        """
        page_count = PDFProcessor.get_pdf_page_count(pdf_path)
        images = []
        
        for page_num in range(1, page_count + 1):
            try:
                image = PDFProcessor.render_pdf_page_to_image(
                    pdf_path, page_num, target_longest_image_dim
                )
                images.append(image)
            except Exception as e:
                raise ValueError(f"Failed to render page {page_num}: {str(e)}")
        
        return images
    
    @staticmethod
    def is_olmocr_available() -> bool:
        """
        Check if olmocr PDF processing is available.
        
        Returns:
            True if olmocr is available, False otherwise
        """
        return OLMOC_PDF_AVAILABLE