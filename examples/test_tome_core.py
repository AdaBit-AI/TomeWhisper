#!/usr/bin/env python3
"""
Example script to test tome_core functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import tome_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from tome_core.models import TransformersOCRModel
from tome_core.processors import ImageProcessor, PDFProcessor
from tome_core.utils.prompt_utils import list_available_prompt_modes


def test_transformers_ocr():
    """Test the Transformers OCR model."""
    print("Testing Transformers OCR Model...")
    
    try:
        # Initialize model (this will download if not cached)
        print("Initializing model...")
        model = TransformersOCRModel()
        print("Model initialized successfully!")
        
        # List available prompt modes
        print("\nAvailable prompt modes:")
        for mode in list_available_prompt_modes():
            print(f"  - {mode}")
        
        print("\nTransformers OCR test completed successfully!")
        
    except Exception as e:
        print(f"Transformers OCR test failed: {e}")
        return False
    
    return True


def test_image_processor():
    """Test the image processor."""
    print("\nTesting Image Processor...")
    
    try:
        # Create image processor
        processor = ImageProcessor(max_dimension=1024)
        print("Image processor created successfully!")
        
        # Test with a simple image (create a dummy image for testing)
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # Process the image
        info = processor.validate_and_get_info(test_image)
        print(f"Image validation info: {info}")
        
        processed_image = processor.process_image(test_image)
        print(f"Processed image size: {processed_image.size}")
        
        # Convert to base64
        base64_string = processor.process_image_to_base64(test_image)
        print(f"Base64 string length: {len(base64_string)}")
        
        print("\nImage processor test completed successfully!")
        
    except Exception as e:
        print(f"Image processor test failed: {e}")
        return False
    
    return True


def test_pdf_processor():
    """Test the PDF processor."""
    print("\nTesting PDF Processor...")
    
    try:
        from tome_core.processors import PDFProcessor
        
        # Check if olmocr is available
        if not PDFProcessor.is_olmocr_available():
            print("olmocr is not available, skipping PDF processor test")
            return True
        
        print("olmocr is available!")
        
        # Test with the sample PDF if it exists
        pdf_path = Path(__file__).parent.parent / "paper.pdf"
        if pdf_path.exists():
            print(f"Testing with PDF: {pdf_path}")
            
            # Render first page
            base64_image = PDFProcessor.render_pdf_page_to_base64(str(pdf_path), 1)
            print(f"First page rendered to base64, length: {len(base64_image)}")
            
            # Get page count
            page_count = PDFProcessor.get_pdf_page_count(str(pdf_path))
            print(f"PDF has {page_count} pages")
            
        else:
            print("No test PDF found, skipping PDF rendering test")
        
        print("\nPDF processor test completed successfully!")
        
    except Exception as e:
        print(f"PDF processor test failed: {e}")
        return False
    
    return True


async def main():
    """Main test function."""
    print("=== Testing TomeWhisper Core Functionality ===\n")
    
    tests = [
        ("Transformers OCR", test_transformers_ocr),
        ("Image Processor", test_image_processor),
        ("PDF Processor", test_pdf_processor),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
            print(f"✓ {test_name} test {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            results[test_name] = False
            print(f"✗ {test_name} test FAILED with exception: {e}\n")
    
    # Summary
    print("=== Test Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)