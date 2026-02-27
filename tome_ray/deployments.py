"""
Ray Serve deployments for OCR models.
"""

import asyncio
from typing import Optional
from fastapi import UploadFile, File, Form, HTTPException
from PIL import Image
import io

from ray import serve

from tome_core.models import VLLMOCRModel, TransformersOCRModel
from tome_core.utils.prompt_utils import get_prompt_by_mode
from tome_core.processors.image_processor import ImageProcessor


@serve.deployment(
    num_replicas=1, 
    max_ongoing_requests=3, 
    max_queued_requests=20
)
class VLLMOCRDeployment:
    """Ray Serve deployment for VLLM OCR model."""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.95):
        """
        Initialize VLLM OCR deployment.
        
        Args:
            model_path: Path to the VLLM model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model = VLLMOCRModel(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization
        )
        self.image_processor = ImageProcessor()
    
    async def __call__(self, image: UploadFile = File(...), prompt_mode: str = Form("prompt_layout_all_en")):
        """
        Process OCR request.
        
        Args:
            image: Uploaded image file
            prompt_mode: Prompt mode for OCR
            
        Returns:
            OCR result
            
        Raises:
            HTTPException: If processing fails
        """
        try:
            # Read image data
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Process image
            processed_image = self.image_processor.process_image(pil_image)
            
            # Get prompt
            prompt = get_prompt_by_mode(prompt_mode)
            
            # Generate OCR text
            result = await self.model.generate_async(processed_image, prompt)
            
            return {"response": result}
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@serve.deployment(
    num_replicas=1, 
    max_ongoing_requests=5, 
    max_queued_requests=50
)
class TransformersOCRDeployment:
    """Ray Serve deployment for Transformers OCR model."""
    
    def __init__(self, model_path: str = "allenai/olmOCR-2-7B-1025", 
                 processor_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize Transformers OCR deployment.
        
        Args:
            model_path: Path to the Transformers model
            processor_path: Path to the processor model
        """
        self.model = TransformersOCRModel(
            model_path=model_path,
            processor_path=processor_path
        )
        self.image_processor = ImageProcessor()
    
    async def __call__(self, image: UploadFile = File(...), prompt_mode: str = Form("prompt_no_anchoring_v4_yaml")):
        """
        Process OCR request.
        
        Args:
            image: Uploaded image file
            prompt_mode: Prompt mode for OCR
            
        Returns:
            OCR result
            
        Raises:
            HTTPException: If processing fails
        """
        try:
            # Read image data
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Process image
            processed_image = self.image_processor.process_image(pil_image)
            
            # Get prompt
            prompt = get_prompt_by_mode(prompt_mode)
            
            # Generate OCR text
            result = await self.model.generate_async(processed_image, prompt)
            
            return {"response": result}
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


def create_vllm_deployment(model_path: str, tensor_parallel_size: int = 1, 
                           gpu_memory_utilization: float = 0.95):
    """
    Create a VLLM OCR deployment.
    
    Args:
        model_path: Path to the VLLM model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio
        
    Returns:
        Ray Serve deployment
    """
    return VLLMOCRDeployment.bind(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )


def create_transformers_deployment(model_path: str = "allenai/olmOCR-2-7B-1025", 
                                   processor_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Create a Transformers OCR deployment.
    
    Args:
        model_path: Path to the Transformers model
        processor_path: Path to the processor model
        
    Returns:
        Ray Serve deployment
    """
    return TransformersOCRDeployment.bind(
        model_path=model_path,
        processor_path=processor_path
    )