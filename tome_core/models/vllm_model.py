"""
VLLM-based OCR model implementation.
"""

import asyncio
import base64
import sys
import os
from typing import Any, Dict, Optional
from PIL import Image
import io

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from .base import BaseOCRModel
from ..utils.image_utils import PILimage_to_base64


class VLLMOCRModel(BaseOCRModel):
    """VLLM-based OCR model implementation."""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, 
                 gpu_memory_utilization: float = 0.95, **kwargs):
        """
        Initialize VLLM OCR model.
        
        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio
            **kwargs: Additional VLLM engine arguments
        """
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine = None
        super().__init__(model_path, **kwargs)
    
    def _load_model(self, **kwargs):
        """Load the VLLM engine."""
        # Add the model path to the Python path to allow importing custom code
        sys.path.insert(0, self.model_path)
        
        # Dynamically import the custom modeling code
        try:
            import DotsOCR.modeling_dots_ocr_vllm
        except ImportError:
            print("Could not import custom modeling code. Make sure the model is downloaded and the path is correct.")
            raise
        
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            **kwargs
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    async def generate_async(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generate OCR text from an image asynchronously using VLLM.
        
        Args:
            image: PIL Image object
            prompt: Prompt text for OCR
            **kwargs: Additional generation parameters (temperature, top_p, max_tokens)
            
        Returns:
            Generated OCR text
        """
        self.validate_inputs(image, prompt)
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Convert image to base64
        image_b64 = PILimage_to_base64(processed_image)
        
        # Create prompt with image
        prompt_text = f"<|img|><|imgpad|><|endofimg|>{prompt}"
        
        # Set up sampling parameters
        temperature = kwargs.get('temperature', 0.1)
        top_p = kwargs.get('top_p', 0.9)
        max_tokens = kwargs.get('max_tokens', 32768)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        request_id = random_uuid()
        
        # Generate results
        results_generator = self.engine.generate(
            prompt_text, 
            sampling_params, 
            request_id, 
            images=[image_b64]
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output:
            return final_output.outputs[0].text
        else:
            raise RuntimeError("Failed to get a response from the model")
    
    def generate_sync(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generate OCR text from an image synchronously using VLLM.
        
        Args:
            image: PIL Image object
            prompt: Prompt text for OCR
            **kwargs: Additional generation parameters
            
        Returns:
            Generated OCR text
        """
        # Run the async method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_async(image, prompt, **kwargs))
        finally:
            loop.close()