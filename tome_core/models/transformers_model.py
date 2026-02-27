"""
Transformers-based OCR model implementation for OLM-OCR.
"""

import torch
import base64
from io import BytesIO
from typing import Any, Dict, Optional
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .base import BaseOCRModel


class TransformersOCRModel(BaseOCRModel):
    """Transformers-based OCR model implementation for OLM-OCR."""
    
    def __init__(self, model_path: str = "allenai/olmOCR-2-7B-1025", 
                 processor_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 torch_dtype: torch.dtype = torch.bfloat16, **kwargs):
        """
        Initialize Transformers OCR model.
        
        Args:
            model_path: Path to the OLM-OCR model
            processor_path: Path to the processor model
            torch_dtype: PyTorch data type for the model
            **kwargs: Additional model parameters
        """
        self.processor_path = processor_path
        self.torch_dtype = torch_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_path, **kwargs)
    
    def _load_model(self, **kwargs):
        """Load the Transformers model and processor."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=self.torch_dtype
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.processor_path)
        self.model.to(self.device)
    
    async def generate_async(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generate OCR text from an image asynchronously.
        
        Note: This implementation runs synchronously since Transformers
        doesn't have native async support, but wrapped in async interface.
        
        Args:
            image: PIL Image object
            prompt: Prompt text for OCR
            **kwargs: Additional generation parameters (temperature, max_new_tokens, etc.)
            
        Returns:
            Generated OCR text
        """
        # Run sync method in thread pool to make it async
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, image, prompt, **kwargs)
    
    def generate_sync(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generate OCR text from an image synchronously using Transformers.
        
        Args:
            image: PIL Image object
            prompt: Prompt text for OCR
            **kwargs: Additional generation parameters
            
        Returns:
            Generated OCR text
        """
        self.validate_inputs(image, prompt)
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Build the full prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": processed_image},
                ],
            }
        ]
        
        # Apply the chat template and processor
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[processed_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for (key, value) in inputs.items()}
        
        # Set generation parameters
        temperature = kwargs.get('temperature', 0.1)
        max_new_tokens = kwargs.get('max_new_tokens', 8000)
        num_return_sequences = kwargs.get('num_return_sequences', 1)
        do_sample = kwargs.get('do_sample', True)
        
        # Generate the output
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
            )
        
        # Decode the output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = self.processor.tokenizer.batch_decode(
            new_tokens, 
            skip_special_tokens=True
        )
        
        return text_output[0] if text_output else ""
    
    def process_batch(self, images: list, prompts: list, **kwargs) -> list:
        """
        Process a batch of images and prompts.
        
        Args:
            images: List of PIL Image objects
            prompts: List of prompt texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated OCR texts
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")
        
        results = []
        for image, prompt in zip(images, prompts):
            result = self.generate_sync(image, prompt, **kwargs)
            results.append(result)
        
        return results