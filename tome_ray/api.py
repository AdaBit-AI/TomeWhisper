"""
FastAPI integration for Ray Serve OCR deployments.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from ray import serve


class OCRResponse(BaseModel):
    """Response model for OCR requests."""
    response: str
    status: str = "success"


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    status: str = "error"


def create_ocr_app(deployment) -> FastAPI:
    """
    Create a FastAPI app for OCR processing.
    
    Args:
        deployment: Ray Serve deployment to handle OCR requests
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="TomeWhisper OCR API",
        description="OCR processing API using Ray Serve",
        version="0.1.0"
    )
    
    @app.post("/ocr", response_model=OCRResponse)
    async def ocr_endpoint(
        image: UploadFile = File(..., description="Image file to process"),
        prompt_mode: str = Form("prompt_layout_all_en", description="Prompt mode for OCR processing")
    ):
        """
        Process OCR on an uploaded image.
        
        Args:
            image: Image file to process (JPEG, PNG, etc.)
            prompt_mode: Prompt mode for OCR processing
            
        Returns:
            OCR result with extracted text
            
        Raises:
            HTTPException: If processing fails
        """
        try:
            # Call the deployment
            result = await deployment(image, prompt_mode)
            return OCRResponse(response=result["response"])
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "tome-whisper-ocr"}
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "TomeWhisper OCR API",
            "version": "0.1.0",
            "endpoints": {
                "ocr": "POST /ocr - Process OCR on uploaded image",
                "health": "GET /health - Health check"
            }
        }
    
    return app


def create_vllm_app(model_path: str, tensor_parallel_size: int = 1, 
                    gpu_memory_utilization: float = 0.95) -> FastAPI:
    """
    Create a FastAPI app for VLLM OCR processing.
    
    Args:
        model_path: Path to the VLLM model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization ratio
        
    Returns:
        FastAPI application with VLLM deployment
    """
    from .deployments import create_vllm_deployment
    
    deployment = create_vllm_deployment(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    return create_ocr_app(deployment)


def create_transformers_app(model_path: str = "allenai/olmOCR-2-7B-1025", 
                           processor_path: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> FastAPI:
    """
    Create a FastAPI app for Transformers OCR processing.
    
    Args:
        model_path: Path to the Transformers model
        processor_path: Path to the processor model
        
    Returns:
        FastAPI application with Transformers deployment
    """
    from .deployments import create_transformers_deployment
    
    deployment = create_transformers_deployment(
        model_path=model_path,
        processor_path=processor_path
    )
    
    return create_ocr_app(deployment)