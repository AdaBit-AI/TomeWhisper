"""
FastAPI application factory for OCR serving.

Usage:
    python -m tome_core.serving.server --model infly/Infinity-Parser2-Pro --port 8000
"""

import contextlib
import io
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from pydantic import BaseModel

from .model_manager import ModelManager


class OCRResponse(BaseModel):
    text: str
    status: str = "success"
    model_used: str
    elapsed_ms: float


DEFAULT_OCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

manager = ModelManager()
_default_model: Optional[str] = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    manager.load(_default_model)
    yield


def create_app(model_name: str = "infly/Infinity-Parser2-Pro") -> FastAPI:
    """Create the FastAPI application with all OCR endpoints."""
    global _default_model
    _default_model = model_name

    app = FastAPI(title="TomeWhisper OCR Server", version="0.3.0", lifespan=lifespan)

    @app.post("/ocr", response_model=OCRResponse)
    async def ocr_endpoint(
        image: UploadFile = File(...),
        prompt: str = Form(None),
        max_new_tokens: int = Form(2048),
    ):
        if manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            t0 = time.time()
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if not prompt:
                prompt = DEFAULT_OCR_PROMPT
            result = await manager.generate(pil_image, prompt, max_new_tokens)
            elapsed = (time.time() - t0) * 1000
            return OCRResponse(text=result, model_used=manager.model_name, elapsed_ms=round(elapsed, 1))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ocr/batch")
    async def ocr_batch_endpoint(
        images: list[UploadFile] = File(...),
        prompt: str = Form(None),
        max_new_tokens: int = Form(2048),
    ):
        if manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            t0 = time.time()
            pil_images = []
            for img in images:
                img_bytes = await img.read()
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            if not prompt:
                prompt = DEFAULT_OCR_PROMPT
            results = await manager.generate_batch(pil_images, [prompt] * len(pil_images), max_new_tokens)
            elapsed = (time.time() - t0) * 1000
            return {
                "texts": results,
                "status": "success",
                "model_used": manager.model_name,
                "elapsed_ms": round(elapsed, 1),
                "batch_size": len(images),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": manager.model_name,
            "device": manager.device,
            "model_loaded": manager.model is not None,
        }

    @app.get("/")
    async def root():
        return {
            "service": "TomeWhisper OCR Server",
            "endpoints": {
                "POST /ocr": "Single-page OCR",
                "POST /ocr/batch": "Batch OCR (multiple images)",
                "GET /health": "Health check",
            },
        }

    return app
