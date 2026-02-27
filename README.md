# TomeWhisper - Modularized OCR Processing System

A production-ready, modularized OCR processing system that leverages vision language models for extracting text from PDFs and images. Built with Ray Serve for scalability and FastAPI for easy integration.

## 🚀 Features

- **Multi-Backend Support**: VLLM and Transformers model backends
- **PDF Processing**: Native PDF rendering using olmocr
- **Scalable Deployment**: Ray Serve integration for production workloads
- **REST API**: FastAPI endpoints for easy integration
- **Modular Architecture**: Clean separation between core functionality and deployment
- **Production Ready**: Health checks, error handling, and monitoring

## 📦 Installation

### Prerequisites
- Python 3.8 or newer
- CUDA-compatible GPU (recommended)

### Quick Install with uv
```bash
# Clone repository
git clone https://github.com/AdaBit-AI/TomeWhisper.git
cd TomeWhisper

# Create and activate uv environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Optional Features
```bash
# For development
uv pip install -r requirements-dev.txt

# For PDF processing
uv pip install olmocr

# For high-performance backend
uv pip install vllm

# For scalable deployment
uv pip install "ray[serve]"
```

## 🔧 Initial Setup

### 1. Verify Installation
Run the test script to ensure everything is working:
```bash
python examples/test_tome_core.py
```

Expected output:
```
=== Testing TomeWhisper Core Functionality ===
✓ Transformers OCR test PASSED
✓ Image Processor test PASSED  
✓ PDF Processor test PASSED
✓ Full OCR Workflow test PASSED
Overall: 4/4 tests passed
```

### 2. Download Models (Automatic)
Models are downloaded automatically on first use:
- **Transformers**: `allenai/olmOCR-2-7B-1025` (~14GB)
- **VLLM**: Configurable model selection

### 3. Environment Variables (Optional)
```bash
# Set CUDA device (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Set Ray configuration
export RAY_DISABLE_IMPORT_WARNING=1
```

## 🎯 Quick Usage Examples

### Basic OCR with Transformers
```python
from tome_core.models import TransformersOCRModel
from tome_core.processors import ImageProcessor
from PIL import Image

# Initialize model and processor
model = TransformersOCRModel()
processor = ImageProcessor()

# Load and process image
image = Image.open("document.png")
processed_image = processor.process_image(image)

# Perform OCR
result = model.generate_sync(processed_image, "Extract all text from this image")
print(result)
```

### PDF Processing with OCR
```python
from tome_core.models import TransformersOCRModel
from tome_core.processors import PDFProcessor, ImageProcessor
from tome_core.utils.prompt_utils import get_prompt_by_mode

# Initialize components
model = TransformersOCRModel()
pdf_processor = PDFProcessor()
image_processor = ImageProcessor()

# Process PDF
pdf_path = "research_paper.pdf"
image = pdf_processor.render_pdf_page_to_image(pdf_path, page_number=1)
processed_image = image_processor.process_image(image)

# Get OCR prompt
prompt = get_prompt_by_mode("prompt_no_anchoring_v4_yaml")

# Perform OCR
result = model.generate_sync(processed_image, prompt)
print(f"Extracted text: {result[:500]}...")
```

### Ray Serve Deployment
```python
from tome_ray.deployments import create_transformers_deployment
from ray import serve

# Create and run deployment
deployment = create_transformers_deployment(
    num_replicas=2,
    max_concurrent_queries=10
)

serve.run(deployment, port=8000)
```

### FastAPI Service
```python
from tome_ray.api import create_transformers_app
import uvicorn

# Create and run API app
app = create_transformers_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📋 API Usage

### OCR Endpoint
```bash
# Upload image for OCR processing
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@document.png" \
  -F "prompt=Extract all text from this image"
```

### Health Check
```bash
# Check service health
curl "http://localhost:8000/health"
```

## 🔍 Configuration Options

### Model Configuration
```python
# Transformers model with custom settings
model = TransformersOCRModel(
    model_name="allenai/olmOCR-2-7B-1025",
    device="cuda",
    torch_dtype="float16",
    max_new_tokens=2048
)

# VLLM model with performance tuning
model = VLLMOCRModel(
    model_name="allenai/olmOCR-2-7B-1025",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    max_model_len=4096
)
```

### Deployment Configuration
```python
# Ray Serve deployment with scaling
deployment = create_transformers_deployment(
    num_replicas=4,
    max_concurrent_queries=20,
    ray_actor_options={"num_gpus": 1}
)
```

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or use smaller model
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**2. Model Download Issues**
```bash
# Set Hugging Face mirror (for China users)
export HF_ENDPOINT=https://hf-mirror.com
```

**3. Ray Serve Port Conflicts**
```bash
# Use different port
serve.run(deployment, port=8001)
```

**4. olmocr Not Available**
```bash
# Install olmocr separately
pip install olmocr --no-deps
# Or use fallback PDF processing
```

### Performance Optimization

**1. GPU Memory Management**
```python
# Use mixed precision
model = TransformersOCRModel(torch_dtype="float16")

# Enable gradient checkpointing (if training)
model.model.gradient_checkpointing_enable()
```

**2. Batch Processing**
```python
# Process multiple images
results = await model.generate_async_batch(images, prompts)
```

**3. Caching**
```python
# Enable response caching (implementation dependent)
# Consider Redis for distributed caching
```

## 📊 Performance Benchmarks

Typical performance on NVIDIA A100 GPU:
- **Single Image OCR**: ~2-3 seconds
- **PDF Page Processing**: ~3-5 seconds per page
- **Throughput**: 10-20 pages per minute (depending on complexity)

## 🔗 Related Documentation

- [MODULARIZATION.md](./MODULARIZATION.md) - Detailed modularization overview
- [MODULARIZATION_SUMMARY.md](./MODULARIZATION_SUMMARY.md) - Complete implementation summary
- [examples/test_tome_core.py](./examples/test_tome_core.py) - Comprehensive usage examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check the FAQ section above
- Review the troubleshooting guide

---

**Happy OCR processing! 🎉**