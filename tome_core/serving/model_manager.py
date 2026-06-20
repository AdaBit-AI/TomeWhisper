"""
ModelManager: loads a VLM, serves single + batched OCR requests.

Uses AutoModelForImageTextToText for broad model support.
"""

import asyncio
import time

import torch
from PIL import Image


class ModelManager:
    """Holds the loaded VLM and processor."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.model_name = None

    def load(self, model_name: str):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model_name = model_name
        print(f"Device: {self.device}", flush=True)
        print(f"Loading: {model_name} ...", flush=True)
        t0 = time.time()

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if self.device == "cuda" else None,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    async def generate(
        self, pil_image: Image.Image, prompt: str, max_new_tokens: int = 2048,
    ) -> str:
        return (await self.generate_batch([pil_image], [prompt], max_new_tokens))[0]

    async def generate_batch(
        self, images: list[Image.Image], prompts: list[str], max_new_tokens: int = 2048,
    ) -> list[str]:
        """Process multiple images in one batched GPU call."""
        print(f"  [server] generate_batch: {len(images)} images, max_tokens={max_new_tokens}",
              flush=True)
        t0 = time.time()

        from qwen_vl_utils import process_vision_info

        min_pixels, max_pixels = 2048, 16777216

        messages_list = []
        for img, prompt in zip(images, prompts):
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": prompt},
                ],
            }])

        chat_kwargs = {"enable_thinking": False}
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
            for m in messages_list
        ]

        all_images = []
        for m in messages_list:
            imgs, _ = process_vision_info(m, image_patch_size=16)
            all_images.extend(imgs)

        print(f"  [server] tokenizing...", flush=True)
        inputs = self.processor(
            text=texts, images=all_images, do_resize=False, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        print(f"  [server] input_ids shape: {inputs['input_ids'].shape}, generating...", flush=True)

        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self._generate_blocking, inputs, max_new_tokens)

        results = []
        for i in range(len(images)):
            prompt_len = (inputs["input_ids"][i] != self.processor.tokenizer.pad_token_id).sum().item()
            new_tokens = output[i, prompt_len:]
            results.append(self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True))

        print(f"  [server] done in {time.time() - t0:.1f}s", flush=True)
        return results

    def _generate_blocking(self, inputs: dict, max_new_tokens: int):
        with torch.no_grad():
            return self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0,
            )
