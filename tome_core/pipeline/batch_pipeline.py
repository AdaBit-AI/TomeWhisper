"""
BatchPipeline: process PDF pages through remote OCR with optional figure extraction.

Usage:
    pipeline = BatchPipeline(pdf_path, out_dir)
    pipeline.run(start_page=11, end_page=111, batch_size=16, with_figures=True)
"""

import io
import os
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from ..config import config
from ..processors import PDFProcessor, ImageProcessor
from ..processors.figure_extractor import FigureExtractor
from ..utils.prompt_utils import get_prompt_by_mode


def _build_ocr_prompt() -> str:
    base = get_prompt_by_mode("prompt_no_anchoring_v4_yaml")
    return (
        base
        + "\nCRITICAL MATH FORMATTING RULES:\n"
        + "- Wrap EVERY mathematical expression in $...$ (inline) or $$...$$ (display).\n"
        + "- Variables: $m$, $w$, $n$, $M$, $W$, $S$, $T$\n"
        + "- Expressions with operators: $m \\in M$, $w \\in W$, $O(n^2)$, $m'$, $w'$\n"
        + "- Sets and tuples: $\\{m_1, \\dots, m_n\\}$, $(m, w)$\n"
        + "- NEVER leave math unmarked — $m'$, $w'$, $S'$, $\\{x,y\\}$, $O(n)$, etc."
    )


class BatchPipeline:
    """Batch OCR pipeline: PDF rendering → remote inference → markdown + figures."""

    def __init__(
        self,
        pdf_path: str,
        out_dir: Path,
        ocr_url: Optional[str] = None,
        batch_url: Optional[str] = None,
    ):
        self.pdf_path = pdf_path
        self.out_dir = Path(out_dir)
        self.ocr_url = ocr_url or config.ocr_url
        self.batch_url = batch_url or config.batch_url
        self.prompt = _build_ocr_prompt()
        self.image_proc = ImageProcessor(max_dimension=1024)

    def _render_pages(self, pages: list[int]) -> dict[int, io.BytesIO]:
        """Render all pages to in-memory PNG buffers."""
        buffers = {}
        for p in pages:
            img = PDFProcessor.render_pdf_page_to_image(
                self.pdf_path, page_number=p, target_longest_image_dim=1024,
            )
            processed = self.image_proc.process_image(img)
            buf = io.BytesIO()
            processed.save(buf, format="PNG")
            buf.seek(0)
            buffers[p] = buf
        return buffers

    def _clean_text(self, text: str) -> str:
        """Strip markdown fences from OCR output."""
        text = text.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    def run(
        self,
        start_page: int = 1,
        end_page: int = 10,
        batch_size: int = 16,
        with_figures: bool = False,
        figures_hires: int = 4096,
        timeout: int = 3600,
    ) -> dict:
        """
        Process a range of PDF pages.

        Returns dict with keys: pages_processed, figures_extracted, elapsed_s, output_dir.
        """
        if not PDFProcessor.is_olmocr_available():
            raise RuntimeError("olmocr not available")

        page_count = PDFProcessor.get_pdf_page_count(self.pdf_path)
        pages = list(range(start_page, min(end_page, page_count) + 1))
        self.out_dir.mkdir(exist_ok=True)

        print(f"PDF: {os.path.basename(self.pdf_path)}")
        print(f"Pages: {start_page}-{pages[-1]} ({len(pages)} total)")
        print(f"Batch size: {batch_size}, figures: {with_figures}")
        print(f"Output: {self.out_dir}")
        print(f"Server: {self.batch_url}")
        print()

        # Connection check
        health_url = self.batch_url.replace("/ocr/batch", "/health")
        try:
            h = requests.get(health_url, timeout=10)
            info = h.json()
            print(f"✓ Server reachable: {info.get('status')}, "
                  f"model={info.get('model')}, device={info.get('device')}")
        except Exception as e:
            print(f"✗ Cannot reach server at {health_url}: {e}")
            print(f"  Is the SSH tunnel up? Run: ssh -L 8001:127.0.0.1:8000 murphy@192.168.31.156")
            raise

        # Quick functional test
        try:
            test_img = PDFProcessor.render_pdf_page_to_image(
                self.pdf_path, page_number=pages[0], target_longest_image_dim=256,
            )
            buf = io.BytesIO()
            test_img.save(buf, format="PNG")
            buf.seek(0)
            resp = requests.post(
                self.ocr_url,
                files={"image": ("test.png", buf, "image/png")},
                data={"prompt": "Say hello", "max_new_tokens": 8},
                timeout=120,
            )
            if resp.status_code == 200:
                print(f"✓ OCR functional test passed ({resp.json().get('elapsed_ms', 0):.0f}ms)\n")
            else:
                print(f"✗ OCR test returned HTTP {resp.status_code}\n")
        except Exception as e:
            print(f"✗ OCR functional test failed: {e}")
            print(f"  Health check passed but POST requests are blocked (firewall?)")
            print(f"  Use SSH tunnel: ssh -L 8001:127.0.0.1:8000 murphy@192.168.31.156")
            print(f"  Then set: export TOMEWHISPER_HOST=127.0.0.1 TOMEWHISPER_PORT=8001")
            raise

        # Pre-render
        print(f"Rendering {len(pages)} pages locally...", flush=True)
        t_render = time.time()
        page_buffers = self._render_pages(pages)
        print(f"  Rendered in {time.time() - t_render:.1f}s\n", flush=True)

        figure_extractor = None
        if with_figures:
            figure_extractor = FigureExtractor(self.pdf_path, self.ocr_url)

        total_start = time.time()
        total_figures = 0
        total_batches = (len(pages) + batch_size - 1) // batch_size

        for batch_start in range(0, len(pages), batch_size):
            batch_pages = pages[batch_start:batch_start + batch_size]
            batch_idx = batch_start // batch_size + 1

            # Build multipart batch request
            files = []
            for p in batch_pages:
                files.append(("images", (f"page{p}.png", page_buffers[p].getvalue(), "image/png")))

            print(f"→ Sending batch {batch_idx}/{total_batches} "
                  f"(pages {batch_pages[0]}-{batch_pages[-1]}, {len(batch_pages)} pages)...",
                  end="", flush=True)
            t0 = time.time()
            resp = requests.post(
                self.batch_url,
                files=files,
                data={"prompt": self.prompt, "max_new_tokens": 2048},
                timeout=timeout,
            )
            batch_time = time.time() - t0
            print(f" done in {batch_time:.0f}s", flush=True)

            if resp.status_code != 200:
                print(f"  ⚠ batch failed: HTTP {resp.status_code}", flush=True)
                for p in batch_pages:
                    print(f"    Page {p:>3}: ❌")
                continue

            data = resp.json()
            texts = data.get("texts", [])
            server_ms = data.get("elapsed_ms", 0)

            for i, p in enumerate(batch_pages):
                t_page = time.time()
                text = self._clean_text(texts[i]) if i < len(texts) else ""
                if not text:
                    print(f"    Page {p:>3}: ❌ empty")
                    continue

                # Figure extraction
                figures = []
                if figure_extractor:
                    try:
                        figures = figure_extractor.extract(p, self.out_dir, hires_dim=figures_hires)
                        total_figures += len(figures)
                    except Exception as e:
                        print(f"    Page {p:>3}: ⚠ figure extraction failed: {e}")

                if figures:
                    text = FigureExtractor.embed_inline(text, figures)

                out_path = self.out_dir / f"page_{p:04d}.md"
                out_path.write_text(text)

                elapsed = time.time() - t_page
                preview = text[:100].replace("\n", " ")[:100]
                fig_info = f" +{len(figures)}figs" if figures else ""
                fig_time = f" ({elapsed:.0f}s fig)" if figures else ""
                print(f"    Page {p:>3}: {len(text):>5} chars{fig_info}{fig_time} | {preview}...",
                      flush=True)

            # Progress summary
            total_done = min(batch_start + batch_size, len(pages))
            elapsed = time.time() - total_start
            rate = elapsed / total_done if total_done > 0 else 0
            remaining = rate * (len(pages) - total_done)
            per_page = server_ms / len(batch_pages) if batch_pages else 0
            print(f"  → [{total_done}/{len(pages)}] server {server_ms:.0f}ms "
                  f"({per_page:.0f}ms/page) | elapsed {elapsed:.0f}s | ETA {remaining/60:.0f}min\n",
                  flush=True)

        total_time = time.time() - total_start
        result = {
            "pages_processed": len(pages),
            "figures_extracted": total_figures,
            "elapsed_s": total_time,
            "output_dir": str(self.out_dir),
        }
        print(f"✓ Done: {len(pages)} pages in {total_time/60:.1f}min "
              f"({total_time / len(pages):.1f}s/page)" +
              (f", {total_figures} figures" if with_figures else ""))
        return result
