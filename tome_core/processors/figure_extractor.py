"""
Figure extraction from PDF pages via layout detection + high-res cropping.

Workflow:
  1. Render page at 1024px → send to OCR server for layout detection
  2. Parse layout JSON → identify figure/table elements with bboxes
  3. Render page at high resolution → scale bboxes → crop → trim whitespace
  4. Return cropped figures with metadata for inline embedding
"""

import io
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image

from ..config import config as global_config
from .pdf_processor import PDFProcessor

LAYOUT_PROMPT = (
    "Analyze this page image. Return a JSON array where each element has:\n"
    '  - "bbox": [x1, y1, x2, y2] in pixels\n'
    '  - "category": one of "figure", "table", "text", "title", "header", "equation"\n'
    '  - "text": extracted text content (caption for figures/tables)\n'
    "Return ONLY valid JSON. No markdown. No explanation."
)


class FigureExtractor:
    """Extract figures from PDF pages using remote OCR layout detection."""

    def __init__(self, pdf_path: str, ocr_url: Optional[str] = None):
        self.pdf_path = pdf_path
        self.ocr_url = ocr_url or global_config.ocr_url

    def detect_layout(self, page_image: Image.Image, timeout: int = 300) -> list[dict]:
        """Send page to OCR server for layout detection. Returns list of elements."""
        buf = io.BytesIO()
        page_image.save(buf, format="PNG")
        buf.seek(0)

        resp = requests.post(
            self.ocr_url,
            files={"image": ("page.png", buf, "image/png")},
            data={"prompt": LAYOUT_PROMPT, "max_new_tokens": 4096},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["text"].strip()
        text = re.sub(r'^```(?:json)?\s*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("elements", data.get("layout", []))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def extract(
        self,
        page_num: int,
        out_dir: Path,
        hires_dim: int = 4096,
    ) -> list[dict]:
        """
        Extract figures from a single PDF page.

        Args:
            page_num: 1-indexed page number.
            out_dir: Directory to save cropped figures (in out_dir/figures/).
            hires_dim: Longest side for high-res render (default 4096).

        Returns:
            List of dicts: {file, category, caption, size, y_frac}.
        """
        # Low-res render for layout detection
        lowres = PDFProcessor.render_pdf_page_to_image(
            self.pdf_path, page_number=page_num, target_longest_image_dim=1024,
        )
        elements = self.detect_layout(lowres)

        figure_elems = [e for e in elements
                        if e.get("category", "").lower() in ("figure", "table")]
        if not figure_elems:
            return []

        # Find captions just below each figure
        captions = [e for e in elements
                    if "caption" in e.get("category", "").lower()]
        fig_caption_top = {}
        for i, fig in enumerate(figure_elems):
            fb = fig.get("bbox", [])
            if len(fb) != 4:
                continue
            for cap in captions:
                cb = cap.get("bbox", [])
                if len(cb) != 4:
                    continue
                x_overlap = min(fb[2], cb[2]) - max(fb[0], cb[0])
                gap = cb[1] - fb[3]
                if x_overlap > 0 and gap >= 0 and gap < (fb[3] - fb[1]):
                    fig_caption_top[i] = cb[1]
                    break

        # High-res render
        hires = PDFProcessor.render_pdf_page_to_image(
            self.pdf_path, page_number=page_num, target_longest_image_dim=hires_dim,
        )
        scale_x = hires.width / lowres.width
        scale_y = hires.height / lowres.height

        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        saved = []
        for i, elem in enumerate(figure_elems):
            bbox = elem.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)

            # Generous top margin, tight bottom
            w_fig, h_fig = x2 - x1, y2 - y1
            fixed = int(40 * scale_x)
            x1 = max(0, x1 - max(fixed, int(w_fig * 0.08)))
            y1 = max(0, y1 - max(fixed, int(h_fig * 0.35)))
            x2 = min(hires.width, x2 + max(fixed, int(w_fig * 0.04)))
            y2 = min(hires.height, y2 + int(h_fig * 0.02))

            # Tighten to caption top
            if i in fig_caption_top:
                cap_y = int(fig_caption_top[i] * scale_y)
                if cap_y < y2:
                    y2 = cap_y

            cropped = hires.crop((x1, y1, x2, y2))

            # Trim bottom/right whitespace only
            arr = np.array(cropped)
            content_mask = (arr < 220).any(axis=2)
            rows = np.where(content_mask.any(axis=1))[0]
            cols = np.where(content_mask.any(axis=0))[0]
            if len(rows) > 0 and len(cols) > 0:
                r2 = min(arr.shape[0], rows[-1] + 16)
                c1 = max(0, cols[0] - 8)
                c2 = min(arr.shape[1], cols[-1] + 8)
                cropped = cropped.crop((c1, 0, c2, r2))

            # Guarantee minimum padding
            pw, ph = 20, 20
            padded = Image.new("RGB", (cropped.width + 2 * pw, cropped.height + 2 * ph), color="white")
            padded.paste(cropped, (pw, ph))
            cropped = padded

            cat = elem.get("category", "figure")
            caption = (elem.get("text") or f"{cat}_{i}")[:60]
            safe = re.sub(r'[^\w\s-]', '', caption).strip().replace(' ', '_')[:40]
            fname = f"page_{page_num:04d}_{cat}_{i:02d}_{safe}.png"
            out_path = fig_dir / fname
            cropped.save(out_path)

            fig_y_frac = (bbox[1] + bbox[3]) / 2 / lowres.height
            saved.append({
                "file": f"figures/{fname}",
                "category": cat,
                "caption": caption,
                "size": list(cropped.size),
                "y_frac": fig_y_frac,
            })

        return saved

    @staticmethod
    def embed_inline(text: str, figures: list[dict]) -> str:
        """
        Insert figure images into OCR text at correct positions based on
        y_frac (fractional y-position on page).
        """
        if not figures:
            return text

        figs_sorted = sorted(figures, key=lambda f: f.get("y_frac", 0))
        lines = text.split("\n")
        total = len(lines)

        insertions = []
        for fig in figs_sorted:
            y_frac = fig.get("y_frac", 0.5)
            line_idx = int(y_frac * total)
            line_idx = max(0, min(total, line_idx))
            md = f"\n\n![{fig['caption']}]({fig['file']})\n"
            insertions.append((line_idx, md))

        # Insert bottom-up to preserve indices
        for line_idx, md in sorted(insertions, key=lambda x: x[0], reverse=True):
            lines.insert(line_idx, md)

        return "\n".join(lines)
