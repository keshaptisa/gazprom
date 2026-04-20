from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import fitz


FontWeight = Literal["normal", "semibold", "bold"]


@dataclass
class NativeTextBlock:
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    font_weight: FontWeight = "normal"


def _detect_font_weight(font_name: str) -> FontWeight:
    name = font_name.lower()

    if any(token in name for token in ("bold", "black", "heavy", "extrabold", "demi")):
        return "bold"

    if any(token in name for token in ("semibold", "medium", "book")):
        return "semibold"

    return "normal"


def _max_font_weight(weights: list[FontWeight]) -> FontWeight:
    if "bold" in weights:
        return "bold"
    if "semibold" in weights:
        return "semibold"
    return "normal"


def extract_native_text_blocks(pdf_path: str) -> list[NativeTextBlock]:
    doc = fitz.open(pdf_path)
    results: list[NativeTextBlock] = []

    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            lines = block.get("lines", [])

            spans_text: list[str] = []
            font_sizes: list[float] = []
            font_weights: list[FontWeight] = []

            for line in lines:
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if not text:
                        continue

                    spans_text.append(text)
                    font_sizes.append(float(span.get("size", 0.0)))

                    font_name = str(span.get("font", ""))
                    font_weights.append(_detect_font_weight(font_name))

            if not spans_text:
                continue

            text = " ".join(spans_text).strip()
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
            font_weight = _max_font_weight(font_weights)

            results.append(
                NativeTextBlock(
                    page_number=page_index + 1,
                    bbox=bbox,
                    text=text,
                    font_size=avg_font_size,
                    font_weight=font_weight,
                )
            )

    return results
