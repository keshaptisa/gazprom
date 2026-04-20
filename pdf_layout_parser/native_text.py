from __future__ import annotations

from dataclasses import dataclass

import fitz


@dataclass
class NativeTextBlock:
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    is_bold: bool = False


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
            bold_flags: list[bool] = []

            for line in lines:
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if not text:
                        continue

                    spans_text.append(text)
                    font_sizes.append(float(span.get("size", 0.0)))

                    font_name = str(span.get("font", "")).lower()
                    bold_flags.append("bold" in font_name or "black" in font_name)

            if not spans_text:
                continue

            text = " ".join(spans_text).strip()
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
            is_bold = any(bold_flags)

            results.append(
                NativeTextBlock(
                    page_number=page_index + 1,
                    bbox=bbox,
                    text=text,
                    font_size=avg_font_size,
                    is_bold=is_bold,
                )
            )

    return results
