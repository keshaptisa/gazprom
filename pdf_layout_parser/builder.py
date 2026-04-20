from __future__ import annotations

from pdf_image_parser.image_extractor import ExtractedImageBlock, parse_doc_id

from .cleanup import clean_element_text
from .models import Document, Element, Page


def infer_element_type(block: ExtractedImageBlock) -> str:
    label = (block.predicted_label or "").lower()

    if block.block_type == "image":
        return "image"

    content = (block.content or "").strip()
    if "|" in content and "---" in content:
        return "table"

    if "handwritten" in label:
        return "handwritten"

    if len(content.splitlines()) <= 2 and len(content) < 120:
        return "heading"

    return "paragraph"


def build_document_from_blocks(
    pdf_path: str,
    blocks: list[ExtractedImageBlock],
) -> Document:
    document = Document(document_id=parse_doc_id(pdf_path))
    pages_map: dict[int, Page] = {}

    sorted_blocks = sorted(blocks, key=lambda block: (block.page_number, block.bbox[1], block.bbox[0]))

    order_per_page: dict[int, int] = {}

    for block in sorted_blocks:
        page = pages_map.setdefault(block.page_number, Page(page_number=block.page_number))
        order_per_page[block.page_number] = order_per_page.get(block.page_number, 0) + 1
        order = order_per_page[block.page_number]

        element_type = infer_element_type(block)
        text = clean_element_text(block.content if block.block_type == "text" else "", block.page_number)

        if block.block_type == "text" and not text and element_type != "table":
            continue

        element = Element(
            type=element_type,  # type: ignore[arg-type]
            order=order,
            bbox=block.bbox,
            text=text if element_type != "table" else "",
            markdown=block.content if element_type == "table" else "",
            image_path=block.image_path,
            source_label=block.predicted_label,
            confidence=block.confidence,
        )
        page.elements.append(element)

    document.pages = [pages_map[key] for key in sorted(pages_map)]
    return document
