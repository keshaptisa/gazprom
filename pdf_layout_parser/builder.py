from __future__ import annotations

from pdf_image_parser.image_extractor import ExtractedImageBlock, parse_doc_id
from pdf_table_parser.table_extractor import ExtractedTable

from .cleanup import build_repeat_index, clean_element_text
from .models import Document, Element, Page
from .native_text import NativeTextBlock


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    return (x1 - x0) * (y1 - y0)


def _overlap_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    inter = _bbox_intersection(a, b)
    if inter <= 0:
        return 0.0

    min_area = min(_bbox_area(a), _bbox_area(b))
    if min_area <= 0:
        return 0.0

    return inter / min_area


def infer_element_type_from_block(block: ExtractedImageBlock) -> str:
    label = (block.predicted_label or "").lower()
    content = (block.content or "").strip()

    if block.block_type == "image":
        return "image"

    if "|" in content and "---" in content:
        return "table"

    if "handwritten" in label:
        return "handwritten"

    if len(content.splitlines()) <= 2 and len(content) < 120:
        return "heading"

    return "paragraph"


def infer_text_element_type(block: NativeTextBlock) -> str:
    text = block.text.strip()

    if len(text) < 200 and (block.font_size >= 15 or (block.font_size >= 13 and block.is_bold)):
        return "heading"

    return "paragraph"


def build_document_from_sources(
    pdf_path: str,
    blocks: list[ExtractedImageBlock],
    tables: list[ExtractedTable],
    native_text_blocks: list[NativeTextBlock],
) -> Document:
    document = Document(document_id=parse_doc_id(pdf_path))
    pages_map: dict[int, Page] = {}

    items: list[dict] = []

    all_texts_for_repeat_index: list[str] = []

    for block in native_text_blocks:
        if block.text.strip():
            all_texts_for_repeat_index.append(block.text)

    for block in blocks:
        if block.block_type == "text" and block.content.strip():
            all_texts_for_repeat_index.append(block.content)

    repeat_index = build_repeat_index(all_texts_for_repeat_index)

    # 1. Нативные таблицы из PDF
    for table in tables:
        items.append(
            {
                "kind": "table",
                "page_number": table.page_number,
                "bbox": table.bbox,
                "text": "",
                "markdown": table.markdown.strip(),
                "image_path": None,
                "confidence": table.confidence,
                "source_label": "pdf_table_parser",
                "font_size": 0.0,
            }
        )

    # 2. Нативный текст из PDF
    for block in native_text_blocks:
        text = clean_element_text(
            block.text,
            block.page_number,
            bbox=block.bbox,
            repeat_index=repeat_index,
        )
        if not text:
            continue

        overlaps_native_table = any(
            table.page_number == block.page_number
            and _overlap_ratio(block.bbox, table.bbox) > 0.5
            for table in tables
        )
        if overlaps_native_table:
            continue

        element_type = infer_text_element_type(block)

        items.append(
            {
                "kind": element_type,
                "page_number": block.page_number,
                "bbox": block.bbox,
                "text": text,
                "markdown": "",
                "image_path": None,
                "confidence": 1.0,
                "source_label": "native_text",
                "font_size": block.font_size,
            }
        )

    # 3. Блоки из image parser
    for block in blocks:
        block_type = infer_element_type_from_block(block)

        if block_type == "table":
            overlaps_native_table = any(
                table.page_number == block.page_number
                and _overlap_ratio(block.bbox, table.bbox) > 0.5
                for table in tables
            )
            if overlaps_native_table:
                continue

        if block.block_type == "text":
            overlaps_native_text = any(
                text_block.page_number == block.page_number
                and _overlap_ratio(block.bbox, text_block.bbox) > 0.5
                for text_block in native_text_blocks
            )
            if overlaps_native_text:
                continue

        text = clean_element_text(
            block.content if block.block_type == "text" else "",
            block.page_number,
            bbox=block.bbox,
            repeat_index=repeat_index,
        )

        if block.block_type == "text" and not text and block_type != "table":
            continue

        items.append(
            {
                "kind": block_type,
                "page_number": block.page_number,
                "bbox": block.bbox,
                "text": text if block_type != "table" else "",
                "markdown": block.content if block_type == "table" else "",
                "image_path": block.image_path,
                "confidence": block.confidence,
                "source_label": block.predicted_label,
                "font_size": 0.0,
            }
        )

    items.sort(key=lambda item: (item["page_number"], item["bbox"][1], item["bbox"][0]))

    order_per_page: dict[int, int] = {}

    for item in items:
        page_number = item["page_number"]
        page = pages_map.setdefault(page_number, Page(page_number=page_number))
        order_per_page[page_number] = order_per_page.get(page_number, 0) + 1

        element = Element(
            type=item["kind"],  # type: ignore[arg-type]
            order=order_per_page[page_number],
            bbox=item["bbox"],
            text=item.get("text", ""),
            markdown=item.get("markdown", ""),
            image_path=item.get("image_path"),
            source_label=item.get("source_label"),
            confidence=float(item.get("confidence", 1.0)),
        )
        page.elements.append(element)

    document.pages = [pages_map[key] for key in sorted(pages_map)]
    return document
