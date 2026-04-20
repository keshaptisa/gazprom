from __future__ import annotations

import json
import os

from .models import Document, Element


def save_document_json(document: Document, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)


def _heading_level(element: Element) -> int | None:
    text = element.text.strip().lower()

    if not text:
        return None

    if text.startswith("глава"):
        return 1

    if text.startswith("раздел"):
        return 2

    if element.font_weight == "bold":
        return 2

    if element.font_weight == "semibold":
        return 3

    if element.font_size >= 17:
        return 2

    if element.font_size >= 14:
        return 3

    if element.font_size >= 12 and len(element.text) <= 120:
        return 4

    return None


def _normalize_image_path(path: str) -> str:
    path = path.replace("\\", "/")
    marker = "/output/images/"
    idx = path.lower().find(marker)
    if idx >= 0:
        return "images/" + path[idx + len(marker):]
    if path.lower().startswith("output/images/"):
        return "images/" + path[len("output/images/"):]
    return path


def _is_caption(element: Element) -> bool:
    text = element.text.strip().lower()
    return text.startswith("рис.") or text.startswith("рис ") or text.startswith("figure ")


def document_to_markdown(document: Document) -> str:
    parts: list[str] = []

    for page in document.pages:
        parts.append(f"<!-- page: {page.page_number} -->")

        page_elements = sorted(page.elements, key=lambda item: item.order)
        used_indexes: set[int] = set()

        for i, element in enumerate(page_elements):
            if i in used_indexes:
                continue

            if element.type == "image" and element.image_path:
                normalized_path = _normalize_image_path(element.image_path)
                parts.append(f"![image]({normalized_path})")

                # Ищем ближайшую следующую подпись на этой же странице
                for j in range(i + 1, min(i + 4, len(page_elements))):
                    candidate = page_elements[j]
                    if j in used_indexes:
                        continue

                    if _is_caption(candidate):
                        parts.append(candidate.text)
                        used_indexes.add(j)
                        break

                    # Если встретили другой крупный структурный блок, дальше уже не ищем подпись
                    if candidate.type in {"image", "table", "heading"}:
                        break

                continue

            if _is_caption(element):
                # Если подпись не была привязана к картинке, всё равно выводим её как обычный текст
                parts.append(element.text)
                continue

            if element.type == "heading" and element.text:
                level = _heading_level(element)
                if level is not None:
                    parts.append(f"{'#' * level} {element.text}")
                else:
                    parts.append(f"**{element.text}**")
                continue

            if element.type == "table" and element.markdown:
                parts.append(element.markdown)
                continue

            if element.type in {"paragraph", "handwritten"} and element.text:
                parts.append(element.text)
                continue

        parts.append("")

    return "\n\n".join(part for part in parts if part.strip())


def save_document_markdown(document: Document, output_path: str) -> None:
    markdown = document_to_markdown(document)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
