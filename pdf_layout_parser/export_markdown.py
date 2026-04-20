from __future__ import annotations

import json
import os

from .models import Document


def save_document_json(document: Document, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)


def document_to_markdown(document: Document) -> str:
    parts: list[str] = []

    for page in document.pages:
        parts.append(f"<!-- page: {page.page_number} -->")

        for element in sorted(page.elements, key=lambda item: item.order):
            if element.type == "heading" and element.text:
                parts.append(f"## {element.text}")
            elif element.type == "table" and element.markdown:
                parts.append(element.markdown)
            elif element.type in {"paragraph", "handwritten"} and element.text:
                parts.append(element.text)
            elif element.type == "image" and element.image_path:
                parts.append(f"![image]({element.image_path})")

        parts.append("")

    return "\n\n".join(part for part in parts if part.strip())


def save_document_markdown(document: Document, output_path: str) -> None:
    markdown = document_to_markdown(document)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
