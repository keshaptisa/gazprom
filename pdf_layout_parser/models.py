from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


ElementType = Literal[
    "heading",
    "paragraph",
    "table",
    "image",
    "handwritten",
    "garbage",
]

FontWeight = Literal["normal", "semibold", "bold"]


@dataclass
class Element:
    type: ElementType
    order: int
    bbox: tuple[float, float, float, float]
    text: str = ""
    markdown: str = ""
    image_path: str | None = None
    source_label: str | None = None
    confidence: float = 1.0
    font_size: float = 0.0
    font_weight: FontWeight = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Page:
    page_number: int
    elements: list[Element] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "elements": [element.to_dict() for element in self.elements],
        }


@dataclass
class Document:
    document_id: int
    pages: list[Page] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "pages": [page.to_dict() for page in self.pages],
        }
