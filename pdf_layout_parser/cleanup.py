from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(len(text), 1)


def canonicalize_text_for_repeat_detection(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_repeat_index(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        normalized = canonicalize_text_for_repeat_detection(text)
        if normalized:
            counter[normalized] += 1
    return counter


def is_probable_junk(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True

    if len(text) <= 2:
        return True

    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)

    if letters == 0 and digits < 3:
        return True

    if _alpha_ratio(text) < 0.20 and len(text) < 25:
        return True

    if len(text.split()) <= 2 and len(text) < 12 and letters < 4:
        return True

    return False


def is_probable_header_or_footer(text: str, page_number: int) -> bool:
    text = normalize_text(text)
    lower = text.lower()

    patterns = [
        rf"^{page_number}$",
        r"^page\s*\d+$",
        rf".*\bpage\s*{page_number}\b.*",
        r"^\[\d+\]$",
        r"^рис\.\s*\d+.*$",
        r".*\b\d{4}-\d{2}-\d{2}\b.*",
    ]

    if any(re.match(pattern, lower) for pattern in patterns):
        return True

    if "page" in lower and re.search(r"\d{4}-\d{2}-\d{2}", lower):
        return True

    if "·" in text and ("page" in lower or re.search(r"\d{4}-\d{2}-\d{2}", lower)):
        return True

    return False


def is_probable_vertical_margin_text(
    text: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> bool:
    text = normalize_text(text)
    lower = text.lower()

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)

        # Узкий и высокий блок у поля страницы
        if height / width > 4.5 and len(text) < 80:
            return True

    if len(text) < 30 and len(text.split()) <= 3 and _alpha_ratio(text) > 0.5:
        if not re.search(r"[.!?]$", text) and not re.search(r"\d{4}", text):
            return True

    if len(text.split()) >= 3 and len(text) < 25:
        return True

    keywords = [
        "черновик",
        "draft",
        "копия",
    ]
    if any(keyword in lower for keyword in keywords):
        return True

    return False


def is_probable_watermark(text: str) -> bool:
    text = normalize_text(text)
    lower = text.lower()

    watermark_keywords = [
        "черновик",
        "draft",
        "sample",
        "demo",
        "confidential",
    ]
    if any(keyword in lower for keyword in watermark_keywords):
        return True

    if len(text.split()) <= 2 and len(text) < 20 and _alpha_ratio(text) > 0.6:
        return True

    return False


def is_probable_repeated_document_noise(
    text: str,
    repeat_index: Counter[str] | None = None,
    min_repeats: int = 2,
) -> bool:
    if repeat_index is None:
        return False

    normalized = canonicalize_text_for_repeat_detection(text)
    if not normalized:
        return False

    count = repeat_index.get(normalized, 0)
    if count < min_repeats:
        return False

    # Повторы считаем шумом только для коротких/служебных строк,
    # чтобы не выкинуть реальные повторяющиеся абзацы
    raw = normalize_text(text)
    if len(raw) <= 120:
        return True

    return False


def clean_element_text(
    text: str,
    page_number: int,
    bbox: tuple[float, float, float, float] | None = None,
    repeat_index: Counter[str] | None = None,
) -> str:
    text = normalize_text(text)

    if is_probable_header_or_footer(text, page_number):
        return ""

    if is_probable_vertical_margin_text(text, bbox):
        return ""

    if is_probable_watermark(text):
        return ""

    if is_probable_repeated_document_noise(text, repeat_index):
        return ""

    if is_probable_junk(text):
        return ""

    return text
