from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_probable_junk(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True

    if len(text) <= 2:
        return True

    # Часто OCR-мусор: почти нет букв, много случайных символов
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    if letters == 0 and digits < 3:
        return True

    return False


def is_probable_header_or_footer(text: str, page_number: int) -> bool:
    text = normalize_text(text).lower()

    if not text:
        return True

    patterns = [
        rf"^{page_number}$",
        r"^рис\.\s*\d+",
        r"^page\s*\d+",
        r"^\[\d+\]$",
    ]
    return any(re.match(pattern, text) for pattern in patterns)


def clean_element_text(text: str, page_number: int) -> str:
    text = normalize_text(text)
    if is_probable_header_or_footer(text, page_number):
        return ""
    if is_probable_junk(text):
        return ""
    return text
