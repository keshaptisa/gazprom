from __future__ import annotations

import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from pdf_layout_parser.native_text import extract_native_text_blocks


ZIP_PATH = Path(r"C:\Users\User\Downloads\pdfs-20260421T170621Z-3-001.zip")

WATERMARK_PATTERNS = {
    "черновик": r"\bчерновик\b",
    "draft": r"\bdraft\b",
    "не для распространения": r"не\s+для\s+распространения",
    "образец": r"\bобразец\b",
    "конфиденциально": r"\bконфиденциально\b",
}

MIN_WORD_LEN = 3
MIN_REPEAT_COUNT = 2
TOP_WORDS_LIMIT = 30


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip().lower()


def extract_zip_to_temp(zip_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="pdf_repeat_scan_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    return temp_dir


def find_pdfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pdf"))


def extract_document_text(pdf_path: Path) -> str:
    blocks = extract_native_text_blocks(str(pdf_path))
    texts = [block.text for block in blocks if block.text.strip()]
    return normalize_text("\n".join(texts))


def count_watermarks(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, pattern in WATERMARK_PATTERNS.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        counts[label] = len(matches)
    return counts


def tokenize_words(text: str) -> list[str]:
    words = re.findall(r"[a-zа-яё]+", text.lower())
    return [word for word in words if len(word) >= MIN_WORD_LEN]


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    extracted_root = extract_zip_to_temp(ZIP_PATH)
    pdf_paths = find_pdfs(extracted_root)

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found inside: {ZIP_PATH}")

    print(f"Found {len(pdf_paths)} PDF files\n")

    total_watermarks = Counter()

    for pdf_path in pdf_paths:
        text = extract_document_text(pdf_path)

        watermark_counts = count_watermarks(text)
        total_watermarks.update(watermark_counts)

        word_counter = Counter(tokenize_words(text))
        repeated_words = {
            word: count
            for word, count in word_counter.most_common()
            if count >= MIN_REPEAT_COUNT
        }

        print("=" * 100)
        print(pdf_path.name)

        found_any_watermark = any(count > 0 for count in watermark_counts.values())
        if found_any_watermark:
            print("Watermarks:")
            for label, count in watermark_counts.items():
                if count > 0:
                    print(f"  {label}: {count}")
        else:
            print("Watermarks:")
            print("  nothing found")

        if repeated_words:
            print(f"Repeated words (top {TOP_WORDS_LIMIT}):")
            shown = 0
            for word, count in repeated_words.items():
                print(f"  {word}: {count}")
                shown += 1
                if shown >= TOP_WORDS_LIMIT:
                    break
        else:
            print("Repeated words:")
            print("  nothing repeated")

        print()

    print("=" * 100)
    print("TOTAL WATERMARKS:")
    for label in WATERMARK_PATTERNS:
        print(f"  {label}: {total_watermarks[label]}")


if __name__ == "__main__":
    main()
