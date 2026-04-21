from __future__ import annotations

import csv
import hashlib
import io
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from pdf_image_parser.image_extractor import (
        DiTClassifier,
        classify_image,
        compute_visual_features,
    )
except ModuleNotFoundError:
    from image_extractor import (
        DiTClassifier,
        classify_image,
        compute_visual_features,
    )


PDF_DIR = Path(r"C:\Users\User\gazprom\pdfs")
OUT_DIR = Path(r"C:\Users\User\gazprom\output\ocr_queue")

ALL_DIR = OUT_DIR / "all_images"
OCR_DIR = OUT_DIR / "ocr_candidates"
TABLE_DIR = OUT_DIR / "table_candidates"
NON_TEXT_DIR = OUT_DIR / "non_text_images"

MANIFEST_CSV = OUT_DIR / "image_queue_manifest.csv"


@dataclass
class QueueRecord:
    image_id: str
    pdf_file: str
    page_number: int
    image_index: int
    action: str
    predicted_label: str
    confidence: float
    final_bucket: str
    width: int
    height: int
    saved_path: str


def reset_output() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    ALL_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    NON_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def file_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def safe_open_image(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def save_manifest(records: list[QueueRecord], csv_path: Path) -> None:
    if not records:
        return

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_bucket_features(img_bgr: np.ndarray) -> dict[str, float]:
    feats = compute_visual_features(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    feats["dark_pixel_ratio"] = float((gray < 120).mean())
    feats["light_pixel_ratio"] = float((gray > 200).mean())
    feats["mid_pixel_ratio"] = float(((gray >= 120) & (gray <= 200)).mean())

    h, w = gray.shape[:2]
    feats["aspect_ratio"] = float(w / max(h, 1))

    return feats


def is_monochrome_text_like_color(feats: dict[str, float]) -> bool:
    return (
        feats["mean_saturation"] < 16
        and feats["dominant_colors_count"] <= 3
        and feats["light_pixel_ratio"] > 0.62
        and feats["dark_pixel_ratio"] > 0.03
        and feats["dark_pixel_ratio"] < 0.28
        and feats["small_components"] > 70
        and feats["contour_count"] > 140
    )


def is_small_text_candidate(feats: dict[str, float]) -> bool:
    return (
        feats["area"] < 120000
        and feats["aspect_ratio"] > 1.3
        and feats["mean_saturation"] < 18
        and feats["dominant_colors_count"] <= 3
        and feats["light_pixel_ratio"] > 0.60
        and feats["dark_pixel_ratio"] > 0.015
        and feats["dark_pixel_ratio"] < 0.30
        and feats["small_components"] > 20
        and feats["contour_count"] > 45
        and feats["hline_ratio"] < 0.015
        and feats["vline_ratio"] < 0.015
    )


def is_printed_text_rescue(feats: dict[str, float]) -> bool:
    return (
        feats["mean_saturation"] < 20
        and feats["dominant_colors_count"] <= 4
        and feats["light_pixel_ratio"] > 0.55
        and feats["dark_pixel_ratio"] > 0.015
        and feats["black_ratio"] < 0.32
        and feats["small_components"] > 80
        and feats["contour_count"] > 120
        and feats["hline_ratio"] < 0.02
        and feats["vline_ratio"] < 0.02
    )


def is_colored_diagram_like(feats: dict[str, float]) -> bool:
    return (
        feats["area"] > 50000
        and feats["mean_saturation"] > 28
        and feats["high_saturation_ratio"] > 0.05
        and feats["dominant_colors_count"] >= 3
        and feats["small_components"] < 180
        and feats["hline_ratio"] < 0.02
        and feats["vline_ratio"] < 0.02
    )


def is_strict_table_candidate(feats: dict[str, float]) -> bool:
    strong_grid = (
        feats["area"] > 50000
        and feats["hline_ratio"] > 0.010
        and feats["vline_ratio"] > 0.006
    )

    structured_grid = (
        feats["row_transitions"] > 8
        and feats["col_transitions"] > 5
        and feats["small_components"] > 60
    )

    low_color = (
        feats["mean_saturation"] < 22
        and feats["dominant_colors_count"] <= 4
    )

    return strong_grid and structured_grid and low_color


def is_soft_table_candidate(feats: dict[str, float]) -> bool:
    return (
        feats["area"] > 45000
        and feats["hline_ratio"] > 0.008
        and feats["vline_ratio"] > 0.004
        and feats["row_transitions"] > 7
        and feats["col_transitions"] > 4
        and feats["mean_saturation"] < 25
        and feats["dominant_colors_count"] <= 4
    )


def decide_bucket(action: str, predicted_label: str, img_bgr: np.ndarray) -> str:
    label = (predicted_label or "").lower().strip()
    feats = build_bucket_features(img_bgr)

    text_priority_labels = {
        "handwritten_like",
        "handwritten",
        "small_text_or_handwritten_like",
        "text_like_dense",
        "printed_text_like",
        "monochrome_text_like",
        "page_like",
        "scientific report",
        "scientific publication",
        "questionnaire",
        "memo",
        "letter",
        "invoice",
    }

    save_priority_labels = {
        "presentation",
        "advertisement",
        "scheme_like",
        "scheme_like_color",
        "file folder",
        "form",
    }

    if action == "drop":
        return "drop"

    if label == "table_like":
        return "table"

    if is_strict_table_candidate(feats):
        return "table"

    if is_soft_table_candidate(feats) and action == "ocr":
        return "table"

    if label in text_priority_labels:
        return "ocr"

    if is_printed_text_rescue(feats):
        return "ocr"

    if is_monochrome_text_like_color(feats):
        return "ocr"

    if is_small_text_candidate(feats):
        return "ocr"

    if label in save_priority_labels:
        return "non_text"

    if is_colored_diagram_like(feats) and action != "ocr":
        return "non_text"

    if action == "ocr":
        return "ocr"

    return "non_text"


def copy_to_bucket(saved_path: Path, file_name: str, bucket: str) -> Path | None:
    if bucket == "ocr":
        target = OCR_DIR / file_name
    elif bucket == "table":
        target = TABLE_DIR / file_name
    elif bucket == "non_text":
        target = NON_TEXT_DIR / file_name
    else:
        return None

    shutil.copy2(saved_path, target)
    return target


def process_pdf(pdf_path: Path, dit: DiTClassifier) -> list[QueueRecord]:
    doc = fitz.open(pdf_path)
    records: list[QueueRecord] = []

    for page_index in tqdm(range(len(doc)), desc=f"Pages {pdf_path.name}", leave=False):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for image_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image.get("image")
            ext = base_image.get("ext", "png")
            if not image_bytes:
                continue

            pil_img = safe_open_image(image_bytes)
            if pil_img is None:
                continue

            rgb = pil_img.convert("RGB")
            img_bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

            action, confidence, predicted_label = classify_image(img_bgr, dit)

            width, height = pil_img.size
            image_hash = file_sha1(image_bytes)
            image_id = f"{pdf_path.stem}_p{page_index + 1:03d}_i{image_index:03d}_{image_hash}"
            file_name = f"{image_id}.{ext}"

            saved_path = ALL_DIR / file_name
            with open(saved_path, "wb") as f:
                f.write(image_bytes)

            bucket = decide_bucket(action, predicted_label, img_bgr)
            if bucket == "drop":
                continue

            final_path = copy_to_bucket(saved_path, file_name, bucket)
            if final_path is None:
                continue

            records.append(
                QueueRecord(
                    image_id=image_id,
                    pdf_file=pdf_path.name,
                    page_number=page_index + 1,
                    image_index=image_index,
                    action=action,
                    predicted_label=predicted_label,
                    confidence=float(confidence),
                    final_bucket=bucket,
                    width=width,
                    height=height,
                    saved_path=str(final_path),
                )
            )

    return records


def main() -> None:
    reset_output()

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF dir not found: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    dit = DiTClassifier()
    all_records: list[QueueRecord] = []

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        all_records.extend(process_pdf(pdf_path, dit))

    save_manifest(all_records, MANIFEST_CSV)

    ocr_count = sum(1 for r in all_records if r.final_bucket == "ocr")
    table_count = sum(1 for r in all_records if r.final_bucket == "table")
    non_text_count = sum(1 for r in all_records if r.final_bucket == "non_text")

    label_counter = Counter(r.predicted_label for r in all_records)
    bucket_counter = Counter(r.final_bucket for r in all_records)

    print()
    print(f"Saved manifest: {MANIFEST_CSV}")
    print(f"All images: {ALL_DIR}")
    print(f"OCR candidates: {OCR_DIR}")
    print(f"Table candidates: {TABLE_DIR}")
    print(f"Non-text images: {NON_TEXT_DIR}")
    print(f"Total queued images: {len(all_records)}")
    print(f"OCR candidates: {ocr_count}")
    print(f"Table candidates: {table_count}")
    print(f"Non-text images: {non_text_count}")

    print("\nTop labels:")
    for label, count in label_counter.most_common(20):
        print(f"{label}: {count}")

    print("\nBuckets:")
    for bucket, count in bucket_counter.items():
        print(f"{bucket}: {count}")


if __name__ == "__main__":
    main()
