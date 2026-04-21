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

DEBUG_MAX_PRINTS = 60
DEBUG_TABLE_LABEL_PRINTS = 0
DEBUG_TO_TABLE_PRINTS = 0
DEBUG_NEAR_TABLE_PRINTS = 0


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

    return feats


def is_monochrome_text_like_color(feats: dict[str, float]) -> bool:
    return (
        feats["mean_saturation"] < 18
        and feats["dominant_colors_count"] <= 3
        and feats["light_pixel_ratio"] > 0.55
        and feats["dark_pixel_ratio"] > 0.03
        and feats["dark_pixel_ratio"] < 0.35
    )


def is_colored_diagram_like(feats: dict[str, float]) -> bool:
    return (
        feats["area"] > 50000
        and feats["mean_saturation"] > 30
        and feats["high_saturation_ratio"] > 0.06
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
    global DEBUG_TABLE_LABEL_PRINTS, DEBUG_TO_TABLE_PRINTS, DEBUG_NEAR_TABLE_PRINTS

    label = (predicted_label or "").lower().strip()
    feats = build_bucket_features(img_bgr)

    if action == "drop":
        return "drop"

    if label == "table_like" and DEBUG_TABLE_LABEL_PRINTS < DEBUG_MAX_PRINTS:
        print(
            "DEBUG_TABLE_LABEL",
            label,
            f"h={feats['hline_ratio']:.4f}",
            f"v={feats['vline_ratio']:.4f}",
            f"row={feats['row_transitions']:.2f}",
            f"col={feats['col_transitions']:.2f}",
            f"sat={feats['mean_saturation']:.2f}",
            f"colors={int(feats['dominant_colors_count'])}",
        )
        DEBUG_TABLE_LABEL_PRINTS += 1

    near_table = (
        feats["hline_ratio"] > 0.004
        or feats["vline_ratio"] > 0.003
    )
    if near_table and DEBUG_NEAR_TABLE_PRINTS < DEBUG_MAX_PRINTS:
        print(
            "DEBUG_NEAR_TABLE",
            label,
            f"h={feats['hline_ratio']:.4f}",
            f"v={feats['vline_ratio']:.4f}",
            f"row={feats['row_transitions']:.2f}",
            f"col={feats['col_transitions']:.2f}",
            f"sat={feats['mean_saturation']:.2f}",
            f"colors={int(feats['dominant_colors_count'])}",
            f"small={int(feats['small_components'])}",
            f"contours={int(feats['contour_count'])}",
        )
        DEBUG_NEAR_TABLE_PRINTS += 1

    # 1. Явные цветные схемы/диаграммы
    if is_colored_diagram_like(feats):
        return "non_text"

    # 2. Таблицы
    if label == "table_like":
        if DEBUG_TO_TABLE_PRINTS < DEBUG_MAX_PRINTS:
            print("DEBUG_TO_TABLE label=table_like")
            DEBUG_TO_TABLE_PRINTS += 1
        return "table"

    if is_strict_table_candidate(feats):
        if DEBUG_TO_TABLE_PRINTS < DEBUG_MAX_PRINTS:
            print(
                "DEBUG_TO_TABLE strict",
                label,
                f"h={feats['hline_ratio']:.4f}",
                f"v={feats['vline_ratio']:.4f}",
                f"row={feats['row_transitions']:.2f}",
                f"col={feats['col_transitions']:.2f}",
                f"sat={feats['mean_saturation']:.2f}",
                f"colors={int(feats['dominant_colors_count'])}",
            )
            DEBUG_TO_TABLE_PRINTS += 1
        return "table"

    if is_soft_table_candidate(feats) and action == "ocr":
        if DEBUG_TO_TABLE_PRINTS < DEBUG_MAX_PRINTS:
            print(
                "DEBUG_TO_TABLE soft",
                label,
                f"h={feats['hline_ratio']:.4f}",
                f"v={feats['vline_ratio']:.4f}",
                f"row={feats['row_transitions']:.2f}",
                f"col={feats['col_transitions']:.2f}",
                f"sat={feats['mean_saturation']:.2f}",
                f"colors={int(feats['dominant_colors_count'])}",
            )
            DEBUG_TO_TABLE_PRINTS += 1
        return "table"

    # 3. Всё OCR-подобное и монохромные текстовые блоки
    if action == "ocr":
        return "ocr"

    if is_monochrome_text_like_color(feats):
        return "ocr"

    # 4. Остальное считаем нетекстовыми картинками
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
