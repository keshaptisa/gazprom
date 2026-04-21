from __future__ import annotations

import csv
import hashlib
import io
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm

from pdf_image_parser.image_extractor import DiTClassifier, classify_image, compute_visual_features


PDF_DIR = Path(r"C:\Users\User\gazprom\pdfs")
OUT_DIR = Path(r"C:\Users\User\gazprom\output\ocr_queue")

ALL_DIR = OUT_DIR / "all_images"
OCR_TEXT_DIR = OUT_DIR / "ocr_text_images"
TABLE_DIR = OUT_DIR / "table_images"
NON_TEXT_DIR = OUT_DIR / "non_text_images"

MANIFEST_CSV = OUT_DIR / "ocr_queue_manifest.csv"


@dataclass
class ImageRecord:
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
    OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)
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


def save_manifest(records: list[ImageRecord], csv_path: Path) -> None:
    if not records:
        return

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def looks_like_table_second_pass(img_bgr: np.ndarray) -> bool:
    feats = compute_visual_features(img_bgr)

    if (
        feats["area"] > 80000
        and feats["hline_ratio"] > 0.015
        and feats["vline_ratio"] > 0.010
        and feats["small_components"] > 120
    ):
        return True

    if (
        feats["hline_ratio"] > 0.025
        and feats["vline_ratio"] > 0.015
    ):
        return True

    return False


def is_text_candidate(action: str, predicted_label: str) -> bool:
    if action != "ocr":
        return False

    label = (predicted_label or "").lower().strip()

    if label == "table_like":
        return False

    if "table" in label:
        return False

    if label in {
        "handwritten",
        "handwritten_like",
        "small_text_or_handwritten_like",
        "printed_text_like",
        "text_like_dense",
        "invoice",
        "memo",
        "letter",
        "rasterized_pdf",
        "scan_image",
        "unknown_large",
    }:
        return True

    if any(x in label for x in ["handwritten", "text", "scan", "invoice", "memo", "letter", "raster"]):
        return True

    return False


def initial_bucket(action: str, predicted_label: str) -> str:
    label = (predicted_label or "").lower().strip()

    if action == "drop":
        return "drop"

    if label == "table_like" or "table" in label:
        return "table"

    if is_text_candidate(action, predicted_label):
        return "ocr_text"

    return "non_text"


def final_bucket_with_second_pass(
    img_bgr: np.ndarray,
    action: str,
    predicted_label: str,
) -> str:
    bucket = initial_bucket(action, predicted_label)

    if bucket == "ocr_text":
        if looks_like_table_second_pass(img_bgr):
            return "table"

    return bucket


def copy_to_bucket(saved_path: Path, file_name: str, bucket: str) -> Path | None:
    if bucket == "ocr_text":
        target = OCR_TEXT_DIR / file_name
    elif bucket == "table":
        target = TABLE_DIR / file_name
    elif bucket == "non_text":
        target = NON_TEXT_DIR / file_name
    else:
        return None

    shutil.copy2(saved_path, target)
    return target


def extract_queue_from_pdf(pdf_path: Path, dit: DiTClassifier) -> list[ImageRecord]:
    doc = fitz.open(pdf_path)
    records: list[ImageRecord] = []

    page_iter = tqdm(range(len(doc)), desc=f"Pages {pdf_path.name}", leave=False)

    for page_index in page_iter:
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

            bucket = final_bucket_with_second_pass(
                img_bgr=img_bgr,
                action=action,
                predicted_label=predicted_label,
            )

            if bucket == "drop":
                continue

            final_path = copy_to_bucket(saved_path, file_name, bucket)
            if final_path is None:
                continue

            records.append(
                ImageRecord(
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

            page_iter.set_postfix({
                "queued": len(records),
                "last": bucket,
            })

    return records


def main() -> None:
    reset_output()

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF dir not found: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    dit = DiTClassifier()
    all_records: list[ImageRecord] = []

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        records = extract_queue_from_pdf(pdf_path, dit)
        all_records.extend(records)

    save_manifest(all_records, MANIFEST_CSV)

    ocr_text_count = sum(1 for r in all_records if r.final_bucket == "ocr_text")
    table_count = sum(1 for r in all_records if r.final_bucket == "table")
    non_text_count = sum(1 for r in all_records if r.final_bucket == "non_text")

    print()
    print(f"Saved manifest: {MANIFEST_CSV}")
    print(f"All images: {ALL_DIR}")
    print(f"OCR text images: {OCR_TEXT_DIR}")
    print(f"Table images: {TABLE_DIR}")
    print(f"Non-text images: {NON_TEXT_DIR}")
    print(f"Total queued images: {len(all_records)}")
    print(f"OCR text count: {ocr_text_count}")
    print(f"Table count: {table_count}")
    print(f"Non-text count: {non_text_count}")


if __name__ == "__main__":
    main()
