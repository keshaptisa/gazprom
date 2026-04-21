from __future__ import annotations

import csv
import hashlib
import io
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm


PDF_DIR = Path(r"C:\Users\User\gazprom\pdfs")
OUT_DIR = Path(r"C:\Users\User\gazprom\output\ocr_queue")

ALL_DIR = OUT_DIR / "all_images"
HANDWRITTEN_DIR = OUT_DIR / "handwritten"
PRINTED_DIR = OUT_DIR / "printed"

MANIFEST_CSV = OUT_DIR / "image_manifest.csv"
FINAL_CSV = OUT_DIR / "image_manifest_with_text.csv"
OCR_RESULTS_CSV = OUT_DIR / "ocr_results.csv"


@dataclass
class ImageRecord:
    image_id: str
    pdf_file: str
    page_number: int
    image_index: int
    class_label: str
    width: int
    height: int
    saved_path: str
    ocr_text: str = ""


def ensure_dirs() -> None:
    for path in [OUT_DIR, ALL_DIR, HANDWRITTEN_DIR, PRINTED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def file_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def safe_open_image(data: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def fallback_classify(img: Image.Image) -> str:
    width, height = img.size
    ratio = width / max(height, 1)

    if ratio > 2.5:
        return "handwritten_or_text_like"
    return "image_like"


def resolve_project_classifier() -> Callable[[Image.Image], str] | None:
    try:
        from pdf_image_parser.image_extractor import classify_image_block  # type: ignore

        def wrapper(img: Image.Image) -> str:
            try:
                return str(classify_image_block(img))
            except Exception:
                return fallback_classify(img)

        return wrapper
    except Exception:
        return None


def normalize_label(label: str) -> str:
    lower = label.lower()

    if any(x in lower for x in ["hand", "рукоп", "small_text_or_handwritten_like"]):
        return "handwritten"
    if any(x in lower for x in ["scan", "printed", "text", "rasterized_pdf"]):
        return "printed"
    if "image_with_table" in lower:
        return "printed"
    return label


def route_copy(saved_path: Path, class_label: str) -> None:
    normalized = normalize_label(class_label)
    if normalized == "handwritten":
        shutil.copy2(saved_path, HANDWRITTEN_DIR / saved_path.name)
    elif normalized == "printed":
        shutil.copy2(saved_path, PRINTED_DIR / saved_path.name)


def extract_images_from_pdf(
    pdf_path: Path,
    classifier: Callable[[Image.Image], str] | None,
) -> list[ImageRecord]:
    doc = fitz.open(pdf_path)
    records: list[ImageRecord] = []

    page_iter = tqdm(
        range(len(doc)),
        desc=f"Pages {pdf_path.name}",
        leave=False,
    )

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

            width, height = pil_img.size
            image_hash = file_sha1(image_bytes)
            image_id = f"{pdf_path.stem}_p{page_index + 1:03d}_i{image_index:03d}_{image_hash}"
            file_name = f"{image_id}.{ext}"
            saved_path = ALL_DIR / file_name

            with open(saved_path, "wb") as f:
                f.write(image_bytes)

            if classifier is not None:
                class_label = classifier(pil_img)
            else:
                class_label = fallback_classify(pil_img)

            route_copy(saved_path, class_label)

            records.append(
                ImageRecord(
                    image_id=image_id,
                    pdf_file=pdf_path.name,
                    page_number=page_index + 1,
                    image_index=image_index,
                    class_label=class_label,
                    width=width,
                    height=height,
                    saved_path=str(saved_path),
                )
            )

            page_iter.set_postfix({"images": len(records)})

    return records


def save_manifest(records: list[ImageRecord], csv_path: Path) -> None:
    if not records:
        return

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_ocr_results(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}

    result: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            text = (row.get("ocr_text") or "").strip()
            if image_id:
                result[image_id] = text
    return result


def merge_ocr_text(records: list[ImageRecord], ocr_map: dict[str, str]) -> list[ImageRecord]:
    merged: list[ImageRecord] = []
    for record in records:
        record.ocr_text = ocr_map.get(record.image_id, "")
        merged.append(record)
    return merged


def main() -> None:
    ensure_dirs()

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF dir not found: {PDF_DIR}")

    classifier = resolve_project_classifier()
    if classifier is None:
        print("Project classifier import failed, using fallback classifier.")
    else:
        print("Using project classifier from pdf_image_parser.image_extractor.")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    all_records: list[ImageRecord] = []

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in tqdm(pdf_files, desc="PDFs"):
        records = extract_images_from_pdf(pdf_path, classifier)
        all_records.extend(records)

    save_manifest(all_records, MANIFEST_CSV)

    ocr_map = load_ocr_results(OCR_RESULTS_CSV)
    merged_records = merge_ocr_text(all_records, ocr_map)
    save_manifest(merged_records, FINAL_CSV)

    print()
    print(f"Saved manifest: {MANIFEST_CSV}")
    print(f"Saved merged manifest: {FINAL_CSV}")
    print(f"All images: {ALL_DIR}")
    print(f"Handwritten images: {HANDWRITTEN_DIR}")
    print(f"Printed images: {PRINTED_DIR}")
    print(f"Total extracted images: {len(all_records)}")
    print(f"OCR texts loaded from table: {sum(1 for r in merged_records if r.ocr_text.strip())}")


if __name__ == "__main__":
    main()
