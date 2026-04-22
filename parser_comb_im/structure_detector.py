#!/usr/bin/env python3
"""
Detect document structures in a JPG and save each as a separate cropped file.

Pipeline (fixed order)
----------------------
0. Preprocess  — denoise + contrast normalisation before any model
1. Table Transformer (Microsoft) — find tables on clean image
2. EasyOCR     — OCR only on image with tables masked out → no table text leaks
3. Contour     — find figures in areas not covered by text or tables

Output filenames
----------------
  text_001.jpg   title_001.jpg   table_001.jpg   figure_001.jpg

Requirements (already installed):
    easyocr>=1.7   transformers>=4.38   torch>=2.0
    Pillow>=10     opencv-python-headless>=4.8

Usage
-----
    python structure_detector.py photo.jpg
    python structure_detector.py photo.jpg -o ./crops --lang en ru
    python structure_detector.py photo.jpg --table-threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import easyocr
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Run: pip install easyocr transformers torch Pillow opencv-python-headless"
    )

# ── tunables ────────────────────────────────────────────────────────────────
TABLE_MODEL    = "microsoft/table-transformer-detection"
PADDING_PX     = 6       # extra pixels around every crop
MIN_DIM_PX     = 20      # discard crops smaller than this in any dimension
TITLE_MIN_W    = 300     # merged text block wider than this → "title"
TITLE_MAX_H    = 60      # …and shorter than this → "title"
MIN_FIG_AREA   = 40_000  # minimum figure area in px²
MAX_FIG_FRAC   = 0.55    # skip regions covering > 55 % of the full image
FIG_FILL_FRAC  = 0.10    # skip regions with < 10 % non-white pixels
OCR_MERGE_GAP  = 14      # vertical gap (px) to merge adjacent text lines
# ────────────────────────────────────────────────────────────────────────────


# ── Step 0: preprocessing ────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Denoise + CLAHE contrast normalisation.
    Returns a cleaned BGR image suitable for both Table Transformer and EasyOCR.
    """
    # Fast bilateral denoise: preserves edges better than Gaussian
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Per-channel CLAHE for contrast normalisation
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    clean = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clean


# ── Step 1: Table Transformer ────────────────────────────────────────────────

def _detect_tables(img_pil: Image.Image, threshold: float) -> list[dict]:
    """Detect tables using microsoft/table-transformer-detection."""
    processor = AutoImageProcessor.from_pretrained(TABLE_MODEL)
    model     = TableTransformerForObjectDetection.from_pretrained(TABLE_MODEL)
    model.eval()

    inputs = processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img_pil.size[::-1]])  # (H, W)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    tables = []
    for box, lid in zip(results["boxes"], results["labels"]):
        x1, y1, x2, y2 = (int(v) for v in box.tolist())
        label = model.config.id2label[lid.item()].split()[0]  # "table"
        tables.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return tables


# ── Step 2: EasyOCR on table-masked image ───────────────────────────────────

def _mask_regions(img_bgr: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Paint detected table areas white so EasyOCR ignores them."""
    masked = img_bgr.copy()
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        masked[y1:y2, x1:x2] = 255
    return masked


def _ocr_to_raw_blocks(raw: list, min_conf: float) -> list[dict]:
    blocks = []
    for pts, text, conf in raw:
        if conf < min_conf or not text.strip():
            continue
        arr = np.array(pts, dtype=int)
        x1, y1 = arr.min(axis=0).tolist()
        x2, y2 = arr.max(axis=0).tolist()
        blocks.append({"label": "text", "bbox": [x1, y1, x2, y2]})
    return blocks


def _merge_text_blocks(blocks: list[dict]) -> list[dict]:
    """
    Merge EasyOCR word-level detections into paragraph-level blocks.
    Uses the median line height to set a dynamic vertical merge gap.
    """
    if not blocks:
        return []

    heights = [b["bbox"][3] - b["bbox"][1] for b in blocks]
    med_h   = float(np.median(heights)) if heights else OCR_MERGE_GAP
    v_gap   = max(OCR_MERGE_GAP, int(med_h * 0.8))   # ~80 % of a line height
    h_gap   = max(40, int(med_h * 2.0))               # wide horizontal tolerance

    items  = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    merged = [dict(items[0])]

    for cur in items[1:]:
        prev = merged[-1]
        px1, py1, px2, py2 = prev["bbox"]
        cx1, cy1, cx2, cy2 = cur["bbox"]

        h_overlap = cx1 < px2 + h_gap and cx2 > px1 - h_gap
        v_close   = cy1 <= py2 + v_gap

        if h_overlap and v_close:
            prev["bbox"] = [
                min(px1, cx1), min(py1, cy1),
                max(px2, cx2), max(py2, cy2),
            ]
        else:
            merged.append(dict(cur))

    # Classify wide short blocks as "title"
    for b in merged:
        x1, y1, x2, y2 = b["bbox"]
        w, h = x2 - x1, y2 - y1
        if w >= TITLE_MIN_W and h <= TITLE_MAX_H:
            b["label"] = "title"

    return merged


# ── Step 3: figure detection ─────────────────────────────────────────────────

def _covered_mask(h: int, w: int, regions: list[dict]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        mask[y1:y2, x1:x2] = 1
    return mask


def _detect_figures(img_bgr: np.ndarray, covered: np.ndarray) -> list[dict]:
    """
    Find large non-text, non-table blobs — likely figures / charts / photos.
    Works on the CLEAN (preprocessed) image.
    """
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle varying backgrounds
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # Remove already-covered (text + table) areas
    free = bin_img.copy()
    free[covered == 1] = 0

    # Close gaps inside the same figure
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilated = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    figures = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_FIG_AREA:
            continue
        if area > img_area * MAX_FIG_FRAC:
            continue
        fill = bin_img[y: y + h, x: x + w].sum() / 255 / area
        if fill < FIG_FILL_FRAC:
            continue
        figures.append({"label": "figure", "bbox": [x, y, x + w, y + h]})

    return figures


# ── NMS helper ───────────────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _suppress_duplicates(regions: list[dict], iou_thr: float = 0.4) -> list[dict]:
    """Tables beat text/figure; title beats text."""
    priority = {"table": 0, "title": 1, "text": 2, "figure": 3}
    ordered  = sorted(regions, key=lambda r: priority.get(r["label"], 9))
    kept: list[dict] = []
    for r in ordered:
        if not any(_iou(r["bbox"], k["bbox"]) > iou_thr for k in kept):
            kept.append(r)
    return kept


# ── Main ─────────────────────────────────────────────────────────────────────

def detect_and_crop(
    image_path: str,
    output_dir: str | None = None,
    languages: list[str] | None = None,
    table_threshold: float = 0.7,
    ocr_confidence: float = 0.3,
) -> list[str]:
    languages = languages or ["en"]

    src = Path(image_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Not found: {src}")

    dst = (
        Path(output_dir).resolve()
        if output_dir
        else src.parent / f"{src.stem}_structures"
    )
    dst.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(src))
    if img_bgr is None:
        raise ValueError(f"OpenCV cannot read: {src}")

    img_h, img_w = img_bgr.shape[:2]
    print(f"Image : {src.name}  ({img_w}×{img_h} px)")
    print(f"Output: {dst}\n")

    # ── 0. Preprocess ────────────────────────────────────────────────────────
    print("[0/3] Preprocessing (denoise + contrast) …")
    clean_bgr = preprocess(img_bgr)
    clean_pil = Image.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))

    # ── 1. Table detection on clean image ────────────────────────────────────
    print(f"[1/3] Table Transformer  threshold={table_threshold} …")
    tables = _detect_tables(clean_pil, threshold=table_threshold)
    print(f"      {len(tables)} table(s): {[t['bbox'] for t in tables]}")

    # ── 2. EasyOCR on image with tables masked out ───────────────────────────
    print(f"[2/3] EasyOCR  lang={languages} (tables masked) …")
    masked_bgr = _mask_regions(clean_bgr, tables)
    gpu    = torch.cuda.is_available()
    reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
    raw    = reader.readtext(masked_bgr, paragraph=False)
    text_blocks = _ocr_to_raw_blocks(raw, min_conf=ocr_confidence)
    text_blocks = _merge_text_blocks(text_blocks)
    print(f"      {len(text_blocks)} text block(s)")

    # ── 3. Figure detection in uncovered areas ───────────────────────────────
    print("[3/3] Figure detection …")
    covered = _covered_mask(img_h, img_w, text_blocks + tables)
    figures = _detect_figures(clean_bgr, covered)
    print(f"      {len(figures)} figure(s)")

    # ── Combine → NMS → sort → crop ─────────────────────────────────────────
    all_regions = _suppress_duplicates(tables + text_blocks + figures)
    all_regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    counters: dict[str, int] = {}
    saved: list[str] = []

    for region in all_regions:
        label = region["label"]
        x1, y1, x2, y2 = region["bbox"]
        x1 = max(0, x1 - PADDING_PX)
        y1 = max(0, y1 - PADDING_PX)
        x2 = min(img_w, x2 + PADDING_PX)
        y2 = min(img_h, y2 + PADDING_PX)

        if (x2 - x1) < MIN_DIM_PX or (y2 - y1) < MIN_DIM_PX:
            continue

        counters[label] = counters.get(label, 0) + 1
        fname = f"{label}_{counters[label]:03d}.jpg"
        crop  = img_bgr[y1:y2, x1:x2]   # crop from ORIGINAL (unprocessed) image
        cv2.imwrite(str(dst / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(str(dst / fname))
        print(f"  [{label:<8s}]  {fname}  ({x2-x1}×{y2-y1} px)")

    print(f"\n✓ {len(saved)} structure(s) saved → {dst}")
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect text / title / table / figure in a JPG and crop each region."
    )
    ap.add_argument("image", help="Path to input JPG")
    ap.add_argument("-o", "--output", metavar="DIR",
                    help="Output directory (default: <stem>_structures/)")
    ap.add_argument("--lang", nargs="+", default=["en"], metavar="CODE",
                    help="EasyOCR languages, e.g. --lang en ru  (default: en)")
    ap.add_argument("--table-threshold", type=float, default=0.7, metavar="F",
                    help="Table Transformer confidence 0–1 (default: 0.7)")
    ap.add_argument("--ocr-confidence", type=float, default=0.3, metavar="F",
                    help="Min EasyOCR word confidence (default: 0.3)")
    args = ap.parse_args()

    try:
        detect_and_crop(
            args.image, args.output, args.lang,
            args.table_threshold, args.ocr_confidence,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()