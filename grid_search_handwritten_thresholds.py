from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


OCR_QUEUE_DIR = Path(r"C:\Users\User\gazprom\output\ocr_queue")
HANDWRITTEN_GT_DIR = Path(r"C:\Users\User\gazprom\handwritten_buhands")

MANIFEST_IN = OCR_QUEUE_DIR / "text_split_manifest.csv"
FEATURES_CSV = OCR_QUEUE_DIR / "handwritten_grid_features.csv"
RESULTS_CSV = OCR_QUEUE_DIR / "handwritten_grid_search_results.csv"

EXPECTED_GT_COUNT = 40
DHASH_THRESHOLD = 6


@dataclass
class FeatureRow:
    file_name: str
    image_id: str
    predicted_label: str
    confidence: float
    width: int
    height: int
    tiny_dots: int
    contour_count: int
    small_components: int
    black_ratio: float
    is_handwritten_gt: int
    matched_gt_name: str


def load_manifest(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_features(img_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    black_ratio = float((bw == 0).mean())

    contours, _ = cv2.findContours(255 - bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(255 - bw, 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])

    small_components = int(np.sum((component_areas >= 5) & (component_areas <= 400)))
    tiny_dots = int(np.sum(component_areas <= 12))

    return {
        "tiny_dots": tiny_dots,
        "contour_count": contour_count,
        "small_components": small_components,
        "black_ratio": black_ratio,
    }


def dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as img:
        gray = img.convert("L")
        resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.int16)

    diff = pixels[:, 1:] > pixels[:, :-1]

    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_gt_hashes() -> list[tuple[str, int]]:
    if not HANDWRITTEN_GT_DIR.exists():
        raise FileNotFoundError(f"GT dir not found: {HANDWRITTEN_GT_DIR}")

    gt_hashes: list[tuple[str, int]] = []
    gt_files = sorted([p for p in HANDWRITTEN_GT_DIR.glob("*.*") if p.is_file()])

    for p in tqdm(gt_files, desc="Hashing GT"):
        try:
            gt_hashes.append((p.name, dhash(p)))
        except Exception:
            continue

    return gt_hashes


def match_gt(path: Path, gt_hashes: list[tuple[str, int]]) -> tuple[int, str]:
    try:
        current_hash = dhash(path)
    except Exception:
        return 0, ""

    best_name = ""
    best_dist = 10**9

    for gt_name, gt_hash in gt_hashes:
        dist = hamming_distance(current_hash, gt_hash)
        if dist < best_dist:
            best_dist = dist
            best_name = gt_name

    if best_dist <= DHASH_THRESHOLD:
        return 1, best_name

    return 0, ""


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_feature_table() -> list[FeatureRow]:
    gt_hashes = build_gt_hashes()
    manifest_rows = load_manifest(MANIFEST_IN)

    feature_rows: list[FeatureRow] = []

    for row in tqdm(manifest_rows, desc="Computing features"):
        saved_path = Path(row["saved_path"])
        if not saved_path.exists():
            continue

        img_bgr = cv2.imread(str(saved_path))
        if img_bgr is None:
            continue

        feats = compute_features(img_bgr)
        is_gt, matched_gt_name = match_gt(saved_path, gt_hashes)

        feature_rows.append(
            FeatureRow(
                file_name=saved_path.name,
                image_id=row["image_id"],
                predicted_label=row["predicted_label"],
                confidence=float(row["confidence"]),
                width=int(row["width"]),
                height=int(row["height"]),
                tiny_dots=int(feats["tiny_dots"]),
                contour_count=int(feats["contour_count"]),
                small_components=int(feats["small_components"]),
                black_ratio=float(feats["black_ratio"]),
                is_handwritten_gt=is_gt,
                matched_gt_name=matched_gt_name,
            )
        )

    save_csv([asdict(r) for r in feature_rows], FEATURES_CSV)
    return feature_rows


def predict_handwritten(
    row: FeatureRow,
    tiny_dots_max: int,
    contour_count_min: int,
    small_components_min: int,
    black_ratio_max: float,
) -> int:
    label = row.predicted_label.lower().strip()

    if label in {"handwritten", "handwritten_like"}:
        return 1

    if label != "small_text_or_handwritten_like":
        return 0

    if (
        row.tiny_dots <= tiny_dots_max
        and row.contour_count >= contour_count_min
        and row.small_components >= small_components_min
        and row.black_ratio <= black_ratio_max
    ):
        return 1

    return 0


def evaluate(feature_rows: list[FeatureRow]) -> list[dict]:
    results: list[dict] = []

    tiny_dots_grid = [300, 400, 500, 700, 900, 1100, 1300]
    contour_grid = [350, 450, 550, 650, 750, 850, 950]
    small_comp_grid = [100, 130, 160, 190, 220, 250, 280]
    black_ratio_grid = [0.14, 0.16, 0.18, 0.20, 0.22, 0.24]

    total = (
        len(tiny_dots_grid)
        * len(contour_grid)
        * len(small_comp_grid)
        * len(black_ratio_grid)
    )

    for tiny_dots_max, contour_count_min, small_components_min, black_ratio_max in tqdm(
        product(tiny_dots_grid, contour_grid, small_comp_grid, black_ratio_grid),
        desc="Grid search",
        total=total,
    ):
        tp = fp = fn = tn = 0

        for row in feature_rows:
            pred = predict_handwritten(
                row=row,
                tiny_dots_max=tiny_dots_max,
                contour_count_min=contour_count_min,
                small_components_min=small_components_min,
                black_ratio_max=black_ratio_max,
            )
            gt = row.is_handwritten_gt

            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 1:
                fn += 1
            else:
                tn += 1

        pred_count = tp + fp
        gt_count = tp + fn

        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        count_diff = abs(pred_count - EXPECTED_GT_COUNT)

        objective = (
            precision * 1000
            - count_diff * 10
            + recall * 100
            + f1 * 10
        )

        results.append(
            {
                "tiny_dots_max": tiny_dots_max,
                "contour_count_min": contour_count_min,
                "small_components_min": small_components_min,
                "black_ratio_max": black_ratio_max,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "pred_count": pred_count,
                "gt_count": gt_count,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "count_diff": count_diff,
                "objective": round(objective, 4),
            }
        )

    results.sort(
        key=lambda x: (
            -x["precision"],
            x["count_diff"],
            -x["recall"],
            -x["f1"],
            -x["objective"],
        )
    )

    save_csv(results, RESULTS_CSV)
    return results


def main() -> None:
    feature_rows = build_feature_table()

    gt_count = sum(r.is_handwritten_gt for r in feature_rows)
    print(f"GT handwritten matched by dhash: {gt_count}")

    results = evaluate(feature_rows)

    print()
    print(f"Saved features: {FEATURES_CSV}")
    print(f"Saved grid results: {RESULTS_CSV}")
    print()
    print("Top 10 threshold sets:")
    for row in results[:10]:
        print(row)


if __name__ == "__main__":
    main()
