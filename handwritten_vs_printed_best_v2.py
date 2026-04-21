from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def classify_handwritten_vs_printed(img_bgr: np.ndarray) -> tuple[str, float, dict]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = bw.shape[:2]
    if h < 40 or w < 80:
        return "printed", 0.0, {"reason": "too_small"}

    row_proj = bw.mean(axis=1)
    row_peaks = row_proj > row_proj.mean()
    row_transitions = int(np.sum(row_peaks[1:] != row_peaks[:-1]))
    transition_density = row_transitions / max(h, 1)

    col_proj = bw.mean(axis=0)
    col_peaks = col_proj > col_proj.mean()
    col_transitions = int(np.sum(col_peaks[1:] != col_peaks[:-1]))
    col_transition_density = col_transitions / max(w, 1)

    dist_transform = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    stroke_pixels = dist_transform[bw > 0]
    if len(stroke_pixels) > 100:
        stroke_mean = float(np.mean(stroke_pixels))
        stroke_std = float(np.std(stroke_pixels))
        stroke_cv = stroke_std / (stroke_mean + 1e-6)
    else:
        stroke_mean = 0.0
        stroke_std = 0.0
        stroke_cv = 0.0

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    valid_boxes = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 8:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 2 or ch < 2:
            continue

        valid_boxes += 1

        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            angles.append(float(ellipse[2]))

    angle_std = float(np.std(angles)) if angles else 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    small_components = int(np.sum((comp_areas >= 5) & (comp_areas <= 120)))
    medium_components = int(np.sum((comp_areas > 120) & (comp_areas <= 1200)))

    ink_ratio = float((bw > 0).mean())

    score = 0.0

    if stroke_cv > 0.50:
        score += 2.0
    elif stroke_cv > 0.40:
        score += 1.0

    if angle_std > 28:
        score += 2.0
    elif angle_std > 18:
        score += 1.0

    if transition_density < 0.05:
        score += 1.0

    if col_transition_density < 0.18:
        score += 1.0

    if medium_components > small_components * 0.18:
        score += 1.0

    if small_components > 250 and angle_std < 15:
        score -= 1.5

    if stroke_cv < 0.28:
        score -= 1.5

    if col_transition_density > 0.24:
        score -= 1.0

    if small_components > 180 and angle_std < 20:
        score -= 1.0

    if valid_boxes > 140 and stroke_cv < 0.42:
        score -= 1.0

    if small_components > 220 and transition_density > 0.045:
        score -= 0.8

    if medium_components <= max(8, small_components * 0.10):
        score -= 0.7

    printed_override = (
        (
            stroke_cv < 0.34
            and angle_std > 25
            and col_transition_density > 0.12
        )
        or (
            stroke_cv < 0.34
            and medium_components < max(18, small_components * 0.35)
        )
        or (
            small_components > 170
            and angle_std < 18
            and stroke_cv < 0.48
            and col_transition_density > 0.16
        )
    )

    handwritten_rescue = (
        score >= 3.0
        and stroke_cv > 0.35
        and angle_std > 30
        and medium_components > max(20, small_components * 0.75)
    )

    if printed_override:
        label = "printed"
    elif score >= 4.0:
        label = "handwritten"
    elif handwritten_rescue:
        label = "handwritten"
    else:
        label = "printed"

    features = {
        "transition_density": transition_density,
        "col_transition_density": col_transition_density,
        "stroke_mean": stroke_mean,
        "stroke_std": stroke_std,
        "stroke_cv": stroke_cv,
        "angle_std": angle_std,
        "small_components": small_components,
        "medium_components": medium_components,
        "ink_ratio": ink_ratio,
        "valid_boxes": valid_boxes,
        "reason": "threshold_4_filtered_rescue_v2",
        "printed_override": int(printed_override),
        "handwritten_rescue": int(handwritten_rescue),
    }

    return label, score, features


def run_folder(input_dir: str, out_csv: str) -> pd.DataFrame:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    rows = []

    for path in sorted(Path(input_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        label, score, feats = classify_handwritten_vs_printed(img)

        row = {
            "file_name": path.name,
            "source_path": str(path),
            "predicted_label": label,
            "score": score,
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    input_dir = r"C:\Users\User\gazprom\output\ocr_queue\ocr_candidates"
    out_csv = r"C:\Users\User\gazprom\output\ocr_queue\handwritten_vs_printed_best_v2.csv"

    df = run_folder(input_dir, out_csv)

    if df.empty:
        print("No images found.")
        return

    print(df["predicted_label"].value_counts())
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
