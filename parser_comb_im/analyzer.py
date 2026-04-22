"""
Layout analyzer: detect text / table / figure regions in an image,
crop each one, save them into class-specific subfolders.

Uses only open-source models:
    * EasyOCR                                     -> word-level text
    * Microsoft Table Transformer (transformers)  -> visual-table hints
    * OpenCV grid / contour analysis              -> grid-based tables + figures

Pipeline:
    0. Denoise (bilateralFilter) + CLAHE contrast normalisation
    1. EasyOCR on clean image -> word-level boxes
    2. Table detection:
        a) grid pattern on word boxes (short cells, >=4 cols, >=3 rows)
        b) Table Transformer — validated against word-density / structure
       Grid and TT results merged via NMS.
    3. Strip words inside tables, cluster remaining words into lines,
       then merge lines into paragraphs (respecting columns).
    4. Figure detection in uncovered regions (contours).
    5. NMS (tables > text > figure) + crop + save.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import easyocr
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

_TABLE_MODEL = "microsoft/table-transformer-detection"

# tunables
_PADDING_PX      = 6
_MIN_DIM_PX      = 20
_MIN_FIG_AREA    = 40_000
_MAX_FIG_FRAC    = 0.55
_FIG_FILL_FRAC   = 0.10

_TABLE_MIN_ROWS  = 3        # min seed rows with >= MIN_COLS cells
_TABLE_MIN_COLS  = 3        # min cells per seed row
_TABLE_ROW_GAP   = 5        # allow this many non-seed rows between seeds
_TABLE_CELL_GAP  = 1.6      # gap / median-height threshold for "new cell"
_TABLE_MAX_CELLW = 180      # median cell width limit (distinguishes from multi-col text)
_TABLE_MIN_WORDS = 8        # a table must contain this many word boxes

_LINE_GAP        = 1.8      # horizontal gap / med-height → word boundary on line
_PARA_V_GAP      = 1.3      # vertical gap / min(local line height) → new paragraph
_PARA_H_OVERLAP  = 0.15     # min horizontal overlap fraction for paragraph merge
_PARA_H_RATIO    = 2.2      # max height ratio between mergeable lines (heading guard)

_COLORS = {
    "text":   (0, 200, 0),
    "figure": (255, 100, 0),
    "table":  (0, 0, 255),
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside_fraction(inner: list[int], outer: list[int]) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter = max(0, min(ix2, ox2) - max(ix1, ox1)) * max(0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / inner_area


def _cluster_rows(blocks: list[dict], tol_factor: float = 0.6) -> list[list[dict]]:
    """Cluster boxes into rows by y-center."""
    if not blocks:
        return []
    items = sorted(blocks, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)
    heights = [b["bbox"][3] - b["bbox"][1] for b in items]
    med_h = float(np.median(heights)) if heights else 20.0
    tol = max(8.0, med_h * tol_factor)

    rows: list[list[dict]] = [[items[0]]]
    for b in items[1:]:
        y_prev = np.mean([(x["bbox"][1] + x["bbox"][3]) / 2 for x in rows[-1]])
        y_cur = (b["bbox"][1] + b["bbox"][3]) / 2
        if abs(y_cur - y_prev) <= tol:
            rows[-1].append(b)
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b["bbox"][0])
    return rows


def _row_cells(row_blocks: list[dict], gap_factor: float = _TABLE_CELL_GAP) -> list[list[dict]]:
    """Split a row into cells (clusters of words separated by large horizontal gaps)."""
    if not row_blocks:
        return []
    heights = [b["bbox"][3] - b["bbox"][1] for b in row_blocks]
    med_h = float(np.median(heights)) if heights else 20.0
    gap_thr = med_h * gap_factor

    cells: list[list[dict]] = [[row_blocks[0]]]
    for b in row_blocks[1:]:
        gap = b["bbox"][0] - cells[-1][-1]["bbox"][2]
        if gap > gap_thr:
            cells.append([b])
        else:
            cells[-1].append(b)
    return cells


def _cell_bbox(cell: list[dict]) -> list[int]:
    xs = [b["bbox"][0] for b in cell] + [b["bbox"][2] for b in cell]
    ys = [b["bbox"][1] for b in cell] + [b["bbox"][3] for b in cell]
    return [min(xs), min(ys), max(xs), max(ys)]


# ── LayoutAnalyzer ──────────────────────────────────────────────────────────

class LayoutAnalyzer:
    """Open-source layout analyzer (no PaddleOCR)."""

    def __init__(self, languages: list[str] | None = None,
                 table_threshold: float = 0.5,
                 ocr_confidence: float = 0.3,
                 use_table_transformer: bool = True) -> None:
        self.languages = languages or ["en", "ru"]
        self.table_threshold = table_threshold
        self.ocr_confidence = ocr_confidence
        self.use_tt = use_table_transformer

        gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)

        if self.use_tt:
            self.processor = AutoImageProcessor.from_pretrained(_TABLE_MODEL)
            self.model = TableTransformerForObjectDetection.from_pretrained(_TABLE_MODEL)
            self.model.eval()

    # ── step 0: preprocessing ───────────────────────────────────────────────
    @staticmethod
    def preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """Bilateral denoise + per-channel CLAHE contrast normalisation."""
        denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    # ── step 1: OCR ─────────────────────────────────────────────────────────
    def _ocr_words(self, img_bgr: np.ndarray) -> list[dict]:
        raw = self.reader.readtext(img_bgr, paragraph=False)
        words: list[dict] = []
        for pts, text, conf in raw:
            if conf < self.ocr_confidence or not text.strip():
                continue
            arr = np.array(pts, dtype=int)
            x1, y1 = arr.min(axis=0).tolist()
            x2, y2 = arr.max(axis=0).tolist()
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            words.append({"class": "text", "bbox": [x1, y1, x2, y2], "text": text})
        return words

    # ── step 2: table detection ─────────────────────────────────────────────
    @staticmethod
    def _detect_tables_from_grid(words: list[dict]) -> list[dict]:
        """
        Grid-based table detection.
        Find 'seed' rows (>=MIN_COLS cells). Group seeds that are within
        ROW_GAP row-positions of each other (tolerating multi-line cell
        continuations). A group of >=MIN_ROWS seeds whose median cell width
        is small (< MAX_CELLW) and has >=MIN_WORDS words becomes a table.
        """
        rows = _cluster_rows(words)
        if len(rows) < _TABLE_MIN_ROWS:
            return []

        cells_per_row = [_row_cells(r) for r in rows]
        seeds = [i for i, cs in enumerate(cells_per_row) if len(cs) >= _TABLE_MIN_COLS]
        if len(seeds) < _TABLE_MIN_ROWS:
            return []

        # Group seeds with allowed gap
        groups: list[list[int]] = [[seeds[0]]]
        for s in seeds[1:]:
            if s - groups[-1][-1] <= _TABLE_ROW_GAP:
                groups[-1].append(s)
            else:
                groups.append([s])

        tables: list[dict] = []
        for group in groups:
            if len(group) < _TABLE_MIN_ROWS:
                continue
            first, last = group[0], group[-1]

            all_words: list[dict] = []
            for k in range(first, last + 1):
                all_words.extend(rows[k])
            if len(all_words) < _TABLE_MIN_WORDS:
                continue

            cell_widths: list[int] = []
            for k in range(first, last + 1):
                for cell in cells_per_row[k]:
                    bx = _cell_bbox(cell)
                    cell_widths.append(bx[2] - bx[0])
            if not cell_widths or float(np.median(cell_widths)) >= _TABLE_MAX_CELLW:
                continue

            xs = [w["bbox"][0] for w in all_words] + [w["bbox"][2] for w in all_words]
            ys = [w["bbox"][1] for w in all_words] + [w["bbox"][3] for w in all_words]
            tables.append({
                "class": "table",
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })
        return tables

    def _detect_tables_tt(self, img_pil: Image.Image) -> list[dict]:
        """Table Transformer boxes (raw, to be validated)."""
        if not self.use_tt:
            return []
        inputs = self.processor(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([img_pil.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.table_threshold, target_sizes=target_sizes,
        )[0]
        return [
            {"class": "table", "bbox": [int(v) for v in box.tolist()]}
            for box in results["boxes"]
        ]

    @staticmethod
    def _validate_table(table: dict, words: list[dict]) -> bool:
        """Keep a candidate only if it looks like a real table."""
        inside = [w for w in words if _inside_fraction(w["bbox"], table["bbox"]) > 0.5]
        if len(inside) < _TABLE_MIN_WORDS:
            return False
        rows = _cluster_rows(inside)
        if len(rows) < 2:
            return False
        cell_widths: list[int] = []
        max_cols = 0
        for row in rows:
            cells = _row_cells(row)
            max_cols = max(max_cols, len(cells))
            for cell in cells:
                bx = _cell_bbox(cell)
                cell_widths.append(bx[2] - bx[0])
        if max_cols < 2 or not cell_widths:
            return False
        return float(np.median(cell_widths)) < _TABLE_MAX_CELLW

    # ── step 3: text merging ────────────────────────────────────────────────
    @staticmethod
    def _merge_words_to_lines(words: list[dict]) -> list[dict]:
        """Group words in same row into line-bboxes, split on wide horizontal gaps."""
        rows = _cluster_rows(words, tol_factor=0.5)
        lines: list[dict] = []
        for row in rows:
            if not row:
                continue
            heights = [b["bbox"][3] - b["bbox"][1] for b in row]
            med_h = float(np.median(heights)) if heights else 20.0
            gap_thr = med_h * _LINE_GAP

            segment: list[dict] = [row[0]]
            for b in row[1:]:
                if b["bbox"][0] - segment[-1]["bbox"][2] <= gap_thr:
                    segment.append(b)
                else:
                    lines.append({"class": "text", "bbox": _cell_bbox(segment)})
                    segment = [b]
            lines.append({"class": "text", "bbox": _cell_bbox(segment)})
        return lines

    @staticmethod
    def _merge_lines_to_paragraphs(lines: list[dict]) -> list[dict]:
        """Merge vertically-adjacent lines sharing a column into paragraphs."""
        if not lines:
            return []
        # Stamp each line with its intrinsic line height — survives merges.
        items = []
        for b in lines:
            d = dict(b)
            d["line_h"] = max(1, b["bbox"][3] - b["bbox"][1])
            items.append(d)
        items.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(items):
                j = i + 1
                while j < len(items):
                    ax1, ay1, ax2, ay2 = items[i]["bbox"]
                    bx1, by1, bx2, by2 = items[j]["bbox"]
                    a_lh = items[i]["line_h"]
                    b_lh = items[j]["line_h"]

                    if by1 < ay1:
                        ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = (
                            bx1, by1, bx2, by2, ax1, ay1, ax2, ay2
                        )

                    # Guard against heading↔body merges via line-height disparity.
                    if max(a_lh, b_lh) / min(a_lh, b_lh) > _PARA_H_RATIO:
                        j += 1
                        continue

                    local_h = max(min(a_lh, b_lh), 10)
                    v_gap_thr = max(6.0, local_h * _PARA_V_GAP)

                    v_gap = by1 - ay2
                    h_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                    min_w = max(1, min(ax2 - ax1, bx2 - bx1))

                    close_vertically = (-local_h <= v_gap <= v_gap_thr)
                    same_column = h_overlap / min_w >= _PARA_H_OVERLAP

                    if close_vertically and same_column:
                        items[i]["bbox"] = [
                            min(ax1, bx1), min(ay1, by1),
                            max(ax2, bx2), max(ay2, by2),
                        ]
                        items[i]["line_h"] = min(a_lh, b_lh)
                        items.pop(j)
                        changed = True
                        continue
                    j += 1
                i += 1
        # Drop the helper field before returning.
        for b in items:
            b.pop("line_h", None)
        return items

    # ── step 4: figures ─────────────────────────────────────────────────────
    @staticmethod
    def _covered_mask(h: int, w: int, regions: list[dict]) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return mask

    @staticmethod
    def _detect_figures(img_bgr: np.ndarray, covered: np.ndarray) -> list[dict]:
        img_h, img_w = img_bgr.shape[:2]
        img_area = img_h * img_w

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4,
        )
        free = bin_img.copy()
        free[covered == 1] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        figures: list[dict] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < _MIN_FIG_AREA or area > img_area * _MAX_FIG_FRAC:
                continue
            fill = bin_img[y:y + h, x:x + w].sum() / 255 / area
            if fill < _FIG_FILL_FRAC:
                continue
            figures.append({"class": "figure", "bbox": [x, y, x + w, y + h]})
        return figures

    # ── NMS ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _nms(regions: list[dict], iou_thr: float = 0.3) -> list[dict]:
        priority = {"table": 0, "text": 1, "figure": 2}
        ordered = sorted(regions, key=lambda r: priority.get(r["class"], 9))
        kept: list[dict] = []
        for r in ordered:
            if not any(_iou(r["bbox"], k["bbox"]) > iou_thr for k in kept):
                kept.append(r)
        return kept

    # ── public API ──────────────────────────────────────────────────────────
    def analyze(self, image_path: str, output_dir: str, stem: str | None = None) -> list[dict]:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_h, img_w = img_bgr.shape[:2]

        # Если stem не передан, используем имя файла без расширения
        if stem is None:
            stem = Path(image_path).stem

        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        for cls_name in ("text", "table", "figure"):
            (out_root / cls_name).mkdir(exist_ok=True)

        clean_bgr = self.preprocess(img_bgr)
        clean_pil = Image.fromarray(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))

        words = self._ocr_words(clean_bgr)

        grid_tables = self._detect_tables_from_grid(words)
        tt_tables = [t for t in self._detect_tables_tt(clean_pil)
                     if self._validate_table(t, words)]
        tables = self._nms(grid_tables + tt_tables, iou_thr=0.2)
        tables = [dict(t, **{"class": "table"}) for t in tables]

        # Expand table bboxes slightly so multi-line cell tails are captured.
        pad = 18
        for t in tables:
            x1, y1, x2, y2 = t["bbox"]
            t["bbox"] = [
                max(0, x1 - pad), max(0, y1 - pad),
                min(img_w, x2 + pad), min(img_h, y2 + pad),
            ]

        words_free = [
            w for w in words
            if not any(_inside_fraction(w["bbox"], t["bbox"]) > 0.3 for t in tables)
        ]
        lines = self._merge_words_to_lines(words_free)
        paragraphs = self._merge_lines_to_paragraphs(lines)

        covered = self._covered_mask(img_h, img_w, tables + paragraphs)
        figures = self._detect_figures(clean_bgr, covered)

        all_regions = self._nms(tables + paragraphs + figures)
        all_regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        counters: dict[str, int] = {}
        saved: list[dict] = []
        for region in all_regions:
            cls = region["class"]
            x1, y1, x2, y2 = region["bbox"]
            x1 = max(0, x1 - _PADDING_PX)
            y1 = max(0, y1 - _PADDING_PX)
            x2 = min(img_w, x2 + _PADDING_PX)
            y2 = min(img_h, y2 + _PADDING_PX)
            if (x2 - x1) < _MIN_DIM_PX or (y2 - y1) < _MIN_DIM_PX:
                continue

            counters[cls] = counters.get(cls, 0) + 1
            fname = f"{stem}_{cls}_{counters[cls]:03d}.png"
            out_path = out_root / cls / fname
            crop = img_bgr[y1:y2, x1:x2]
            cv2.imwrite(str(out_path), crop)
            saved.append({"class": cls, "bbox": [x1, y1, x2, y2], "path": str(out_path)})
        return saved

    # ── visualisation ───────────────────────────────────────────────────────
    @staticmethod
    def visualize(image_path: str, regions: list[dict], output_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            color = _COLORS.get(r["class"], (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, r["class"], (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, img)
        return output_path