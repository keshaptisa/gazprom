from __future__ import annotations

import os
import re
import tempfile
from collections import Counter

import cv2
import numpy as np
import easyocr
from img2table.document import Image as Img2TableImage
from img2table.ocr import EasyOCR as Img2TableEasyOCR

from preprocess import preprocess_for_ocr, preprocess_table_variants, resize_for_ocr


class OCRRouter:
    """
    Упрощенный OCR-роутер:
    хорошо старается извлекать обычные таблицы с явной сеткой,
    а всё остальное отправляет в обычный OCR.
    """

    def __init__(self) -> None:
        self.reader = easyocr.Reader(["ru", "en"])
        self.table_ocr = Img2TableEasyOCR(lang=["ru", "en"])

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = text.replace("|", " ")
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _score_text(self, text: str) -> int:
        text = self._normalize_text(text)
        if not text:
            return 0

        alnum = sum(ch.isalnum() for ch in text)
        lines = len([line for line in text.splitlines() if line.strip()])
        return alnum + lines * 8

    def _best_text_candidate(self, images: list[np.ndarray]) -> str:
        best_text = ""
        best_score = -1

        for image in images:
            result = self.reader.readtext(image, detail=0, paragraph=False)
            text = self._normalize_text("\n".join(result))
            score = self._score_text(text)
            if score > best_score:
                best_score = score
                best_text = text

        return best_text

    def extract_printed_text(self, img_bgr: np.ndarray) -> str:
        variants = preprocess_table_variants(img_bgr)
        candidates = [
            variants["adaptive"],
            variants["otsu"],
            preprocess_for_ocr(img_bgr),
            resize_for_ocr(img_bgr),
        ]
        return self._best_text_candidate(candidates)

    def extract_handwritten_text(self, img_bgr: np.ndarray) -> str:
        variants = preprocess_table_variants(img_bgr)
        return self._best_text_candidate(
            [
                resize_for_ocr(img_bgr),
                variants["gray"],
                variants["adaptive"],
            ]
        )

    def _dataframe_to_markdown(self, df) -> str:
        if df is None or df.empty:
            return ""

        rows_raw = df.fillna("").values.tolist()
        if not rows_raw:
            return ""

        current_headers = [str(col).strip() for col in df.columns]
        auto_headers = all(
            h == str(i) or h.startswith("Unnamed:")
            for i, h in enumerate(current_headers)
        )

        if auto_headers and rows_raw:
            headers = [self._normalize_text(str(cell)) for cell in rows_raw[0]]
            data_rows = rows_raw[1:]
        else:
            headers = [self._normalize_text(str(col)) or f"Колонка {i + 1}" for i, col in enumerate(df.columns)]
            data_rows = rows_raw

        if not any(headers):
            headers = [f"Колонка {i + 1}" for i in range(len(headers))]

        rows: list[list[str]] = []
        for raw_row in data_rows:
            row = [self._normalize_text(str(cell)) for cell in raw_row]
            if any(cell for cell in row):
                rows.append(row)

        if not rows:
            return ""

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for row in rows:
            padded = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded[: len(headers)]) + " |")

        return "\n".join(lines)

    def _is_good_markdown_table(self, markdown: str) -> bool:
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        if "|" not in lines[0] or "|" not in lines[1]:
            return False

        column_count = len([part for part in lines[0].split("|") if part.strip()])
        return column_count >= 2

    def _extract_table_markdown_with_img2table(self, image: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cv2.imwrite(tmp_path, image)
            doc = Img2TableImage(src=tmp_path)
            tables = doc.extract_tables(
                ocr=self.table_ocr,
                implicit_rows=False,
                implicit_columns=False,
                borderless_tables=False,
            )

            markdown_tables = []
            for table in tables:
                markdown = self._dataframe_to_markdown(table.df)
                if self._is_good_markdown_table(markdown):
                    markdown_tables.append(markdown)

            return "\n\n".join(markdown_tables).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _extract_cell_boxes(self, grid_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            grid_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        img_h, img_w = grid_mask.shape[:2]
        min_w = max(25, img_w // 40)
        min_h = max(18, img_h // 60)
        max_area = img_h * img_w * 0.8

        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if w < min_w or h < min_h:
                continue
            if area >= max_area:
                continue
            boxes.append((x, y, w, h))

        unique: list[tuple[int, int, int, int]] = []
        for box in sorted(boxes, key=lambda b: (b[1], b[0], b[2], b[3])):
            if any(
                abs(box[0] - prev[0]) < 6
                and abs(box[1] - prev[1]) < 6
                and abs(box[2] - prev[2]) < 6
                and abs(box[3] - prev[3]) < 6
                for prev in unique
            ):
                continue
            unique.append(box)
        return unique

    def _group_boxes_to_rows(
        self,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[list[tuple[int, int, int, int]]]:
        if not boxes:
            return []

        heights = [h for _, _, _, h in boxes]
        row_tolerance = max(12, int(np.median(heights) * 0.6))

        rows: list[list[tuple[int, int, int, int]]] = []
        for box in sorted(boxes, key=lambda b: (b[1], b[0])):
            _, y, _, _ = box
            placed = False
            for row in rows:
                ref_y = int(np.median([candidate[1] for candidate in row]))
                if abs(y - ref_y) <= row_tolerance:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        for row in rows:
            row.sort(key=lambda b: b[0])

        return [row for row in rows if len(row) >= 2]

    def _ocr_cell(self, img_bgr: np.ndarray) -> str:
        variants = preprocess_table_variants(img_bgr)
        return self._best_text_candidate(
            [
                variants["adaptive"],
                variants["otsu"],
                resize_for_ocr(img_bgr, min_width=900, min_height=200),
            ]
        )

    def _rows_to_markdown(self, rows: list[list[str]]) -> str:
        cleaned_rows: list[list[str]] = []
        for row in rows:
            cleaned = [self._normalize_text(cell) for cell in row]
            if any(cell for cell in cleaned):
                cleaned_rows.append(cleaned)

        if len(cleaned_rows) < 2:
            return ""

        width_counts = Counter(len(row) for row in cleaned_rows if len(row) >= 2)
        if not width_counts:
            return ""
        target_width = width_counts.most_common(1)[0][0]

        normalized_rows: list[list[str]] = []
        for row in cleaned_rows:
            if len(row) > target_width:
                row = row[:target_width]
            elif len(row) < target_width:
                row = row + [""] * (target_width - len(row))
            normalized_rows.append(row)

        header = normalized_rows[0]
        if all(not cell for cell in header):
            header = [f"Колонка {i + 1}" for i in range(target_width)]
            normalized_rows = [header] + normalized_rows[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * target_width) + " |",
        ]
        for row in normalized_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        markdown = "\n".join(lines)
        return markdown if self._is_good_markdown_table(markdown) else ""

    def _extract_table_markdown_by_cells(self, img_bgr: np.ndarray) -> str:
        variants = preprocess_table_variants(img_bgr)
        boxes = self._extract_cell_boxes(variants["grid"])
        rows = self._group_boxes_to_rows(boxes)
        if len(rows) < 2:
            return ""

        width_counts = Counter(len(row) for row in rows)
        target_width = width_counts.most_common(1)[0][0]
        rows = [row for row in rows if len(row) == target_width]
        if len(rows) < 2:
            return ""

        text_rows: list[list[str]] = []
        for row in rows:
            text_row: list[str] = []
            for x, y, w, h in row:
                pad = 4
                x0 = max(x + 1, 0)
                y0 = max(y + 1, 0)
                x1 = min(x + w - pad, img_bgr.shape[1])
                y1 = min(y + h - pad, img_bgr.shape[0])

                if x1 <= x0 or y1 <= y0:
                    text_row.append("")
                    continue

                cell = img_bgr[y0:y1, x0:x1]
                text_row.append(self._ocr_cell(cell))
            text_rows.append(text_row)

        return self._rows_to_markdown(text_rows)

    def extract_table_markdown(self, img_bgr: np.ndarray) -> str:
        variants = preprocess_table_variants(img_bgr)

        for name in ("original", "adaptive", "otsu"):
            markdown = self._extract_table_markdown_with_img2table(variants[name])
            if markdown:
                return markdown

        markdown = self._extract_table_markdown_by_cells(variants["original"])
        if markdown:
            return markdown

        return ""

    def route(self, img_bgr: np.ndarray, predicted_label: str) -> str:
        label = predicted_label.lower().strip()

        if label in {
            "table_like",
            "small_text_or_handwritten_like",
        }:
            markdown = self.extract_table_markdown(img_bgr)
            if markdown:
                return markdown
            return self.extract_printed_text(img_bgr)

        if label in {"handwritten", "handwritten_like"}:
            return self.extract_handwritten_text(img_bgr)

        return self.extract_printed_text(img_bgr)
