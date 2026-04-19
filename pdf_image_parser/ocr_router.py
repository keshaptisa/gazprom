from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import easyocr
from img2table.document import Image as Img2TableImage
from img2table.ocr import EasyOCR as Img2TableEasyOCR

from preprocess import preprocess_for_ocr


class OCRRouter:
    """
    Роутер OCR-обработки для блоков, которые image_extractor классифицировал как OCR-like.

    Поддерживаемые ветки:
    - table_like -> img2table + EasyOCR -> markdown
    - handwritten -> EasyOCR fallback
    - page_like / printed_text_like / small_text_or_handwritten_like -> EasyOCR + preprocess
    """

    def __init__(self) -> None:
        self.reader = easyocr.Reader(["ru", "en"])
        self.table_ocr = Img2TableEasyOCR(lang=["ru"])

    def extract_printed_text(self, img_bgr: np.ndarray) -> str:
        """
        OCR для печатного/сканированного текста.
        """
        processed = preprocess_for_ocr(img_bgr)
        result = self.reader.readtext(processed, detail=0)
        text = "\n".join(result).strip()
        return text

    def extract_handwritten_text(self, img_bgr: np.ndarray) -> str:
        """
        Пока fallback-ветка для рукописного текста.
        Используем EasyOCR напрямую без сильного препроцессинга.
        """
        result = self.reader.readtext(img_bgr, detail=0)
        text = "\n".join(result).strip()
        return text

    def extract_table_markdown(self, img_bgr: np.ndarray) -> str:
        """
        Пытается извлечь таблицу из изображения и вернуть markdown.
        Если таблица не найдена, вернёт пустую строку.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cv2.imwrite(tmp_path, img_bgr)

            doc = Img2TableImage(src=tmp_path)
            tables = doc.extract_tables(ocr=self.table_ocr)

            markdown_tables = []
            for table in tables:
                df = table.df
                md = df.to_markdown(index=False)
                markdown_tables.append(md)

            return "\n\n".join(markdown_tables).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def route(self, img_bgr: np.ndarray, predicted_label: str) -> str:
        """
        Выбирает стратегию OCR по типу блока.

        Args:
            img_bgr: изображение в формате OpenCV BGR
            predicted_label: label из image_extractor

        Returns:
            Текст или markdown-таблица
        """
        label = predicted_label.lower().strip()

        if label == "table_like":
            md = self.extract_table_markdown(img_bgr)
            if md:
                return md
            return self.extract_printed_text(img_bgr)

        if label == "handwritten":
            return self.extract_handwritten_text(img_bgr)

        if label in {
            "page_like",
            "printed_text_like",
            "small_text_or_handwritten_like",
            "text_like_dense",
        }:
            return self.extract_printed_text(img_bgr)

        # fallback
        return self.extract_printed_text(img_bgr)