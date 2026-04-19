"""
Главный пайплайн для извлечения таблиц из PDF и сохранения в Markdown.
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

from .table_extractor import TableExtractor, ExtractedTable
# В table_to_markdown.py нет функции table_to_markdown, она используется как модуль или нужно импортировать конкретные функции
from .table_to_markdown import extract_all_tables

logger = logging.getLogger(__name__)


@dataclass
class TablePipeline:
    """
    Пайплайн: PDF → таблицы → Markdown.

    Использование:
        pipeline = TablePipeline()
        result = pipeline.process("document.pdf")
        pipeline.save("document.pdf", "output/")
    """

    extractor: TableExtractor = field(default_factory=TableExtractor)
    include_page_info: bool = False

    def process(self, pdf_path: str) -> str:
        """
        Обработать PDF файл и вернуть Markdown с таблицами.

        Args:
            pdf_path: путь к PDF файлу

        Returns:
            Markdown строка с таблицами
        """
        pdf_path = str(pdf_path)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Processing: {pdf_path}")

        # Извлекаем таблицы
        tables = self.extractor.extract_all(pdf_path)

        logger.info(f"Found {len(tables)} tables in {pdf_path}")

        for i, t in enumerate(tables):
            logger.info(
                f"  Table {i + 1}: page={t.page_number}, "
                f"size={t.dataframe.shape}, "
                f"merged={t.has_merged_cells}, "
                f"method={t.method}, "
                f"confidence={t.confidence:.2f}"
            )

        # Конвертируем в Markdown
        md = "\n\n".join(t.markdown for t in tables)
        return md

    def process_to_tables(self, pdf_path: str) -> list[ExtractedTable]:
        """
        Обработать PDF файл и вернуть список ExtractedTable.
        """
        pdf_path = str(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        return self.extractor.extract_all(pdf_path)

    def save(self, pdf_path: str, output_dir: str) -> str:
        """
        Обработать PDF и сохранить результат в файл.

        Args:
            pdf_path: путь к PDF файлу
            output_dir: директория для сохранения

        Returns:
            Путь к сохранённому MD файлу
        """
        md_content = self.process(pdf_path)

        os.makedirs(output_dir, exist_ok=True)

        # Имя файла без расширения
        pdf_name = Path(pdf_path).stem
        md_path = os.path.join(output_dir, f"{pdf_name}.md")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Saved: {md_path}")
        return md_path

    def process_directory(
            self, pdf_dir: str, output_dir: str
    ) -> list[str]:
        """
        Обработать все PDF файлы в директории.

        Returns:
            Список путей к сохранённым MD файлам
        """
        pdf_dir = Path(pdf_dir)
        saved_files = []

        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            try:
                md_path = self.save(str(pdf_file), output_dir)
                saved_files.append(md_path)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")

        return saved_files
