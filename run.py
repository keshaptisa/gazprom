#!/usr/bin/env python3
"""
Скрипт для запуска парсинга таблиц из PDF.

Использование:
    python run.py input.pdf                    # вывод в stdout
    python run.py input.pdf -o output/         # сохранить в директорию
    python run.py input_dir/ -o output/        # обработать директорию
"""

import argparse
import logging
import sys
from pathlib import Path

# Добавляем путь к корню проекта, чтобы импорт работал
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_table_parser import TablePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract tables from PDF to Markdown"
    )
    parser.add_argument("input", help="PDF file or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "--page-info", action="store_true",
        help="Include page info comments"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pipeline = TablePipeline(include_page_info=args.page_info)
    input_path = Path(args.input)

    if input_path.is_file():
        if args.output:
            md_path = pipeline.save(str(input_path), args.output)
            print(f"Saved: {md_path}")
        else:
            md = pipeline.process(str(input_path))
            print(md)
    elif input_path.is_dir():
        if not args.output:
            print("Error: --output required for directory input", file=sys.stderr)
            sys.exit(1)
        saved = pipeline.process_directory(str(input_path), args.output)
        print(f"Processed {len(saved)} files")
    else:
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
