"""
Конвертер безрамочных таблиц (результат Table Transformer) в Markdown.
"""
from __future__ import annotations

from pathlib import Path


def _normalize_cell(cell) -> str:
    if cell is None:
        return ""
    text = str(cell).replace("\r", " ").replace("\n", " ")
    # схлопываем множественные пробелы
    text = " ".join(text.split())
    return text.strip()


def _normalize_grid(data):
    if not data:
        return []
    n_cols = max(len(row) for row in data)
    result = []
    for row in data:
        norm = [_normalize_cell(c) for c in row]
        while len(norm) < n_cols:
            norm.append("")
        result.append(norm)
    return result


def _drop_empty_columns(grid):
    if not grid:
        return grid
    n_cols = len(grid[0])
    keep = []
    for c in range(n_cols):
        if any(row[c] for row in grid):
            keep.append(c)
    return [[row[c] for c in keep] for row in grid]


def _drop_empty_rows(grid):
    return [row for row in grid if any(cell for cell in row)]


def table_to_markdown(table_data) -> str:
    data = table_data.get("data") if isinstance(table_data, dict) else table_data
    grid = _normalize_grid(data or [])
    grid = _drop_empty_rows(grid)
    grid = _drop_empty_columns(grid)
    if not grid:
        return ""

    n_cols = len(grid[0])
    header = grid[0]
    rest = grid[1:] if len(grid) > 1 else []

    def render_row(row):
        escaped = [cell.replace("|", "\\|") for cell in row]
        return "| " + " | ".join(escaped) + " |"

    lines = [render_row(header), "| " + " | ".join(["---"] * n_cols) + " |"]
    for row in rest:
        lines.append(render_row(row))
    return "\n".join(lines)


def convert_all_tables(tables_dict):
    result = {}
    for page_num in sorted(tables_dict.keys()):
        md_tables = []
        for table in tables_dict[page_num]:
            md = table_to_markdown(table)
            if md:
                md_tables.append(md)
        result[page_num] = md_tables
    return result


def save_markdown_to_file(tables_dict, output_path):
    output_path = Path(output_path)
    pages_with_tables = [p for p in sorted(tables_dict.keys()) if tables_dict[p]]
    with open(output_path, "w", encoding="utf-8") as f:
        if not pages_with_tables:
            f.write("_Таблицы не найдены._\n")
            return
        for page_num in pages_with_tables:
            f.write(f"# Страница {page_num}\n\n")
            for i, md in enumerate(tables_dict[page_num], 1):
                f.write(f"## Таблица {i}\n\n")
                f.write(md + "\n\n")

