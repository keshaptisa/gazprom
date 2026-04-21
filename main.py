"""
Единый пайплайн: читает все PDF из `pdf_files/` и сохраняет в `output/`
по одному Markdown-файлу на PDF. В MD-файле — все найденные таблицы:
сначала таблицы с рамками (pdfplumber, стратегия «lines»), затем
безрамочные (Table Transformer + вывод структуры по словам PDF).

Таблицы-сканы и таблицы с высоким уровнем шума (водяные знаки,
повёрнутый текст, мало нативных слов) отбрасываются автоматически
фильтром качества в table_detector/table_text.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from table_detector.table_lines import find_clean_tables as find_lined_tables
from table_detector.table_text import find_clean_tables as find_borderless_tables
from formatter.table_lines_to_md import convert_all_tables as convert_lined
from formatter.table_text_to_md import convert_all_tables as convert_borderless


INPUT_DIR = Path(__file__).parent / "pdf_files"
OUTPUT_DIR = Path(__file__).parent / "output"


def _collect_bboxes(lined: dict) -> dict:
    """{page: [bbox]} — чтобы безрамочный поиск не накрывал то же место."""
    result = {}
    for page, tables in lined.items():
        result[page] = [t["bbox"] for t in tables if t.get("bbox")]
    return result


def _merge_pages(
    lined_md: dict[int, list[str]],
    borderless_md: dict[int, list[str]],
) -> dict[int, list[dict]]:
    """
    На каждой странице сначала таблицы с рамками, затем безрамочные.
    Возвращает {page: [{"kind": str, "md": str}]}.
    """
    pages = sorted(set(lined_md) | set(borderless_md))
    out = {}
    for p in pages:
        items = []
        for md in lined_md.get(p, []) or []:
            if md.strip():
                items.append({"kind": "lines", "md": md})
        for md in borderless_md.get(p, []) or []:
            if md.strip():
                items.append({"kind": "text", "md": md})
        out[p] = items
    return out


def _save_md(pages: dict[int, list[dict]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_with_tables = [p for p in sorted(pages) if pages[p]]
    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        if not pages_with_tables:
            f.write("_Таблицы не найдены._\n")
            return 0
        for page_num in pages_with_tables:
            f.write(f"# Страница {page_num}\n\n")
            for i, item in enumerate(pages[page_num], 1):
                kind_label = "с рамками" if item["kind"] == "lines" else "безрамочная"
                f.write(f"## Таблица {i} ({kind_label})\n\n")
                f.write(item["md"].rstrip() + "\n\n")
                total += 1
    return total


def process_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Обрабатывает один PDF-файл. Возвращает сводку."""
    try:
        lined = find_lined_tables(pdf_path) or {}
    except Exception as e:
        print(f"  [ERROR] Lined detection failed for {pdf_path.name}: {e}")
        lined = {}

    skip = _collect_bboxes(lined)
    
    try:
        borderless = find_borderless_tables(pdf_path, skip_bboxes=skip) or {}
    except Exception as e:
        print(f"  [ERROR] Borderless detection failed for {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        borderless = {}

    lined_md = convert_lined(lined)
    borderless_md = convert_borderless(borderless)
    merged = _merge_pages(lined_md, borderless_md)

    out_path = output_dir / f"{pdf_path.stem}.md"
    total = _save_md(merged, out_path)

    return {
        "pdf": pdf_path.name,
        "out": out_path,
        "lined": sum(len(v) for v in lined.values()),
        "borderless": sum(len(v) for v in borderless.values()),
        "saved": total,
    }


def main():
    if not INPUT_DIR.exists():
        print(f"Директория {INPUT_DIR} не найдена.")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"В {INPUT_DIR} нет PDF-файлов.")
        return

    print(f"Обработка {len(pdfs)} PDF из {INPUT_DIR} → {OUTPUT_DIR}\n")

    totals = {"lined": 0, "borderless": 0, "saved": 0}
    for pdf in pdfs:
        print(f"--- {pdf.name} ---")
        try:
            s = process_pdf(pdf, OUTPUT_DIR)
        except Exception as e:
            print(f"  ошибка: {e}")
            continue
        print(
            f"  с рамками: {s['lined']:>3} | "
            f"безрамочных: {s['borderless']:>3} | "
            f"в MD: {s['saved']:>3} → {s['out'].name}"
        )
        for k in totals:
            totals[k] += s[k]

    print("\n" + "=" * 60)
    print(
        f"Итого — файлов: {len(pdfs)}, таблиц с рамками: {totals['lined']}, "
        f"безрамочных: {totals['borderless']}, сохранено в MD: {totals['saved']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()