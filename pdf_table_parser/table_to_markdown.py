"""Обработка таблиц с усиленным поиском."""
import pdfplumber
import re
from typing import List, Dict, Any
from . import config
from pathlib import Path


def _split_merged_tables(grid: List[List[str]], table_bbox, is_borderless: bool = False) -> List[List[List[str]]]:
    """Возвращает таблицу без изменений."""
    if not grid or len(grid) < 2:
        return [grid] if grid else []
    
    return [grid]


def _split_table_by_row_gaps(grid: List[List[str]], table_bbox) -> List[List[List[str]]]:
    """
    Разделяет таблицу на основе адаптивного разреза по высоте строк.
    Разрезает, если зазор > 2.5 × медианную высоту строки.
    """
    if not grid or len(grid) < 2:
        return [grid] if grid else []
    
    # Получаем высоту таблицы из bbox
    table_height = table_bbox[3] - table_bbox[1]
    
    # Вычисляем высоту каждой строки
    row_heights = [table_height / len(grid)] * len(grid)
    
    # Вычисляем медианную высоту строки
    median_height = sorted(row_heights)[len(row_heights) // 2]
    
    # Порог для разреза: 2.5 × медианная высота
    threshold = 2.5 * median_height
    
    # Поскольку у нас нет информации о фактических зазорах между строками,
    # используем простую логику: если таблица слишком большая, разделяем её
    # на части примерно равного размера
    
    max_rows_per_table = 20  # Максимальное количество строк в одной таблице
    
    if len(grid) <= max_rows_per_table:
        return [grid]
    
    # Разделяем таблицу на части
    split_grids = []
    for i in range(0, len(grid), max_rows_per_table):
        split_grid = grid[i:i + max_rows_per_table]
        if split_grid:
            split_grids.append(split_grid)
    
    return split_grids


def extract_all_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Извлекает все таблицы из PDF с множественными стратегиями поиска."""
    all_tables = []

    print(f"   Сканирование {Path(pdf_path).name}...")

    with pdfplumber.open(pdf_path) as pdf:
        print(f"   Всего страниц: {len(pdf.pages)}")
        for page_num, page in enumerate(pdf.pages):
            print(f"   Страница {page_num + 1} из {len(pdf.pages)}: поиск таблиц...")

            # Поворот для album_table
            if page.rotation in (90, 270):
                print(f"     Альбомная ориентация, поворот...")
                page = page.rotate(-page.rotation)

            # СТРАТЕГИЯ 1: Стандартный поиск по линиям
            tables = page.find_tables(table_settings=config.TABLE_SETTINGS)
            print(f"     Стратегия 1 (линии): {len(tables)} таблиц")
            strategy_used = "lines"

            # СТРАТЕГИЯ 1.5: Строгий поиск по тексту (для адаптивного разреза)
            if len(tables) == 0:
                tables_strict = page.find_tables(table_settings=config.STRICT_SETTINGS)
                print(f"     Стратегия 1.5 (строгий текст): {len(tables_strict)} таблиц")
                if tables_strict:
                    tables = tables_strict
                    strategy_used = "strict_text"

            # СТРАТЕГИЯ 2: Поиск по тексту (для borderless)
            if len(tables) == 0:
                settings_text = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "text_tolerance": 5,
                    "join_tolerance": 5,
                }
                tables_text = page.find_tables(table_settings=settings_text)
                print(f"     Стратегия 2 (текст): {len(tables_text)} таблиц")
                if tables_text:
                    tables = tables_text
                    strategy_used = "text"

            # СТРАТЕГИЯ 3: Комбинированный поиск
            if len(tables) == 0:
                settings_combo = {
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                }
                tables_combo = page.find_tables(table_settings=settings_combo)
                print(f"     Стратегия 3 (комбо): {len(tables_combo)} таблиц")
                if tables_combo:
                    tables = tables_combo
                    strategy_used = "combo"

            # Если таблицы найдены
            if tables:
                # Сортировка: Y -> X
                tables_sorted = sorted(
                    tables,
                    key=lambda t: (t.bbox[1], t.bbox[0])
                )

                for i, table in enumerate(tables_sorted, 1):
                    try:
                        grid = _build_grid(table)
                        if not grid:
                            print(f"       Таблица {i}: пропущена (пустая сетка)")
                            continue

                        # Разделяем таблицу по адаптивному разрезу по высоте строк
                        split_grids = _split_table_by_row_gaps(grid, table.bbox)
                        
                        for j, split_grid in enumerate(split_grids, 1):
                            if not split_grid or len(split_grid) < 2:
                                continue
                            
                            # Отладка: покажем размеры
                            print(f"       Таблица {i}: {len(split_grid)} строк x {len(split_grid[0]) if split_grid else 0} колонок")
                            
                            # Валидация таблицы - проверяем, не является ли она ложным положительным результатом
                            if not _is_valid_table_structure(split_grid, table.bbox, page_num + 1):
                                print(f"       Таблица {i}: пропущена (не прошла валидацию структуры)")
                                continue

                            # Обработка заголовков
                            split_grid = _process_headers(split_grid)

                            # Конвертация в Markdown
                            md = _to_markdown(split_grid)

                            all_tables.append({
                                "page": page_num + 1,
                                "bbox": table.bbox,
                                "markdown": md,
                                "grid": split_grid
                            })

                            # Покажем первую строку таблицы
                            first_line = md.split('\n')[0][:60]

                    except Exception as e:
                        print(f"       Таблица {i}: ошибка {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                print(f"     Таблицы не найдены")

    # Отключено автоматическое объединение таблиц для предотвращения некорректных слияний
    # Если нужно объединение, его следует делать вручную или с более строгими критериями
    print(f"   Всего таблиц: {len(all_tables)}")
    return all_tables


def _build_grid(table) -> List[List[str]]:
    """Строит сетку с правильной обработкой merged-ячеек."""
    cells = table.cells
    if not cells:
        return []

    # Получаем raw данные для текста
    raw = table.extract()
    if not raw:
        return []

    # Создаем сетку из raw данных
    grid = [[str(cell) if cell is not None else "" for cell in row] for row in raw]

    # Умная обработка разрывов строк
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            text = grid[r][c]
            if '\n' in text:
                lines = text.split('\n')
                processed = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    if not processed:
                        processed.append(line)
                    else:
                        prev = processed[-1]
                        if prev.endswith('-'):
                            processed[-1] = prev.rstrip('-') + line
                        else:
                            processed.append(line)
                grid[r][c] = ' '.join(processed)
            else:
                grid[r][c] = re.sub(r'\s+', ' ', text).strip()

    # Фильтруем строки с подписями к рисункам
    filtered_grid = []
    for row in grid:
        row_text = ' '.join(row)
        if re.search(r'\bрис\.\s*\d+', row_text, re.IGNORECASE):
            continue
        filtered_grid.append(row)
    grid = filtered_grid

    # Обработка merged ячеек: если мало пропусков (< 30%) - заполняем пустые ячейки из левого/верхнего значения
    total_cells = sum(len(row) for row in grid)
    empty_cells = sum(1 for row in grid for cell in row if not cell.strip() or cell.strip() == "—")
    
    if total_cells > 0:
        empty_ratio = empty_cells / total_cells
        
        # Если мало пропусков (< 30%) - заполняем пустые ячейки из левого/верхнего значения
        if empty_ratio < 0.3:
            # Горизонтальные: заполняем из левого значения
            for r in range(len(grid)):
                for c in range(1, len(grid[r])):
                    if not grid[r][c].strip() and grid[r][c-1].strip():
                        grid[r][c] = grid[r][c-1]
            
            # Вертикальные: заполняем из верхнего значения
            for c in range(len(grid[0]) if grid else 0):
                for r in range(1, len(grid)):
                    if c < len(grid[r]) and not grid[r][c].strip() and grid[r-1][c].strip():
                        grid[r][c] = grid[r-1][c]
        # Если много пропусков - оставляем как есть, ничего не делаем

    # Убираем первую колонку если она содержит краевой текст (буквы слева на краю страницы)
    if len(grid) > 0 and len(grid[0]) > 1:
        # Проверяем длину текста в первой колонке - если в большинстве строк очень короткий текст (1-2 символа)
        short_text_count = sum(1 for row in grid if row[0].strip() and len(row[0].strip()) <= 2)
        total_rows_with_text = sum(1 for row in grid if row[0].strip())
        
        # Если более 70% строк с текстом в первой колонке имеют очень короткий текст - это краевой текст
        if total_rows_with_text > 0 and short_text_count / total_rows_with_text > 0.7:
            grid = [row[1:] for row in grid]

    return grid


def _process_headers(grid: List[List[str]]) -> List[List[str]]:
    """Объединяет multi-level заголовки через _."""
    if len(grid) < 2:
        return grid

    header_rows = []
    for i in range(min(3, len(grid))):
        has_numbers = any(_is_number(cell) for cell in grid[i])
        has_text = any(cell.strip() for cell in grid[i])

        if not has_numbers and has_text:
            header_rows.append(i)
        else:
            break

    if len(header_rows) > 1:
        merged = []
        for c in range(len(grid[0])):
            parts = [grid[r][c].strip() for r in header_rows if grid[r][c].strip()]
            merged.append("_".join(parts))

        return [merged] + [row for i, row in enumerate(grid) if i not in header_rows]

    return grid


def _is_number(text: str) -> bool:
    """Проверяет, число ли в ячейке."""
    text = text.strip()
    if not text:
        return False
    patterns = [
        r'^\d+[\s,.]?\d*$',
        r'^\d+[.,]\d+\s*%$',
        r'^\d{2}\.\d{2}\.\d{4}$',
        r'^\d+[KMB]?\s*$'
    ]
    return any(re.match(p, text) for p in patterns)


def _is_valid_table_structure(grid: List[List[str]], bbox: tuple, page_num: int) -> bool:
    """
    Проверяет, является ли таблица валидной на основе структуры и характеристик.
    
    Args:
        grid: Сетка таблицы
        bbox: Координаты таблицы (x0, y0, x1, y1)
        page_num: Номер страницы
        
    Returns:
        True если таблица валидна, False если должна быть исключена
    """
    if not grid or len(grid) < 2:
        return False
    
    num_rows = len(grid)
    num_cols = len(grid[0]) if grid else 0
    
    print(f"       Валидация таблицы: страница {page_num}, размер {num_rows}x{num_cols}")
    
    # Проверяем, что таблица содержит много пустых ячеек
    empty_count = sum(1 for row in grid for cell in row if not cell or not cell.strip())
    total_cells = num_rows * num_cols
    empty_ratio = empty_count / total_cells if total_cells > 0 else 0
    
    print(f"       Пустые ячейки: {empty_count}/{total_cells} ({empty_ratio:.1%})")
    
    # Если более 50% ячеек пустые, считаем таблицу ложным положительным результатом
    if empty_ratio > 0.5:
        print(f"       Таблица на странице {page_num} исключена: слишком много пустых ячеек ({empty_ratio:.1%})")
        return False
    
    # Проверяем среднюю длину ячеек
    cell_lengths = [len(cell) for row in grid for cell in row if cell]
    avg_length = sum(cell_lengths) / len(cell_lengths) if cell_lengths else 0
    
    print(f"       Средняя длина ячеек: {avg_length:.1f}")
    
    # Если средняя длина ячеек слишком короткая, считаем таблицу ложным положительным результатом
    if avg_length < 2:
        print(f"       Таблица на странице {page_num} исключена: слишком короткие ячейки (средняя длина: {avg_length:.1f})")
        return False
    
    return True


def _to_markdown(grid: List[List[str]]) -> str:
    """Конвертирует сетку в Markdown."""
    if not grid or not grid[0]:
        return ""

    def fmt(row):
        return "| " + " | ".join(cell for cell in row) + " |"

    lines = [fmt(grid[0])]
    lines.append("| " + " | ".join(["---"] * len(grid[0])) + " |")

    for row in grid[1:]:
        lines.append(fmt(row))

    return "\n".join(lines)
