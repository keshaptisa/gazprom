import re
from pathlib import Path


class MarkdownFormatter:
    """Форматирует таблицу в Markdown."""

    def format_table(self, table_data, headers=None):
        """
        Форматирует двумерный массив в Markdown-таблицу.

        Args:
            table_data: list of lists — данные таблицы
                       первая строка считается заголовком
            headers: опционально — отдельный заголовок

        Returns:
            str: таблица в формате Markdown
        """
        if not table_data or len(table_data) == 0:
            return ""

        # Определяем количество столбцов
        num_cols = max(len(row) for row in table_data)

        # Выравниваем строки (дополняем пустыми ячейками если нужно)
        normalized = []
        for row in table_data:
            new_row = list(row) + [""] * (num_cols - len(row))
            normalized.append(new_row)

        if len(normalized) < 1:
            return ""

        # Первая строка — заголовок
        header_row = normalized[0]
        data_rows = normalized[1:] if len(normalized) > 1 else []

        # Формируем Markdown
        lines = []

        # Заголовок
        header_line = "| " + " | ".join(
            self._escape_cell(cell) for cell in header_row
        ) + " |"
        lines.append(header_line)

        # Разделитель
        separator = "| " + " | ".join(["---"] * num_cols) + " |"
        lines.append(separator)

        # Данные
        for row in data_rows:
            row_line = "| " + " | ".join(
                self._escape_cell(cell) for cell in row
            ) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    def format_multilevel_header(self, headers_levels, data_rows):
        """
        Форматирует таблицу с многоуровневым заголовком.
        Уровни объединяются через '_'.

        Args:
            headers_levels: list of lists — уровни заголовков
            data_rows: list of lists — данные

        Returns:
            str: таблица в Markdown
        """
        if not headers_levels:
            return ""

        num_cols = max(
            max((len(row) for row in headers_levels), default=0),
            max((len(row) for row in data_rows), default=0)
        )

        # Объединяем уровни заголовков через '_'
        merged_headers = [""] * num_cols
        for col_idx in range(num_cols):
            parts = []
            for level in headers_levels:
                if col_idx < len(level) and level[col_idx].strip():
                    parts.append(level[col_idx].strip())
            merged_headers[col_idx] = "_".join(parts)

        # Создаём таблицу с объединённым заголовком
        full_table = [merged_headers] + data_rows
        return self.format_table(full_table)

    def handle_merged_cells(self, table_data, merged_info=None):
        """
        Обрабатывает объединённые ячейки: копирует содержимое
        во все составные ячейки.

        Args:
            table_data: list of lists
            merged_info: list of dict с информацией об объединении
                        {'row': r, 'col': c, 'rowspan': rs, 'colspan': cs, 'text': text}

        Returns:
            list of lists — таблица с раскопированными ячейками
        """
        if not merged_info:
            return table_data

        result = [row[:] for row in table_data]

        for merge in merged_info:
            r = merge.get('row', 0)
            c = merge.get('col', 0)
            rs = merge.get('rowspan', 1)
            cs = merge.get('colspan', 1)
            text = merge.get('text', '')

            if r < len(result) and c < len(result[r]):
                # Копируем текст во все ячейки объединения
                for ri in range(r, min(r + rs, len(result))):
                    for ci in range(c, min(c + cs, len(result[ri]))):
                        result[ri][ci] = text

        return result

    def _escape_cell(self, text):
        """Экранирует содержимое ячейки для Markdown."""
        if not text:
            return ""

        text = str(text).strip()

        # Заменяем pipe-символ
        text = text.replace("|", "\\|")

        # Убираем переносы строк внутри ячеек
        text = text.replace("\n", " ").replace("\r", "")

        # Убираем множественные пробелы
        while "  " in text:
            text = text.replace("  ", " ")

        return text.strip()


def _looks_like_header_row(row):
    """
    Строгая проверка: является ли строка заголовком.
    Заголовок НЕ должен содержать чисел/дат/валют/процентов.
    """
    if not row:
        return False
    
    # Паттерны данных (НЕ заголовков)
    data_patterns = [
        r'^\d+[\s,.]?\d*[\s%руб.\$°≤≥×≈\-\^]?$',  # Числа, %, валюты
        r'^\d{2}\.\d{2}\.\d{4}$',  # Даты
        r'^\d{4}-\d{2}-\d{2}$',
        r'^\d+,\d{2}\s*руб\.$',  # 123,45 руб.
        r'^\d+\.\d{2}%$',  # 12.34%
    ]
    
    data_count = 0
    text_count = 0
    long_text_count = 0  # Счетчик длинного текста (>20 символов)
    has_punctuation = False  # Счетчик знаков препинания в конце
    
    for cell in row:
        cell = cell.strip()
        if not cell:
            continue
            
        # Проверяем, выглядит ли как данные
        is_data = any(re.match(p, cell) for p in data_patterns)
        if is_data:
            data_count += 1
        else:
            text_count += 1
            if len(cell) > 20:
                long_text_count += 1
            # Проверяем знаки препинания в конце (точка, запятая и т.д.)
            if cell and cell[-1] in '.,;:!?' or '..' in cell:
                has_punctuation = True
    
    # Заголовок: >95% ячеек - текст (не данные)
    # И не должно быть длинного текста (заголовки обычно короткие)
    # И не должно быть знаков препинания в конце
    total = data_count + text_count
    if total == 0:
        return False
    
    text_ratio = text_count / total
    
    return text_ratio > 0.95 and long_text_count == 0 and not has_punctuation


def merge_multilevel_headers(grid):
    """
    Объединяет заголовки через '_' ТОЛЬКО если найдено 2+ реальных уровня.
    """
    if len(grid) < 2:
        return grid
    
    # 1. Находим строки-заголовки с СТРОГОЙ проверкой
    header_rows = []
    for r in range(min(3, len(grid))):  # Проверяем первые 3 строки
        if _looks_like_header_row(grid[r]):
            header_rows.append(r)
        else:
            # Как только нашли строку с данными - стоп
            break
    
    # 2. Объединяем ТОЛЬКО если найдено ≥2 уровня заголовков
    if len(header_rows) < 2:
        return grid  # Ничего не делаем!
    
    # 3. Дополнительная проверка: заголовки должны быть разной длины
    # (чтобы не объединять дублирующиеся строки)
    first_header = "_".join(grid[header_rows[0]]).strip()
    second_header = "_".join(grid[header_rows[1]]).strip()
    
    if first_header == second_header:
        # Это дубликаты, а не многоуровневые заголовки
        return grid
    
    # 4. Поколоночное слияние
    n_cols = len(grid[0])
    merged_header = []
    
    for c in range(n_cols):
        parts = []
        for r in header_rows:
            if c < len(grid[r]):
                val = grid[r][c].strip()
                # Добавляем только непустые и уникальные части
                if val and (not parts or val != parts[-1]):
                    parts.append(val)
        
        merged_header.append("_".join(parts) if parts else "")
    
    # 5. Пересобираем: новый заголовок + данные
    return [merged_header] + [row for i, row in enumerate(grid) if i not in header_rows]


def table_to_markdown(table_data):
    """
    Конвертирует данные таблицы в Markdown формат.
    
    Args:
        table_data: Словарь с данными таблицы, содержащий:
            - data: список списков с ячейками таблицы
            - cols: количество столбцов
            - rows: количество строк
    
    Returns:
        Строка в формате Markdown таблицы
    """
    formatter = MarkdownFormatter()
    data = table_data["data"]
    
    if not data or not data[0]:
        return ""
    
    # Очищаем ячейки от лишних пробелов и пустых значений
    cleaned_data = []
    for row in data:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                # Обрабатываем переносы строк
                cell_text = str(cell)
                
                # Объединяем перенесенные слова без пробела, если части меньше 6 букв
                parts = cell_text.replace('\r', '').split('\n')
                if len(parts) > 1:
                    merged_parts = []
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if i > 0 and len(part) < 6 and len(merged_parts[-1]) > 0:
                            # Объединяем без пробела
                            merged_parts[-1] += part
                        else:
                            # Добавляем с пробелом
                            if merged_parts and part:
                                merged_parts.append(' ' + part)
                            elif part:
                                merged_parts.append(part)
                    cell_text = ''.join(merged_parts)
                else:
                    cell_text = cell_text.replace('\n', ' ').replace('\r', ' ')
                
                cell_text = ' '.join(cell_text.split())

                # Объединяем части, если они меньше 4 букв
                words = cell_text.split()
                merged_words = []
                for i, word in enumerate(words):
                    if i > 0 and len(word) < 4 and merged_words:
                        # Объединяем с предыдущим словом без пробела
                        merged_words[-1] += word
                    else:
                        merged_words.append(word)
                cell_text = ' '.join(merged_words)
                
                # Убираем заглавные буквы в середине слов и удаляем заглавные перед пробелами
                cleaned_cell = ""
                for i, char in enumerate(cell_text):
                    if char.isupper():
                        # Если это первая буква слова (предыдущий - пробел или i=0), оставляем
                        if i == 0 or cell_text[i-1] == ' ':
                            # Проверяем: после этой заглавной стоит пробел?
                            if i < len(cell_text) - 1 and cell_text[i+1] == ' ':
                                # Заглавная перед пробелом - пропускаем (удаляем)
                                continue
                            else:
                                cleaned_cell += char
                        else:
                            # Заглавная в середине слова - пропускаем (удаляем)
                            continue
                    else:
                        cleaned_cell += char
                
                cleaned_row.append(cleaned_cell)
        # Удаляем полностью пустые строки
        if any(cleaned_row):  # Если хотя бы одна ячейка не пустая
            cleaned_data.append(cleaned_row)
    
    # Новая логика заполнения пустых ячеек
    if cleaned_data:
        # Считаем процент пустых ячеек (включая прочерки)
        total_cells = sum(len(row) for row in cleaned_data)
        empty_cells = 0
        for row in cleaned_data:
            for cell in row:
                if not cell or cell.strip() == '' or cell.strip() == '—' or cell.strip() == '-':
                    empty_cells += 1
        
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 0
        
        if empty_ratio < 0.4:
            # Заполняем пустые ячейки
            for r in range(len(cleaned_data)):
                row = cleaned_data[r]
                
                # Сначала горизонтальное заполнение (слева направо)
                left_value = None
                for c in range(len(row)):
                    if row[c] and row[c].strip() not in ['', '—', '-']:
                        left_value = row[c]
                    elif not row[c] or row[c].strip() in ['', '—', '-']:
                        if left_value:
                            row[c] = left_value
                
                # Затем вертикальное заполнение (сверху вниз)
                for c in range(len(row)):
                    if not row[c] or row[c].strip() == '' or row[c].strip() == '—' or row[c].strip() == '-':
                        # Ищем значение сверху
                        for rr in range(r - 1, -1, -1):
                            if c < len(cleaned_data[rr]) and cleaned_data[rr][c] and cleaned_data[rr][c].strip() not in ['', '—', '-']:
                                row[c] = cleaned_data[rr][c]
                                break
    
    # Обнуляем ячейки с одним символом
    cleaned_data = [
        ["" if len(cell) <= 1 else cell for cell in row]
        for row in cleaned_data
    ]

    # Удаляем строки, где все ячейки пустые
    cleaned_data = [row for row in cleaned_data if any(cell for cell in row)]

    # Удаляем колонки, где все ячейки пустые
    if cleaned_data:
        n_cols = len(cleaned_data[0])
        keep_cols = [c for c in range(n_cols) if any(row[c] for row in cleaned_data if c < len(row))]
        cleaned_data = [[row[c] for c in keep_cols if c < len(row)] for row in cleaned_data]

    # Объединяем многоуровневые заголовки
    cleaned_data = merge_multilevel_headers(cleaned_data)
    
    # Используем formatter для создания Markdown
    return formatter.format_table(cleaned_data)


def convert_all_tables(tables_dict):
    """
    Конвертирует все таблицы из словаря в Markdown формат.
    
    Args:
        tables_dict: Словарь {page_num: [table_data]}
    
    Returns:
        Словарь {page_num: [markdown_strings]}
    """
    result = {}
    
    for page_num in sorted(tables_dict.keys()):
        tables_on_page = tables_dict[page_num]
        markdown_tables = []
        
        for table_data in tables_on_page:
            md_table = table_to_markdown(table_data)
            if md_table:
                markdown_tables.append(md_table)
        
        result[page_num] = markdown_tables
    
    return result


def save_markdown_to_file(tables_dict, output_path):
    """
    Сохраняет все таблицы в Markdown файл.
    
    Args:
        tables_dict: Словарь {page_num: [markdown_strings]}
        output_path: Путь к выходному файлу
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_num in sorted(tables_dict.keys()):
            f.write(f"# Страница {page_num}\n\n")
            
            tables = tables_dict[page_num]
            for i, md_table in enumerate(tables):
                f.write(f"## Таблица {i + 1}\n\n")
                f.write(md_table + "\n\n")
            
            f.write("\n")


