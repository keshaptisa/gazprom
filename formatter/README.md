# Formatter Module

Модуль для конвертации детектированных таблиц в Markdown формат.

## Структура

```
formatter/
├── table_lines_to_md.py    # Конвертация таблиц с рамками в Markdown
└── table_text_to_md.py     # Конвертация безрамочных таблиц в Markdown
```

## Компоненты

### table_lines_to_md.py

Конвертация таблиц с рамками (из pdfplumber) в Markdown.

**Основные функции:**
- `table_to_markdown(table_data)` - конвертирует одну таблицу в Markdown
- `convert_all_tables(tables_dict)` - конвертирует все таблицы из словаря
- `save_markdown_to_file(tables_dict, output_path)` - сохраняет в файл

**Класс MarkdownFormatter:**
- `format_table(table_data, headers=None)` - базовое форматирование
- `format_multilevel_header(headers_levels, data_rows)` - многоуровневые заголовки
- `handle_merged_cells(table_data, merged_info)` - обработка объединенных ячеек

**Особенности обработки:**
- Объединение перенесенных слов (части < 6 букв)
- Удаление лишних заглавных букв в середине слов
- Объединение многоуровневых заголовков через '_'
- Заполнение пустых ячеек (горизонтальное и вертикальное)
- Экранирование pipe-символов для Markdown
- Сжатие множественных пробелов

### table_text_to_md.py

Конвертация безрамочных таблиц (из Table Transformer) в Markdown.

**Основные функции:**
- `table_to_markdown(table_data)` - конвертирует одну таблицу в Markdown
- `convert_all_tables(tables_dict)` - конвертирует все таблицы из словаря
- `save_markdown_to_file(tables_dict, output_path)` - сохраняет в файл

**Особенности:**
- Нормализация ячеек (удаление \r, \n, сжатие пробелов)
- Удаление пустых строк и столбцов
- Простое форматирование без сложной обработки текста

## Использование

```python
from formatter.table_lines_to_md import convert_all_tables as convert_lined
from formatter.table_text_to_md import convert_all_tables as convert_borderless

# Конвертация таблиц с рамками
lined_md = convert_lined(lined_tables_dict)

# Конвертация безрамочных таблиц
borderless_md = convert_borderless(borderless_tables_dict)

# Сохранение в файл
from formatter.table_lines_to_md import save_markdown_to_file
save_markdown_to_file(lined_md, "output.md")
```

## Формат вывода

Markdown таблицы имеют стандартный формат:
```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
```

## Зависимости

- pathlib (стандартная библиотека)
- re (стандартная библиотека)
