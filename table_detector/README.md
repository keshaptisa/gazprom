# Table Detector Module

Модуль для детекции таблиц в PDF-документах. Поддерживает два типа таблиц: с рамками и безрамочные.

## Структура

```
table_detector/
├── table_lines.py      # Детекция таблиц с рамками (pdfplumber)
├── table_text.py       # Детекция безрамочных таблиц (Table Transformer)
└── config.py           # Конфигурации для обоих методов
```

## Компоненты

### table_lines.py

Детекция таблиц с явными границами с использованием pdfplumber.

**Основная функция:**
- `find_clean_tables(pdf_path)` - находит таблицы с рамками в PDF

**Особенности:**
- Использует стратегию "lines" для поиска таблиц по границам
- Автоматическое слияние таблиц между страницами
- Фильтрация по количеству строк/столбцов и заполненности
- Обрезка колонтитулов для улучшения качества детекции

### table_text.py

Детекция безрамочных таблиц с использованием Table Transformer (microsoft/table-transformer-detection).

**Основная функция:**
- `find_clean_tables(pdf_path, skip_bboxes=None)` - находит безрамочные таблицы

**Подход:**
1. Рендер страницы PDF в изображение (300 DPI)
2. Детекция bbox таблиц через Table Transformer
3. Проверка на отсутствие линий (OpenCV HoughLinesP)
4. Фильтрация качества (отсев сканов и шума)
5. Построение структуры по нативным словам PDF

**Особенности:**
- Использует координаты нативных слов PDF для структуры (более стабильно)
- Отсеивает маркированные списки
- Проверка качества текста (длина слов, доля односимвольных)
- Умная кластеризация строк и столбцов

### config.py

Конфигурационные параметры для обоих методов детекции.

**CONFIG_LINES** - настройки для таблиц с рамками:
- `min_rows`, `min_cols` - минимальные размеры таблицы
- `min_fill_ratio` - минимальная заполненность ячеек
- `page_stitch_*_margin` - отступы для склейки между страницами
- `search_strategies` - стратегии поиска таблиц

**TT_CONFIG** - настройки для Table Transformer:
- `render_dpi` - DPI для рендера страницы
- `detection_threshold` - порог уверенности детекции
- `merge_iou` - порог слияния пересекающихся bbox
- `filter` - параметры фильтрации качества
- `borderless` - критерии безрамочности (макс. кол-во линий)
- `clustering` - параметры кластеризации слов

## Использование

```python
from table_detector.table_lines import find_clean_tables as find_lined
from table_detector.table_text import find_clean_tables as find_borderless

# Таблицы с рамками
lined_tables = find_lined("path/to/file.pdf")

# Безрамочные таблицы (с пропуском областей, покрытых lined tables)
skip_bboxes = {page_num: [bbox for bbox in lined_tables[page_num]]}
borderless_tables = find_borderless("path/to/file.pdf", skip_bboxes=skip_bboxes)
```

## Зависимости

- pdfplumber
- PyMuPDF (fitz)
- torch
- transformers
- opencv-python
- Pillow
- numpy
