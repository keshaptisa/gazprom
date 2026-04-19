# PDF Table Parser

Python библиотека для извлечения таблиц из PDF документов с множественными стратегиями поиска и валидацией.

## Особенности

- **Множественные стратегии поиска таблиц:**
  - Поиск по линиям (стандартный)
  - Поиск по тексту (для таблиц без рамок)
  - Строгий поиск по тексту (для сложных случаев)
  - Комбинированный поиск

- **Адаптивная обработка:**
  - Автоматическое разделение больших таблиц по высоте строк
  - Валидация структуры таблиц на всех страницах
  - Обработка таблиц без рамок

- **Конвертация в Markdown:**
  - Автоматическое объединение многоуровневых заголовков
  - Умная обработка разрывов строк
  - Поддержка слияния ячеек

## Установка

### Требования

- Python 3.11+
- Ghostscript (для работы с PDF)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Установка Ghostscript

**Windows:**
1. Скачайте Ghostscript с [официального сайта](https://www.ghostscript.com/releases/gsdnld.html)
2. Установите и добавьте в PATH

**Linux:**
```bash
sudo apt-get install ghostscript
```

**macOS:**
```bash
brew install ghostscript
```

## Использование

### Базовое использование

```python
from pdf_table_parser import TablePipeline

# Создаем pipeline
pipeline = TablePipeline()

# Извлекаем таблицы из PDF
tables = pipeline.extract_all("path/to/document.pdf")

# Сохраняем результат
pipeline.save("path/to/document.pdf", "output/")
```

### Командная строка

```bash
python run.py pdfs/document_001.pdf -o output/
```

## Структура проекта

```
pdf_table_parser/
├── __init__.py              # Пакет инициализация
├── config.py                # Конфигурация параметров
├── pipeline.py              # Основной pipeline
├── table_extractor.py       # Экстрактор таблиц
├── table_to_markdown.py     # Конвертация в Markdown
├── utils.py                 # Утилиты
├── run.py                   # Точка входа CLI
├── requirements.txt         # Зависимости
├── pdfs/                    # Директория с PDF файлами
├── output/                  # Директория для результатов
└── tests/                   # Тесты
```

## Конфигурация

### Параметры таблиц (config.py)

```python
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 8,
    "join_tolerance": 1,
    "edge_min_length": 25,
    "intersection_tolerance": 2,
}

STRICT_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "text_x_tolerance": 7,
    "text_y_tolerance": 6,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3
}
```

### Валидация таблиц

- **Порог пустых ячеек:** 50% (таблицы с более чем 50% пустых ячеек исключаются)
- **Минимальная длина ячейки:** 2 символа (таблицы с очень короткими ячейками исключаются)
- **Минимальное количество строк:** 2

## Стратегии поиска

1. **Стратегия 1 (линии):** Стандартный поиск по линиям PDF
2. **Стратегия 1.5 (строгий текст):** Строгий поиск по тексту для сложных случаев
3. **Стратегия 2 (текст):** Поиск по тексту для таблиц без рамок
4. **Стратегия 3 (комбо):** Комбинированный поиск

## Вывод

Таблицы сохраняются в формате Markdown в указанную директорию:

```bash
output/
├── document_001.md
└── document_002.md
```

## Тестирование

```bash
pytest tests/
```

## Требования к окружению

- **Python:** 3.11+
- **ОС:** Windows, Linux, macOS
- **Память:** Минимум 2 GB RAM
- **Диск:** Минимум 100 MB свободного места

## Лицензия

MIT License

## Поддержка

Для вопросов и предложений создайте issue в репозитории.
