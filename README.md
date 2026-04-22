# Gazprom PDF Processing Pipeline

Проект предназначен для поэтапной обработки PDF-документов и связанных с ними растровых фрагментов. Репозиторий объединяет несколько независимых, но совместимых пайплайнов:

- извлечение нативных таблиц из PDF;
- сборку структурированного layout-документа из текста, таблиц и изображений;
- маршрутизацию растровых блоков в OCR или в сохранение как изображений;
- разделение OCR-кандидатов на рукописные и печатные;
- локальное распознавание рукописного текста;
- разбор комбинированных изображений на текстовые, табличные и графические области.

## Основные модели

### 1. Классификация растровых блоков из PDF

Используется модель:

- `microsoft/dit-base-finetuned-rvlcdip`

Где применяется:

- [pdf_image_parser/image_extractor.py](C:/Users/User/gazprom/pdf_image_parser/image_extractor.py)
- [prepare_image_ocr_queue.py](C:/Users/User/gazprom/prepare_image_ocr_queue.py)

Назначение:

- классификация встроенных изображений из PDF;
- принятие решения, отправлять блок в OCR, сохранять как изображение или отбрасывать.

Локальный кэш моделей:

- `C:\Users\User\gazprom\hf_models`

### 2. OCR для печатного текста и табличных изображений

Используются модели и библиотеки:

- `EasyOCR`
- `img2table` + `EasyOCR`

Где применяется:

- [pdf_image_parser/ocr_router.py](C:/Users/User/gazprom/pdf_image_parser/ocr_router.py)
- [parser_comb_im/analyzer.py](C:/Users/User/gazprom/parser_comb_im/analyzer.py)

Назначение:

- OCR печатных фрагментов;
- OCR табличных изображений;
- извлечение текста из ячеек и преобразование таблиц в Markdown.

### 3. Детекция таблиц на изображениях

Используется модель:

- `microsoft/table-transformer-detection`

Где применяется:

- [parser_comb_im/analyzer.py](C:/Users/User/gazprom/parser_comb_im/analyzer.py)

Назначение:

- поиск табличных областей на обычных изображениях;
- совместная работа с EasyOCR и OpenCV-эвристиками для сегментации изображения на регионы.

### 4. OCR рукописного текста

Используется модель:

- `raxtemur/trocr-base-ru`

Где применяется:

- [recognize_handwritten_local.py](C:/Users/User/gazprom/recognize_handwritten_local.py)

Назначение:

- локальное распознавание рукописных фрагментов после их отбора в OCR-очереди.

### 5. Нативный текст PDF

Модель не используется. Извлечение выполняется напрямую из PDF через `PyMuPDF`.

Где применяется:

- [pdf_layout_parser/native_text.py](C:/Users/User/gazprom/pdf_layout_parser/native_text.py)

Назначение:

- получение текстовых блоков с координатами, размером шрифта и оценкой насыщенности шрифта;
- передача этих блоков в layout-сборку документа.

## Схема проекта

```text
gazprom/
|-- README.md
|-- requirements.txt
|-- run.py
|-- prepare_image_ocr_queue.py
|-- split_text_images_handwritten_vs_printed.py
|-- handwritten_vs_printed_best_v2.py
|-- evaluate_handwritten_vs_printed_local.py
|-- recognize_handwritten_local.py
|-- count_trigger_phrases.py
|-- make_handwritten_contact_sheet.py
|-- rename_images_for_submission.py
|-- pdfs/
|-- output/
|-- hf_models/
|-- handwritten_buhands/
|-- pdf_table_parser/
|   |-- __init__.py
|   |-- config.py
|   |-- pipeline.py
|   |-- table_extractor.py
|   `-- table_to_markdown.py
|-- pdf_image_parser/
|   |-- __init__.py
|   |-- image_extractor.py
|   |-- ocr_router.py
|   |-- preprocess.py
|   |-- build_image_queue.py
|   |-- test_images.py
|   `-- test_ocr.py
|-- pdf_layout_parser/
|   |-- models.py
|   |-- native_text.py
|   |-- cleanup.py
|   |-- builder.py
|   |-- export_markdown.py
|   `-- pipeline.py
`-- parser_comb_im/
    |-- main.py
    |-- analyzer.py
    |-- structure_detector.py
    |-- requirements.txt
    |-- comb_image/
    `-- output/
```

## Логическая архитектура

### 1. `pdf_table_parser`

Зона ответственности:

- извлечение нативных таблиц из PDF;
- преобразование таблиц в Markdown;
- пакетная обработка PDF-файлов и сохранение результата.

Ключевые файлы:

- [pdf_table_parser/pipeline.py](C:/Users/User/gazprom/pdf_table_parser/pipeline.py)
- [pdf_table_parser/table_extractor.py](C:/Users/User/gazprom/pdf_table_parser/table_extractor.py)
- [pdf_table_parser/table_to_markdown.py](C:/Users/User/gazprom/pdf_table_parser/table_to_markdown.py)

Точка входа:

- [run.py](C:/Users/User/gazprom/run.py)

### 2. `pdf_image_parser`

Зона ответственности:

- извлечение встроенных изображений из PDF;
- классификация изображений через DiT;
- маршрутизация блоков в OCR или в сохранение;
- кэширование результатов классификации блоков.

Ключевые файлы:

- [pdf_image_parser/image_extractor.py](C:/Users/User/gazprom/pdf_image_parser/image_extractor.py)
- [pdf_image_parser/ocr_router.py](C:/Users/User/gazprom/pdf_image_parser/ocr_router.py)
- [pdf_image_parser/preprocess.py](C:/Users/User/gazprom/pdf_image_parser/preprocess.py)

Основной выход:

- изображения в `output/images`;
- кэш классификации блоков в `output/block_cache.json`.

### 3. `pdf_layout_parser`

Зона ответственности:

- объединение трех источников данных:
  - нативный текст PDF;
  - таблицы из `pdf_table_parser`;
  - изображения и OCR-блоки из `pdf_image_parser`;
- нормализация порядка элементов на странице;
- экспорт документа в JSON и Markdown.

Ключевые файлы:

- [pdf_layout_parser/pipeline.py](C:/Users/User/gazprom/pdf_layout_parser/pipeline.py)
- [pdf_layout_parser/builder.py](C:/Users/User/gazprom/pdf_layout_parser/builder.py)
- [pdf_layout_parser/models.py](C:/Users/User/gazprom/pdf_layout_parser/models.py)
- [pdf_layout_parser/export_markdown.py](C:/Users/User/gazprom/pdf_layout_parser/export_markdown.py)

Форматы результата:

- `output/layout/*.json`
- `output/layout/*.md`

### 4. OCR-очередь изображений

Назначение:

- массовая выгрузка изображений из всех PDF;
- распределение по каталогам OCR / table / non_text;
- сохранение манифестов для дальнейших этапов.

Ключевой файл:

- [prepare_image_ocr_queue.py](C:/Users/User/gazprom/prepare_image_ocr_queue.py)

Основные каталоги:

- `output/ocr_queue/all_images`
- `output/ocr_queue/ocr_text_images`
- `output/ocr_queue/table_images`
- `output/ocr_queue/non_text_images`

### 5. Разделение на рукописные и печатные изображения

Назначение:

- бинарная классификация OCR-кандидатов на `handwritten` и `printed`;
- подготовка папок для последующего OCR.

Ключевые файлы:

- [split_text_images_handwritten_vs_printed.py](C:/Users/User/gazprom/split_text_images_handwritten_vs_printed.py)
- [handwritten_vs_printed_best_v2.py](C:/Users/User/gazprom/handwritten_vs_printed_best_v2.py)
- [evaluate_handwritten_vs_printed_local.py](C:/Users/User/gazprom/evaluate_handwritten_vs_printed_local.py)

Выходные каталоги:

- `output/ocr_queue/handwritten_final`
- `output/ocr_queue/printed_final`

### 6. Локальное распознавание рукописного текста

Назначение:

- запуск TrOCR по подготовленным рукописным изображениям;
- сохранение итогового CSV с распознанным текстом.

Ключевой файл:

- [recognize_handwritten_local.py](C:/Users/User/gazprom/recognize_handwritten_local.py)

Результат:

- `output/handwritten_ocr/handwritten_ocr_results.csv`

### 7. `parser_comb_im`

Назначение:

- анализ обычных изображений, не PDF;
- разметка регионов `text`, `table`, `figure`;
- сохранение вырезанных областей по классам.

Ключевые файлы:

- [parser_comb_im/main.py](C:/Users/User/gazprom/parser_comb_im/main.py)
- [parser_comb_im/analyzer.py](C:/Users/User/gazprom/parser_comb_im/analyzer.py)
- [parser_comb_im/structure_detector.py](C:/Users/User/gazprom/parser_comb_im/structure_detector.py)

Выходные каталоги:

- `parser_comb_im/output/text`
- `parser_comb_im/output/table`
- `parser_comb_im/output/figure`

## Сквозная схема обработки

```text
PDF
-> pdf_table_parser: нативные таблицы -> Markdown-таблицы
-> pdf_image_parser: встроенные изображения -> image / OCR text / table text
-> pdf_layout_parser: сборка page-level структуры -> JSON + Markdown

Массовая OCR-ветка
PDF -> prepare_image_ocr_queue.py -> OCR queue
-> split_text_images_handwritten_vs_printed.py -> handwritten / printed
-> recognize_handwritten_local.py -> CSV с рукописным OCR
```

## Точки входа

### 1. Извлечение таблиц из PDF

```bash
python run.py pdfs/document_001.pdf -o output
```

### 2. Сборка layout-документа

Программный вход:

- [pdf_layout_parser/pipeline.py](C:/Users/User/gazprom/pdf_layout_parser/pipeline.py)

Основная функция:

- `process_pdf(pdf_path, output_dir="output/layout", images_dir="output/images")`

### 3. Подготовка OCR-очереди

```bash
python prepare_image_ocr_queue.py
```

### 4. Разделение OCR-кандидатов на рукописные и печатные

```bash
python split_text_images_handwritten_vs_printed.py
```

### 5. OCR рукописных фрагментов

```bash
python recognize_handwritten_local.py
```

### 6. Анализ комбинированных изображений

```bash
python parser_comb_im/main.py --input parser_comb_im/comb_image --output parser_comb_im/output
```

## Основные зависимости

Файл зависимостей проекта:

- [requirements.txt](C:/Users/User/gazprom/requirements.txt)

В коде также используются дополнительные библиотеки, которые нужны для полного запуска всех пайплайнов:

- `torch`
- `transformers`
- `easyocr`
- `img2table`
- `pymupdf`
- `opencv-python` или `opencv-python-headless`
- `pandas`
- `numpy`
- `Pillow`
- `symspellpy`
- `wordfreq`
- `tqdm`

## Где лежат данные и артефакты

- `pdfs/` — входные PDF-файлы;
- `hf_models/` — локальный кэш Hugging Face моделей;
- `output/images/` — извлеченные и сохраненные изображения из PDF;
- `output/layout/` — собранные JSON и Markdown документы;
- `output/ocr_queue/` — промежуточные очереди для OCR;
- `output/handwritten_ocr/` — результаты распознавания рукописного текста;
- `parser_comb_im/output/` — регионы, выделенные из обычных изображений.

## Краткое назначение репозитория

Репозиторий решает задачу структурного разбора документов на уровне PDF и изображений. Основная идея архитектуры — не использовать один универсальный обработчик для всех типов контента, а собирать итоговый документ из специализированных модулей:

- таблицы извлекаются отдельно;
- нативный текст читается отдельно;
- встроенные изображения классифицируются отдельно;
- рукописные и печатные растровые фрагменты обрабатываются разными ветками;
- финальная структура документа собирается в общем layout-пайплайне.
