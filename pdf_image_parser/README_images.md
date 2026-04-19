
---

# Блок обработки изображений

Этот модуль отвечает за:

* поиск image-блоков в PDF
* классификацию блоков на:

  * `save` — сохранить как PNG
  * `ocr` — не сохранять, отправить в текстовую ветку
  * `drop` — выбросить
* сохранение изображений в папку `output/images`
* именование файлов в формате `doc_<id>_image_<order>.png`

## Где лежит код

```text
pdf_image_parser/
├── image_extractor.py
└── test_images.py
```

## Что нужно установить

Из корня проекта:

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision transformers pillow opencv-python pymupdf
```

Если зависимости уже частично стоят, достаточно:

```bash
pip install torch torchvision transformers pillow opencv-python pymupdf
```

## Какая модель используется

Для классификации используется:

```text
microsoft/dit-base-finetuned-rvlcdip
```

Модель скачивается автоматически при первом запуске и сохраняется локально в:

```text
./hf_models
```

## Как запустить проверку

Из корня проекта:

```bash
python pdf_image_parser\test_images.py
```

## Что делает тест

Скрипт:

* берёт PDF-файлы из папки `pdfs/`
* извлекает image-блоки
* классифицирует их
* печатает в консоль:

  * страницу
  * bbox
  * размер блока
  * действие (`save/ocr/drop`)
  * метку модели
  * confidence
* сохраняет только нужные изображения в `output/images`

## Формат результата

В консоли будет что-то вроде:

```text
[page=7] bbox=(...) size=(520x380) action=save label=scheme_like conf=0.900
```

Это значит:

* блок найден на странице 7
* классифицирован как картинка
* будет сохранён в `output/images`

Если будет:

```text
action=ocr
```

это значит:

* блок не сохраняется как PNG
* он должен идти в текстовую/OCR-ветку

## Где смотреть сохранённые изображения

После запуска:

```text
output/images/
```

## Как сейчас настроена логика

Сохраняем как картинки:

* схемы
* диаграммы
* иллюстрации
* обычные графические блоки

Не сохраняем как картинки:

* рукописный текст
* печатный текст в растре
* page-like блоки
* таблицы-картинки
* document-like изображения

## Если нужно проверить другой PDF

В файле:

```text
pdf_image_parser/test_images.py
```

замени список PDF на нужный:

```python
PDFS = [
    "pdfs/document_001.pdf",
    "pdfs/document_002.pdf",
]
```

или оставь один конкретный файл:

```python
PDFS = [
    "pdfs/document_002.pdf",
]
```

## Важный момент

Папка `output/images` очищается перед запуском, если в вызове стоит:

```python
reset_output_dir=True
```

Если нужно прогнать несколько PDF подряд и сохранить картинки от всех файлов, используйте:

* один раз очистку в начале
* дальше `reset_output_dir=False`

## Команда для повторного запуска

```bash
python pdf_image_parser\test_images.py
```

## Если модель долго грузится

Это нормально при первом запуске:

* сначала скачиваются веса
* потом они берутся из кэша локально

## Если что-то не работает

Проверь:

* что запускаешь из корня проекта
* что папка `pdfs/` существует
* что установлен `torch`, `transformers`, `opencv-python`, `pymupdf`
* что файл называется:

  * `pdf_image_parser/image_extractor.py`
  * `pdf_image_parser/test_images.py`

## Минимальный рабочий сценарий

```bash
venv\Scripts\activate
pip install torch torchvision transformers pillow opencv-python pymupdf
python pdf_image_parser\test_images.py
```
