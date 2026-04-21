# Конфигурация для сканера таблиц С РАМКАМИ (table_lines.py)
CONFIG_LINES = {
    "min_rows": 2,
    "min_cols": 2,
    "min_fill_ratio": 0.1,
    "max_avg_words_per_cell": 6,
    "min_total_cells": 6,
    "page_stitch_top_margin": 200,
    "page_stitch_bottom_margin": 200,
    "crop_margin_top": 30,
    "crop_margin_bottom": 30,
    "crop_margin_left": 0,
    "crop_margin_right": 0,
    "search_strategies": [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
            "snap_tolerance": 1,
            "join_tolerance": 1,
            "edge_min_length": 10,
        }
    ],
}

CONFIG = CONFIG_LINES


# === Table Transformer (БЕЗРАМОЧНЫЕ) ===
TT_CONFIG = {
    # DPI для рендера страницы PDF в картинку.
    "render_dpi": 300,

    # Пороги уверенности моделей.
    "detection_threshold": 0.70,
    "structure_threshold": 0.40,

    # Слияние пересекающихся bbox'ов детекций (IoU по площади).
    "merge_iou": 0.4,

    # Отступ (в пикселях) вокруг обнаруженной таблицы.
    "crop_pad_px": 12,

    # Отступы колонтитулов (в пунктах PDF), в них поиск отключён.
    "page_margin_top": 20,
    "page_margin_bottom": 20,

    # Фильтр структуры таблицы.
    "filter": {
        "min_rows": 2,
        "min_cols": 2,
        "min_fill_ratio": 0.1,
        # Минимум нативных слов PDF в bbox (меньше — считаем скан).
        "min_native_words": 0,
        # Минимум символов в среднем на слово (защита от битого текста/глифов).
        "min_avg_word_len": 0.3,
        # Максимальная доля однобуквенных слов в bbox (типично для
        # разорванных/повёрнутых глифов и водяных знаков).
        "max_single_char_ratio": 0.95,
    },

    # Признаки "безрамочности" таблицы (если линий внутри мало).
    "borderless": {
        "min_line_length": 30,
        "max_horizontal_lines": 2,
        "max_vertical_lines": 2,
    },

    # Кластеризация по словам для вывода структуры.
    "clustering": {
        # Допуск по Y при объединении слов в одну визуальную строку (пт).
        "row_y_tolerance": 2.5,
        # Если разрыв между строками больше line_height * этого коэф. —
        # считаем, что это новая логическая строка. Иначе — перенос в ячейке.
        "row_gap_factor": 1.6,
        # Минимальная ширина "пустой" вертикальной полосы между столбцами (пт).
        "col_min_gap": 4.0,
    },
}

TABLE_FINDER_CONFIG = TT_CONFIG