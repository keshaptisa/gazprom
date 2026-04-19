"""Настройки парсинга."""

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
    "text_y_tolerance": 6,  # ← Ослаблено для лучшего обнаружения таблиц
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3
}
