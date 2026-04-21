"""
Поиск БЕЗРАМОЧНЫХ таблиц в PDF.

Подход:
  1. Детекция bbox  → Table Transformer (microsoft/table-transformer-detection).
  2. Проверка, что таблица действительно безрамочная и текстовая (не скан,
     не мусор из водяных знаков, нет решётки линий).
  3. Вывод структуры строк/столбцов строится по координатам нативных слов
     PDF внутри bbox, а не по выходу структурной модели — это даёт
     значительно более стабильный результат на тестовом корпусе.

Все веса — open-source (Apache-2.0 / MIT-совместимо).
"""
import warnings
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from .config import TT_CONFIG

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


_DETECTION_MODEL_ID = "microsoft/table-transformer-detection"

_detection_cache = {"processor": None, "model": None, "device": None}


def _patch_processor_size(processor):
    try:
        size = processor.size
        raw = dict(size) if isinstance(size, dict) else {}
        if not raw:
            for k in ("shortest_edge", "longest_edge", "height", "width"):
                v = getattr(size, k, None)
                if v is not None:
                    raw[k] = v
        if raw.get("height") and raw.get("width"):
            processor.size = {"height": raw["height"], "width": raw["width"]}
        else:
            processor.size = {
                "shortest_edge": raw.get("shortest_edge") or 800,
                "longest_edge": raw.get("longest_edge") or 1333,
            }
    except Exception:
        pass


def _get_detection():
    if _detection_cache["model"] is None:
        proc = AutoImageProcessor.from_pretrained(_DETECTION_MODEL_ID)
        _patch_processor_size(proc)
        model = TableTransformerForObjectDetection.from_pretrained(
            _DETECTION_MODEL_ID
        ).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _detection_cache.update({"processor": proc, "model": model, "device": device})
    return _detection_cache["processor"], _detection_cache["model"], _detection_cache["device"]


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img.convert("RGB")


@torch.inference_mode()
def _detect_tables(image: Image.Image, threshold: float):
    processor, model, device = _get_detection()
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    tables = []
    id2label = model.config.id2label
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if id2label[int(label)] != "table":
            continue
        tables.append({
            "score": float(score),
            "bbox_px": [float(v) for v in box.tolist()],
        })
    return tables


def _bbox_iou(a, b):
    # Ensure a and b are tuples/lists of 4 floats
    if not (isinstance(a, (list, tuple)) and len(a) == 4):
        return 0.0
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a_area = (ax1 - ax0) * (ay1 - ay0)
    b_area = (bx1 - bx0) * (by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping(tables_px, iou_thr: float = 0.5):
    kept = []
    # Sort by score descending to keep the most confident detections
    # tables_px is a list of dicts: {"score": float, "bbox_px": [float, float, float, float]}
    try:
        sorted_tables = sorted(tables_px, key=lambda t: float(t.get("score", 0)), reverse=True)
    except Exception as e:
        return tables_px
    
    for t in sorted_tables:
        overlap = False
        t_bbox = t.get("bbox_px")
        if not t_bbox:
            continue
        for k in kept:
            k_bbox = k.get("bbox_px")
            if not k_bbox:
                continue
            try:
                if _bbox_iou(t_bbox, k_bbox) >= iou_thr:
                    overlap = True
                    break
            except Exception as e:
                continue
        if not overlap:
            kept.append(t)
    return kept


def _has_lines_opencv(image: Image.Image) -> bool:
    """
    Проверяет наличие горизонтальных или вертикальных линий в изображении с помощью HoughLinesP.
    Если линий почти нет — считаем таблицу безрамочной (или это скан, требующий VLM).
    """
    from .config import TT_CONFIG
    # Конвертируем PIL Image в numpy array (OpenCV формат)
    open_cv_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    
    # Адаптивный порог для выделения границ
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # HoughLinesP для поиска сегментов линий
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=100, # Increased further to only see long table lines
        maxLineGap=5
    )
    
    if lines is None:
        return False
        
    h_lines = 0
    v_lines = 0
    
    # Считаем уникальные Y-координаты для горизонтальных и X для вертикальных
    h_ys = set()
    v_xs = set()
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 2:
            h_ys.add(round(y1 / 5) * 5) # Группируем близкие линии
        elif abs(x1 - x2) < 2:
            v_xs.add(round(x1 / 5) * 5)
            
    h_lines = len(h_ys)
    v_lines = len(v_xs)
    
    max_v = TT_CONFIG["borderless"]["max_vertical_lines"]
    # Если количество вертикальных линий превышает порог, считаем что рамки есть.
    return v_lines > max_v


def _call_vlm_for_table(image: Image.Image) -> list[list[str]]:
    """
    Заглушка для вызова VLM (GPT-4o, Claude 3.5 Sonnet и т.д.)
    В реальности здесь должен быть вызов API.
    """
    # Пока возвращаем пустую сетку или текст-заглушку
    return [["[VLM_PENDING]", "Таблица отправлена на VLM"], ["Для работы вставьте API ключ"]]


def _px_bbox_to_pdf(bbox_px, scale):
    return [v / scale for v in bbox_px]


# ----------------------------------------------------------------------
# Фильтры качества: отсев сканов и шумных таблиц.
# ----------------------------------------------------------------------
def _count_grid_lines(page: pdfplumber.pdf.Page, pdf_bbox, min_len: float):
    """Считает горизонтальные/вертикальные линии сетки внутри bbox."""
    x0, top, x1, bottom = pdf_bbox
    h = v = 0
    for ln in list(page.lines) + list(page.edges):
        lx0, ltop = ln.get("x0", 0), ln.get("top", 0)
        lx1, lbottom = ln.get("x1", 0), ln.get("bottom", 0)
        if lx1 < x0 - 2 or lx0 > x1 + 2:
            continue
        if lbottom < top - 2 or ltop > bottom + 2:
            continue
        w, hgt = abs(lx1 - lx0), abs(lbottom - ltop)
        if hgt <= 1 and w >= min_len:
            h += 1
        elif w <= 1 and hgt >= min_len:
            v += 1
    for r in page.rects:
        rx0, rtop = r.get("x0", 0), r.get("top", 0)
        rx1, rbottom = r.get("x1", 0), r.get("bottom", 0)
        if rx1 < x0 - 2 or rx0 > x1 + 2:
            continue
        if rbottom < top - 2 or rtop > bottom + 2:
            continue
        rw, rh = abs(rx1 - rx0), abs(rbottom - rtop)
        if rh <= 1.5 and rw >= min_len:
            h += 1
        elif rw <= 1.5 and rh >= min_len:
            v += 1
    return h, v


def _is_borderless(page: pdfplumber.pdf.Page, pdf_bbox, cfg) -> bool:
    b = cfg["borderless"]
    h, v = _count_grid_lines(page, pdf_bbox, b["min_line_length"])
    return h <= b["max_horizontal_lines"] and v <= b["max_vertical_lines"]


def _words_in_bbox(page: pdfplumber.pdf.Page, pdf_bbox, shrink: float = 0.0):
    """
    Возвращает слова pdfplumber, центры которых попадают в bbox (с небольшим
    отступом внутрь — чтобы не захватить соседний заголовок/подпись).
    """
    x0, top, x1, bottom = pdf_bbox
    x0 += shrink
    top += shrink
    x1 -= shrink
    bottom -= shrink

    words = page.extract_words(
        keep_blank_chars=False,
        use_text_flow=False,
        extra_attrs=["size"],
    )

    out = []
    for w in words:
        # Используем пересечение площадей, а не только центр, если shrink=0
        if shrink == 0:
            # Проверяем пересечение bbox слова с bbox таблицы
            if (w["x0"] < x1 and w["x1"] > x0 and
                w["top"] < bottom and w["bottom"] > top):
                out.append(w)
        else:
            cx = (w["x0"] + w["x1"]) / 2.0
            cy = (w["top"] + w["bottom"]) / 2.0
            if x0 <= cx <= x1 and top <= cy <= bottom:
                out.append(w)

    return out


_BULLET_CHARS = {"•", "◦", "▪", "●", "○", "■", "□", "▸", "▹", "▶", "–"}


def _is_bullet_list(words, min_fraction: float = 0.35) -> bool:
    """
    Возвращает True, если bbox содержит маркированный список, а не таблицу.
    Признак: значительная доля строк начинается с маркера списка (•, ◦, ▪ и т.д.).
    """
    rows = _cluster_rows(words, y_tol=3.0)
    if not rows:
        return False
    bullet_rows = sum(
        1 for r in rows
        if r["words"] and min(r["words"], key=lambda w: w["x0"])["text"] in _BULLET_CHARS
    )
    return bullet_rows / max(len(rows), 1) >= min_fraction


def _quality_passes(words, cfg_filter) -> bool:
    """
    Отсеивает «шум» и сканы.
    """
    n = len(words)
    if n < cfg_filter["min_native_words"]:
        return False

    if n == 0:
        return True # Если слов 0 и min_native_words 0, проходим

    total_chars = sum(len(w["text"]) for w in words)
    avg_len = total_chars / max(n, 1)
    if avg_len < cfg_filter["min_avg_word_len"]:
        return False
    single = sum(1 for w in words if len(w["text"]) <= 1)
    ratio = single / n
    if ratio > cfg_filter["max_single_char_ratio"]:
        return False

    return True


# ----------------------------------------------------------------------
# Вывод структуры таблицы по словам (без структурной модели).
# ----------------------------------------------------------------------
def _cluster_rows(words, y_tol: float):
    """Группирует слова в визуальные строки по y-центру (с допуском)."""
    # Create list of (y_center, word_dict) and sort only by y_center
    # to avoid TypeError if word_dicts are compared
    items = []
    for w in words:
        y_center = (w["top"] + w["bottom"]) / 2.0
        items.append((y_center, w))
    
    items.sort(key=lambda x: x[0])
    
    lines = []
    for y, w in items:
        if lines and abs(y - lines[-1]["y"]) <= y_tol:
            lines[-1]["words"].append(w)
            lines[-1]["ys"].append(y)
            lines[-1]["y"] = sum(lines[-1]["ys"]) / len(lines[-1]["ys"])
        else:
            lines.append({"y": y, "ys": [y], "words": [w]})
    # Внутри строки — слева направо.
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
    return lines


def _estimate_line_height(lines) -> float:
    if not lines:
        return 10.0
    heights = []
    for ln in lines:
        for w in ln["words"]:
            heights.append(w["bottom"] - w["top"])
    heights.sort()
    return heights[len(heights) // 2] if heights else 10.0


def _find_header_line(page: pdfplumber.pdf.Page, pdf_bbox) -> float | None:
    """Ищет горизонтальную линию в верхней трети bbox таблицы."""
    x0, top, x1, bottom = pdf_bbox
    table_height = bottom - top
    header_zone_bottom = top + (table_height * 0.4) # ищем в верхней части
    
    # Собираем линии и прямоугольники, которые выглядят как горизонтальные разделители
    candidates = []
    for ln in list(page.lines) + list(page.edges):
        lx0, ltop = ln.get("x0", 0), ln.get("top", 0)
        lx1, lbottom = ln.get("x1", 0), ln.get("bottom", 0)
        # Горизонтальность и попадание в bbox
        if abs(ltop - lbottom) < 2 and lx1 > x0 and lx0 < x1:
            if top < ltop < header_zone_bottom:
                candidates.append(ltop)
    
    for r in page.rects:
        rtop, rbottom = r.get("top", 0), r.get("bottom", 0)
        rx0, rx1 = r.get("x0", 0), r.get("x1", 0)
        if abs(rtop - rbottom) < 2 and rx1 > x0 and rx0 < x1:
            if top < rtop < header_zone_bottom:
                candidates.append(rtop)
                
    if not candidates:
        return None
    # Возвращаем самую верхнюю подходящую линию
    return min(candidates)


def _group_logical_rows(lines, line_h: float, gap_factor: float, header_line_y: float | None = None):
    """
    Объединяет соседние визуальные строки в одну логическую строку таблицы.
    Если есть header_line_y, то строки выше и ниже этой линии НЕ объединяются.
    """
    if not lines:
        return []
    
    grouped = [[lines[0]]]
    for i in range(1, len(lines)):
        prev = lines[i-1]
        cur = lines[i]
        
        # Проверка разделителя заголовка
        is_header_split = False
        if header_line_y is not None:
            if prev["y"] < header_line_y < cur["y"]:
                is_header_split = True
        
        gap = cur["y"] - prev["y"]
        if gap <= line_h * gap_factor and not is_header_split:
            grouped[-1].append(cur)
        else:
            grouped.append([cur])
    return grouped


def _build_grid(words, bbox, clustering_cfg, page: pdfplumber.pdf.Page | None = None):
    """Строит двумерную сетку строк/столбцов по словам в bbox."""
    if not words:
        return []
    lines = _cluster_rows(words, clustering_cfg["row_y_tolerance"])
    if not lines:
        return []

    header_line_y = None
    if page is not None:
        header_line_y = _find_header_line(page, bbox)

    line_h = _estimate_line_height(lines)
    logical = _group_logical_rows(lines, line_h, clustering_cfg["row_gap_factor"], header_line_y)
    col_edges = _detect_column_edges(words, bbox, clustering_cfg["col_min_gap"])
    if len(col_edges) < 2:
        return []
    grid = []
    for rl in logical:
        grid.append(_assign_columns(rl, col_edges))
    return grid


def _get_row_segments(row_words, min_gap: float = 8.0):
    """Группирует слова строки в сегменты, разделённые пробелами > min_gap."""
    ws = sorted(row_words, key=lambda w: w["x0"])
    if not ws:
        return []
    segs = [[ws[0]]]
    for w in ws[1:]:
        if w["x0"] - segs[-1][-1]["x1"] > min_gap:
            segs.append([w])
        else:
            segs[-1].append(w)
    return segs


def _detect_column_edges(words, bbox, min_gap: float = 10.0):
    """
    Определяет позиции столбцов по «опорной» строке — строке с наибольшим
    числом сегментов (обычно заголовок или самая полная строка данных).
    Остальные строки при назначении ячеек snap к ближайшей опорной позиции,
    что устраняет фантомные столбцы от переносов и разных суб-таблиц.
    """
    if not words:
        return []

    rows = _cluster_rows(words, y_tol=3.0)
    if not rows:
        return []

    # x0 сегментов для каждой строки
    row_segs: list[list[float]] = []
    for row in rows:
        xs = [seg[0]["x0"] for seg in _get_row_segments(row["words"], min_gap=min_gap)]
        row_segs.append(xs)

    # Опорная строка — с наибольшим числом сегментов
    anchor_xs = sorted(max(row_segs, key=len))

    # Кластеризуем опорные x0 с допуском min_gap
    bands: list[float] = []
    for x in anchor_xs:
        if not bands or x - bands[-1] > min_gap:
            bands.append(x)

    if len(bands) < 2:
        # Запасной вариант: берём все уникальные x0
        all_xs = sorted(x for xs in row_segs for x in xs)
        bands = [all_xs[0]]
        for x in all_xs[1:]:
            if x - bands[-1] > min_gap:
                bands.append(x)

    right = max(w["x1"] for w in words) + min_gap
    bands.append(right)
    return bands


def _assign_columns(row, col_edges):
    """
    Распределяет слова логической строки по столбцам.
    Единицей является сегмент (группа слов без большого пробела).
    Каждый сегмент назначается столбцу с БЛИЖАЙШИМ левым краем —
    это корректно работает даже когда строки разных суб-таблиц имеют
    слегка отличающиеся позиции колонок.
    row — список словарей visual-line, каждый содержит ключ 'words'.
    """
    n_cols = len(col_edges) - 1
    if n_cols <= 0:
        return []
    result = [""] * n_cols
    col_starts = col_edges[:-1]  # левые края столбцов

    for line in row:
        for seg in _get_row_segments(line["words"], min_gap=8.0):
            seg_x0 = seg[0]["x0"]
            seg_text = " ".join(w["text"] for w in seg)

            # Ближайший столбец по x0
            col_idx = min(range(n_cols), key=lambda i: abs(col_starts[i] - seg_x0))

            if result[col_idx]:
                result[col_idx] += " " + seg_text
            else:
                result[col_idx] = seg_text

    return result


# ----------------------------------------------------------------------
# Публичный API
# ----------------------------------------------------------------------
def find_clean_tables(pdf_path: Path | str, skip_bboxes: dict | None = None):
    """
    Возвращает {page_num(1-based): [table_info]}, где
    table_info = {"data": [[str, ...]], "rows": int, "cols": int, "bbox": (x0,top,x1,bottom), "score": float}.

    skip_bboxes — опционально {page_num: [pdf_bbox, ...]}, области, которые
    уже покрыты таблицами с линиями (чтобы не дублировать).
    """
    cfg = TT_CONFIG
    skip_bboxes = skip_bboxes or {}
    result: dict[int, list[dict]] = {}

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    try:
        with pdfplumber.open(str(pdf_path)) as plumber:
            for page_num, fitz_page in enumerate(doc, 1):
                plumber_page = plumber.pages[page_num - 1]
                page_h = plumber_page.height

                # Рендерим страницу один раз
                image_full = _render_page(fitz_page, dpi=cfg["render_dpi"])
                scale = cfg["render_dpi"] / 72.0

                tables_px = _detect_tables(image_full, threshold=cfg["detection_threshold"])
                tables_px = _merge_overlapping(tables_px, iou_thr=cfg["merge_iou"])

                page_tables = []
                for t in tables_px:
                    pdf_bbox = tuple(_px_bbox_to_pdf(t["bbox_px"], scale))
                    x0, top, x1, bottom = pdf_bbox
                    score = t["score"]

                    # Отсекаем колонтитулы.
                    if top < cfg["page_margin_top"] or (page_h - bottom) < cfg["page_margin_bottom"]:
                        continue

                    # Кропаем изображение таблицы для OpenCV анализа
                    crop_px = t["bbox_px"]
                    table_img = image_full.crop((crop_px[0], crop_px[1], crop_px[2], crop_px[3]))

                    # Не берём то, что уже покрыто таблицей с линиями.
                    overlap = False
                    for sb in skip_bboxes.get(page_num, []):
                        if not (isinstance(sb, (list, tuple)) and len(sb) == 4):
                            continue
                        if _bbox_iou(pdf_bbox, sb) > 0.5:
                            overlap = True
                            break
                    if overlap:
                        continue

                    # Только безрамочные.
                    has_lines = _has_lines_opencv(table_img)

                    if has_lines:
                        continue

                    words = _words_in_bbox(plumber_page, pdf_bbox)

                    if not words:
                        continue
                    
                    if not _quality_passes(words, cfg["filter"]):
                        continue

                    if _is_bullet_list(words):
                        continue
                    
                    grid = _build_grid(words, pdf_bbox, cfg["clustering"], page=plumber_page)

                    if not grid:
                        continue

                    rows = len(grid)
                    cols = max(len(r) for r in grid) if grid else 0
                    if rows < cfg["filter"]["min_rows"] or cols < cfg["filter"]["min_cols"]:
                        continue

                    total = rows * cols
                    filled = sum(1 for r in grid for c in r if c and c.strip())
                    fill_ratio = filled / total if total > 0 else 0
                    if total == 0 or fill_ratio < cfg["filter"]["min_fill_ratio"]:
                        continue

                    page_tables.append({
                        "data": grid,
                        "rows": rows,
                        "cols": cols,
                        "bbox": pdf_bbox,
                        "score": t["score"],
                    })

                result[page_num] = page_tables
    finally:
        doc.close()

    return result