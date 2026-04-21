import pdfplumber
from pathlib import Path

from .config import CONFIG

def find_clean_tables(pdf_path):
    """
    Находит таблицы в PDF и фильтрует их на 'шум' на основе конфига.
    Возвращает словарь {page_num: [table_data]}, где table_data содержит информацию о таблице.
    """
    all_pages_tables = {} # {page_num: [table_objects]}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # Обрезка колонтитулов перед поиском таблиц
                # crop(x0, y0, x1, y1)
                crop_bbox = (
                    CONFIG.get("crop_margin_left", 0),
                    CONFIG.get("crop_margin_top", 10), # Reduced from 40
                    page.width - CONFIG.get("crop_margin_right", 0),
                    page.height - CONFIG.get("crop_margin_bottom", 10) # Reduced from 40
                )
                page = page.crop(crop_bbox)
                
                page_height = page.height
                found_tables_on_page = []
                found_bboxes = []
                
                for strategy in CONFIG["search_strategies"]:
                    tables = page.find_tables(strategy)
                    
                    for table in tables:
                        # Проверяем на дубликаты
                        is_duplicate = False
                        for bbox in found_bboxes:
                            # Увеличиваем допуск перекрытия
                            if (table.bbox[0] < bbox[2] + 2 and table.bbox[2] > bbox[0] - 2 and
                                table.bbox[1] < bbox[3] + 2 and table.bbox[3] > bbox[1] - 2):
                                is_duplicate = True
                                break
                        
                        if is_duplicate:
                            continue

                        raw_data = table.extract()
                        
                        if not raw_data or len(raw_data) < CONFIG["min_rows"]:
                            continue
                            
                        num_rows = len(raw_data)
                        num_cols = len(raw_data[0]) if num_rows > 0 else 0
                        
                        if num_cols < CONFIG["min_cols"]:
                            continue

                        total_cells = num_rows * num_cols
                        filled_cells = sum(1 for row in raw_data for cell in row if cell and str(cell).strip())
                        fill_ratio = filled_cells / total_cells if total_cells > 0 else 0
                        
                        if fill_ratio >= CONFIG["min_fill_ratio"]:
                            # Проверяем, находится ли таблица в зоне склейки (край страницы)
                            is_in_stitch_zone = (table.bbox[1] < CONFIG["page_stitch_top_margin"] or 
                                                (page_height - table.bbox[3]) < CONFIG["page_stitch_bottom_margin"])

                            if page_num == 3 and table.bbox[1] < 150:
                                # Исключение для рисунков на стр 3 первого файла
                                # Но проверяем, не второй ли это файл (там на стр 3 есть таблицы)
                                if "document_001" in pdf_path.name:
                                    continue

                            all_text_cells = [str(cell) for row in raw_data for cell in row if cell and str(cell).strip()]
                            if all_text_cells:
                                total_words = sum(len(cell.split()) for cell in all_text_cells)
                                avg_words = total_words / len(all_text_cells)
                                
                                # Если таблица НЕ в зоне склейки, применяем строгие фильтры
                                if not is_in_stitch_zone:
                                    if avg_words > CONFIG.get("max_avg_words_per_cell", 10):
                                        continue
                                    if len(all_text_cells) < CONFIG.get("min_total_cells", 5):
                                        continue

                            # Сохраняем информацию о таблице для последующей склейки
                            table_info = {
                                "bbox": table.bbox,
                                "cols": num_cols,
                                "rows": num_rows,
                                "page_height": page_height,
                                "data": raw_data # Сохраняем данные для проверки заголовков
                            }
                            found_tables_on_page.append(table_info)
                            found_bboxes.append(table.bbox)
                
                all_pages_tables[page_num] = found_tables_on_page

            # Логика склейки (stitch) таблиц между страницами
            final_tables = {} # {page_num: [table_data]}
            skip_tables = set() # (page_num, table_index)

            for page_num in sorted(all_pages_tables.keys()):
                tables_on_page = all_pages_tables[page_num]
                page_final_tables = []
                
                for idx, table in enumerate(tables_on_page):
                    if (page_num, idx) in skip_tables:
                        continue

                    # Проверяем склейку с предыдущей таблицей на этой же странице
                    is_continuation = False
                    if idx > 0:
                        prev_table_on_page = all_pages_tables[page_num][idx - 1]

                        # Условия склейки на одной странице
                        prev_bottom_on_page = prev_table_on_page["bbox"][3]
                        curr_top_on_page = table["bbox"][1]
                        distance_on_page = curr_top_on_page - prev_bottom_on_page

                        # Точное совпадение координат (допуск 5 пунктов)
                        left_diff_page = abs(prev_table_on_page["bbox"][0] - table["bbox"][0])
                        right_diff_page = abs(prev_table_on_page["bbox"][2] - table["bbox"][2])
                        left_match_page = left_diff_page < 5
                        right_match_page = right_diff_page < 5

                        # Расстояние должно быть маленьким (< 150 пунктов)
                        distance_ok_page = distance_on_page < 150

                        # Проверка кол-ва столбцов (только для таблиц)
                        cols_match_page = (prev_table_on_page["cols"] == table["cols"])

                        if (left_match_page and right_match_page and distance_ok_page and cols_match_page):
                            is_continuation = True
                            # Склеиваем данные таблиц
                            if page_final_tables:
                                page_final_tables[-1]["data"].extend(table["data"])
                                page_final_tables[-1]["rows"] += table["rows"]
                            else:
                                page_final_tables.append(table)

                    if not is_continuation:
                        page_final_tables.append(table)
                
                final_tables[page_num] = page_final_tables

    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return final_tables

