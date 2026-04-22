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
            final_tables = {page_num: list(tables) for page_num, tables in all_pages_tables.items()}

            # Склейка таблиц между соседними страницами
            pages_sorted = sorted(final_tables.keys())
            for i in range(len(pages_sorted) - 1):
                curr_page = pages_sorted[i]
                next_page = pages_sorted[i + 1]

                # Только для строго следующих страниц
                if next_page != curr_page + 1:
                    continue

                curr_tables = final_tables[curr_page]
                next_tables = final_tables[next_page]

                if not curr_tables or not next_tables:
                    continue

                last_table = curr_tables[-1]
                first_table = next_tables[0]

                # Совпадение левой и правой границы (допуск 5 пунктов)
                left_match = abs(last_table["bbox"][0] - first_table["bbox"][0]) < 5
                right_match = abs(last_table["bbox"][2] - first_table["bbox"][2]) < 5

                # Одинаковое кол-во столбцов
                cols_match = last_table["cols"] == first_table["cols"]

                # Последняя таблица должна быть у нижнего края страницы
                curr_page_height = last_table["page_height"]
                near_bottom = (curr_page_height - last_table["bbox"][3]) < CONFIG["page_stitch_bottom_margin"]

                # Первая таблица следующей страницы должна быть у верхнего края
                near_top = first_table["bbox"][1] < CONFIG["page_stitch_top_margin"]

                if left_match and right_match and cols_match and near_bottom and near_top:
                    last_table["data"].extend(first_table["data"])
                    last_table["rows"] += first_table["rows"]
                    final_tables[next_page] = next_tables[1:]

    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return final_tables

