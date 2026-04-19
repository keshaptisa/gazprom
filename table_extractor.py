"""
Классы и логика для извлечения таблиц.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import pdfplumber
import config
from .table_to_markdown import extract_all_tables, _to_markdown

@dataclass
class ExtractedTable:
    """Данные извлечённой таблицы."""
    page_number: int
    dataframe: pd.DataFrame
    bbox: tuple
    markdown: str
    has_merged_cells: bool = False
    method: str = "pdfplumber"
    confidence: float = 1.0

class TableExtractor:
    """Извлекает таблицы из PDF."""
    
    def extract_all(self, pdf_path: str) -> List[ExtractedTable]:
        """Извлекает все таблицы и возвращает список ExtractedTable."""
        tables_data = extract_all_tables(pdf_path)
        results = []
        
        for t in tables_data:
            # Превращаем grid (List[List[str]]) в DataFrame
            grid = t.get("grid", [])
            if not grid:
                continue
                
            headers = grid[0]
            data = grid[1:] if len(grid) > 1 else []
            
            df = pd.DataFrame(data, columns=headers)
            
            results.append(ExtractedTable(
                page_number=t["page"],
                dataframe=df,
                bbox=t["bbox"],
                markdown=t["markdown"]
            ))
            
        return results
