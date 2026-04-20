from __future__ import annotations

import os

from pdf_image_parser.image_extractor import extract_images
from pdf_table_parser.table_extractor import TableExtractor

from .builder import build_document_from_sources
from .export_markdown import save_document_json, save_document_markdown
from .native_text import extract_native_text_blocks


def process_pdf(
    pdf_path: str,
    output_dir: str = "output/layout",
    images_dir: str = "output/images",
    reset_output_dir: bool = False,
    verbose: bool = False,
):
    image_blocks = extract_images(
        pdf_path=pdf_path,
        output_images_dir=images_dir,
        reset_output_dir=reset_output_dir,
        verbose=verbose,
    )

    table_extractor = TableExtractor()
    native_tables = table_extractor.extract_all(pdf_path)

    native_text_blocks = extract_native_text_blocks(pdf_path)

    document = build_document_from_sources(
        pdf_path=pdf_path,
        blocks=image_blocks,
        tables=native_tables,
        native_text_blocks=native_text_blocks,
    )

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(output_dir, f"{pdf_name}.json")
    md_path = os.path.join(output_dir, f"{pdf_name}.md")

    save_document_json(document, json_path)
    save_document_markdown(document, md_path)

    return {
        "document": document,
        "image_blocks": image_blocks,
        "native_tables": native_tables,
        "native_text_blocks": native_text_blocks,
        "json_path": json_path,
        "md_path": md_path,
    }


if __name__ == "__main__":
    PDFS = [
        "pdfs/document_001.pdf",
        "pdfs/document_002.pdf",
    ]

    for i, pdf_path in enumerate(PDFS):
        result = process_pdf(
            pdf_path=pdf_path,
            output_dir="output/layout",
            images_dir="output/images",
            reset_output_dir=(i == 0),
            verbose=True,
        )

        print("\n" + "=" * 80)
        print("PDF:", pdf_path)
        print("JSON:", result["json_path"])
        print("MD:", result["md_path"])
        print("PAGES:", len(result["document"].pages))
        print("NATIVE TABLES:", len(result["native_tables"]))
        print("NATIVE TEXT BLOCKS:", len(result["native_text_blocks"]))
        print("IMAGE BLOCKS:", len(result["image_blocks"]))
        print("=" * 80)

        for page in result["document"].pages:
            print(f"page={page.page_number} elements={len(page.elements)}")
            for element in page.elements:
                preview = (
                    element.markdown[:80]
                    if element.type == "table"
                    else element.text[:80]
                )
                print(
                    f"  order={element.order} "
                    f"type={element.type} "
                    f"label={element.source_label} "
                    f"conf={element.confidence:.3f} "
                    f"preview={preview!r}"
                )
