from __future__ import annotations

import os

from pdf_image_parser.image_extractor import extract_images

from .builder import build_document_from_blocks
from .export_markdown import save_document_json, save_document_markdown


def process_pdf(
    pdf_path: str,
    output_dir: str = "output/layout",
    images_dir: str = "output/images",
    reset_output_dir: bool = False,
    verbose: bool = False,
):
    blocks = extract_images(
        pdf_path=pdf_path,
        output_images_dir=images_dir,
        reset_output_dir=reset_output_dir,
        verbose=verbose,
    )

    document = build_document_from_blocks(pdf_path, blocks)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(output_dir, f"{pdf_name}.json")
    md_path = os.path.join(output_dir, f"{pdf_name}.md")

    save_document_json(document, json_path)
    save_document_markdown(document, md_path)

    return {
        "document": document,
        "blocks": blocks,
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
