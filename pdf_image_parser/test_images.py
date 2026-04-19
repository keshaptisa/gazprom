from image_extractor import extract_images, reset_images_dir


PDFS = [
    "pdfs/document_001.pdf",
    "pdfs/document_002.pdf",
]

reset_images_dir("output/images")

for i, pdf_path in enumerate(PDFS):
    print("\n" + "=" * 80)
    print("PDF:", pdf_path)
    print("=" * 80)

    blocks = extract_images(
        pdf_path=pdf_path,
        output_images_dir="output/images",
        reset_output_dir=False,
        verbose=True,
    )

    print("TOTAL BLOCKS:", len(blocks))
    print("IMAGE BLOCKS:", sum(1 for b in blocks if b.block_type == "image"))
    print("TEXT BLOCKS:", sum(1 for b in blocks if b.block_type == "text"))

    for block in blocks:
        print(
            block.page_number,
            block.block_type,
            block.predicted_label,
            round(block.confidence, 3),
            block.bbox,
            block.content[:80],
        )