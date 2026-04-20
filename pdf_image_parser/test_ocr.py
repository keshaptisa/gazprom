from image_extractor import extract_images


PDFS = [
    "pdfs/document_001.pdf",
    "pdfs/document_002.pdf",
]


for pdf_path in PDFS:
    print("\n" + "=" * 80)
    print("PDF:", pdf_path)
    print("=" * 80)

    blocks = extract_images(
        pdf_path=pdf_path,
        output_images_dir="output/images",
        reset_output_dir=False,
        verbose=True,
    )

    print("\nSUMMARY")
    print("TOTAL BLOCKS:", len(blocks))
    print("IMAGE BLOCKS:", sum(1 for b in blocks if b.block_type == "image"))
    print("TEXT BLOCKS:", sum(1 for b in blocks if b.block_type == "text"))

    for i, block in enumerate(blocks, 1):
        print("\n" + "-" * 80)
        print(f"BLOCK #{i}")
        print("page:", block.page_number)
        print("type:", block.block_type)
        print("label:", block.predicted_label)
        print("confidence:", round(block.confidence, 3))
        print("bbox:", block.bbox)

        if block.block_type == "image":
            print("content:", block.content)
        else:
            text = (block.content or "").strip()
            if not text:
                print("content: <EMPTY OCR RESULT>")
            else:
                print("content preview:")
                print(text[:1500])