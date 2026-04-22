"""
CLI entry point.

Usage:
    python main.py
    python main.py --output results/ --visualize
    python main.py --lang en ru
"""
import argparse
import os
import sys
from pathlib import Path

from analyzer import LayoutAnalyzer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect text / table / figure regions in images and save crops."
    )
    ap.add_argument("-i", "--input", default="comb_image",
                    help="Directory with input images (default: comb_image)")
    ap.add_argument("-o", "--output", default="output",
                    help="Directory where cropped regions are saved (default: output)")
    ap.add_argument("-v", "--visualize", action="store_true",
                    help="Also save annotated.png with bounding-box overlay")
    ap.add_argument("--lang", nargs="+", default=["en", "ru"], metavar="CODE",
                    help="EasyOCR languages, e.g. --lang en ru (default: en ru)")
    ap.add_argument("--table-threshold", type=float, default=0.5,
                    help="Table Transformer confidence 0-1 (default: 0.5)")
    ap.add_argument("--ocr-confidence", type=float, default=0.3,
                    help="Min EasyOCR word confidence (default: 0.3)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Находим все изображения в папке
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in input_dir.iterdir() if f.suffix in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Loading models (EasyOCR + Table Transformer) …")
    analyzer = LayoutAnalyzer(
        languages=args.lang,
        table_threshold=args.table_threshold,
        ocr_confidence=args.ocr_confidence,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_regions = 0
    for image_path in sorted(image_files):
        print(f"\nAnalysing {image_path.name} …")
        stem = image_path.stem  # имя файла без расширения
        
        regions = analyzer.analyze(str(image_path), str(output_dir), stem)

        if not regions:
            print(f"No regions detected in {image_path.name}")
            continue

        print(f"Detected {len(regions)} region(s):")
        for r in regions:
            print(f"  [{r['class']:6s}]  bbox={r['bbox']}  -> {r['path']}")
        total_regions += len(regions)

        if args.visualize:
            vis_path = output_dir / f"{stem}_annotated.png"
            analyzer.visualize(str(image_path), regions, str(vis_path))
            print(f"Annotated image saved: {vis_path}")

    print(f"\n✓ Processed {len(image_files)} image(s), {total_regions} region(s) total")


if __name__ == "__main__":
    main()