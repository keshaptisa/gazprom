from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from typing import Literal

import cv2
import fitz
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


BlockType = Literal["image", "text"]

MODEL_NAME = "microsoft/dit-base-finetuned-rvlcdip"
CACHE_DIR = "./hf_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Только те классы DiT, которые хотим уводить в OCR
OCR_LABEL_KEYWORDS = [
    "handwritten",
    "invoice",
    "memo",
    "letter",
]

# Те классы DiT, которые хотим сохранять как картинки
SAVE_LABEL_KEYWORDS = [
    "advertisement",
    "form",
    "presentation",
]


@dataclass
class ExtractedImageBlock:
    page_number: int
    bbox: tuple[float, float, float, float]
    block_type: BlockType
    content: str
    image_path: str | None = None
    confidence: float = 1.0
    predicted_label: str | None = None


class DiTClassifier:
    """Классификатор документных изображений на базе DiT."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: str = CACHE_DIR,
        device: str = DEVICE,
    ) -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> tuple[str, float]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        pred_idx = int(logits.argmax(-1).item())
        probs = torch.softmax(logits, dim=1)[0]
        conf = float(probs[pred_idx].item())
        label = str(self.model.config.id2label[pred_idx]).lower()

        return label, conf


def reset_images_dir(images_dir: str) -> None:
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)


def parse_doc_id(pdf_path: str) -> int:
    filename = os.path.basename(pdf_path)
    match = re.match(r"document_(\d+)\.pdf$", filename)
    if not match:
        raise ValueError(f"Expected document_NNN.pdf, got: {filename}")
    return int(match.group(1))


def image_is_too_small(img_bgr: np.ndarray) -> bool:
    h, w = img_bgr.shape[:2]
    return h < 40 or w < 40 or (h * w) < 4000


def image_is_tiny_icon(img_bgr: np.ndarray) -> bool:
    h, w = img_bgr.shape[:2]
    return h < 80 and w < 80


def compute_visual_features(img_bgr: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    area = h * w

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    black_ratio = float((bw == 0).mean())

    edges = cv2.Canny(gray, 80, 180)
    edge_ratio = float((edges > 0).mean())

    row_transitions = float(np.mean(np.sum(bw[:, 1:] != bw[:, :-1], axis=1)))
    col_transitions = float(np.mean(np.sum(bw[1:, :] != bw[:-1, :], axis=0)))

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(
        255 - bw, cv2.MORPH_OPEN, horizontal_kernel
    )
    vertical_lines = cv2.morphologyEx(
        255 - bw, cv2.MORPH_OPEN, vertical_kernel
    )

    hline_ratio = float((horizontal_lines > 0).mean())
    vline_ratio = float((vertical_lines > 0).mean())

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(255 - bw, 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    small_components = int(np.sum((component_areas >= 5) & (component_areas <= 400)))

    contours, _ = cv2.findContours(255 - bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    return {
        "height": float(h),
        "width": float(w),
        "area": float(area),
        "black_ratio": black_ratio,
        "edge_ratio": edge_ratio,
        "row_transitions": row_transitions,
        "col_transitions": col_transitions,
        "hline_ratio": hline_ratio,
        "vline_ratio": vline_ratio,
        "small_components": float(small_components),
        "contour_count": float(contour_count),
    }


def is_page_like(feats: dict[str, float]) -> bool:
    return feats["height"] > 1000 and feats["width"] > 700


def is_table_like(feats: dict[str, float]) -> bool:
    return (
        feats["area"] > 120000
        and feats["hline_ratio"] > 0.030
        and feats["vline_ratio"] > 0.018
        and feats["small_components"] > 220
    )


def is_printed_text_like(feats: dict[str, float]) -> bool:
    return (
        feats["area"] > 100000
        and feats["small_components"] > 280
        and feats["black_ratio"] < 0.30
        and feats["edge_ratio"] > 0.03
    )


def is_handwritten_like(feats: dict[str, float]) -> bool:
    return (
        feats["small_components"] > 150
        and feats["contour_count"] > 500
        and feats["hline_ratio"] < 0.01
        and feats["vline_ratio"] < 0.01
        and feats["black_ratio"] < 0.22
    )


def is_small_text_or_handwritten_like(feats: dict[str, float]) -> bool:
    return (
        feats["area"] > 100000
        and feats["small_components"] > 220
        and feats["contour_count"] > 300
        and feats["black_ratio"] < 0.25
    )


def is_scheme_like(feats: dict[str, float]) -> bool:
    return (
        feats["edge_ratio"] > 0.008
        and feats["edge_ratio"] < 0.08
        and feats["small_components"] < 140
        and feats["hline_ratio"] < 0.015
        and feats["vline_ratio"] < 0.010
        and feats["contour_count"] < 220
        and feats["black_ratio"] < 0.16
    )


def label_is_ocr_like(label: str) -> bool:
    label = label.lower()
    return any(keyword in label for keyword in OCR_LABEL_KEYWORDS)


def label_is_save_like(label: str) -> bool:
    label = label.lower()
    return any(keyword in label for keyword in SAVE_LABEL_KEYWORDS)


def classify_image(
    img_bgr: np.ndarray,
    dit: DiTClassifier,
) -> tuple[str, float, str]:
    if img_bgr is None or img_bgr.size == 0:
        return "drop", 0.0, "invalid"

    if image_is_too_small(img_bgr) or image_is_tiny_icon(img_bgr):
        return "drop", 1.0, "tiny"

    feats = compute_visual_features(img_bgr)

    if is_page_like(feats):
        return "ocr", 0.95, "page_like"

    if is_small_text_or_handwritten_like(feats):
        return "ocr", 0.92, "small_text_or_handwritten_like"

    if is_handwritten_like(feats):
        return "ocr", 0.90, "handwritten_like"

    if feats["small_components"] > 160 and feats["contour_count"] > 260:
        return "ocr", 0.88, "text_like_dense"

    if is_scheme_like(feats):
        return "save", 0.90, "scheme_like"

    if is_table_like(feats):
        return "ocr", 0.95, "table_like"

    if is_printed_text_like(feats):
        return "ocr", 0.90, "printed_text_like"

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    label, conf = dit.predict(pil_image)

    if label_is_save_like(label):
        return "save", conf, label

    if label_is_ocr_like(label):
        return "ocr", conf, label

    if label == "unknown" and feats["area"] > 180000:
        return "ocr", 0.55, "unknown_large"

    return "save", conf, label
def extract_images(
    pdf_path: str,
    output_images_dir: str,
    reset_output_dir: bool = True,
    verbose: bool = False,
) -> list[ExtractedImageBlock]:
    if reset_output_dir:
        reset_images_dir(output_images_dir)
    else:
        os.makedirs(output_images_dir, exist_ok=True)

    dit = DiTClassifier()
    doc = fitz.open(pdf_path)
    doc_id = parse_doc_id(pdf_path)

    blocks: list[ExtractedImageBlock] = []
    seen_hashes: set[str] = set()
    image_order = 1

    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 1:
                continue

            bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            image_bytes = block.get("image")
            if not image_bytes:
                continue

            image_hash = hashlib.md5(image_bytes).hexdigest()
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)

            img_np = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            action, confidence, predicted_label = classify_image(img_bgr, dit)

            if verbose:
                h, w = img_bgr.shape[:2]
                print(
                    f"[page={page_index + 1}] "
                    f"bbox={bbox} size=({w}x{h}) "
                    f"action={action} label={predicted_label} conf={confidence:.3f}"
                )

            if action == "drop":
                continue

            if action == "ocr":
                blocks.append(
                    ExtractedImageBlock(
                        page_number=page_index + 1,
                        bbox=bbox,
                        block_type="text",
                        content="",
                        image_path=None,
                        confidence=confidence,
                        predicted_label=predicted_label,
                    )
                )
                continue

            filename = f"doc_{doc_id}_image_{image_order}.png"
            image_path = os.path.join(output_images_dir, filename)
            cv2.imwrite(image_path, img_bgr)

            blocks.append(
                ExtractedImageBlock(
                    page_number=page_index + 1,
                    bbox=bbox,
                    block_type="image",
                    content=f"![image](images/{filename})",
                    image_path=image_path,
                    confidence=confidence,
                    predicted_label=predicted_label,
                )
            )
            image_order += 1

    return blocks