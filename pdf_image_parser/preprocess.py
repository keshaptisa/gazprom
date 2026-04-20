import cv2
import numpy as np


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def resize_for_ocr(
    img: np.ndarray,
    min_width: int = 1600,
    min_height: int = 1200,
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(min_width / max(w, 1), min_height / max(h, 1), 1.0)
    if scale <= 1.01:
        return img

    interpolation = cv2.INTER_CUBIC if scale > 1.2 else cv2.INTER_LINEAR
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=interpolation,
    )


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise_image(gray: np.ndarray) -> np.ndarray:
    gray = cv2.fastNlMeansDenoising(gray, h=18)
    return cv2.medianBlur(gray, 3)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return bw


def sharpen_image(gray: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    return cv2.filter2D(gray, -1, kernel)


def clean_binary_noise(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


def detect_table_grid(binary: np.ndarray) -> np.ndarray:
    inverted = 255 - binary
    h, w = binary.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, w // 18), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, h // 18)),
    )

    horizontal = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    grid = cv2.add(horizontal, vertical)
    return cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = resize_for_ocr(img_bgr)
    gray = to_gray(img_bgr)
    gray = normalize_contrast(gray)
    gray = denoise_image(gray)
    gray = sharpen_image(gray)
    bw = adaptive_binarize(gray)
    return clean_binary_noise(bw)


def preprocess_table_variants(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    resized = resize_for_ocr(img_bgr, min_width=1800, min_height=1400)
    gray = to_gray(resized)
    normalized = normalize_contrast(gray)
    denoised = denoise_image(normalized)
    sharpened = sharpen_image(denoised)

    adaptive = clean_binary_noise(adaptive_binarize(sharpened))
    otsu = clean_binary_noise(otsu_binarize(sharpened))
    grid = detect_table_grid(adaptive)

    return {
        "original": resized,
        "gray": gray,
        "adaptive": adaptive,
        "otsu": otsu,
        "grid": grid,
    }
