import cv2
import numpy as np


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def denoise_image(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=20)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = to_gray(img_bgr)
    gray = denoise_image(gray)
    bw = adaptive_binarize(gray)
    return bw