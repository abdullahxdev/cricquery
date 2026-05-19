"""OpenCV preprocessing + answer-overlay rendering for the demo."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def clahe_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB — boosts contrast for under-exposed shots."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_pil(pil_img: Image.Image, enhance: bool = True) -> Image.Image:
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if enhance:
        bgr = clahe_enhance(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int, str]]) -> np.ndarray:
    """boxes: list of (x1, y1, x2, y2, label)."""
    out = image.copy()
    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_answer_panel(image: np.ndarray, answer: str, qtype: str, confidence: float) -> np.ndarray:
    """Add a translucent black banner with the answer text at the bottom."""
    h, w = image.shape[:2]
    overlay = image.copy()
    band_h = max(72, int(h * 0.14))
    cv2.rectangle(overlay, (0, h - band_h), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    line1 = f"[{qtype.upper()}]  {answer}"
    line2 = f"confidence: {confidence:.1%}"
    cv2.putText(out, line1, (12, h - band_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, line2, (12, h - band_h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
    return out
