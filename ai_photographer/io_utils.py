# scripts/io_utils.py
import os
import csv
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import imagehash


# ---------- FS helpers ----------

def ensure_dir(p: str) -> None:
    if not p:
        return
    d = p if os.path.isdir(p) else os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def timestamp_name(prefix: str = "shot", ext: str = ".jpg") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"


def append_csv(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(path)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------- Imaging helpers ----------

class Webcam:
    """Simple webcam wrapper that returns a single BGR frame when read()."""

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Webcam index {index} could not be opened.")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read from webcam.")
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def placeholder_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a gray placeholder with timestamp text."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.putText(img, datetime.now().strftime("%H:%M:%S"),
                (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
    return img


def save_image_bgr(path: str, bgr: np.ndarray) -> None:
    ensure_dir(path)
    cv2.imwrite(path, bgr)


# ---------- Quality & duplicate helpers ----------

def sharpness_laplacian(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def phash_from_bgr(bgr: np.ndarray, hash_size: int = 16) -> imagehash.ImageHash:
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil, hash_size=hash_size)


def is_duplicate(curr_hash: imagehash.ImageHash,
                 last_hash: Optional[imagehash.ImageHash],
                 threshold: int) -> Tuple[bool, Optional[int]]:
    if last_hash is None:
        return False, None
    dist = curr_hash - last_hash  # Hamming distance
    return dist <= threshold, dist
