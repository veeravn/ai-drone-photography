import os
import csv
import time
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import imagehash
import cv2
from scipy.spatial import cKDTree

@dataclass
class ShotMeta:
    timestamp: float
    img_path: str
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    phash_hex: str
    sharpness: float
    kept: bool
    reason: str

class ShotDB:
    def __init__(self):
        self.shots: List[ShotMeta] = []
        self._kdtree: Optional[cKDTree] = None
        self._positions: Optional[np.ndarray] = None

    def _rebuild_index(self):
        if not self.shots:
            self._kdtree = None
            self._positions = None
            return
        self._positions = np.array([[s.x, s.y, s.z] for s in self.shots if s.kept])
        if len(self._positions) > 0:
            self._kdtree = cKDTree(self._positions)
        else:
            self._kdtree = None

    def add(self, shot: ShotMeta):
        self.shots.append(shot)
        self._rebuild_index()

    def neighbors_within(self, pos: Tuple[float, float, float], radius: float) -> List[ShotMeta]:
        if self._kdtree is None or self._positions is None or len(self._positions) == 0:
            return []
        idxs = self._kdtree.query_ball_point(np.array(pos), r=radius)
        kept_shots = [s for s in self.shots if s.kept]
        return [kept_shots[i] for i in idxs]

def compute_phash(image_path: str, phash_size: int = 16) -> imagehash.ImageHash:
    img = Image.open(image_path).convert("L").resize((256, 256))
    return imagehash.phash(img, hash_size=phash_size)

def compute_sharpness(image_path: str) -> float:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def angular_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)

def decide_duplicate(
    candidate_img: str,
    candidate_pose: Tuple[float, float, float],
    candidate_yaw_deg: float,
    db: ShotDB,
    phash_thresh: int,
    pos_radius_m: float,
    angle_threshold_deg: float,
    sharpness_replace_delta: float,
    phash_size: int,
) -> Tuple[bool, str, imagehash.ImageHash, float]:
    cand_hash = compute_phash(candidate_img, phash_size)
    cand_sharp = compute_sharpness(candidate_img)
    neighbors = db.neighbors_within(candidate_pose, pos_radius_m)

    for n in neighbors:
        n_hash = imagehash.hex_to_hash(n.phash_hex)
        hdist = cand_hash - n_hash
        if hdist <= phash_thresh:
            # Similar image perceptually — check angle too
            if angular_delta_deg(candidate_yaw_deg, n.yaw_deg) <= angle_threshold_deg:
                # Decide replace vs discard
                if cand_sharp > n.sharpness + sharpness_replace_delta:
                    return False, "replace_better_sharpness", cand_hash, cand_sharp
                return True, f"duplicate(h={int(hdist)})", cand_hash, cand_sharp

    return False, "unique", cand_hash, cand_sharp

def append_csv(csv_path: str, shot: ShotMeta):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(shot).keys()))
        if new_file:
            w.writeheader()
        w.writerow(asdict(shot))
