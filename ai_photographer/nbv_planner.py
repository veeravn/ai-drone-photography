# ai_photographer/nbv_planner.py
from math import cos, sin, pi
from typing import Dict, List


def sample_views(cfg: Dict) -> List[Dict]:
    radii = cfg.get("radii_m", [3, 6, 10])
    heights = cfg.get("heights_m", [1.5, 3.0])
    steps = int(cfg.get("azimuth_steps", 12))
    cands = []
    for r in radii:
        for z in heights:
            for k in range(steps):
                theta = 2 * pi * (k / steps)
                x = r * cos(theta)
                y = r * sin(theta)
                yaw_deg = (theta * 180 / pi) % 360
                cands.append({"pos": (x, y, z), "yaw": yaw_deg})
    return cands


def candidate_score(c: Dict) -> float:
    x, y, z = c["pos"]
    alt_term = -abs(z - 2.0)
    yaw_bins = int(c["yaw"] // 30)
    diversity = 1.0 - (abs(6 - yaw_bins) / 6.0)
    return alt_term + diversity


def pick_next_best(candidates: List[Dict]) -> Dict:
    return max(candidates, key=candidate_score)
