import math
from typing import List, Dict, Optional, Tuple
import numpy as np

def meters_to_latlon_offsets(d_north_m: float, d_east_m: float, ref_lat_deg: float) -> Tuple[float, float]:
    # Simple equirectangular approximation (good for small distances)
    dlat = d_north_m / 111320.0
    dlon = d_east_m / (111320.0 * math.cos(math.radians(ref_lat_deg)))
    return dlat, dlon

def sample_views(poi_xyz: Tuple[float, float, float],
                 radii: List[float],
                 heights: List[float],
                 azimuth_steps: int) -> List[Dict]:
    px, py, pz = poi_xyz
    out = []
    for r in radii:
        for h in heights:
            for k in range(azimuth_steps):
                az = 2.0 * math.pi * k / azimuth_steps
                x = px + r * math.cos(az)
                y = py + r * math.sin(az)
                z = pz + h
                dx, dy, dz = px - x, py - y, pz - z
                dist = max(1e-6, math.sqrt(dx*dx + dy*dy + dz*dz))
                yaw = math.degrees(math.atan2(dy, dx))
                pitch = math.degrees(math.asin(dz / dist))
                out.append({
                    "pos": (x, y, z),
                    "yaw": yaw,
                    "pitch": pitch,
                    "radius": r
                })
    return out

def score_candidate(cand, kept_positions_np: Optional[np.ndarray], current_pos, min_move_m: float):
    # Visibility/occlusion not modeled yet
    novelty = 100.0
    if kept_positions_np is not None and kept_positions_np.size > 0:
        dists = np.linalg.norm(kept_positions_np - np.array(cand["pos"]), axis=1)
        novelty = float(dists.min())
    travel_cost = float(np.linalg.norm(np.array(cand["pos"]) - np.array(current_pos)))
    # Require a minimum move
    if travel_cost < min_move_m:
        return -1e9
    # Aesthetic nudge for middle radii
    aesthetic = 1.0 if 3.0 <= cand["radius"] <= 10.0 else 0.5
    return 2.0 * novelty + 1.0 * aesthetic - 1.5 * (travel_cost / 1.0)  # assume ~1 m/s for scoring

def pick_next_best(candidates: List[Dict],
                   kept_positions_np: Optional[np.ndarray],
                   current_pos,
                   min_move_m: float) -> Optional[Dict]:
    best = None
    best_score = -1e12
    for c in candidates:
        s = score_candidate(c, kept_positions_np, current_pos, min_move_m)
        if s > best_score:
            best_score = s
            best = c
    return best
