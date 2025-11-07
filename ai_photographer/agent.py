# ai_photographer/agent.py
"""
Rule-based adaptive NBV agent for the AI Photographer.

This agent adjusts drone exploration strategy based on:
- Duplicate streaks (expand search area / raise altitude)
- Image sharpness (trigger retake with gimbal pitch change)
- Normal conditions (continue sampling next-best-view candidates)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np


# ---------- Agent State ----------
@dataclass
class AgentState:
    kept_count: int = 0
    dup_streak: int = 0
    last_sharpness: float = 0.0
    last_kept_sharpness: float = 0.0
    current_alt: float = 0.0
    current_pos_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_action: str = "INIT"


# ---------- Agent Parameters ----------
@dataclass
class AgentParams:
    dup_streak_threshold: int
    low_sharpness_threshold: float
    alt_step_m: float
    max_alt_m: float
    radius_scale: float
    retake_pitch_delta_deg: float


# ---------- NBV Agent ----------
class NBVAgent:
    """
    A lightweight rule-based NBV (Next-Best-View) controller.

    Decision logic:
    1. Repeated duplicates -> expand radius, then raise altitude.
    2. Soft (blurry) last kept shot -> retake with pitch offset.
    3. Otherwise -> continue standard NBV move.
    """

    def __init__(self, params: AgentParams):
        self.p = params
        self.s = AgentState()

    # ----- state updates -----
    def update_after_shot(self, kept: bool, sharpness: float, xyz: Tuple[float, float, float]):
        """Update internal memory after each shot."""
        self.s.last_sharpness = sharpness
        self.s.current_pos_xyz = xyz
        if kept:
            self.s.kept_count += 1
            self.s.last_kept_sharpness = sharpness
            self.s.dup_streak = 0
        else:
            self.s.dup_streak += 1

    # ----- decision logic -----
    def decide(self) -> str:
        """Return an action label for the next step."""
        if self.s.dup_streak >= self.p.dup_streak_threshold:
            # Alternate between widening and raising altitude
            if self.s.last_action != "WIDEN_RADIUS":
                self.s.last_action = "WIDEN_RADIUS"
                return "WIDEN_RADIUS"
            else:
                self.s.last_action = "RAISE_ALT"
                return "RAISE_ALT"

        if self.s.last_kept_sharpness and self.s.last_kept_sharpness < self.p.low_sharpness_threshold:
            self.s.last_action = "RETAKE_PITCH"
            return "RETAKE_PITCH"

        self.s.last_action = "NBV_MOVE"
        return "NBV_MOVE"

    # ----- candidate manipulation -----
    def apply_action_to_candidates(
        self,
        action: str,
        candidates: List[Dict],
        current_alt: float
    ) -> Tuple[List[Dict], float, Optional[float]]:
        """
        Optionally modify candidate list (e.g., widen radius, raise altitude, or retake).

        Returns:
            (new_candidates, new_altitude, optional_gimbal_pitch_offset)
        """
        gimbal_pitch_offset = None
        new_alt = current_alt

        if action == "WIDEN_RADIUS":
            # Expand search radius around current point
            scaled = []
            for c in candidates:
                x, y, z = c["pos"]
                cx, cy, cz = 0.0, 0.0, z
                x = (x - cx) * self.p.radius_scale + cx
                y = (y - cy) * self.p.radius_scale + cy
                scaled.append({**c, "pos": (x, y, z)})
            candidates = scaled

        elif action == "RAISE_ALT":
            new_alt = min(current_alt + self.p.alt_step_m, self.p.max_alt_m)
            lifted = []
            for c in candidates:
                x, y, _z = c["pos"]
                lifted.append({**c, "pos": (x, y, new_alt)})
            candidates = lifted

        elif action == "RETAKE_PITCH":
            # Instruct gimbal to slightly change pitch for a retake
            gimbal_pitch_offset = float(self.p.retake_pitch_delta_deg)

        return candidates, new_alt, gimbal_pitch_offset
