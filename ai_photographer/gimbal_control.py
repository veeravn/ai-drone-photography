# ai_photographer/gimbal_control.py
from typing import Optional

class Gimbal:
    """
    Lightweight gimbal abstraction.

    - In SITL (simulation): only stores requested pitch/yaw values.
    - In hardware mode: replace 'set_angles' with MAVSDK gimbal commands.
    """

    def __init__(self):
        self.last_pitch_deg: Optional[float] = None
        self.last_yaw_deg: Optional[float] = None

    async def set_angles(self, pitch_deg: float, yaw_deg: float = 0.0):
        """Set desired pitch/yaw angles (SITL: no-op)."""
        # SITL: remember angles only
        self.last_pitch_deg = float(pitch_deg)
        self.last_yaw_deg = float(yaw_deg)
        return True

    async def set_pitch_yaw(self, pitch_deg: float, yaw_deg: Optional[float] = None):
        """Alias to match future MAVSDK API naming."""
        return await self.set_angles(pitch_deg, yaw_deg or 0.0)

    async def center(self):
        """Return to neutral gimbal position."""
        return await self.set_angles(0.0, 0.0)
