"""
Stub gimbal control for SITL/macOS.

In real hardware (Pixhawk AUX PWM -> gimbal), you'll set:
  - MNT_MODE_OUT = 1 (PWM)
  - AUX1_FUNCTION = 6 (Mount Tilt)
  - And then send MAVLink mount control or servo set commands.

For now, we provide a placeholder API so your main loop doesn't change later.
"""

from typing import Optional

class Gimbal:
    def __init__(self):
        self.last_pitch_deg: Optional[float] = None
        self.last_yaw_deg: Optional[float] = None

    async def set_angles(self, pitch_deg: float, yaw_deg: float = 0.0):
        # In SITL we don't move a real gimbal; just remember requested angles.
        self.last_pitch_deg = float(pitch_deg)
        self.last_yaw_deg = float(yaw_deg)
        return True
