"""
Rangefinder abstraction.

On macOS/SITL you'll likely have no I2C device; we return None safely.
On Raspberry Pi, you can install an actual VL53L0X driver (e.g., via smbus2).
"""

from typing import Optional
import random
import time

class Rangefinder:
    def __init__(self, simulate: bool = True):
        self.simulate = simulate

    def read_meters(self) -> Optional[float]:
        if self.simulate:
            # Simulate a reading between 0.5m and 3.0m
            return round(0.5 + random.random() * 2.5, 2)
        try:
            # TODO: implement real VL53L0X read here when on the Pi
            return None
        except Exception:
            return None
