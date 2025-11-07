# ai_photographer/rangefinder_reader.py
from typing import Optional
import random
import time

class Rangefinder:
    """
    Simple rangefinder abstraction.

    - In simulation mode: returns random distances between 0.5m and 3.0m.
    - In hardware mode: replace with VL53L0X sensor read or MAVSDK telemetry.
    """

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self.last_distance_m: Optional[float] = None

    async def read_distance_m(self) -> Optional[float]:
        """Asynchronously read distance in meters."""
        if self.simulate:
            # Simulate realistic fluctuation
            dist = round(0.5 + random.random() * 2.5, 2)
            self.last_distance_m = dist
            return dist

        try:
            # TODO: Implement real sensor read (e.g., VL53L0X via I2C)
            # Example (pseudo-code):
            # dist = self.sensor.read_range_continuous_millimeters() / 1000.0
            # self.last_distance_m = dist
            # return dist
            return None
        except Exception:
            return None

    def read_meters(self) -> Optional[float]:
        """Synchronous wrapper (for backward compatibility)."""
        if self.simulate:
            dist = round(0.5 + random.random() * 2.5, 2)
            self.last_distance_m = dist
            return dist
        return None
