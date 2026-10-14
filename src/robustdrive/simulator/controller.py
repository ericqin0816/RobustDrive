from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedController:
    """Proportional throttle in simulator mph units; not a learned driving policy."""

    target_speed: float = 12.0
    kp: float = 0.08
    max_throttle: float = 0.3

    def __post_init__(self):
        if self.target_speed <= 0 or self.kp <= 0 or not 0 < self.max_throttle <= 1:
            raise ValueError("Speed, gain, and throttle limits must be positive.")

    def __call__(self, speed: float) -> float:
        if not np.isfinite(speed):
            raise ValueError("Speed must be finite.")
        return float(np.clip(self.kp * (self.target_speed - speed), 0, self.max_throttle))
