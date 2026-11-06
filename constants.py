from dataclasses import dataclass

@dataclass
class Thresholds:
    # seperation threshold
    xi: float

@dataclass
class MALAParams:
    step_size: int
    total_steps: int

@dataclass
class SMCParams:
    num_particles: int
    max_steps: int

thresholds = Thresholds(0.3)
