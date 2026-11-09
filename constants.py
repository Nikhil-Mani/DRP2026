from dataclasses import dataclass

@dataclass
class Thresholds:
    # seperation threshold
    xi: float

@dataclass
class MALAParams:
    step_size: float
    total_steps: int

@dataclass
class SMCParams:
    num_particles: int
    max_steps: int
    ESS_alpha: float

thresholds = Thresholds(0.3)
