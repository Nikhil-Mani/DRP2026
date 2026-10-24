from dataclasses import dataclass

@dataclass
class Thresholds:
    # seperation threshold
    xi: float

thresholds = Thresholds(0.5)
