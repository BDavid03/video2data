"""Frozen trial request for the shipped example task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    arm: str
    seed: int
    budget: dict
    maximum_seconds: float

    def __post_init__(self):
        if self.arm not in ("Z", "L1", "L2", "D1") or self.seed not in (0, 1):
            raise ValueError("Only the protocol's arms and seeds are admitted")
        if (self.budget["world_count"] < 1 or self.maximum_seconds <= 0
                or (self.arm != "D1" and self.budget["world_count"] != 64)):
            raise ValueError("D0 uses 64 worlds; all arms require positive worlds and wall-clock allowance")
