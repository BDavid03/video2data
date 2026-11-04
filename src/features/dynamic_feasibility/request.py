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


@dataclass(frozen=True)
class ResumeRequest:
    """One independently recorded M0b training operation."""

    operation: str
    arm: str
    budget: dict
    maximum_seconds: float
    resume_state: str
    seed: int = 0

    def __post_init__(self):
        if self.operation not in ('smoke', 'probe', 'train') or self.seed != 0:
            raise ValueError('M0b admits only its fixed seed and training operations')
        if self.budget['world_count'] != 1024 or self.maximum_seconds <= 0:
            raise ValueError('M0b requires 1024 worlds and a positive allowance')
