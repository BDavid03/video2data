"""The single frozen M2 request; scored episodes are deliberately unsupported."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    episode: int = 39
    acquisition_seconds: float = 0.5
    seed: int = 0

    def __post_init__(self):
        if (self.episode, self.acquisition_seconds, self.seed) != (39, 0.5, 0):
            raise ValueError('M2 Stage 0 admits only the frozen tuning-39 request')
