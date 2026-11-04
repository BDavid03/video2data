"""Episode and FlashSAC budget for one insurance trajectory."""
from dataclasses import dataclass, field
from src.runs import read_json


@dataclass(frozen=True)
class Request:
    source_split: str = "evaluation"
    episode_index: int = 38
    budget: dict = field(default_factory=lambda: {"environment_steps": 3, "worlds": 1, "replay_minimum": 1})
    contract_only: bool = False

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split not in ("public", "evaluation") or self.episode_index in (41, 43, 44):
            raise PermissionError("Excluded split or episode")
        steps, worlds, minimum = (self.budget[k] for k in ("environment_steps", "worlds", "replay_minimum"))
        if any(type(v) is not int or v < 1 for v in (steps, worlds, minimum)):
            raise ValueError("Budget fields must be positive integers")
        if steps % worlds or minimum > steps or minimum > 64 or worlds > 4:
            raise ValueError("Budget must fit the bounded 64-entry replay and at most four worlds")
