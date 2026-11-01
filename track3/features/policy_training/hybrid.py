"""Admission contract for original residual-policy distillation from executed robot actions."""

from dataclasses import asdict, dataclass
import json
import math


TRAIN_EPISODES = (0, 11, 12, 18, 21, 23, 26, 31)
TUNING_EPISODES = (39, 42)


def _pin(value):
    return (isinstance(value, dict) and isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]))


@dataclass(frozen=True)
class ActionContract:
    """No padding, truncation or dimension-only interpretation of external actions."""
    embodiment: str
    control_mode: str
    control_fps: float
    action_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    runtime_action_schema: dict
    reference_run: dict

    def __post_init__(self):
        if (self.embodiment != "floating_sharpa" or self.control_mode != "normalized_reference_residual"
                or self.control_fps != 20.0 or len(self.action_names) != 56
                or len(set(self.action_names)) != 56 or any(not isinstance(name, str) or not name for name in self.action_names)
                or len(self.lower) != 56 or len(self.upper) != 56
                or not _pin(self.runtime_action_schema) or not _pin(self.reference_run)):
            raise ValueError("Require the pinned ordered 56D floating-Sharpa residual schema at 20 Hz")
        if any(not math.isfinite(x) for x in (*self.lower, *self.upper)) or any(
                lo != -1.0 or hi != 1.0 for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("This adapter admits normalized [-1,1] residual actions only")

    def admit(self, action):
        if len(action) != 56 or any(not math.isfinite(x) or x < lo or x > hi
                                    for x, lo, hi in zip(action, self.lower, self.upper)):
            raise ValueError("Executed action is invalid; no silent clamping or shape conversion")
        return tuple(float(x) for x in action)


def admit_distillation_manifest(manifest, action_contract, *, partition):
    """Check declared manifest provenance; consumers must still hash and read the pinned payloads."""
    if partition not in ("train", "tune"):
        raise ValueError("No eligible held-out policy dataset exists")
    allowed = TRAIN_EPISODES if partition == "train" else TUNING_EPISODES
    if (manifest.get("schema") != "sharpa_achieved_distillation_v1"
            or manifest.get("partition") != partition or not manifest.get("clips")
            or manifest.get("action_source") != "executed_simulator_controls"
            or manifest.get("observation_timing") != "before_action_no_future"
            or manifest.get("action_contract") != json.loads(json.dumps(asdict(action_contract)))
            or not _pin(manifest.get("success_protocol"))):
        raise ValueError("Incomplete achieved-action dataset contract")
    seen = set()
    for clip in manifest["clips"]:
        if (clip.get("source_split") != "public" or clip.get("episode_index") not in allowed
                or clip.get("assistance_scale") != 0.0 or clip.get("complete_duration") is not True
                or clip.get("success") is not True or clip.get("control_fps") != 20.0
                or type(clip.get("transition_count")) is not int or clip["transition_count"] <= 0
                or any(not _pin(clip.get(key)) for key in ("rollout_run", "checkpoint", "reference_run",
                    "success_report", "observations", "executed_actions", "achieved_objects", "timestamps"))):
            raise ValueError("Need successful complete assistance-free achieved robot rollouts")
        identity = clip["rollout_run"]["sha256"]
        if identity in seen:
            raise ValueError("Duplicate rollout must not inflate the corpus")
        seen.add(identity)
    normalization = manifest.get("normalization", {})
    if (normalization.get("fit_episode_indices") != list(TRAIN_EPISODES)
            or normalization.get("action_transform") != "identity_normalized_residual"
            or not _pin(normalization.get("observation_statistics"))):
        raise ValueError("Observation statistics must fit training only; action units stay unchanged")
    return {"partition": partition, "clip_count": len(seen),
            "transition_count": sum(clip["transition_count"] for clip in manifest["clips"]),
            "admission_scope": "Declared manifest only; payload hashing and row admission required before fitting"}


def make_tiny_residual_policy(observation_width, hidden_width=64, *, request):
    """Tiny MLP emitting the upstream normalized residual; PyTorch is imported only after authorization."""
    if type(observation_width) is not int or not 0 < observation_width <= 4096 or hidden_width != 64:
        raise ValueError("Bounded tiny-policy dimensions required")
    from ._common import ROOT, PolicyRequest, authorize
    from track3.paths import output_path
    import os
    import sys
    import tempfile
    if (not isinstance(request, PolicyRequest) or request.source_split != "public"
            or request.episode_index not in (*TRAIN_EPISODES, *TUNING_EPISODES)):
        raise ValueError("Hybrid requests require explicit public training/tuning episode identity")
    authorize("hybrid_training", request)
    cache = output_path(ROOT, ".cache/policy-hybrid")
    for name, relative in {
        "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "XDG_CACHE_HOME": "xdg",
        "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
    }.items():
        os.environ[name] = str(cache / relative)
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cache / "tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import torch
    return torch.nn.Sequential(torch.nn.Linear(observation_width, hidden_width), torch.nn.Tanh(),
                               torch.nn.Linear(hidden_width, 56), torch.nn.Tanh())
