"""Bounded recipe space shared by the baseline queue, the heuristic proposer and the advisor.

A recipe fully specifies one experiment: technique, architecture parameters,
optimisation, data preparation and seed. Every value is validated against finite
bounds before any execution; the advisor can only choose inside this space.
"""

from copy import deepcopy
import random

from track3.features.learning_experiments._config import BOUNDS, DEFAULTS, parameters
from track3.features.depth_student._config import ARCHITECTURES, BOUNDS as STUDENT_BOUNDS, DEFAULTS as STUDENT_DEFAULTS, architecture_config
from ._common import TECHNIQUES, digest

TRAINING_BOUNDS = {"epochs": (1, 100), "batch_size": (1, 8), "learning_rate": (1e-6, 0.1),
                   "weight_decay": (0.0, 1.0)}
SCRATCH_OPTIONS = ("bf16_autocast", "channels_last", "fused_adam", "torch_compile")
DATA_BOUNDS = {"teacher_samples_per_episode": (4, 64), "prepared_max_image_side": (64, 512)}
RESEARCH_NET_MAX_SIDE = 128  # ann/cnn/transformer/gradient_boosting/scratch keep the 65536-pixel guard
STUDENT_TRAINING_BOUNDS = {"warmup_epochs": (0, 20), "ema_decay": (0.0, 0.9999), "augment_color": (0.0, 0.5),
                           "augment_crop": (0.0, 0.5), "grad_clip": (0.0, 100.0), "loss_log_l1": (0.0, 10.0),
                           "loss_silog": (0.0, 10.0), "loss_gradient": (0.0, 10.0)}
STUDENT_TRAINING_DEFAULTS = {"warmup_epochs": 1, "schedule": "cosine", "ema_decay": 0.0, "channels_last": False,
                             "augment_flip": True, "augment_color": 0.1, "augment_crop": 0.0, "grad_clip": 1.0,
                             "loss_log_l1": 1.0, "loss_silog": 0.0, "loss_gradient": 0.0, "flip_tta": False}
REPLICATES = (1, 3)
TRAINING_DEFAULTS = {"epochs": 20, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.00001,
                     "bf16_autocast": False}
DATA_DEFAULTS = {"teacher_samples_per_episode": 16, "prepared_max_image_side": 128}


def _check_number(value, name, low, high, integer=False):
    if integer and (type(value) is not int or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if not integer and (type(value) not in (int, float) or not low <= value <= high):
        raise ValueError(f"{name} must be a number in [{low}, {high}]")
    return int(value) if integer else float(value)


def validate_recipe(recipe):
    """Return a normalised copy or raise ValueError; never executes anything."""
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be an object")
    technique = recipe.get("technique")
    if technique not in TECHNIQUES:
        raise ValueError(f"technique must be one of {TECHNIQUES}")
    result = {"technique": technique, "note": str(recipe.get("note", ""))[:400],
              "seed": _check_number(recipe.get("seed", 31), "seed", 0, 2**31 - 1, integer=True),
              "replicates": _check_number(recipe.get("replicates", 1), "replicates", *REPLICATES, integer=True)}
    params = recipe.get("parameters", {}) or {}
    if technique == "student":
        params = dict(params)
        arch = params.pop("arch", "unet")
        result["parameters"] = {"arch": arch, **architecture_config(arch, params)}
    elif technique == "scratch":
        if params:
            raise ValueError("scratch takes no architecture parameters (fixed coursework CNN)")
        result["parameters"] = {}
    elif technique == "gradient_boosting":
        result["parameters"] = parameters(technique, dict(params))
    else:
        allowed = set(BOUNDS[technique])
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"Unknown {technique} parameters: {sorted(unknown)}")
        result["parameters"] = parameters(technique, dict(params))
    training = {**TRAINING_DEFAULTS, **(recipe.get("training", {}) or {})}
    if technique == "gradient_boosting":
        result["training"] = {}
    elif technique == "student":
        training = {**STUDENT_TRAINING_DEFAULTS, **training}
        allowed = set(TRAINING_BOUNDS) | set(STUDENT_TRAINING_BOUNDS) | {"bf16_autocast", "channels_last", "augment_flip",
                                                                         "schedule", "flip_tta"}
        unknown = set(training) - allowed
        if unknown:
            raise ValueError(f"Unknown student training options: {sorted(unknown)}")
        out = {}
        for name, (low, high) in {**TRAINING_BOUNDS, **STUDENT_TRAINING_BOUNDS}.items():
            out[name] = _check_number(training[name], name, low, high, integer=name in ("epochs", "batch_size", "warmup_epochs"))
        for name in ("bf16_autocast", "channels_last", "augment_flip", "flip_tta"):
            if type(training[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
            out[name] = training[name]
        if training["schedule"] not in ("cosine", "constant"):
            raise ValueError("schedule must be cosine or constant")
        out["schedule"] = training["schedule"]
        if not any(out[k] > 0 for k in ("loss_log_l1", "loss_silog", "loss_gradient")):
            raise ValueError("At least one loss weight must be positive")
        result["training"] = out
    else:
        allowed = set(TRAINING_BOUNDS) | {"bf16_autocast"}
        if technique == "scratch":
            allowed |= set(SCRATCH_OPTIONS) | {"brightness_jitter"}
        unknown = set(training) - allowed
        if unknown:
            raise ValueError(f"Unknown training options: {sorted(unknown)}")
        out = {}
        for name, (low, high) in TRAINING_BOUNDS.items():
            out[name] = _check_number(training[name], name, low, high, integer=name in ("epochs", "batch_size"))
        for name in ("bf16_autocast", *(SCRATCH_OPTIONS if technique == "scratch" else ())):
            value = training.get(name, False)
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            out[name] = value
        if technique == "scratch":
            out["brightness_jitter"] = _check_number(training.get("brightness_jitter", 0.1), "brightness_jitter", 0.0, 0.5)
        result["training"] = out
    data = {**DATA_DEFAULTS, **(recipe.get("data", {}) or {})}
    unknown = set(data) - set(DATA_BOUNDS)
    if unknown:
        raise ValueError(f"Unknown data options: {sorted(unknown)}")
    result["data"] = {name: _check_number(data[name], name, low, high, integer=True)
                      for name, (low, high) in DATA_BOUNDS.items()}
    if technique != "student" and result["data"]["prepared_max_image_side"] > RESEARCH_NET_MAX_SIDE:
        raise ValueError(f"{technique} supports prepared_max_image_side <= {RESEARCH_NET_MAX_SIDE}; use student for larger grids")
    return result


def recipe_identity(recipe):
    """Hash of everything that changes the experiment; the note is excluded."""
    clean = {key: value for key, value in validate_recipe(recipe).items() if key != "note"}
    if clean.get("replicates", 1) == 1:
        clean.pop("replicates", None)  # keeps identities of pre-replication ledger entries stable
    return digest(clean)


def space_description():
    return {"techniques": list(TECHNIQUES), "architecture_bounds": BOUNDS, "architecture_defaults": DEFAULTS,
            "training_bounds": TRAINING_BOUNDS, "training_booleans": ["bf16_autocast"],
            "scratch_only_booleans": list(SCRATCH_OPTIONS), "data_bounds": DATA_BOUNDS,
            "student": {"parameters": {"arch": list(ARCHITECTURES), "bounds": STUDENT_BOUNDS, "defaults_per_arch": STUDENT_DEFAULTS,
                                       "norm": ["group", "batch", "none"], "bottleneck (unet)": ["dilated", "plain"], "attention (unet)": "bool"},
                        "training_extra_bounds": STUDENT_TRAINING_BOUNDS, "training_extra_defaults": STUDENT_TRAINING_DEFAULTS,
                        "schedule": ["cosine", "constant"], "booleans": ["bf16_autocast", "channels_last", "augment_flip", "flip_tta"],
                        "prepared_max_image_side": "64-512 (fully convolutional); other techniques max 128"},
            "replicates": "1-3 seeds per recipe; mean/std of every metric is recorded and the mean is ranked",
            "notes": [
                "student = fully convolutional UNet/dilated/coursework family: the main PhD-level model line (losses: log_l1, silog, gradient; cosine warmup; EMA; augmentation; flip TTA)","transformer width must be divisible by heads",
                      "transformer tokens: ceil(H/patch)*ceil(W/patch) <= max_tokens; prepared side 128 gives 128x72 at most",
                      "gradient_boosting ignores training options",
                      "scratch is the fixed coursework CNN; only training/data options vary",
                      "changing data options regenerates teachers/datasets (cached once per setting)"]}


def baseline_recipes():
    """Opening queue: one honest default per technique plus the two strongest prior variants."""
    return [
        {"technique": "cnn", "parameters": {"width": 32, "layers": 3}, "note": "baseline cnn"},
        {"technique": "scratch", "training": {"epochs": 10}, "note": "baseline coursework scratch CNN, eager fp32"},
        {"technique": "transformer", "parameters": {"width": 32, "layers": 2, "heads": 4, "patch_size": 16},
         "note": "baseline transformer (prior tuning winner recipe)"},
        {"technique": "scratch", "training": {"epochs": 10, "bf16_autocast": True}, "note": "baseline scratch bf16"},
        {"technique": "gradient_boosting", "parameters": {"estimators": 16, "max_thresholds": 16, "learning_rate": 0.1},
         "note": "baseline gradient boosting"},
        {"technique": "ann", "parameters": {"width": 32, "layers": 2}, "note": "baseline ann"},
        {"technique": "student", "parameters": {"arch": "unet", "width": 32, "stages": 4},
         "training": {"epochs": 30, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.0001, "ema_decay": 0.999},
         "data": {"teacher_samples_per_episode": 32, "prepared_max_image_side": 128},
         "note": "student baseline: residual UNet, log-L1 only, 128px, EMA"},
        {"technique": "student", "parameters": {"arch": "unet", "width": 32, "stages": 4},
         "training": {"epochs": 30, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.0001, "ema_decay": 0.999,
                      "loss_silog": 0.5, "loss_gradient": 0.5},
         "data": {"teacher_samples_per_episode": 32, "prepared_max_image_side": 256},
         "note": "student: UNet with SILog + gradient loss at 256px"},
        {"technique": "student", "parameters": {"arch": "unet", "width": 32, "stages": 4, "attention": True},
         "training": {"epochs": 30, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.0001, "ema_decay": 0.999,
                      "loss_silog": 0.5, "loss_gradient": 0.5},
         "data": {"teacher_samples_per_episode": 32, "prepared_max_image_side": 256},
         "note": "student: UNet + bottleneck attention (global scale cues) at 256px"},
        {"technique": "student", "parameters": {"arch": "dilated", "width": 32, "depth": 8},
         "training": {"epochs": 30, "batch_size": 4, "learning_rate": 0.001, "weight_decay": 0.0001, "ema_decay": 0.999,
                      "loss_silog": 0.5, "loss_gradient": 0.5},
         "data": {"teacher_samples_per_episode": 32, "prepared_max_image_side": 256},
         "note": "student: dilated single-resolution stack at 256px (edge-fidelity control)"},
    ]


def _choices(name, low, high, current, integer):
    if integer:
        steps = sorted({max(low, min(high, v)) for v in
                        (current // 2, current - 1, current + 1, current * 2, current + current // 2)} - {current})
        return steps
    steps = sorted({max(low, min(high, v)) for v in (current / 3, current / 2, current * 2, current * 3)} - {current})
    return steps


def mutate(recipe, rng: random.Random, tried):
    """Local-search fallback: change one bounded dimension of a known-good recipe."""
    base = validate_recipe(recipe)
    for _ in range(64):
        candidate = deepcopy(base)
        technique = candidate["technique"]
        dims = []
        if technique == "student":
            for name in ("width", "stages", "depth", "blocks_per_stage"):
                if name in candidate["parameters"]:
                    dims.append(("parameters", name, *STUDENT_BOUNDS[name], True))
            for name, (low, high) in STUDENT_TRAINING_BOUNDS.items():
                if name != "grad_clip":
                    dims.append(("training", name, low, high, name == "warmup_epochs"))
            for name in ("augment_flip", "flip_tta", "channels_last"):
                dims.append(("training", name, None, None, None))
            if "attention" in candidate["parameters"]:
                dims.append(("parameters", "attention", None, None, None))
        for name, (low, high) in BOUNDS.get(technique, {}).items():
            if name in ("heads", "max_tokens"):
                continue
            dims.append(("parameters", name, low, high, True))
        if technique != "gradient_boosting":
            for name, (low, high) in TRAINING_BOUNDS.items():
                dims.append(("training", name, low, high, name in ("epochs", "batch_size")))
            dims.append(("training", "bf16_autocast", None, None, None))
        for name, (low, high) in DATA_BOUNDS.items():
            dims.append(("data", name, low, high, True))
        group, name, low, high, integer = rng.choice(dims)
        if low is None:
            candidate[group][name] = not candidate[group][name]
        else:
            current = candidate[group][name]
            if not integer and current == 0:
                options = [low + (high - low) * f for f in (0.05, 0.1, 0.25, 0.5)]
            else:
                options = _choices(name, low, high, current, integer)
            if not options:
                continue
            candidate[group][name] = rng.choice(options)
        if technique == "transformer" and candidate["parameters"]["width"] % candidate["parameters"]["heads"]:
            candidate["parameters"]["width"] = candidate["parameters"]["heads"] * max(
                1, round(candidate["parameters"]["width"] / candidate["parameters"]["heads"]))
        candidate["note"] = f"heuristic mutation of {name} from {base[group][name]!r} to {candidate[group][name]!r}"
        try:
            normalised = validate_recipe(candidate)
        except ValueError:
            continue
        if recipe_identity(normalised) not in tried:
            return normalised
    return None


def describe(recipe):
    """Compact label: technique plus every value that differs from the technique's defaults."""
    recipe = validate_recipe(recipe)
    technique = recipe["technique"]
    bits = [technique]
    if technique == "student":
        arch = recipe["parameters"]["arch"]
        defaults = {"arch": "unet", **STUDENT_DEFAULTS[arch]}
        bits.append(arch)
        bits += [f"{k}={v}" for k, v in recipe["parameters"].items() if k != "arch" and defaults.get(k) != v]
        training_defaults = {**TRAINING_DEFAULTS, **STUDENT_TRAINING_DEFAULTS}
    else:
        bits += [f"{k}={v}" for k, v in recipe["parameters"].items()]
        training_defaults = dict(TRAINING_DEFAULTS)
    bits += [f"{k}={v}" for k, v in recipe["training"].items() if training_defaults.get(k) != v]
    bits += [f"{k}={v}" for k, v in recipe["data"].items() if DATA_DEFAULTS.get(k) != v]
    if recipe.get("replicates", 1) > 1:
        bits.append(f"replicates={recipe['replicates']}")
    return " ".join(bits)


__all__ = ["validate_recipe", "recipe_identity", "space_description", "baseline_recipes", "mutate", "describe",
           "canonical"]
