"""Torch-free architecture bounds and validation shared by recipes and the training request."""

ARCHITECTURES = ("unet", "dilated", "coursework")
BOUNDS = {"width": (8, 96), "stages": (2, 5), "depth": (2, 12), "blocks_per_stage": (1, 3), "dropout": (0.0, 0.5)}
DEFAULTS = {"unet": {"width": 32, "stages": 4, "norm": "group", "bottleneck": "dilated", "attention": False,
                     "dropout": 0.0, "blocks_per_stage": 1},
            "dilated": {"width": 32, "depth": 6, "norm": "group", "dropout": 0.0},
            "coursework": {"width": 24, "dropout": 0.1}}


def architecture_config(arch, values):
    """Validate and normalise architecture parameters without importing anything heavy."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {ARCHITECTURES}")
    config = {**DEFAULTS[arch], **(values or {})}
    unknown = set(config) - set(DEFAULTS[arch])
    if unknown:
        raise ValueError(f"Unknown {arch} parameters: {sorted(unknown)}")
    for name, value in config.items():
        if name in ("width", "stages", "depth", "blocks_per_stage"):
            low, high = BOUNDS[name]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        elif name == "dropout":
            if type(value) not in (int, float) or not 0 <= value <= 0.5:
                raise ValueError("dropout must be in [0, 0.5]")
            config[name] = float(value)
        elif name == "norm":
            if value not in ("group", "batch", "none"):
                raise ValueError("norm must be group, batch or none")
        elif name == "bottleneck":
            if value not in ("dilated", "plain"):
                raise ValueError("bottleneck must be dilated or plain")
        elif name == "attention":
            if type(value) is not bool:
                raise ValueError("attention must be boolean")
    if arch == "unet" and config["width"] * 2 ** config["stages"] > 1024:
        raise ValueError("unet bottleneck width exceeds 1024 channels; reduce width or stages")
    return config
