"""Isolated optional MANO integration; no provisioning or implicit asset downloads."""

import importlib.metadata
import os
from pathlib import Path
import sys

from track3.paths import project_path
from track3.provenance import input_file
from ._common import FEATURE, ROOT, read_json, record_input

# Pinned upstream retarget/params.py maps each tip to its distal joint frame.
JOINT_TRANSFORMS = (0, 13, 14, 15, 15, 1, 2, 3, 3, 4, 5, 6, 6, 10, 11, 12, 12, 7, 8, 9, 9)


def load_runtime(record):
    """Verify the hash-pinned MANO environment, then import and return its layers."""
    record_input(FEATURE / "backends.json", record)
    config = read_json(FEATURE / "backends.json")["mano"]
    environment = project_path(ROOT, config["environment"])
    if Path(sys.prefix).resolve() != environment:
        raise PermissionError("Run MANO operations only in the isolated project MANO environment")
    if not config.get("installation_record") or not config.get("installation_record_sha256"):
        raise PermissionError("The MANO backend has not been approved and provisioned")
    installation_path = project_path(ROOT, config["installation_record"])
    if record_input(installation_path, record)["sha256"] != config["installation_record_sha256"]:
        raise ValueError("MANO installation record differs from its approved pin")
    installation = read_json(installation_path)
    if (installation.get("status") != "completed" or installation.get("backend") != "manotorch" or
            installation.get("user_approved") is not True or
            project_path(ROOT, installation["environment"]) != environment):
        raise PermissionError("A completed approved MANO installation record is required")
    packages = installation.get("packages", {})
    if not {"numpy", "torch", "manotorch"}.issubset(packages) or any(not isinstance(value, str) or not value for value in packages.values()):
        raise ValueError("MANO installation must pin its package versions")
    for name, version in packages.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"MANO package version differs from its installation record: {name}")
    files = installation.get("files", [])
    if not files:
        raise ValueError("MANO installation requires an explicit installed-file hash inventory")
    entries = {}
    for item in files:
        path = project_path(ROOT, item["path"])
        if not path.is_relative_to(environment) or path in entries:
            raise ValueError("MANO installed-file inventory must be unique and inside its environment")
        entry = record_input(path, record)
        if entry["sha256"] != item["sha256"]:
            raise ValueError("MANO installed source differs from its installation record")
        entries[path] = entry
    assets_root = project_path(ROOT, config["assets_root"])
    if installation.get("licensed_assets_acknowledged") is not True:
        raise PermissionError("User-provided licensed MANO assets must be recorded by provisioning")
    assets = installation.get("assets", {})
    for side in ("LEFT", "RIGHT"):
        name = f"MANO_{side}.pkl"
        entry = record_input(assets_root / "models" / name, record)
        if assets.get(name) != entry:
            raise ValueError(f"MANO asset differs from the provisioned licensed file: {name}")
    for field in ("reference", "joint_mapping_reference"):
        record["sources"].append(input_file(ROOT, project_path(ROOT, config[field])))

    # Keep optional-library caches and temporary writes in project storage.
    for variable, relative in {
        "TORCH_HOME": ".cache/hand-motion/torch", "XDG_CACHE_HOME": ".cache/hand-motion",
        "HF_HOME": ".cache/hand-motion/huggingface", "MPLCONFIGDIR": ".cache/hand-motion/matplotlib",
        "TMPDIR": ".runtime/tmp/hand-motion", "TMP": ".runtime/tmp/hand-motion",
        "TEMP": ".runtime/tmp/hand-motion",
    }.items():
        path = project_path(ROOT, relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True

    # Optional dependencies stay local to the explicitly provisioned CPU backend.
    import torch
    import manotorch.axislayer as axis_module
    import manotorch.manolayer as mano_module

    for module in (axis_module, mano_module):
        if Path(module.__file__).resolve() not in entries:
            raise ValueError("Imported MANO source is not in the installed-file hash inventory")
    record["backend"] = {"name": "manotorch", "packages": packages, "assets": assets,
                         "environment": environment.relative_to(ROOT).as_posix(), "device": "cpu"}
    return torch, mano_module.ManoLayer, axis_module.AxisLayerFK, assets_root


class ManoKinematics:
    def __init__(self, runtime, side):
        self.torch, ManoLayer, AxisLayerFK, assets_root = runtime
        self.layer = ManoLayer(use_pca=False, side=side, gender="neutral", center_idx=None,
                               mano_assets_root=str(assets_root), flat_hand_mean=True).to("cpu")
        self.axes = AxisLayerFK(side=side, mano_assets_root=str(assets_root)).to("cpu")
        self.layer.eval()
        self.axes.eval()

    def forward(self, wrist, fingers, betas, translation, np):
        torch = self.torch
        pose = np.concatenate((wrist, fingers), axis=1).astype(np.float32)
        with torch.inference_mode():
            result = self.layer(pose_coeffs=torch.from_numpy(pose),
                                betas=torch.from_numpy(np.broadcast_to(betas, (len(pose), 10)).copy().astype(np.float32)))
            frames = result.transforms_abs.clone()
            anatomical_frames, _, _ = self.axes(frames)
            rotations = anatomical_frames[:, JOINT_TRANSFORMS, :3, :3].detach().cpu().numpy().astype(np.float64)
            joints = result.joints.detach().cpu().numpy().astype(np.float64)
            vertices = result.verts.detach().cpu().numpy().astype(np.float64)
        # Translation is the MANO model origin, not a request to recenter the wrist.
        return joints + translation[:, None], rotations, vertices + translation[:, None]
