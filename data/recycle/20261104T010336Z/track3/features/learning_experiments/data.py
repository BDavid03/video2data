"""Read-only adapter for public ``scratch_depth.prepare`` dataset.json artifacts.

``PreparedPairs(dataset_path, partition, group, np)`` selects every manifest row
belonging to the chosen train/tuning/holdout episodes. ``partition`` uses the
public planner fields dataset_revision, split_id and {group}_episodes. Groups
must be nonempty, disjoint subsets of frozen public development, excluding 41;
undeclared episodes and missing coverage of the selected group are refused.

Only stable path/provenance primitives are shared. No producer-private module,
NumPy, model, compiler or notebook is imported here. The caller supplies NumPy
after its own execution authorization. Construction reads manifests and hashes
actual artifacts; array decoding is lazy. This adapter grants no execution
permission and never writes, downloads, resamples, augments or creates caches.

The dataset, owning completed run, sample inventory, provenance, pinned split,
calibration and canonical optical grid must agree. Samples must live directly
in the owning run's samples directory. Each read rechecks manifest/run bytes and
hashes the same open sample stream used for decoding with allow_pickle=False.
RGB is uint8 HWC, depth is floating HW, masks must already have boolean dtype,
and intrinsics describe the same positive-focal, zero-distortion optical grid.
Grids are limited to 512 pixels per side / 262144 pixels (raised from 256 on
2026-10-29 for the fully convolutional depth_student family; the research
ann/cnn/transformer factories keep their own 65536-pixel forward guard).
Larger outputs require separately authorized preparation, never implicit resizing.

Returned dictionaries contain rgb, float32 depth_m, bool valid,
input_intrinsics_pixels and a copy of the manifest row. Usable labels are finite,
positive and within the recorded inclusive depth bounds; at least 64 are needed.
Every returned array is an independent C-contiguous copy for torch.from_numpy.
Unusable labels, including originally invalid NaNs, become NaN in depth_m.
``valid`` is target supervision validity; the teacher mask cannot establish an
RGB input_valid mask. Research prediction validity must instead describe the
prediction's own finite, bounded values and must not inherit this target mask.
These are MoGe2 pseudo-labels in estimated metres, not measured ground truth.
Exposed metadata includes dataset_path, producer_path, dataset_sha256, document,
selected rows, data_contract and input_paths (dataset plus selected artifacts).
Missing real artifacts raise FileNotFoundError; inconsistent contracts raise
ValueError. No synthetic data or missing-artifact fallback is provided.
"""

from copy import deepcopy
import hashlib
import io
import json
import math
import operator
from pathlib import Path, PurePosixPath

from track3.paths import project_path
from track3.provenance import inherited_provenance, read_run, sha256_file


ROOT = Path(__file__).resolve().parents[3]
_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
_SPLIT_ID = "track3-public-dev13-val7-v1"
_DEVELOPMENT = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42, 43, 44)
_DATA_FIELDS = (
    "dataset_revision", "split_id", "target_kind", "geometry", "max_image_side",
    "calibration_sha256", "minimum_depth_m", "maximum_depth_m",
)


def _json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required actual artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _integer(value, name, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _size(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [width, height]")
    return tuple(_integer(x, name, 1) for x in value)


def _matrix(value, name):
    if (not isinstance(value, list) or len(value) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in value)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    for row in value:
        for entry in row:
            _number(entry, name)
    if (value[0][0] <= 0 or value[1][1] <= 0 or value[2] != [0, 0, 1]
            or value[0][1] != 0 or value[1][0] != 0):
        raise ValueError(f"{name} must be canonical positive-focal pinhole intrinsics")
    return value


def _relative(directory, value):
    if (not isinstance(value, str) or not value or "\\" in value
            or ":" in value or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Artifact must use a confined relative POSIX path")
    path = project_path(ROOT, directory / value)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact escapes its owning run")
    if not path.is_file():
        raise FileNotFoundError(f"Required actual artifact is missing: {path}")
    return path


def pinhole_input_validity(contract, cv2, np):
    """Reconstruct input-ray support from the admitted camera, without label masks.

    Caller supplies the already approved OpenCV/NumPy modules after operation
    authorization. Output rays landing outside the original RGB are invalid.
    This uses the same calibrated mapping that preparation records as geometry.
    """
    geometry = contract["geometry"]
    width, height = geometry["source_size_width_height"]
    size = tuple(geometry["output_size_width_height"])
    x, y = cv2.initUndistortRectifyMap(
        np.asarray(geometry["factory_intrinsics_pixels"], dtype=np.float64),
        np.asarray(geometry["factory_distortion_coefficients"], dtype=np.float64),
        np.eye(3), np.asarray(geometry["input_intrinsics_pixels"], dtype=np.float64),
        size, cv2.CV_32FC1)
    return (np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1)
            & (y >= 0) & (y <= height - 1))


class PreparedPairs:
    """Map-style prepared pairs; NumPy is supplied, never imported internally."""

    def __init__(self, dataset_path, partition, group, np):
        self.np = np
        if group not in ("train", "tuning", "holdout") or not isinstance(partition, dict):
            raise ValueError("Supply a declared partition and train/tuning/holdout group")
        split_path = project_path(ROOT, "configs/splits/public-v1.json")
        split = _json(split_path)
        lock_path = project_path(ROOT, "configs/sources.lock.json")
        lock = _json(lock_path)
        if (split.get("dataset_revision") != _REVISION or split.get("split_id") != _SPLIT_ID
                or split.get("source_split") != "public"
                or split.get("development_episode_indices") != [0, 11, 12, 18, 21, 23, 26, 31, 39, 41, 42, 43, 44]
                or lock.get("dataset", {}).get("revision") != _REVISION
                or partition.get("dataset_revision") != _REVISION
                or partition.get("split_id") != _SPLIT_ID):
            raise ValueError("Prepared pairs require the pinned frozen public development policy")
        groups, declared = {}, set()
        for name in ("train", "tuning", "holdout"):
            episodes = partition.get(f"{name}_episodes")
            if (not isinstance(episodes, (list, tuple)) or not episodes
                    or any(type(e) is not int or e not in _DEVELOPMENT for e in episodes)
                    or len(set(episodes)) != len(episodes) or declared.intersection(episodes)):
                raise ValueError("Episode groups must be nonempty, unique, disjoint and exclude 41")
            groups[name] = set(episodes)
            declared.update(episodes)

        self.dataset_path = project_path(ROOT, dataset_path)
        if self.dataset_path.name != "dataset.json":
            raise ValueError("Expected the public preparation dataset.json artifact")
        directory = self.dataset_path.parent
        self.producer_path = project_path(ROOT, directory / "run.json")
        if self.producer_path.parent != directory or not self.producer_path.is_file():
            raise FileNotFoundError("Required owning preparation run.json is missing or relocated")
        self.document = _json(self.dataset_path)
        doc = self.document
        producer = read_run(ROOT, self.producer_path, kind="scratch_depth.prepare")
        inherited_provenance([doc])
        self.dataset_sha256 = sha256_file(self.dataset_path)
        self._producer_sha256 = sha256_file(self.producer_path)
        artifacts = producer.get("artifacts")
        if (not isinstance(artifacts, dict) or producer.get("schema_version") != 1
                or producer.get("feature") != "scratch_depth" or producer.get("operation") != "prepare"
                or producer.get("dataset_sha256") != self.dataset_sha256
                or artifacts.get("dataset.json") != self.dataset_sha256
                or _relative(directory, producer.get("dataset")) != self.dataset_path
                or not isinstance(doc.get("producer_run"), str)
                or project_path(ROOT, doc["producer_run"]) != self.producer_path
                or doc.get("provenance") != producer.get("provenance")):
            raise ValueError("Dataset is not bound to its completed tagged preparation producer")
        if (doc.get("schema_version") != 1 or doc.get("kind") != "scratch_depth_training_pairs"
                or doc.get("status") != "completed" or doc.get("source_split") != "public"
                or doc.get("subset") != "development" or doc.get("camera") != "ego_cam_a"
                or doc.get("dataset_revision") != _REVISION or doc.get("split_id") != _SPLIT_ID
                or doc.get("target_kind") != "moge2_pseudo_depth"
                or doc.get("target_units") != "estimated metres"
                or producer.get("target_kind") != doc["target_kind"]):
            raise ValueError("Expected pinned public development MoGe2 pseudo-depth pairs")

        maximum_side = _integer(doc.get("max_image_side"), "max_image_side", 64, 512)
        minimum = _number(doc.get("minimum_depth_m"), "minimum_depth_m")
        maximum = _number(doc.get("maximum_depth_m"), "maximum_depth_m")
        if not 0 < minimum < maximum:
            raise ValueError("Depth bounds must be positive and ordered")
        request = producer.get("request")
        if not isinstance(request, dict) or any(request.get(key) != doc[key] for key in (
                "max_image_side", "minimum_depth_m", "maximum_depth_m")):
            raise ValueError("Dataset preparation bounds differ from the producer request")
        calibration_hash = _digest(doc.get("calibration_sha256"))
        policy = _json(project_path(ROOT, "configs/calibration/ego.json"))
        if (policy.get("dataset_revision") != _REVISION
                or policy.get("source_sha256") != calibration_hash
                or policy.get("orientation", {}).get("additional_pixel_rotation_deg") != 0):
            raise ValueError("Dataset calibration/orientation differs from the pinned public policy")
        calibration_path = project_path(ROOT, Path(lock["dataset"]["cache_path"]) / "snapshots"
                                        / _REVISION / "track_3/public/meta/camera_calibration.json")
        calibration = _json(calibration_path)
        inputs = producer.get("inputs")
        if not isinstance(inputs, list) or any(not isinstance(entry, dict) for entry in inputs):
            raise ValueError("Producer requires a public input hash inventory")
        for path, expected in ((calibration_path, calibration_hash), (split_path, sha256_file(split_path)),
                               (lock_path, sha256_file(lock_path))):
            matches = [entry for entry in inputs if isinstance(entry.get("path"), str)
                       and project_path(ROOT, entry["path"]) == path]
            if not matches or any(entry.get("sha256") != expected for entry in matches):
                raise ValueError(f"Producer did not bind the current pinned input: {path}")
            if sha256_file(path) != expected:
                raise ValueError(f"Pinned input bytes changed: {path}")
        camera = calibration.get("cameras", {}).get("ego_cam_a")
        geometry = doc.get("geometry")
        if not isinstance(camera, dict) or not isinstance(geometry, dict):
            raise ValueError("Missing public RGB calibration or prepared geometry")
        if (geometry.get("camera") != "ego_cam_a" or geometry.get("output_distortion") != "zero"
                or type(geometry.get("additional_rotation_degrees")) is not int
                or geometry["additional_rotation_degrees"] != 0
                or geometry.get("camera_frame") != "OpenCV optical x-right y-down z-forward"
                or camera.get("distortion_model") != "Perspective"):
            raise ValueError("Expected unrotated, zero-distortion OpenCV optical RGB geometry")
        source_width, source_height = _size(geometry.get("source_size_width_height"), "source size")
        width, height = _size(geometry.get("output_size_width_height"), "output size")
        if max(width, height) > maximum_side or max(width, height) > 512 or width * height > 512 * 512:
            raise ValueError("Prepared geometry exceeds model limits; implicit resizing is forbidden")
        factory = _matrix(geometry.get("factory_intrinsics_pixels"), "factory intrinsics")
        intrinsics = _matrix(geometry.get("input_intrinsics_pixels"), "input intrinsics")
        distortion = geometry.get("factory_distortion_coefficients")
        if not isinstance(distortion, list) or len(distortion) != 14:
            raise ValueError("Expected all 14 factory distortion coefficients")
        for coefficient in distortion:
            _number(coefficient, "factory distortion")
        if (camera.get("resolution") != [source_width, source_height]
                or camera.get("intrinsic_matrix") != factory
                or camera.get("distortion_coefficients") != distortion):
            raise ValueError("Prepared factory geometry differs from the hashed RGB calibration")
        scale = min(1.0, maximum_side / max(source_width, source_height))
        focal = factory[0][0] * width / source_width
        expected_k = [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]
        if ([width, height] != [round(source_width * scale), round(source_height * scale)]
                or any(not math.isclose(intrinsics[i][j], expected_k[i][j], rel_tol=1e-9, abs_tol=1e-9)
                       for i in range(3) for j in range(3))):
            raise ValueError("Prepared grid differs from the public canonical pinhole encoding")
        self.data_contract = {key: deepcopy(doc[key]) for key in _DATA_FIELDS}
        self._contract = deepcopy(self.data_contract)

        samples = doc.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Dataset has no actual prepared samples")
        seen, paths, selected, covered = set(), set(), [], set()
        samples_directory = directory / "samples"
        if samples_directory.resolve() != samples_directory:
            raise ValueError("Owning samples directory must not be redirected")
        for row in samples:
            if not isinstance(row, dict):
                raise ValueError("Sample rows must be manifest objects")
            episode = _integer(row.get("episode_index"), "episode_index")
            frame = _integer(row.get("frame_index"), "frame_index")
            key = (episode, frame)
            if key in seen or episode not in declared:
                raise ValueError("Duplicate episode/frame or episode outside the declared partition")
            seen.add(key)
            name = f"samples/episode_{episode:06d}_frame_{frame:06d}.npz"
            if row.get("artifact") != name:
                raise ValueError("Sample does not use its owning public artifact name")
            path = _relative(directory, name)
            if path.parent != samples_directory or path != directory / name or path in paths:
                raise ValueError("Sample is aliased or escapes the exact owning samples directory")
            paths.add(path)
            digest = _digest(row.get("sha256"))
            _integer(row.get("valid_pixels"), "valid_pixels", 64, width * height)
            if artifacts.get(name) != digest or sha256_file(path) != digest:
                raise ValueError(f"Sample differs from dataset/producer artifact hashes: {path}")
            if episode in groups[group]:
                selected.append((deepcopy(row), path))
                covered.add(episode)
        if covered != groups[group]:
            raise ValueError(f"Prepared samples do not cover every {group} episode: {sorted(groups[group] - covered)}")
        if producer.get("prepared_samples") != len(samples):
            raise ValueError("Prepared sample count differs from the producer")
        self._selected = tuple(selected)
        self.rows = [deepcopy(row) for row, _ in selected]
        self.input_paths = [self.dataset_path, *(path for _, path in selected)]

    def __len__(self):
        return len(self._selected)

    def __getitem__(self, index):
        if isinstance(index, bool):
            raise TypeError("PreparedPairs indices must be integers")
        index = operator.index(index)
        row, path = self._selected[index]
        np = self.np
        raw = path.read_bytes()  # the dataset and producer were hashed once at admission
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Prepared sample hash changed: {path}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            required = {"rgb", "depth_m", "valid", "input_intrinsics_pixels"}
            if not required.issubset(arrays.files):
                raise ValueError(f"Prepared sample is missing required arrays: {path}")
            rgb = arrays["rgb"].copy()
            depth = arrays["depth_m"].copy()
            valid = arrays["valid"].copy()
            intrinsics = arrays["input_intrinsics_pixels"].copy()
        contract = self._contract
        width, height = contract["geometry"]["output_size_width_height"]
        if (rgb.shape != (height, width, 3) or rgb.dtype != np.dtype("uint8")
                or depth.shape != (height, width) or depth.dtype.kind != "f"
                or valid.shape != (height, width) or valid.dtype != np.dtype("bool")
                or intrinsics.shape != (3, 3) or intrinsics.dtype.kind != "f"
                or not np.isfinite(intrinsics).all()
                or not np.allclose(intrinsics, contract["geometry"]["input_intrinsics_pixels"],
                                   rtol=1e-5, atol=1e-5)):
            raise ValueError("Prepared sample shape, dtype or optical intrinsics are inconsistent")
        if int(valid.sum()) != row["valid_pixels"]:
            raise ValueError("Boolean mask count differs from the dataset manifest")
        # Check both stored and float32 values: conversion must not make a
        # supposedly usable label infinite, zero, or outside the declared bounds.
        depth64 = depth.astype(np.float64)
        usable = (valid & np.isfinite(depth64) & (depth64 > 0)
                  & (depth64 >= contract["minimum_depth_m"])
                  & (depth64 <= contract["maximum_depth_m"]))
        depth32 = np.where(usable, depth, np.nan).astype(np.float32)
        stored_depth64 = depth32.astype(np.float64)
        usable &= (np.isfinite(stored_depth64) & (stored_depth64 > 0)
                   & (stored_depth64 >= contract["minimum_depth_m"])
                   & (stored_depth64 <= contract["maximum_depth_m"]))
        if int(usable.sum()) < 64:
            raise ValueError("Prepared sample has fewer than 64 usable bounded positive depth labels")
        return {"rgb": rgb.copy(order="C"),
                "depth_m": np.where(usable, depth32, np.float32(np.nan)).copy(order="C"),
                "valid": usable.copy(order="C"),
                "input_intrinsics_pixels": intrinsics.copy(order="C"),
                "row": deepcopy(row)}
