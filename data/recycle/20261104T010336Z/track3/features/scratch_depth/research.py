"""Bounded, opt-in original-CNN research; historical training stays unchanged.

Importing this module performs no heavy imports, fitting or artifact writes.
"""

from dataclasses import asdict, dataclass
import math
import shutil
import time

from track3.gpu import GpuLease
from track3.provenance import input_file, read_run, sha256_file, write_json
from track3.timing import PhaseRecorder
from ._common import (FEATURE, ROOT, authorize, local_path, operation_run, read_json,
                      relative, save_npz, select_episode, within)

TRAIN = (0, 11, 12, 18, 21, 23, 26, 31)
TUNE = (39, 42)
FIELDS = ("rgb", "base_depth_m", "base_valid", "confidence", "target_depth_m",
          "target_valid", "intrinsics")
OPTIONAL = ("normal_support", "temporal_depth_m", "temporal_support")
ARCHITECTURE = "coursework-admitted-depth-cnn-v1"


@dataclass(frozen=True)
class PrepareLearningRequest:
    sources: str
    sources_sha256: str

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


@dataclass(frozen=True)
class LearningRequest:
    dataset: str
    dataset_sha256: str
    mode: str = "direct"
    context: bool = False
    robust_log: bool = False
    boundary: bool = False
    normals: bool = False
    temporal: bool = False
    trace_runtime: bool = False
    epochs: int = 10
    seed: int = 31

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.mode not in ("direct", "residual"):
            raise ValueError("mode must be direct or residual")
        if any(type(getattr(self, name)) is not bool for name in
               ("context", "robust_log", "boundary", "normals", "temporal", "trace_runtime")):
            raise ValueError("Objective/context flags must be explicit booleans")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 10 or self.seed != 31:
            raise ValueError("Research fixes seed 31 and caps training at ten epochs")
        if sum((self.boundary, self.normals, self.temporal)) > 1:
            raise ValueError("Compare boundary, normal and temporal objectives separately")
        if (self.boundary or self.normals or self.temporal) and not self.robust_log:
            raise ValueError("Additional objectives compare against fixed robust-log baseline")


@dataclass(frozen=True)
class PredictLearningRequest:
    checkpoint: str
    checkpoint_sha256: str
    dataset: str
    dataset_sha256: str
    episode_index: int
    calibration_depth_run: str

    def __post_init__(self):
        if type(self.episode_index) is not int or self.episode_index not in TUNE:
            raise PermissionError("Research prediction admits tuning episodes 39/42 only")

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


def _partition(row):
    episode = row["episode_index"]
    if type(episode) is not int or episode not in (*TRAIN, *TUNE):
        raise PermissionError("New research admits only public train/tuning episodes, excluding 41/43/44")
    return "train" if episode in TRAIN else "tuning"


def _reference_receipt(reference, field, row, revision, split_id):
    """Bind arrays to producer episode/frame/grid before opening array bytes.

    Known legacy single-frame producers are admitted through their own recorded
    frame inventories. Future confidence/warped-context producers must emit the
    per-artifact learning_inputs receipt; an author-supplied row is insufficient.
    """
    path, producer_path = local_path(reference["artifact"]), local_path(reference["producer_run"])
    producer = read_run(ROOT, producer_path, allowed=("video_only",))
    episode = producer.get("episode_index", producer.get("request", {}).get("episode_index"))
    if episode is not None and (episode not in (*TRAIN, *TUNE) or episode != row["episode_index"]):
        raise PermissionError("Array producer episode is excluded or differs from the declared clip")
    if not path.is_relative_to(producer_path.parent):
        raise ValueError("Array artifact escapes its producer")
    name = path.relative_to(producer_path.parent).as_posix()
    digest = producer.get("artifacts", {}).get(name)
    if digest != reference["sha256"]:
        raise ValueError("Source reference is not owned by its completed producer")
    if producer["kind"] == "scratch_depth.prepare":
        dataset_path = within(producer_path.parent, producer["dataset"])
        if (sha256_file(dataset_path) != producer.get("dataset_sha256")
                or producer.get("artifacts", {}).get(dataset_path.name) != producer["dataset_sha256"]):
            raise ValueError("Legacy pair inventory is not hash-owned")
        dataset = read_json(dataset_path)
        samples = [sample for sample in dataset["samples"] if sample.get("artifact") == name]
        if len(samples) != 1:
            raise ValueError("Array lacks a unique owning pair sample")
        sample = samples[0]
        if (sample["episode_index"] not in (*TRAIN, *TUNE) or sample["episode_index"] != row["episode_index"]
                or row["frame_indices"] != [sample["frame_index"]]
                or row["pts_seconds"] != [sample["video_pts_seconds"]] or sample["sha256"] != digest
                or dataset.get("source_split") != "public" or dataset.get("dataset_revision") != revision
                or dataset.get("split_id") != split_id):
            raise PermissionError("Legacy pair episode, frame, PTS, split or revision mismatch")
        allowed = {"rgb": "rgb", "base_depth_m": "depth_m", "base_valid": "valid",
                   "target_depth_m": "depth_m", "target_valid": "valid", "intrinsics": "input_intrinsics_pixels"}
        if allowed.get(field) != reference["key"] or row["geometry"] != dataset["geometry"]:
            raise ValueError("Legacy array field/grid relabeling is prohibited; no confidence is available")
    elif producer["kind"] in ("depth.generate", "stereo_depth.generate"):
        if (producer.get("split") != "public" or producer.get("dataset_revision") != revision
                or len(row["frame_indices"]) != 1):
            raise ValueError("Depth producer split/revision/context mismatch")
        frames = [item for item in producer["frames"] if item.get("artifact") == name]
        if (len(frames) != 1 or frames[0].get("status") != "completed" or frames[0].get("sha256") != digest
                or row["frame_indices"] != [frames[0]["frame_index"]]
                or row["pts_seconds"] != [frames[0]["video_pts_seconds"]]):
            raise ValueError("Depth source frame/PTS is not bound to its owning producer")
        keys = ({"base_depth_m": "rgb_depth_m", "base_valid": "rgb_valid", "confidence": "rgb_match_confidence", "target_depth_m": "rgb_depth_m",
                 "target_valid": "rgb_valid", "intrinsics": "rgb_intrinsics_pixels"}
                if producer["kind"] == "stereo_depth.generate" else
                {"base_depth_m": "depth_m", "base_valid": "valid", "target_depth_m": "depth_m",
                 "target_valid": "valid", "intrinsics": "input_intrinsics_pixels"})
        if keys.get(field) != reference["key"]:
            raise ValueError("Depth field alias is unsupported; validity cannot stand in for confidence")
        if field == "confidence" and (producer.get("confidence_field") != "rgb_match_confidence"
                or producer.get("confidence_semantics") != "uncalibrated_matching_quality_not_probability"
                or producer.get("confidence_origin") != "linear_left_right_disparity_residual"
                or row.get("confidence_semantics") != producer["confidence_semantics"]):
            raise ValueError("Stereo confidence requires explicit continuous matching-quality producer semantics")
    else:
        receipt = producer.get("learning_inputs", {}).get(name)
        if (not isinstance(receipt, dict) or receipt.get("episode_index") != row["episode_index"]
                or receipt.get("episode_index") not in (*TRAIN, *TUNE) or receipt.get("source_split") != "public"
                or receipt.get("dataset_revision") != revision or receipt.get("split_id") != split_id
                or receipt.get("frame_indices") != row["frame_indices"] or receipt.get("pts_seconds") != row["pts_seconds"]
                or receipt.get("geometry") != row["geometry"] or receipt.get("sha256") != digest
                or receipt.get("fields", {}).get(field) != reference["key"]):
            raise ValueError("Producer lacks a matching per-artifact learning-input receipt")
        for key in ("context_alignment", "normal_boundaries_excluded", "temporal_depth_frame",
                    "temporal_visibility_and_motion_admitted", "confidence_semantics", "target_source",
                    "reserved_correspondences_excluded"):
            if key in row and receipt.get(key) != row[key]:
                raise ValueError(f"Producer does not admit declared {key}")
    if field in ("target_depth_m", "target_valid"):
        kind = producer["kind"]
        expected = "admitted_training_stereo" if kind == "stereo_depth.generate" else "moge2_pseudo_depth"
        if kind in ("stereo_depth.generate", "depth.generate", "scratch_depth.prepare") and row["target_source"] != expected:
            raise ValueError("Supervision source relabeled")
        if kind == "stereo_depth.generate":
            raise ValueError("Raw dense stereo includes reserved support: supply a producer-owned training-support exclusion receipt")
    if sha256_file(path) != digest:
        raise ValueError("Source artifact changed after its completed producer")
    return path, producer_path


def _sample(arrays, row, np, *, require_target=True):
    """Fail closed on unsupported grids, gaps, confidence and geometry."""
    required = set(FIELDS) - (set() if require_target else {"target_depth_m", "target_valid"})
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Missing explicitly produced inputs: {sorted(missing)}")
    rgb = arrays["rgb"]
    if (rgb.ndim != 4 or rgb.shape[0] not in (1, 3) or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8 or min(rgb.shape[1:3]) < 8 or max(rgb.shape[1:3]) > 128):
        raise ValueError("RGB must be one/three admitted centre-grid frames, THW3 uint8 at side <=128")
    frames, height, width, _ = rgb.shape
    if row["geometry"].get("output_size_width_height") != [width, height]:
        raise ValueError("Array shape differs from its producer-bound geometry")
    if (len(row["frame_indices"]) != frames or len(row["pts_seconds"]) != frames
            or any(type(index) is not int or index < 0 for index in row["frame_indices"])
            or any(type(value) not in (float, int) or not math.isfinite(value) for value in row["pts_seconds"])):
        raise ValueError("Every frame needs its inventoried index and finite actual PTS")
    if frames == 3:
        if (any(b - a != 1 for a, b in zip(row["frame_indices"], row["frame_indices"][1:]))
                or any(not 0 < b - a <= 0.1 for a, b in zip(row["pts_seconds"], row["pts_seconds"][1:]))
                or row.get("context_alignment") != "visibility-masked-motion-warp-to-centre-optical-grid"):
            raise ValueError("Context requires contiguous, gap-free, motion/visibility-admitted warps")
    for name in ("base_depth_m", "base_valid", "confidence"):
        if arrays[name].shape != (frames, height, width):
            raise ValueError(f"{name} is not on the matched input grid")
    for name in ("target_depth_m", "target_valid"):
        if require_target and arrays[name].shape != (height, width):
            raise ValueError(f"{name} is not on the centre target grid")
    for name in ("base_valid", "target_valid", "normal_support", "temporal_support"):
        if name in arrays and arrays[name].dtype != np.dtype("bool"):
            raise ValueError(f"{name} must be a producer boolean support mask")
    for depth_name, mask_name in (("base_depth_m", "base_valid"), ("target_depth_m", "target_valid"),
                                  ("temporal_depth_m", "temporal_support")):
        if depth_name not in arrays and mask_name not in arrays:
            continue
        depth, mask = arrays[depth_name], arrays[mask_name]
        if depth.shape != mask.shape or depth.dtype.kind != "f" or not np.all(np.isfinite(depth[mask]) & (depth[mask] > 0)):
            raise ValueError(f"{depth_name} has invalid positive-metre support")
    confidence, valid = arrays["confidence"], arrays["base_valid"]
    if (confidence.dtype.kind != "f" or
            not np.all(np.isfinite(confidence[valid]) & (confidence[valid] >= 0) & (confidence[valid] <= 1))):
        raise ValueError("Confidence must be explicitly produced finite [0,1] values; validity cannot substitute")
    k = arrays["intrinsics"]
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("A calibrated invertible centre-grid pinhole matrix is required")
    if "normal_support" in arrays:
        if arrays["normal_support"].shape != (height, width) or row.get("normal_boundaries_excluded") is not True:
            raise ValueError("Normal support must explicitly exclude instance boundaries and depth holes")
    if "temporal_depth_m" in arrays or "temporal_support" in arrays:
        if (frames != 3 or arrays["temporal_depth_m"].shape != (2, height, width)
                or row.get("temporal_depth_frame") != "centre-camera-optical-z-metres"
                or row.get("temporal_visibility_and_motion_admitted") is not True):
            raise ValueError("Temporal supervision needs two motion-transformed neighbour z maps and visibility")
    centre = frames // 2
    matched = arrays["target_valid"] & valid[centre] if require_target else valid[centre]
    if require_target and int(matched.sum()) < 64:
        raise ValueError("Fewer than 64 matched base/label pixels")


def prepare_learning(request: PrepareLearningRequest):
    """Adapt explicit named arrays from completed producer artifacts, never labels.

    The source JSON maps every field to artifact, key, sha256 and producer_run.
    Existing prepared RGB/pseudo-target arrays may be named alongside stereo
    depth/confidence arrays, but their pixel grids must already match. This
    adapter performs no resizing, correspondence estimation or motion warping.
    """
    authorize("research_prepare")
    import numpy as np
    source_path = local_path(request.sources)
    if sha256_file(source_path) != request.sources_sha256:
        raise ValueError("Source protocol bytes differ from the predeclared request digest")
    source = read_json(source_path)
    revision = read_json("configs/sources.lock.json")["dataset"]["revision"]
    split = read_json("configs/splits/public-v1.json")
    if (source.get("dataset_revision") != revision or source.get("split_id") != split["split_id"]
            or source.get("source_split") != "public" or source.get("real_captured_only") is not True
            or source.get("pixel_geometry") != "undistorted-centre-pinhole-optical-z-metres"
            or source.get("frozen_protocol_before_fitting") is not True):
        raise ValueError("Require frozen real-public protocol, pinned split/revision and metric pixel geometry")
    rows = source["samples"]
    if not rows or len(rows) > 256 or sum(_partition(row) == "train" for row in rows) > 128:
        raise ValueError("Bounded research permits <=128 training clip centres and <=256 total")
    seen, parents, artifacts = set(), set(), set()
    for row in rows:
        if (not isinstance(row.get("frame_indices"), list) or len(row["frame_indices"]) not in (1, 3)
                or not isinstance(row.get("pts_seconds"), list) or len(row["pts_seconds"]) != len(row["frame_indices"])):
            raise ValueError("A research sample requires exactly one or three inventoried frame/PTS entries")
        selection = select_episode(row["episode_index"])
        clock = selection["clock"]
        for index, pts in zip(row["frame_indices"], row["pts_seconds"]):
            if (type(index) is not int or not 0 <= index < clock["decoded_frame_count"]
                    or pts != clock["pts_seconds"][index]):
                raise ValueError("Clip source PTS/index differs from pinned RGB inventory")
        key = (row["episode_index"], row["frame_indices"][len(row["frame_indices"]) // 2])
        if key in seen:
            raise ValueError("Duplicate research clip centre")
        seen.add(key)
        if set(row["arrays"]) - set((*FIELDS, *OPTIONAL)):
            raise ValueError("Unknown source array field")
        if row.get("target_source") not in ("moge2_pseudo_depth", "admitted_training_stereo"):
            raise ValueError("Target source must distinguish imitation from admitted training stereo")
        if row.get("target_source") == "admitted_training_stereo" and row.get("reserved_correspondences_excluded") is not True:
            raise ValueError("Stereo training support must exclude reserved evaluation correspondences")
        if not isinstance(row.get("confidence_semantics"), str) or not row["confidence_semantics"].strip():
            raise ValueError("Producer confidence semantics are required and cannot be invented")
        for field, reference in row["arrays"].items():
            path, producer_path = _reference_receipt(reference, field, row, revision, split["split_id"])
            parents.add(producer_path)
            artifacts.add(path)
    with operation_run("research_prepare", request, inputs=[source_path, *sorted(artifacts)],
                       parents=sorted(parents)) as (directory, record):
        record["architecture"] = ARCHITECTURE
        samples, count, log_sum, log_square = [], 0, 0.0, 0.0
        for index, row in enumerate(rows):
            arrays = {}
            for field, ref in row["arrays"].items():
                path = local_path(ref["artifact"])
                if sha256_file(path) != ref["sha256"]:
                    raise ValueError("Source changed after admission")
                with np.load(path, allow_pickle=False) as data:
                    arrays[field] = data[ref["key"]].copy()
                    producer = read_run(ROOT, local_path(ref["producer_run"]), allowed=("video_only",))
                    if producer["kind"] in ("scratch_depth.prepare", "depth.generate", "stereo_depth.generate"):
                        intrinsic_key = "rgb_intrinsics_pixels" if producer["kind"] == "stereo_depth.generate" else "input_intrinsics_pixels"
                        if (data[intrinsic_key].shape != (3, 3)
                                or not np.allclose(data[intrinsic_key], row["geometry"]["input_intrinsics_pixels"], rtol=1e-6, atol=1e-6)):
                            raise ValueError("Source array calibration differs from the matched centre grid")
                if ref.get("add_frame_axis", False):
                    if ref["add_frame_axis"] is not True or field not in ("rgb", "base_depth_m", "base_valid", "confidence"):
                        raise ValueError("Only single-frame input arrays permit explicit frame-axis expansion")
                    arrays[field] = arrays[field][None]
            _sample(arrays, row, np)
            # Fixed preprocessing uses admitted input depth from training only.
            if _partition(row) == "train":
                values = np.log(arrays["base_depth_m"][arrays["base_valid"]].astype(np.float64))
                count += values.size
                log_sum += float(values.sum())
                log_square += float(np.square(values).sum())
            path = directory / f"sample_{index:04d}.npz"
            save_npz(path, np, **arrays)
            samples.append({**row, "partition": _partition(row), "artifact": path.name, "sha256": sha256_file(path)})
            if sum(file.stat().st_size for file in directory.iterdir() if file.is_file()) > 4 * 1024**3:
                raise RuntimeError("Preparation artifact ceiling exceeded")
        if not count:
            raise ValueError("No real training depth for frozen preprocessing")
        mean = log_sum / count
        preprocessing = {"fit_episode_indices": list(TRAIN), "fit_partition": "train",
                         "input_log_depth_mean": mean,
                         "input_log_depth_std": max((max(log_square / count - mean * mean, 0)) ** 0.5, 1e-6),
                         "rgb_divisor": 255, "supported_input_pixels": count}
        document = {"schema_version": 1, "kind": "scratch_depth_admitted_learning_data", "status": "completed",
                    "provenance": record["provenance"], "source_split": "public", "real_captured_only": True,
                    "dataset_revision": revision, "split_id": split["split_id"], "samples": samples,
                    "preprocessing": preprocessing, "producer_run": relative(directory / "run.json"),
                    "source_manifest": input_file(ROOT, source_path)}
        path = directory / "dataset.json"
        write_json(path, document)
        record.update(dataset=path.name, dataset_sha256=sha256_file(path), preprocessing=preprocessing,
                      sample_count=len(samples), no_holdout_access=True)
    return path


def _input_batch(arrays, row, preprocessing, request, np, torch):
    _sample(arrays, row, np, require_target=False)
    if request.context and arrays["rgb"].shape[0] != 3:
        raise ValueError("Context ablation requires three admitted frames")
    centre = arrays["rgb"].shape[0] // 2
    frames = range(3) if request.context else (centre,)
    channels = []
    for index in frames:
        valid = arrays["base_valid"][index]
        depth = np.where(valid, arrays["base_depth_m"][index], 1).astype(np.float32)
        logz = (np.log(depth) - preprocessing["input_log_depth_mean"]) / preprocessing["input_log_depth_std"]
        rgb = arrays["rgb"][index].astype(np.float32).transpose(2, 0, 1) / 255.0
        channels.extend((rgb * valid[None], np.where(valid, logz, 0)[None], valid[None].astype(np.float32),
                         np.where(valid, arrays["confidence"][index], 0)[None].astype(np.float32)))
    batch = {"inputs": np.concatenate(channels)[None],
             "base_depth_m": np.where(arrays["base_valid"][centre], arrays["base_depth_m"][centre], 1)[None, None],
             "base_valid": arrays["base_valid"][centre][None, None],
             "intrinsics": arrays["intrinsics"][None]}
    return {name: torch.from_numpy(value).to(dtype=torch.bool if value.dtype == np.bool_ else torch.float32)
            for name, value in batch.items()}


def _batch(arrays, row, preprocessing, request, np, torch):
    _sample(arrays, row, np)
    batch = _input_batch(arrays, row, preprocessing, request, np, torch)
    labels = {
             "target_depth_m": np.where(arrays["target_valid"], arrays["target_depth_m"], 1)[None, None],
             "target_valid": arrays["target_valid"][None, None]}
    for option, field in ((request.normals, "normal_support"), (request.temporal, "temporal_support")):
        if option and field not in arrays:
            raise ValueError(f"Enabled objective requires explicit {field}")
    if request.normals:
        labels["normal_support"] = arrays["normal_support"][None, None]
    if request.temporal:
        labels["temporal_support"] = arrays["temporal_support"][None]
        labels["temporal_depth_m"] = np.where(arrays["temporal_support"], arrays["temporal_depth_m"], 1)[None]
    batch.update({name: torch.from_numpy(value).to(dtype=torch.bool if value.dtype == np.bool_ else torch.float32)
                  for name, value in labels.items()})
    return batch


def train_learning(request: LearningRequest):
    """Train one declared arm; no tuning/holdout loader or implicit best epoch."""
    runtime = authorize("research_train")
    import numpy as np
    import torch
    from ._research_network import AdmittedDepthCNN, objectives

    dataset_path = local_path(request.dataset)
    if sha256_file(dataset_path) != request.dataset_sha256:
        raise ValueError("Prepared dataset differs from the predeclared training request digest")
    doc = read_json(dataset_path)
    producer_path = local_path(doc["producer_run"])
    producer = read_run(ROOT, producer_path, allowed=("video_only",), kind="scratch_depth.research_prepare")
    if (doc.get("kind") != "scratch_depth_admitted_learning_data" or doc.get("status") != "completed"
            or producer_path != dataset_path.parent / "run.json" or producer.get("dataset") != dataset_path.name
            or producer.get("dataset_sha256") != sha256_file(dataset_path)
            or producer.get("artifacts", {}).get(dataset_path.name) != sha256_file(dataset_path)
            or doc.get("real_captured_only") is not True or doc.get("provenance") != "video_only"
            or producer.get("preprocessing") != doc["preprocessing"]):
        raise ValueError("Dataset/preprocessing is not owned by completed real-only research preparation")
    pp = doc["preprocessing"]
    if pp.get("fit_partition") != "train" or pp.get("fit_episode_indices") != list(TRAIN):
        raise ValueError("Preprocessing must be frozen using training episodes only")
    if (any(type(pp.get(key)) not in (float, int) or not math.isfinite(pp[key])
            for key in ("input_log_depth_mean", "input_log_depth_std"))
            or pp["input_log_depth_std"] <= 0 or pp.get("rgb_divisor") != 255
            or doc.get("dataset_revision") != read_json("configs/sources.lock.json")["dataset"]["revision"]
            or doc.get("split_id") != read_json("configs/splits/public-v1.json")["split_id"]):
        raise ValueError("Frozen preprocessing/split is invalid or stale")
    rows = []
    for row in doc["samples"]:
        if row.get("partition") != _partition(row):
            raise ValueError("Episode partition mismatch")
        if row["partition"] == "train":
            rows.append(row)
    if not 1 <= len(rows) <= 128:
        raise ValueError("Training requires 1..128 predeclared real clip centres")
    with operation_run("research_train", request, inputs=[dataset_path], parents=[producer_path]) as (directory, record):
        record.update(architecture=ARCHITECTURE, pretrained_weights=False, initialization="original random",
                      source_dataset=input_file(ROOT, dataset_path), preprocessing=pp, epochs=[], optimizer_steps=0,
                      no_tuning_or_holdout_loader=True, output_support="admitted centre-depth support only",
                      objective_weights={"boundary": 0.1, "normals": 0.1, "temporal": 0.1},
                      budget={"gpu_seconds": 1800, "peak_reserved_bytes": 12 * 1024**3,
                              "artifact_bytes": 4 * 1024**3, "batch_size": 4},
                      model_sources=[input_file(ROOT, FEATURE / name) for name in
                                     ("_network.py", "_research_network.py", "research.py")])
        record["model_config"] = {"mode": request.mode, "context": request.context,
                                  "base_channels": 24, "dropout": 0.1, "correction_bound": 0.5}
        record["sources"].append(input_file(ROOT, ROOT / "track3/timing.py"))
        trace = PhaseRecorder(enabled=request.trace_runtime)
        sync = lambda: torch.cuda.synchronize(0)
        record["inputs"] += [input_file(ROOT, within(dataset_path.parent, row["artifact"])) for row in rows]
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                         stage="scratch_depth_research")
        started, model, optimizer, examples = time.monotonic(), None, None, 0
        def budget():
            lease.require_headroom()
            if time.monotonic() - started > 1800 or torch.cuda.max_memory_reserved(0) > 12 * 1024**3:
                raise RuntimeError("Declared arm time/device-memory ceiling exceeded")
        try:
            with lease, trace.span("end_to_end", "cold", 0):
                torch.cuda.set_device(0)
                torch.cuda.reset_peak_memory_stats(0)
                torch.manual_seed(31)
                model = AdmittedDepthCNN(mode=request.mode, context=request.context).to("cuda:0").train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
                record["parameter_count"] = sum(value.numel() for value in model.parameters())
                rng = np.random.default_rng(31)
                for epoch in range(1, request.epochs + 1):
                    counters, sums, steps = {}, {}, 0
                    order = rng.permutation(len(rows))
                    for offset in range(0, len(order), 4):
                        budget()
                        optimizer.zero_grad(set_to_none=True)
                        indices = order[offset:offset + 4]
                        # Accumulation matches batch four, including the fixed final partial batch.
                        for index in indices:
                            iteration = examples
                            regime = "cold" if iteration == 0 else "warmup" if iteration < 3 else "warm"
                            # Bound observer overhead; keep all samples in training.
                            observer = trace if iteration < 32 else PhaseRecorder()
                            row = rows[int(index)]
                            path = within(dataset_path.parent, row["artifact"])
                            if sha256_file(path) != row["sha256"] or producer.get("artifacts", {}).get(path.name) != row["sha256"]:
                                raise ValueError("Admitted sample changed after preparation")
                            with observer.span("decode", regime, iteration):
                                with np.load(path, allow_pickle=False) as archive:
                                    arrays = {name: archive[name].copy() for name in archive.files}
                            with observer.span("preprocess", regime, iteration):
                                batch = _batch(arrays, row, pp, request, np, torch)
                            with observer.span("transfer_to_device", regime, iteration, device="gpu", synchronize=sync):
                                batch = {name: value.to("cuda:0") for name, value in batch.items()}
                            with observer.span("model", regime, iteration, device="gpu", synchronize=sync):
                                prediction = model(batch["inputs"], batch["base_depth_m"])
                            with observer.span("loss_backward_update", regime, iteration, device="gpu", synchronize=sync):
                                loss, terms, support = objectives(prediction, batch, request)
                                if not bool(torch.isfinite(loss)):
                                    raise FloatingPointError("Nonfinite research objective")
                                (loss / len(indices)).backward()
                            for name, value in terms.items():
                                sums[name] = sums.get(name, 0.0) + value
                            for name, value in support.items():
                                counters[name] = counters.get(name, 0) + value
                            steps += 1
                            examples += 1
                            del batch, prediction, loss
                            budget()
                        with observer.span("loss_backward_update", regime, iteration, device="gpu", synchronize=sync):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                            optimizer.step()
                        record["optimizer_steps"] += 1
                    record["epochs"].append({"epoch": epoch, "mean_per_clip_objectives": {k: v / steps for k, v in sums.items()},
                                             "native_support_counts": counters})
                    write_json(directory / "run.json", record)
                with trace.span("transfer_to_host", "warm", examples, device="gpu", synchronize=sync):
                    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                if any(not bool(torch.isfinite(value).all()) for value in state.values()):
                    raise FloatingPointError("Nonfinite research checkpoint")
                path = directory / "research-checkpoint.pt"
                payload = {"schema_version": 1, "architecture": ARCHITECTURE, "request": asdict(request),
                           "state_dict": state, "pretrained_weights": False, "producer_run": relative(directory / "run.json"),
                           "dataset": record["source_dataset"], "preprocessing": pp, "model_sources": record["model_sources"],
                           "optimizer_steps": record["optimizer_steps"], "trained_epochs": request.epochs,
                           "model_config": record["model_config"], "production_admitted": False}
                with trace.span("artifact_io", "warm", examples):
                    with path.open("xb") as stream:
                        torch.save(payload, stream)
                if path.stat().st_size > 4 * 1024**3:
                    raise RuntimeError("Checkpoint artifact ceiling exceeded")
                budget()
                record["checkpoint"] = input_file(ROOT, path)
        finally:
            record["gpu"] = lease.report()
            if request.trace_runtime:
                write_json(directory / "runtime_trace.json", {
                    "schema_version": 1, "producer_run": relative(directory / "run.json"),
                    "scope": "whole_training_arm", "source_split": "public",
                    "episode_indices": sorted({row["episode_index"] for row in rows}),
                    "episode_groups": {"train": list(TRAIN), "tuning": list(TUNE)},
                    "workload": {"architecture": ARCHITECTURE, "request": asdict(request),
                                 "dataset_sha256": sha256_file(dataset_path), "parameter_count": record.get("parameter_count"),
                                 "training_clip_centres": len(rows), "processed_examples": examples,
                                 "optimizer_steps": record["optimizer_steps"], "traced_examples": min(examples, 32)},
                    "spans": trace.snapshot(), "timing_source": input_file(ROOT, ROOT / "track3/timing.py"),
                    "semantics": "First example cold, next two warmup, next 29 warm; synchronized GPU host wall. "
                                 "Decode means prepared NPZ read; RGB video decode happened upstream. "
                                 "End-to-end covers arm after admission, including initialization and checkpoint. "
                                 "Nested spans overlap; accumulation backward and optimizer update are separate spans. "
                                 "No compilation or complete frame pipeline timing; failed spans remain visible."})
            if torch.cuda.is_initialized():
                record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                model = optimizer = None
                torch.cuda.empty_cache()
    return path


def predict_learning(request: PredictLearningRequest):
    """Emit every declared sample of one tuning episode; never use target masks."""
    runtime = authorize("research_predict")
    checkpoint, dataset_path = local_path(request.checkpoint), local_path(request.dataset)
    if (checkpoint.name != "research-checkpoint.pt" or not 0 < checkpoint.stat().st_size <= 128 * 1024**2
            or sha256_file(checkpoint) != request.checkpoint_sha256
            or sha256_file(dataset_path) != request.dataset_sha256):
        raise ValueError("Research checkpoint/dataset byte limits or request digests differ")
    training_path = checkpoint.parent / "run.json"
    training = read_run(ROOT, training_path, allowed=("video_only",), kind="scratch_depth.research_train")
    if (training.get("checkpoint") != input_file(ROOT, checkpoint)
            or training.get("artifacts", {}).get(checkpoint.name) != request.checkpoint_sha256
            or training.get("source_dataset") != input_file(ROOT, dataset_path)
            or training.get("architecture") != ARCHITECTURE or training.get("pretrained_weights") is not False):
        raise ValueError("Checkpoint is not owned by the exact completed original training run")
    recipe = LearningRequest(**training["request"])
    if local_path(recipe.dataset) != dataset_path or recipe.dataset_sha256 != request.dataset_sha256:
        raise ValueError("Training request identifies a different dataset")
    sources = [input_file(ROOT, FEATURE / name) for name in ("_network.py", "_research_network.py", "research.py")]
    if (training.get("model_sources") != sources
            or any(training.get("source_hashes", {}).get(source["path"]) != source["sha256"] for source in sources)):
        raise ValueError("Research implementation changed; use the checkpoint's frozen original sources")
    document = read_json(dataset_path)
    preparation_path = local_path(document["producer_run"])
    preparation = read_run(ROOT, preparation_path, allowed=("video_only",), kind="scratch_depth.research_prepare")
    revision = read_json("configs/sources.lock.json")["dataset"]["revision"]
    if (preparation_path != dataset_path.parent / "run.json"
            or preparation.get("dataset") != dataset_path.name
            or preparation.get("dataset_sha256") != request.dataset_sha256
            or preparation.get("artifacts", {}).get(dataset_path.name) != request.dataset_sha256
            or document.get("kind") != "scratch_depth_admitted_learning_data"
            or document.get("status") != "completed" or document.get("provenance") != "video_only"
            or document.get("source_split") != "public" or document.get("real_captured_only") is not True
            or document.get("dataset_revision") != revision
            or document.get("split_id") != read_json("configs/splits/public-v1.json")["split_id"]
            or document.get("preprocessing") != training.get("preprocessing")
            or preparation.get("preprocessing") != document["preprocessing"]):
        raise ValueError("Prepared inference input lineage or frozen preprocessing differs")
    pp = document["preprocessing"]
    if (pp.get("fit_partition") != "train" or pp.get("fit_episode_indices") != list(TRAIN)
            or pp.get("rgb_divisor") != 255
            or any(type(pp.get(key)) not in (float, int) or not math.isfinite(pp[key]) for key in
                   ("input_log_depth_mean", "input_log_depth_std")) or pp["input_log_depth_std"] <= 0):
        raise ValueError("Invalid frozen train-only preprocessing")
    rows, seen, training_samples = [], set(), 0
    for row in document["samples"]:
        partition = _partition(row)
        if partition != row.get("partition"):
            raise ValueError("Prepared sample partition differs from its episode")
        training_samples += partition == "train"
        if row["episode_index"] == request.episode_index:
            index = row["frame_indices"][len(row["frame_indices"]) // 2]
            if index in seen:
                raise ValueError("Duplicate tuning centre; complete prediction inventory is ambiguous")
            seen.add(index)
            rows.append(row)
    steps = recipe.epochs * ((training_samples + 3) // 4)
    if (not rows or len(rows) > 128 or not 1 <= training_samples <= 128
            or training.get("optimizer_steps") != steps or len(training.get("epochs", [])) != recipe.epochs):
        raise ValueError("Incomplete training counters or missing/beyond-budget declared tuning samples")
    calibration_path = local_path(request.calibration_depth_run)
    calibration = read_run(ROOT, calibration_path, allowed=("video_only",), kind="depth.generate")
    if (calibration.get("episode_index") != request.episode_index or calibration.get("split") != "public"
            or calibration.get("dataset_revision") != revision or calibration.get("camera") != "ego_cam_a"):
        raise PermissionError("Calibration run must match the selected public tuning episode/camera/revision")
    geometry_path = within(calibration_path.parent, "preprocessing.npz")
    grid = sha256_file(geometry_path)
    if grid != calibration.get("preprocessing_artifact_sha256") or calibration.get("artifacts", {}).get(geometry_path.name) != grid:
        raise ValueError("Calibrated preprocessing bytes are not producer-owned")
    clock = select_episode(request.episode_index)["clock"]
    import numpy as np
    import torch
    from ._research_network import AdmittedDepthCNN

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = {"mode": recipe.mode, "context": recipe.context, "base_channels": 24,
              "dropout": 0.1, "correction_bound": 0.5}
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or payload.get("architecture") != ARCHITECTURE or payload.get("request") != training["request"]
            or payload.get("producer_run") != relative(training_path) or payload.get("dataset") != training["source_dataset"]
            or payload.get("model_sources") != sources or payload.get("preprocessing") != pp
            or payload.get("model_config") != config or training.get("model_config") != config
            or payload.get("pretrained_weights") is not False or payload.get("production_admitted") is not False
            or payload.get("trained_epochs") != recipe.epochs or payload.get("optimizer_steps") != steps):
        raise ValueError("Safe-loaded checkpoint metadata differs from its completed owner/configuration")
    state = payload.get("state_dict")
    if (not isinstance(state, dict) or not state or len(state) > 128
            or any(not isinstance(name, str) or not torch.is_tensor(value) or value.layout != torch.strided
                   or value.dtype != torch.float32 or value.numel() > 10_000_000 or not bool(torch.isfinite(value).all())
                   for name, value in state.items()) or sum(value.numel() for value in state.values()) > 10_000_000):
        raise ValueError("Checkpoint tensors exceed the finite dense float32 state contract")
    with np.load(geometry_path, allow_pickle=False) as arrays:
        input_valid = arrays["input_valid"].copy()
        intrinsics = arrays["input_intrinsics_pixels"].copy()
    if input_valid.dtype != np.dtype("bool") or input_valid.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("Invalid calibrated pinhole ray grid")
    with operation_run("research_predict", request,
                       inputs=[checkpoint, dataset_path, geometry_path],
                       parents=[training_path, preparation_path, calibration_path]) as (directory, record):
        record.update(architecture=ARCHITECTURE, split="public", episode_index=request.episode_index,
                      dataset_revision=revision, split_id=document["split_id"], camera="ego_cam_a",
                      depth_kind="metric_optical_z_estimate", preprocessing=calibration["preprocessing"],
                      preprocessing_artifact_sha256=grid, grid=grid, model_sources=sources, model_config=config,
                      source_dataset=input_file(ROOT, dataset_path), source_checkpoint=input_file(ROOT, checkpoint),
                      output_support="admitted centre input-depth/ray support and finite positive prediction only",
                      frames=[], declared_sample_count=len(rows), production_admitted=False,
                      no_target_masks_consumed=True)
        shutil.copyfile(geometry_path, directory / "preprocessing.npz")
        record["inputs"] += [input_file(ROOT, within(dataset_path.parent, row["artifact"])) for row in rows]
        (directory / "frames").mkdir()
        lease = GpuLease(ROOT, runtime["gpu_headroom_mib"], runtime["gpu_admission_allowance_mib"],
                         stage="scratch_depth_research_predict")
        model, started = None, time.monotonic()
        try:
            with lease:
                torch.cuda.set_device(0)
                torch.cuda.reset_peak_memory_stats(0)
                model = AdmittedDepthCNN(mode=recipe.mode, context=recipe.context)
                expected = model.state_dict()
                if set(state) != set(expected) or any(state[name].shape != expected[name].shape for name in expected):
                    raise ValueError("Checkpoint state keys/shapes differ from the bounded original architecture")
                model.load_state_dict(state, strict=True)
                model = model.to("cuda:0").eval()
                with torch.inference_mode():
                    for row in sorted(rows, key=lambda sample: sample["frame_indices"][len(sample["frame_indices"]) // 2]):
                        lease.require_headroom()
                        if time.monotonic() - started > 1800 or torch.cuda.max_memory_reserved(0) > 12 * 1024**3:
                            raise RuntimeError("Research prediction time/device-memory ceiling exceeded")
                        path = within(dataset_path.parent, row["artifact"])
                        if sha256_file(path) != row["sha256"] or preparation.get("artifacts", {}).get(path.name) != row["sha256"]:
                            raise ValueError("Prepared tuning input changed")
                        # Deliberately do not read target, target validity, normals or temporal labels.
                        with np.load(path, allow_pickle=False) as archive:
                            arrays = {name: archive[name].copy() for name in FIELDS if name not in ("target_depth_m", "target_valid")}
                        if (arrays["base_valid"].shape[1:] != input_valid.shape
                                or not np.allclose(arrays["intrinsics"], intrinsics, rtol=1e-6, atol=1e-6)
                                or bool((arrays["base_valid"] & ~input_valid[None]).any())):
                            raise ValueError("Prepared source input rays differ from the calibrated inference grid")
                        batch = {key: value.to("cuda:0") for key, value in _input_batch(arrays, row, pp, recipe, np, torch).items()}
                        depth = model(batch["inputs"], batch["base_depth_m"])[0, 0].cpu().numpy()
                        centre = len(row["frame_indices"]) // 2
                        valid = arrays["base_valid"][centre] & input_valid & np.isfinite(depth) & (depth > 0)
                        index, pts = row["frame_indices"][centre], row["pts_seconds"][centre]
                        if pts != clock["pts_seconds"][index]:
                            raise ValueError("Tuning sample PTS differs from the pinned RGB source clock")
                        destination = directory / "frames" / f"frame_{index:06d}.npz"
                        save_npz(destination, np, depth_m=np.where(valid, depth, np.nan).astype(np.float32), valid=valid,
                                 input_intrinsics_pixels=intrinsics, frame_index=np.int64(index),
                                 pts=np.int64(clock["pts_integer"][index]), episode_index=np.int64(request.episode_index))
                        record["frames"].append({"frame_index": index, "video_pts": clock["pts_integer"][index],
                                                 "video_pts_seconds": pts, "status": "completed", "valid_pixels": int(valid.sum()),
                                                 "artifact": destination.relative_to(directory).as_posix(), "sha256": sha256_file(destination)})
                        del batch
                        write_json(directory / "run.json", record)
                record["episodes"] = [{"episode_index": request.episode_index, "split": "public", "frames": record["frames"]}]
                if sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) > 4 * 1024**3:
                    raise RuntimeError("Research prediction artifact ceiling exceeded")
        finally:
            record["gpu"] = lease.report()
            model = None
            if torch.cuda.is_initialized():
                record["gpu_torch"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
                torch.cuda.empty_cache()
    return directory / "run.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "train", "predict"))
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request_type, handler = {"prepare": (PrepareLearningRequest, prepare_learning),
                             "train": (LearningRequest, train_learning),
                             "predict": (PredictLearningRequest, predict_learning)}[args.operation]
    print(handler(request_type.from_file(args.request)))


if __name__ == "__main__":
    main()
