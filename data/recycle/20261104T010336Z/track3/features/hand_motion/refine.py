"""Conservative, depth-supported wrist refinement; articulated joints remain inferred."""

from dataclasses import dataclass
from fractions import Fraction

from track3.provenance import sha256_file
from ._common import (arrays, authorize, is_number, operation_run, producer, read_json,
                      require_input, rgb_clock, save_npz)

PUBLIC_RESEARCH_EPISODES = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)


@dataclass(frozen=True)
class RefineHandsRequest:
    source_split: str = "public"
    episode_index: int = 39
    hand_run: str | None = None
    frame_start: int = 0
    frame_stop: int = 384
    window_frames: int = 5
    maximum_frames: int = 384
    maximum_gap_seconds: float = 0.1
    maximum_correction_m: float = 0.005
    rapid_acceleration_m_s2: float = 5.0
    minimum_depth_support: int = 20
    maximum_depth_residual_m: float = 0.025

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        research_episode(self)
        if (type(self.frame_start) is not int or type(self.frame_stop) is not int or
                self.frame_start < 0 or not self.frame_start < self.frame_stop or
                self.frame_stop - self.frame_start > self.maximum_frames):
            raise ValueError("Supply an explicit bounded source-frame interval")
        if type(self.window_frames) is not int or not 3 <= self.window_frames <= 15 or self.window_frames % 2 != 1:
            raise ValueError("Temporal windows must have 3–15 odd frames")
        if type(self.maximum_frames) is not int or not 1 <= self.maximum_frames <= 384:
            raise ValueError("Hand refinement is bounded to 384 frames")
        if type(self.minimum_depth_support) is not int or self.minimum_depth_support < 1:
            raise ValueError("Depth support must be positive")
        for key in ("maximum_gap_seconds", "maximum_correction_m", "rapid_acceleration_m_s2", "maximum_depth_residual_m"):
            if not is_number(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid hand refinement bound: {key}")


def research_episode(request):
    if request.source_split != "public" or type(request.episode_index) is not int or request.episode_index not in PUBLIC_RESEARCH_EPISODES:
        raise PermissionError("Hand research admits public training/tuning only; 41/43/44 excluded")


def source_seconds(request, data, record, np):
    clock = rgb_clock(request, record)
    bases = {row[0] for row in clock["time_base_and_frame_rate"]}
    if len(bases) != 1:
        raise ValueError("Hand refinement needs one exact source PTS time base")
    base = Fraction(next(iter(bases)))
    frames, pts = data["frame_index"], data["pts"]
    if (base <= 0 or frames.dtype != np.dtype("int64") or pts.dtype != np.dtype("int64") or
            frames.ndim != 1 or pts.shape != frames.shape or not len(pts) or
            (np.diff(frames) <= 0).any() or (np.diff(pts) <= 0).any() or
            (frames < 0).any() or (frames >= len(clock["pts_integer"])).any()):
        raise ValueError("Invalid hand source clock")
    if not np.array_equal(pts, np.asarray(clock["pts_integer"], dtype=np.int64)[frames]):
        raise ValueError("Hand PTS differ from the pinned RGB clock")
    record["time_base"] = str(base)
    # Subtract integer origin before conversion to avoid large-PTS cancellation.
    return (pts - pts[0]).astype(np.float64) * float(base)


def derivatives(points, supported, seconds, frames, gap, np):
    """Backward velocity and nonuniform centred acceleration, never across gaps."""
    points = np.asarray(points, dtype=np.float64)
    supported = np.asarray(supported, dtype=bool)
    dt = np.diff(seconds)
    step = dt.reshape(-1, *[1] * (points.ndim - 1))
    contiguous = (np.diff(frames) == 1) & ~(dt > gap)
    vvalid = np.zeros(supported.shape, dtype=bool)
    vvalid[1:] = supported[1:] & supported[:-1] & contiguous.reshape(-1, *[1] * (supported.ndim - 1))
    velocity = np.full(points.shape, np.nan)
    velocity[1:][vvalid[1:]] = ((points[1:] - points[:-1]) / step)[vvalid[1:]]
    avalid = np.zeros(supported.shape, dtype=bool)
    avalid[2:] = vvalid[2:] & vvalid[1:-1]
    acceleration = np.full(points.shape, np.nan)
    acceleration[2:][avalid[2:]] = (2 * (velocity[2:] - velocity[1:-1]) / (step[1:] + step[:-1]))[avalid[2:]]
    return velocity, acceleration, vvalid, avalid


def _wrist_corrections(wrists, supported, seconds, frames, residual, raw_acceleration, raw_acceleration_valid,
                       request, np):
    """Bounded weighted-linear wrist translation over full, contiguous, non-rapid windows."""
    count = len(seconds)
    correction = np.zeros((count, 2, 3), dtype=np.float64)
    dispositions = np.full((count, 2), "unsupported", dtype="U32")
    half = request.window_frames // 2
    for index, side in zip(*np.nonzero(supported)):
        dispositions[index, side] = "insufficient_window"
        low, high = index - half, index + half + 1
        if low < 0 or high > count:
            continue
        ids = np.arange(low, high)
        if (not supported[ids, side].all() or (np.diff(frames[ids]) != 1).any() or
                (np.diff(seconds[ids]) > request.maximum_gap_seconds).any()):
            dispositions[index, side] = "gap_or_support_loss"
            continue
        rapid = raw_acceleration_valid[ids, side] & (np.linalg.norm(raw_acceleration[ids, side], axis=-1) > request.rapid_acceleration_m_s2)
        if rapid.any():
            dispositions[index, side] = "rapid_motion_preserved"
            continue
        design = np.stack((np.ones(len(ids)), seconds[ids] - seconds[index]), axis=1)
        weights = 1 / np.maximum(residual[ids, side], 0.001)
        prediction = np.linalg.lstsq(design * weights[:, None], wrists[ids, side] * weights[:, None], rcond=None)[0][0]
        delta = prediction - wrists[index, side]
        if np.linalg.norm(delta) > request.maximum_correction_m:
            dispositions[index, side] = "correction_limit"
            continue
        correction[index, side] = delta
        dispositions[index, side] = "bounded_translation"
    return correction, dispositions


def refine(request):
    research_episode(request)
    authorize(request, "refine")
    hand = producer(request.hand_run, "hand_motion.lift", request)
    camera = producer(hand[1]["request"].get("camera_run"), "camera_motion.estimate", request)
    require_input(hand, camera)
    if hand[1].get("grid") != camera[1].get("grid"):
        raise ValueError("Lifted hand and camera grids differ")
    with operation_run(request, [hand, camera], "refine") as (directory, record):
        import numpy as np

        data = arrays(hand, "hands.npz", record, np)
        selected = (data["frame_index"] >= request.frame_start) & (data["frame_index"] < request.frame_stop)
        source_count = len(data["frame_index"])
        data = {key: value[selected].copy() if key not in ("sides", "betas") and value.ndim and value.shape[0] == source_count
                else value for key, value in data.items()}
        seconds = source_seconds(request, data, record, np)
        count = len(seconds)
        if count > request.maximum_frames:
            raise ValueError("Selected hand interval exceeds its resource budget")
        if (data["joints_world"].shape != (count, 2, 21, 3) or data["world_valid"].shape != (count, 2) or
                data["world_valid"].dtype != np.dtype("bool") or data["sides"].tolist() != ["left", "right"]):
            raise ValueError("Expected anatomical MANO21 world hands")
        if hand[1].get("joint_order") != "MANO21" or hand[1].get("units") != "metres":
            raise ValueError("Hand refinement requires metric MANO21 input")
        valid = data["world_valid"]
        if not np.isfinite(data["joints_world"][valid]).all():
            raise ValueError("Nonfinite valid hand joints")
        supported = (valid & (data["depth_support"] >= request.minimum_depth_support) &
                     np.isfinite(data["depth_residual_m"]) &
                     (data["depth_residual_m"] <= request.maximum_depth_residual_m))
        original = data["joints_world"].copy()
        _, raw_acceleration, _, raw_acceleration_valid = derivatives(
            original[:, :, 0], supported, seconds, data["frame_index"], request.maximum_gap_seconds, np)
        correction, dispositions = _wrist_corrections(
            original[:, :, 0], supported, seconds, data["frame_index"], data["depth_residual_m"],
            raw_acceleration, raw_acceleration_valid, request, np)
        data["joints_world"] = original + correction[:, :, None]
        data.update(unrefined_joints_world=original, correction_world_m=correction,
                    wrist_depth_supported=supported,
                    joint_observed=np.zeros((count, 2, 21), dtype=bool),
                    joint_inferred=np.broadcast_to(valid[:, :, None], (count, 2, 21)).copy(),
                    refinement_status=dispositions)
        joint_valid = data["joint_inferred"]
        velocity, acceleration, vvalid, avalid = derivatives(data["joints_world"], joint_valid, seconds,
                                                            data["frame_index"], request.maximum_gap_seconds, np)
        data.update(velocity_world_m_s=velocity, acceleration_world_m_s2=acceleration,
                    velocity_valid=vvalid, acceleration_valid=avalid)
        parents = np.asarray([0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        lengths = np.linalg.norm(data["joints_world"][:, :, 1:] - data["joints_world"][:, :, parents], axis=-1)
        data["bone_length_m"] = lengths
        baseline_lengths = np.linalg.norm(original[:, :, 1:] - original[:, :, parents], axis=-1)
        data["bone_length_correction_m"] = lengths - baseline_lengths
        # CamA fields retain the initializer; corrections are world-only and explicit.
        record.update(grid=hand[1]["grid"], units="metres", joint_order="MANO21",
                      orientation_source=hand[1]["orientation_source"],
                      joint_frame_convention=hand[1]["joint_frame_convention"],
                      output_parameter_form=hand[1]["output_parameter_form"],
                      input_parameter_form=hand[1]["input_parameter_form"],
                      unmirroring_applied=hand[1]["unmirroring_applied"], hand_model=hand[1]["hand_model"],
                      method="bounded_depth_supported_wrist_translation", articulation="unchanged_inferred_fk",
                      camera_space_fields="unrefined_initializer", interpolation="none",
                      shape="unchanged_episode_betas", observation_semantics="depth-supported wrist is not an observed joint landmark")
        for index in range(count):
            record["frames"].append({"frame_index": int(data["frame_index"][index]), "video_pts": int(data["pts"][index]),
                                     "status": "completed", "hands": dispositions[index].tolist()})
        save_npz(directory / "hands.npz", np, **data)
        record["artifacts"]["hands.npz"] = sha256_file(directory / "hands.npz")
    return directory / "run.json"
