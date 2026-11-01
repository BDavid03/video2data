"""Bounded opt-in scene refinement; raw producers and assembly stay unchanged.

An observation-space solver (not MANO fitting or a simulator) over hashed completed producer artifacts.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import json
import math
import time
from uuid import uuid4

from track3.paths import output_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from ._common import (ROOT, FEATURE, arrays, artifact, clock_row, dataset_revision, producer, read_json, recorded,
                      require_input, runtime_flag, same_revision, save_npz)

FAMILIES = ("point3d", "reprojection", "depth", "correspondence", "bone_length", "contact")
PRODUCERS = ("camera_motion.estimate", "object_pose.refine", "object_pose.compose",
             "hand_motion.refine", "hand_motion.lift", "object_masks.correspondences",
             "scene_assembly.prepare_refinement")
STOPPING = ("cpu_budget", "increasing_fixed_objective", "invalid_trial_or_objective", "underconstrained_observations")


@dataclass(frozen=True)
class RefineSceneRequest:
    camera_run: str
    pose_run: str
    observation_sources: tuple
    hand_run: str | None = None
    source_split: str = "public"
    episode_index: int = 39
    window_starts: tuple = (0, 12, 24, 36)
    window_frames: int = 16
    overlap_frames: int = 4
    iterations: int = 15
    cpu_seconds_per_arm: float = 600.
    maximum_generated_bytes: int = 2147483648
    maximum_variables: int = 768
    temporal_sigma_m_s: float = 0.2
    temporal_sigma_rad_s: float = 1.
    translation_bound_m: float = 0.025
    rotation_bound_rad: float = 0.1

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.source_split != "public" or type(self.episode_index) is not int or self.episode_index not in (0, 11, 12, 18, 21, 23, 26, 31, 39, 42):
            raise ValueError("Joint research admits only frozen training/tuning episodes, never 41/43/44")
        for name, low, high in (("window_frames", 2, 16), ("overlap_frames", 1, 4),
                                ("iterations", 1, 15), ("maximum_variables", 1, 768),
                                ("maximum_generated_bytes", 1, 2147483648)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid bounded field: {name}")
        if self.overlap_frames >= self.window_frames:
            raise ValueError("Windows must advance")
        starts = self.window_starts
        if not 1 <= len(starts) <= 4 or any(type(i) is not int or i < 0 for i in starts):
            raise ValueError("Supply one to four nonnegative window starts")
        if any(b - a != self.window_frames - self.overlap_frames for a, b in zip(starts, starts[1:])):
            raise ValueError("Windows must be ordered with exactly the declared overlap")
        for name in ("cpu_seconds_per_arm", "temporal_sigma_m_s", "temporal_sigma_rad_s",
                     "translation_bound_m", "rotation_bound_rad"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid positive parameter: {name}")
        if self.cpu_seconds_per_arm > 600 or not 1 <= len(self.observation_sources) <= 16:
            raise ValueError("At most 600 CPU seconds per arm and 16 observation artifacts")


def _parent(value, request, allowed):
    path, record = producer(value, None, request)
    if record["kind"] not in allowed:
        raise ValueError("Unsupported or diagnostic refinement input")
    return path, record


def _entity(value):
    parts = value.split(":")
    if parts == ["world"] or parts == ["camera"]:
        return tuple(parts)
    if len(parts) == 2 and parts[0] == "object" and parts[1].isdigit():
        return "object", int(parts[1])
    if len(parts) == 3 and parts[0] == "hand" and parts[1] in ("0", "1") and parts[2].isdigit() and int(parts[2]) < 21:
        return "hand", int(parts[1]), int(parts[2])
    raise ValueError("Unknown refinement entity")


def _admit_row(row, frame_count, pts, valid, intrinsics, np):
    """Validate one observation row; return its parsed fields and a rejection reason (None if admitted)."""
    frame = row["frame_index"]
    if type(frame) is not int or not 0 <= frame < frame_count or row["video_pts"] != int(pts[frame]):
        raise ValueError("Observation source clock mismatch")
    family = row["family"]
    if family not in FAMILIES:
        raise ValueError("Unsupported residual family must be absent, never silently ignored")
    if row.get("role") != "optimization":
        raise ValueError("Reserved/assessment observations cannot enter optimization")
    # Explicit rejected records remain visible; missing masks never default true.
    needed = ["observed", "visible", "accepted"]
    if family == "depth":
        needed.append("depth_valid")
    if family in ("correspondence", "contact"):
        needed.append("identity_supported")
    if any(type(row.get(key)) is not bool for key in (*needed, "occluded")):
        raise ValueError("Explicit observation/occlusion/support masks required")
    reason = None
    if row["occluded"] or not all(row[key] for key in needed):
        reason = "unsupported_or_occluded"
    if row.get("motion_class") not in ("static", "dynamic"):
        raise ValueError("Every residual needs an explicit static/dynamic mask classification")
    if family == "contact" and row.get("physical_contact_supported") is not True:
        reason = "proximity_is_not_contact"
    entity = _entity(row["entity"])
    if entity[0] != "world" and (frame, entity) not in valid:
        reason = "producer_state_unobserved"
    sigma = row["sigma"]
    if not isinstance(sigma, (float, int)) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Residual needs a fixed positive noise scale")
    fields = {"entity": entity}
    if family in ("reprojection", "depth"):
        if (frame, ("camera",)) not in valid:
            reason = "camera_unobserved"
        k = np.asarray(row["intrinsics_pixels"], dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Fixed calibrated pixel intrinsics required")
        if not np.array_equal(k, intrinsics):
            raise ValueError("Residual intrinsics differ from the exact frozen input pixel grid")
    if family in ("correspondence", "bone_length", "contact"):
        other = _entity(row["other_entity"])
        other_frame = row.get("other_frame_index", frame)
        if type(other_frame) is not int or not 0 <= other_frame < frame_count or row.get("other_video_pts") != int(pts[other_frame]):
            raise ValueError("Paired residual needs both explicit source clocks")
        if other[0] != "world" and (other_frame, other) not in valid:
            reason = "paired_state_unobserved"
        if any(type(row.get(key)) is not bool for key in ("other_observed", "other_visible", "other_occluded")):
            raise ValueError("Paired residual needs explicit endpoint observation masks")
        if not row["other_observed"] or not row["other_visible"] or row["other_occluded"]:
            reason = "paired_observation_unsupported"
        fields.update(other_entity=other, other_frame_index=other_frame)
    return fields, reason


def _load_observations(request, parents, record, pts, valid, grid, intrinsics, np):
    admitted, rejected, seen = [], [], set()
    pose_sha256 = input_file(ROOT, request.pose_run)["sha256"]
    camera_sha256 = input_file(ROOT, request.camera_run)["sha256"]
    for spec, parent in zip(request.observation_sources, parents):
        name = spec["artifact"]
        digest = parent[1]["artifacts"].get(name)
        if digest != spec["sha256"]:
            raise ValueError("Observation artifact hash must match both request and completed producer")
        path = artifact(parent[0], name, record, digest)
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Observation JSON exceeds bounded 64 MiB admission")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if (bundle.get("schema") != "scene_refinement_observations_v1" or
                bundle.get("units") != "metres" or bundle.get("clock") != "camA_rgb_pts" or
                bundle.get("time_base") != [1, 10240] or bundle.get("grid_sha256") != grid or
                bundle.get("pose_run_sha256") != pose_sha256 or bundle.get("camera_run_sha256") != camera_sha256):
            raise ValueError("Observation schema, grid, camera/pose lineage or source clock missing")
        rows = bundle["observations"]
        if len(rows) > len(pts) * 256:
            raise ValueError("Observation input exceeds 256 records/frame")
        source = {"path": str(path.relative_to(ROOT)), "sha256": digest}
        for row_index, row in enumerate(rows):
            if (digest, row_index) in seen:
                raise ValueError("Duplicated observation source")
            seen.add((digest, row_index))
            fields, reason = _admit_row(row, len(pts), pts, valid, intrinsics, np)
            entry = dict(row, source=dict(source, row_index=row_index), **fields)
            if reason:
                rejected.append(dict(source=entry["source"], frame_index=row["frame_index"], reason=reason))
            else:
                admitted.append(entry)
    if not admitted or max(Counter(row["frame_index"] for row in admitted).values()) > 256:
        raise ValueError("Require admitted evidence, with at most 256 records/frame across all sources")
    return admitted, rejected


def _point(state, frame, entity, local, np):
    point = np.asarray(local, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Finite three-dimensional residual point required")
    if entity[0] == "world":
        return point
    if entity[0] == "hand":
        if not np.array_equal(point, [0., 0., 0.]):
            raise ValueError("Hand residuals address joint centres only")
        return state["hand"][frame, entity[1], entity[2]]
    pose = state[entity[0]][(frame, *entity[1:])]
    return pose[:3, :3] @ point + pose[:3, 3]


def _velocity(state, a, b, entity, dt, request, np, Rotation):
    if entity[0] == "hand":
        delta = state["hand"][b, entity[1], entity[2]] - state["hand"][a, entity[1], entity[2]]
        return delta / dt / request.temporal_sigma_m_s
    pa, pb = state[entity[0]][(a, *entity[1:])], state[entity[0]][(b, *entity[1:])]
    return np.r_[(pb[:3, 3] - pa[:3, 3]) / dt / request.temporal_sigma_m_s,
                 Rotation.from_matrix(pa[:3, :3].T @ pb[:3, :3]).as_rotvec() / dt / request.temporal_sigma_rad_s]


def _residual(state, row, np):
    family = row["family"]
    point = _point(state, row["frame_index"], row["entity"], row["local_point_m"], np)
    if family in ("reprojection", "depth"):
        pose = state["camera"][row["frame_index"]]
        optical = pose[:3, :3].T @ (point - pose[:3, 3])
        if optical[2] <= 1e-6:
            raise ValueError("Fixed-support point moved behind camera")
        if family == "depth":
            if row["target"] <= 0:
                raise ValueError("Optical-z observation must be positive metres")
            return np.asarray([optical[2] - row["target"]])
        projected = np.asarray(row["intrinsics_pixels"], dtype=float) @ optical
        residual = projected[:2] / projected[2] - np.asarray(row["target"], dtype=float)
        if residual.shape != (2,):
            raise ValueError("Reprojection targets use two pixel coordinates")
        return residual
    if family == "point3d":
        residual = point - np.asarray(row["target"], dtype=float)
        if residual.shape != (3,):
            raise ValueError("Point3D targets use three world coordinates")
        return residual
    other = _point(state, row["other_frame_index"], row["other_entity"], row["other_local_point_m"], np)
    if family in ("bone_length", "contact"):
        if row["target"] < 0:
            raise ValueError("Distance targets cannot be negative")
        return np.asarray([np.linalg.norm(point - other) - row["target"]])
    residual = point - other - np.asarray(row["target"], dtype=float)
    if residual.shape != (3,):
        raise ValueError("Correspondence displacement uses world metres")
    return residual


def _residuals(state, rows, temporal, request, np, Rotation):
    result, families = [], []
    for row in rows:
        residual = _residual(state, row, np)
        result.extend(residual / row["sigma"])
        families.extend([row["family"]] * len(residual))
    for a, b, entity, dt, measured_velocity in temporal:
        # Preserve measured rapid motion; regularize changes to its velocity.
        values = _velocity(state, a, b, entity, dt, request, np, Rotation) - measured_velocity
        result.extend(values)
        families.extend(["temporal_prior"] * len(values))
    residual = np.asarray(result, dtype=float)
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Empty or nonfinite fixed-support objective")
    # Equal family weight, normalized by the original fixed scalar support count.
    counts = {family: families.count(family) for family in set(families)}
    normalization = np.asarray([1. / counts[family] for family in families])
    return residual, normalization, counts


def _objective(residual, normalization, np):
    absolute = np.abs(residual)
    return float(np.sum(np.where(absolute <= 1, .5 * residual ** 2, absolute - .5) * normalization))


def _apply(base, variables, vector, np, Rotation):
    state = {name: value.copy() for name, value in base.items()}
    offset = 0
    for frame, entity in variables:
        if entity[0] == "hand":
            state["hand"][frame, entity[1], entity[2]] += vector[offset:offset + 3]
            offset += 3
        else:
            tangent = vector[offset:offset + 6]
            increment = np.eye(4)
            increment[:3, :3] = Rotation.from_rotvec(tangent[:3]).as_matrix()
            increment[:3, 3] = tangent[3:]
            # Right-local rotation and translation; body origins are fixed.
            index = (frame, *entity[1:])
            state[entity[0]][index] = base[entity[0]][index] @ increment
            offset += 6
    return state


def _clip(trial, sizes, request, np):
    """Scale each rotation/translation block of a trial in place onto its norm bound."""
    offset = 0
    for size in sizes:
        parts = (((0, request.translation_bound_m),) if size == 3 else
                 ((0, request.rotation_bound_rad), (3, request.translation_bound_m)))
        for start, bound in parts:
            part = trial[offset + start:offset + start + 3]
            norm = np.linalg.norm(part)
            if norm > bound:
                part *= bound / norm
        offset += size


def _solve(base, variables, rows, temporal, request, arm, spent, emit, np, Rotation):
    from scipy.linalg import lstsq

    sizes = [3 if entity[0] == "hand" else 6 for _, entity in variables]
    dimensions = sum(sizes)
    if dimensions > request.maximum_variables:
        raise ValueError("Window exceeds declared variable budget; reduce admitted scope explicitly")
    limits = np.concatenate([np.full(3, request.translation_bound_m) if n == 3 else
                             np.r_[np.full(3, request.rotation_bound_rad), np.full(3, request.translation_bound_m)]
                             for n in sizes]) if sizes else np.zeros(0)
    vector, damping = np.zeros(dimensions), .01
    residual, normalization, support = _residuals(base, rows, temporal, request, np, Rotation)
    initial = objective = _objective(residual, normalization, np)
    summary = {"initial_objective": initial, "support_scalar_counts": support,
               "accepted_steps": 0, "rejected_steps": 0, "ambiguity": []}
    if dimensions == 0:
        summary.update(final_objective=initial, stop="fixed_or_no_supported_variables")
        return base, summary
    columns = {kind: [] for kind in ("camera", "object", "hand")}
    offset = 0
    for (_, entity), size in zip(variables, sizes):
        columns[entity[0]].extend(range(offset, offset + size))
        offset += size
    listed = [[frame, list(entity)] for frame, entity in variables]
    # Sequential stages share the same total iteration and CPU cap as the joint arm.
    stages = [kind for kind in ("camera", "object", "hand") if columns[kind]]
    stop = "iteration_budget"
    for iteration in range(request.iterations):
        if time.process_time() - spent > request.cpu_seconds_per_arm:
            stop = "cpu_budget"
            break
        stage = "joint" if arm == "joint" else stages[min(len(stages) - 1, iteration * len(stages) // request.iterations)]
        selected = list(range(dimensions)) if arm == "joint" else columns[stage]
        jacobian = np.empty((len(residual), len(selected)))
        try:
            for column, index in enumerate(selected):
                if time.process_time() - spent > request.cpu_seconds_per_arm:
                    raise TimeoutError("Arm CPU budget")
                perturbed = vector.copy()
                perturbed[index] += 1e-6
                values, _, counts = _residuals(_apply(base, variables, perturbed, np, Rotation), rows, temporal, request, np, Rotation)
                if counts != support:
                    raise ValueError("Fixed residual support changed")
                jacobian[:, column] = (values - residual) / 1e-6
            weights = np.sqrt(normalization / np.maximum(1., np.abs(residual)))
            weighted = jacobian * weights[:, None]
            # Diagnose rank before damping; damping cannot create observation support.
            _, _, rank, _ = lstsq(weighted, -residual * weights, lapack_driver="gelsd")
            if rank < len(selected):
                summary["ambiguity"].append({"iteration": iteration, "rank": int(rank), "columns": len(selected)})
                emit(dict(iteration=iteration, accepted=False, failure="underconstrained_observations",
                          rank=int(rank), columns=len(selected), current_vector=vector.tolist(), variables=listed))
                summary["rejected_steps"] += 1
                stop = "underconstrained_observations"
                break
            matrix = np.vstack([weighted, math.sqrt(damping) * np.eye(len(selected))])
            step = lstsq(matrix, np.r_[-residual * weights, np.zeros(len(selected))], lapack_driver="gelsd")[0]
            trial = vector.copy()
            trial[selected] += step
            _clip(trial, sizes, request, np)
            if np.any(np.abs(trial) > limits + 1e-12):
                raise ValueError("Trial exceeds frozen correction bounds")
            candidate_r, _, counts = _residuals(_apply(base, variables, trial, np, Rotation), rows, temporal, request, np, Rotation)
            next_objective = _objective(candidate_r, normalization, np)
            accepted = next_objective <= objective
            emit(dict(iteration=iteration, stage=stage, accepted=accepted, objective_before=objective,
                      objective_trial=next_objective, vector=trial.tolist(), variables=listed))
            summary["accepted_steps" if accepted else "rejected_steps"] += 1
            if not accepted:
                stop = "increasing_fixed_objective"
                break
            improvement = objective - next_objective
            vector, residual, objective = trial, candidate_r, next_objective
            damping = max(1e-6, damping / 2)
            # A converged sequential stage must not suppress later stages.
            if improvement <= 1e-9 and arm == "joint":
                stop = "objective_converged"
                break
        except TimeoutError:
            stop = "cpu_budget"
            break
        except ValueError as error:
            emit(dict(iteration=iteration, accepted=False, failure=str(error),
                      current_vector=vector.tolist(), variables=listed))
            summary["rejected_steps"] += 1
            stop = "invalid_trial_or_objective"
            break
    summary.update(final_objective=objective, stop=stop)
    return _apply(base, variables, vector, np, Rotation), summary


def _admit(request):
    """Resolve camera/depth/pose/hand/observation producers and check revision, lineage and grids."""
    camera = _parent(request.camera_run, request, ("camera_motion.estimate",))
    depth = _parent(camera[1]["request"]["depth_run"], request, ("depth.generate",))
    require_input(camera, depth)
    pose = _parent(request.pose_run, request, ("object_pose.compose", "object_pose.refine"))
    hand = _parent(request.hand_run, request, ("hand_motion.lift", "hand_motion.refine")) if request.hand_run else None
    if hand and (hand[1].get("units") != "metres" or hand[1].get("joint_order") != "MANO21" or
                 hand[1].get("joint_frame_convention") != "MANO" or
                 hand[1].get("output_parameter_form") != "anatomical_unmirrored"):
        raise ValueError("Joint refinement requires exact metric MANO21 source frames")
    observations = [_parent(spec["run"], request, PRODUCERS) for spec in request.observation_sources]
    parents = [camera, depth, pose, *([hand] if hand else []), *observations]
    revision = same_revision(parents, dataset_revision())
    camera_entry = input_file(ROOT, camera[0])
    for parent in [pose, *([hand] if hand else [])]:
        if camera_entry not in parent[1]["inputs"]:
            raise ValueError("Pose/hand estimate is not bound to the exact frozen camera producer")
        if parent[1].get("grid") != camera[1].get("grid"):
            raise ValueError("Camera/pose/hand source pixel grids differ")
    return camera, depth, pose, hand, observations, parents, revision


def _valid_states(base, motion, objects, hand_data, np):
    """Observed (frame, entity) keys; every admitted transform must be rigid and finite."""
    count = len(base["camera"])
    observed = motion["observed"]
    cameras = np.flatnonzero(motion["valid"] & observed)
    bodies = np.argwhere(objects["world_valid"] & observed[:, None])
    valid = {(i, ("camera",)) for i in cameras.tolist()}
    valid.update((i, ("object", slot)) for i, slot in bodies.tolist())
    joints = np.zeros((0, 3), dtype=np.int64)
    if hand_data is not None:
        joint_observed = hand_data.get("joint_observed", np.zeros((count, 2, 21), dtype=bool))
        if joint_observed.shape != (count, 2, 21) or joint_observed.dtype != np.bool_:
            raise ValueError("Explicit boolean joint observation support required")
        joints = np.argwhere(joint_observed & hand_data["world_valid"].astype(bool)[:, :, None] & observed[:, None, None])
        valid.update((i, ("hand", side, joint)) for i, side, joint in joints.tolist())
    transforms = np.concatenate((base["camera"][cameras], base["object"][bodies[:, 0], bodies[:, 1]]))
    rotations = transforms[:, :3, :3]
    rigid = (np.isclose(transforms[:, 3], [0, 0, 0, 1]).all(axis=1)
             & np.isclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-6).all(axis=(1, 2))
             & np.isclose(np.linalg.det(rotations), 1., atol=1e-6))
    if not rigid.all():
        raise ValueError("Admitted state is not a rigid metre transform")
    if not np.isfinite(transforms).all() or not np.isfinite(base["hand"][tuple(joints.T)]).all():
        raise ValueError("Admitted state must be finite")
    return valid


def _source_state(request, camera, depth, pose, hand, record, revision, np):
    """Load producer arrays on one dense source clock; return base state, observed keys, pts and arrays."""
    motion, objects = arrays(camera, "trajectory.npz", record, np), arrays(pose, "poses.npz", record, np)
    preprocessing = arrays(depth, "preprocessing.npz", record, np)
    if depth[1]["preprocessing_artifact_sha256"] != camera[1]["grid"]:
        raise ValueError("Camera grid must identify the exact calibrated preprocessing artifact")
    intrinsics = preprocessing["input_intrinsics_pixels"]
    pts = objects["pts"]
    if pts.dtype.kind not in "iu" or not np.array_equal(motion["pts"], pts) or pts.ndim != 1 or len(pts) < 2 or np.any(np.diff(pts) <= 0):
        raise ValueError("Source RGB clocks differ or do not increase")
    count = len(pts)
    bodies = objects["world_T_object"].shape[1]
    if (motion["valid"].shape != (count,) or motion["observed"].shape != (count,) or
            objects["world_valid"].shape != (count, bodies) or objects["valid"].shape != (count, bodies) or
            not np.array_equal(objects["world_valid"], objects["valid"] & motion["valid"][:, None])):
        raise ValueError("Object world validity must preserve exact measured object/camera support")
    record.update(grid=camera[1]["grid"], object_names=pose[1]["object_names"],
                  mesh_body_frames=pose[1]["mesh_body_frames"],
                  hand_observation_semantics="observed joint centres only; inferred FK unchanged")
    error = "Require complete dense source-aligned camera/pose clocks at fixed 1/10240; sparse inputs need an adapter"
    clock = clock_row(request, revision, error)
    if (clock["time_base_and_frame_rate"] != [["1/10240", "20/1"]] or
            not np.array_equal(pts, np.asarray(clock["pts_integer"], dtype=np.int64)) or
            not np.array_equal(objects["frame_index"], np.arange(count)) or
            not np.array_equal(motion["frame_index"], np.arange(count))):
        raise ValueError(error)
    hand_data = arrays(hand, "hands.npz", record, np) if hand else None
    if hand_data is not None and (not np.array_equal(hand_data["pts"], pts) or
                                  not np.array_equal(hand_data["frame_index"], np.arange(count))):
        raise ValueError("Hand source RGB clock differs")
    base = {"camera": motion["world_T_camA"].copy(), "object": objects["world_T_object"].copy(),
            "hand": (hand_data["joints_world"] if hand else np.full((count, 2, 21, 3), np.nan)).copy()}
    if (base["camera"].shape != (count, 4, 4) or base["object"].ndim != 4 or
            base["object"].shape[0] != count or base["object"].shape[2:] != (4, 4) or
            base["hand"].shape != (count, 2, 21, 3)):
        raise ValueError("Invalid source state dimensions")
    if motion["valid"].dtype != np.bool_ or motion["observed"].dtype != np.bool_ or objects["world_valid"].dtype != np.bool_:
        raise ValueError("Producer observation/validity masks must be boolean")
    valid = _valid_states(base, motion, objects, hand_data, np)
    return base, valid, pts, motion, objects, intrinsics


def _window_problem(rows, start, end, window, valid, base, pts, request, np, Rotation):
    """Window support rows, free (frame, entity) variables and measured-velocity temporal priors."""
    support = [row for row in rows if start <= row["frame_index"] < end and
               ("other_frame_index" not in row or start <= row["other_frame_index"] < end)]
    if not support:
        raise ValueError("Declared window has no admitted optimization observations")
    involved = set()
    for row in support:
        involved.add((row["frame_index"], row["entity"]))
        if row["family"] in ("reprojection", "depth"):
            involved.add((row["frame_index"], ("camera",)))
        if "other_entity" in row:
            involved.add((row["other_frame_index"], row["other_entity"]))
    involved &= valid
    fixed_until = start + request.overlap_frames if window else start
    fixed_camera = min((frame for frame, entity in involved if entity == ("camera",)), default=None)
    variables = sorted((frame, entity) for frame, entity in involved
                       if frame >= fixed_until and not (entity[0] == "camera" and frame == fixed_camera))
    temporal = []
    for a, entity in sorted(involved):
        if (a + 1, entity) in involved:
            dt = float(pts[a + 1] - pts[a]) / 10240
            temporal.append((a, a + 1, entity, dt, _velocity(base, a, a + 1, entity, dt, request, np, Rotation)))
    return support, variables, temporal


def _ceiling(directory, limit):
    if sum(path.stat().st_size for path in directory.iterdir() if path.is_file()) > limit:
        raise RuntimeError("Generated artifact budget exceeded; preserve failed run")


def _persist(directory, name, state, fixed, record, limit, np):
    # Conservative uncompressed allocation bound before writing; compressed files can be smaller.
    if sum(value.nbytes for value in state.values()) + sum(path.stat().st_size for path in directory.iterdir() if path.is_file()) + 65536 > limit:
        raise RuntimeError("Output allocation would exceed declared generated-byte ceiling")
    save_npz(directory / name, np, **fixed, **state)
    record["artifacts"][name] = sha256_file(directory / name)
    _ceiling(directory, limit)


def _emit(directory, limit, arm, window, step):
    with (directory / "steps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(arm=arm, window=window, **step), allow_nan=False) + "\n")
    _ceiling(directory, limit)


def refine(request):
    """Retain independent/sequential/joint candidates; never promote a winner."""
    runtime = runtime_flag(FEATURE, "refine_authorized", "Joint scene refinement")
    if request.episode_index not in runtime.get("refine_episode_indices", []):
        raise PermissionError("Named training/tuning episode is not authorized for refinement")
    directory = output_path(ROOT, f"outputs/scene_assembly/{request.source_split}/episode_{request.episode_index:06d}/refine-" +
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    import numpy as np
    from scipy.spatial.transform import Rotation

    camera, depth, pose, hand, observations, parents, revision = _admit(request)
    record = run_fields(ROOT, "scene_assembly", "refine", asdict(request), parents=[path for path, _ in parents],
                        inputs=[FEATURE / "runtime.json", "reports/data-audit/videos.json"], allowed=("video_only",))
    record.update(split=request.source_split, episode_index=request.episode_index, dataset_revision=revision,
                  artifacts={}, frames=[], independent_success_established=False)
    base, valid, pts, motion, objects, intrinsics = _source_state(request, camera, depth, pose, hand, record, revision, np)
    rows, rejected = _load_observations(request, observations, record, pts, valid, camera[1]["grid"], intrinsics, np)
    with recorded(directory, record):
        started = time.process_time()
        limit = request.maximum_generated_bytes
        fixed = {"pts": pts, "frame_index": objects["frame_index"], "camera_valid": motion["valid"],
                 "object_valid": objects["world_valid"], "object_names": objects["object_names"]}
        report = {"fit_statistics_only": True, "no_candidate_selected": True, "rejected_observations": rejected,
                  "unsupported": ["silhouette_distance", "MANO_pose_and_shape", "joint_limits", "penetration", "contact_sliding", "temporal_acceleration"],
                  "merge_rule": "Earlier window owns all overlap; overlap and earliest supported camera of each window fixed",
                  "arms": {}, "coverage": {"admitted_records": len(rows), "rejected_records": len(rejected)},
                  "independent_assessment": "Deferred: reserved evidence not consumed by fitting; item02 assessment required"}
        try:
            _persist(directory, "original.npz", base, fixed, record, limit, np)
            for arm in ("independent", "sequential", "joint"):
                state = {name: value.copy() for name, value in base.items()}
                arm_started = time.process_time()
                summaries = []
                for window, start in enumerate(request.window_starts):
                    end = start + request.window_frames
                    if end > len(pts):
                        raise ValueError("Declared window exceeds source clip; no silent truncation")
                    support, variables, temporal = _window_problem(rows, start, end, window, valid, base, pts, request, np, Rotation)
                    pre = {name: value.copy() for name, value in state.items()}
                    _persist(directory, f"{arm}-window{window}-before.npz", pre, fixed, record, limit, np)
                    if arm == "independent":
                        residual, normalization, counts = _residuals(state, support, temporal, request, np, Rotation)
                        score = _objective(residual, normalization, np)
                        summary = {"initial_objective": score, "final_objective": score,
                                   "support_scalar_counts": counts, "stop": "unchanged_independent"}
                    else:
                        emit = partial(_emit, directory, limit, arm, window)
                        state, summary = _solve(pre, variables, support, temporal, request, arm, arm_started, emit, np, Rotation)
                    _persist(directory, f"{arm}-window{window}-after.npz", state, fixed, record, limit, np)
                    summary.update(start=start, end=end, fixed_overlap_frames=request.overlap_frames if window else 0)
                    summaries.append(summary)
                    if summary["stop"] in STOPPING:
                        break
                _persist(directory, f"{arm}.npz", state, fixed, record, limit, np)
                report["arms"][arm] = {"windows": summaries, "cpu_seconds": time.process_time() - arm_started}
            report["cpu_seconds_total"] = time.process_time() - started
        finally:
            write_json(directory / "refinement.json", report)
        _ceiling(directory, limit)
    return directory / "run.json"
