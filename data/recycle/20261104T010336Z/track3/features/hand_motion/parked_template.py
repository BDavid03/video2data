"""A model-derived neutral template, independent of any episode or observation."""

from dataclasses import asdict, dataclass

from track3.paths import output_path
from track3.provenance import run_fields, sha256_file, write_json
from ._common import FEATURE, ROOT, new_run_id, read_json, recorded, require_rotations, runtime_flag, save_npz
from ._mano import ManoKinematics, load_runtime


@dataclass(frozen=True)
class ParkedTemplateRequest:
    method: str = "mano_flat_rest"

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))

    def __post_init__(self):
        if self.method != "mano_flat_rest":
            raise ValueError("Only the licensed MANO flat-rest template is implemented")


def _align(joints, rotations, surface, side, np):
    if (joints.shape != (21, 3) or rotations.shape != (21, 3, 3) or surface.shape != (778, 3) or
            not all(np.isfinite(value).all() for value in (joints, rotations, surface))):
        raise ValueError("Malformed MANO template geometry")
    require_rotations(rotations, np)
    wrist = joints[0].copy()
    points = joints - wrist
    # MANO21 indices: wrist=0, index MCP=5, middle MCP=9, little MCP=17.
    # Set the distal direction to +x. The transverse direction is radial for
    # the right hand and ulnar for the left, preserving chirality with SO(3).
    distal = points[9]
    length = float(np.linalg.norm(distal))
    if not 0.02 < length < 0.2:
        raise ValueError("MANO template palm length is incompatible with metres")
    distal = distal / length
    transverse = points[5] - points[17]
    transverse = transverse - np.dot(transverse, distal) * distal
    width = float(np.linalg.norm(transverse))
    if not 0.01 < width < 0.2:
        raise ValueError("MANO template palm basis is degenerate or not metric")
    transverse = transverse / width * (1 if side == "right" else -1)
    dorsal = np.cross(distal, transverse)
    world_from_model = np.stack((distal, transverse, dorsal))
    if not np.isclose(np.linalg.det(world_from_model), 1, atol=1e-6, rtol=0):
        raise ValueError("Template alignment must be a rotation, never a reflection")
    aligned_joints = points @ world_from_model.T
    aligned_surface = (surface - wrist) @ world_from_model.T
    aligned_rotations = world_from_model @ rotations
    radius = float(np.max(np.linalg.norm(aligned_surface, axis=1)))
    if not 0.05 < radius < 0.4:
        raise ValueError("MANO template surface radius is incompatible with metres")
    return aligned_joints, aligned_rotations, aligned_surface, world_from_model, wrist


def parked_template(request: ParkedTemplateRequest):
    runtime_flag(FEATURE, "parked_template_authorized", "Parked-hand template generation")
    record = run_fields(ROOT, "hand_motion", "parked_template", asdict(request),
                        inputs=[FEATURE / "runtime.json", FEATURE / "backends.json"])
    run_id = new_run_id()
    record.update(run_id=run_id, scope="model_asset_template", gpu=None, artifacts={},
                  hand_model="MANO", units="metres", joint_order="MANO21",
                  joint_frame_convention="anatomical_unmirrored", observed=False)
    with recorded(output_path(ROOT, f"outputs/hand_motion/templates/{run_id}"), record) as (directory, record):
        runtime = load_runtime(record)
        import numpy as np

        raw, aligned = [], []
        zeros3, fingers, betas = np.zeros((1, 3)), np.zeros((1, 45)), np.zeros(10)
        for side in ("left", "right"):
            joints, rotations, surface = ManoKinematics(runtime, side).forward(zeros3, fingers, betas, zeros3, np)
            raw.append((joints[0], rotations[0], surface[0]))
            aligned.append(_align(joints[0], rotations[0], surface[0], side, np))
        raw_joints, raw_rotations, raw_surfaces = (np.asarray(value) for value in zip(*raw))
        joint_rows, rotation_rows, surfaces, alignments, wrists = (np.asarray(value) for value in zip(*aligned))
        save_npz(directory / "model_rest.npz", np, sides=np.asarray(["left", "right"]),
                 joints_model=raw_joints, joint_rotations_model=raw_rotations,
                 surface_points_model=raw_surfaces, world_from_model_rotation=alignments,
                 model_wrist_m=wrists, betas=np.zeros((2, 10)), pose_coeffs=np.zeros((2, 48)))
        record["artifacts"]["model_rest.npz"] = sha256_file(directory / "model_rest.npz")
        document = {"schema_version": 1, "source": {"method": request.method, "backend": record["backend"],
                     "model_rest_sha256": record["artifacts"]["model_rest.npz"],
                     "shape": "zero_betas", "pose": "zero_axis_angle_flat_hand_mean"},
                    "scope": "model_asset_template", "observed": False, "units": "metres",
                    "joint_order": "MANO21", "joint_frame_convention": "anatomical_unmirrored",
                    "sides": ["left", "right"], "palms": "down",
                    "orientation_rule": "Distal +x; right radial/left ulnar +y; dorsal +z; no reflection",
                    "joints_relative_to_wrist_m": joint_rows.tolist(),
                    "joint_rotations_world": rotation_rows.tolist(),
                    "surface_points_relative_to_wrist_m": surfaces.tolist(),
                    "world_from_model_rotation": alignments.tolist(),
                    "provenance": record["provenance"]}
        write_json(directory / "parked_hand.json", document)
        record["artifacts"]["parked_hand.json"] = sha256_file(directory / "parked_hand.json")
    return directory / "run.json"
