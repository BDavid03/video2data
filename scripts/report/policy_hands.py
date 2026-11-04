"""CPU diagnostic of a saved FlashSAC policy; never scores or exports a submission.

Replay with the installed FlashCHORD interpreter; render saved arrays with the
project interpreter. See docs/report/nontechnical/README.md for exact commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FC = ROOT / "data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord"
DEFAULT_OUTPUT = ROOT / "data/outputs/visuals/policy-hands/m0c"
DEFAULT_CHECKPOINT = ROOT / "data/outputs/dynamic_feasibility/m0c/trials/train/training/policy_66763776.safetensors"
DEFAULT_OFFICIAL = ROOT / "data/outputs/dynamic_feasibility/m0c/scoring/runs/steps-66763776/attempt0/trajectory.parquet"
TEMPLATE = ROOT / ("data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/"
                   "robotic_grounding/assets/human_motion_data/ego_recon/processed/"
                   "sequence_id=tissue_box_simple/robot_name=sharpa_wave")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def clustered_mesh(vertices, faces, spacing=.0015):
    """Pure vertex clustering for presentation, retaining mesh surfaces at 1.5 mm."""
    import numpy as np
    _, inverse = np.unique(np.round(vertices / spacing).astype(int), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    points = np.column_stack([np.bincount(inverse, weights=vertices[:, a]) / count for a in range(3)])
    triangles = inverse[faces]
    keep = ((triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
            & (triangles[:, 0] != triangles[:, 2]))
    return points, triangles[keep]


def replay(args):
    """Orchestrate upstream composition, inference and physics without editing them."""
    output = args.output.resolve()
    output.relative_to(ROOT / "data/outputs/visuals")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", WANDB_MODE="disabled",
                      PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for key, name in {"WARP_CACHE_PATH": "warp", "FLASH_CHORD_CACHE_DIR": "flash-chord",
                      "JAX_COMPILATION_CACHE_DIR": "jax", "XDG_CACHE_HOME": "xdg",
                      "MPLCONFIGDIR": "matplotlib", "TMPDIR": "tmp"}.items():
        path = output / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.path.insert(0, str(FC / "src"))
    import warp as wp
    wp.config.enable_cuda = False
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    wp.set_device("cpu")
    import numpy as np
    import jax
    import jax.numpy as jnp
    import newton
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    from flash_chord.evaluation import harness as h
    from flash_chord.evaluation.rollout import reference_object_pose
    from flash_chord.runtime.contact import read_contacts_w
    from flash_chord.assets import registry

    task = args.task or next(TEMPLATE.glob("*.parquet"))
    task = task.resolve()
    forbidden_episode = re.search(r"episode_0*(2|13|22|24|27|41|43|44)(?:\D|$)", str(task))
    if forbidden_episode or any(token in str(task).lower() for token in ("scan", "/mnt/z")):
        raise ValueError("Disallowed diagnostic input")
    row = pq.ParquetFile(task).read().to_pylist()[0]
    # Assets in historical checkpoints were moved by the owner. Resolve only
    # missing paths to the same named files, never rewrite the input parquet.
    relocation = {}
    for field in ("object_mesh_paths", "object_urdf_paths"):
        for value in row[field]:
            if "scan" in value.lower():
                raise ValueError("A diagnostic cannot read a scan")
            if not Path(value).is_file():
                candidate = task.parents[2] / Path(value).name
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                relocation[value] = str(candidate)
    resolve = registry.resolve_asset_path
    registry.resolve_asset_path = lambda path, **kw: relocation[path] if path in relocation else resolve(path, **kw)

    metadata = h.read_checkpoint_metadata(args.checkpoint)
    config = h.checkpoint_config(metadata)
    if h.checkpoint_training_algorithm(config) != "flash_sac":
        raise ValueError("This diagnostic supports FlashSAC checkpoints")
    original_task = config["task"]["parquet"]
    h.apply_evaluation_composition(config, world_count=1, reset_mode="explicit")
    config["task"]["parquet"] = str(task)
    config["training"]["device"] = "cpu"
    cfg = OmegaConf.create(config)
    env_config = h.instantiate_typed(cfg.env, h.RLEnvConfig)
    training = h.instantiate_typed(cfg.training, h.FlashSACTrainingConfig)
    started = time.monotonic()
    write(output / "status.json", {"status": "running", "pid": os.getpid(), "diagnostic_only": True})
    setup = h.setup_scene(
        parquet=str(task), control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed,
        embodiment=h.instantiate_typed(cfg.embodiment, h.Embodiment),
        collision=h.instantiate_typed(cfg.collision, h.CollisionPolicy), world_count=1,
        include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
        **{key: cfg.scene[key] for key in ("object_scale_min", "object_scale_max", "object_scale_seed",
                                           "contact_friction", "object_free_joint_damping") if key in cfg.scene},
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)))
    env = h.RLEnv(setup.scene, setup.reference, config=env_config, device="cpu")
    env.reset()
    # CUDA graph capture is a GPU optimization; the CPU uses the same env.step.
    jax_env = h.WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    if training.curriculum is not None:
        stage = training.curriculum.stages[int(metadata["curriculum_stage"])]
        jax_env.set_curriculum(h._evaluation_curriculum_stage(stage, "explicit"))
    env.reset(frame_id=0)
    observation = jax.device_put(jnp.array(env.observation.numpy().reshape(1, env.observation_dim), copy=True), jax.devices("cpu")[0])
    actor, policy_action, key, _ = h.load_flash_sac_policy(
        str(args.checkpoint), training, observation, env.action.action_dim,
        deterministic=True, device=jax.devices("cpu")[0])
    model = env.model
    labels = [str(label) for label in model.body_label]
    object_ids = np.asarray(env.command.layout.body_ids, dtype=int)
    initial = env.state_0.body_q.numpy().copy()
    geometry = {}
    shape_bodies, shape_types = model.shape_body.numpy(), model.shape_type.numpy()
    scales, transforms = model.shape_scale.numpy(), model.shape_transform.numpy()
    mesh_info = []
    for shape, flag in enumerate(model.shape_flags.numpy()):
        body = int(shape_bodies[shape])
        source = model.shape_source[shape]
        if (body < 0 or not int(flag) & int(newton.ShapeFlags.VISIBLE)
                or int(shape_types[shape]) != int(newton.GeoType.MESH) or source is None):
            continue
        vertices = np.asarray(source.vertices, float) * scales[shape]
        vertices = Rotation.from_quat(transforms[shape, 3:]).apply(vertices) + transforms[shape, :3]
        faces = np.asarray(source.indices).reshape(-1, 3)
        original_faces = len(faces)
        vertices, faces = clustered_mesh(vertices, faces)
        index = len(mesh_info)
        geometry[f"vertices_{index}"] = vertices.astype(np.float32)
        geometry[f"faces_{index}"] = faces.astype(np.int32)
        mesh_info.append(dict(body=body, label=labels[body], original_faces=original_faces,
                              drawn_faces=len(faces), object=bool(body in object_ids)))
    np.savez_compressed(output / "geometry.npz", **geometry)
    poses, contacts, done = [], [], []
    for step in range(setup.reference.num_frames):
        action, key = policy_action(actor, observation, key)
        observation = jax_env.step(action).observation
        poses.append(env.state_0.body_q.numpy().copy())
        readout = read_contacts_w(model, env.state_0, env.contacts)
        pair = np.column_stack([readout.body0_ids, readout.body1_ids])
        touches = np.isin(pair, object_ids).any(axis=1)
        names = [labels[b] if b >= 0 else "world" for b in np.where(np.isin(pair[:, 0], object_ids), pair[:, 1], pair[:, 0])]
        selected = np.flatnonzero(touches)
        contacts.append({"step": step, "object_contacts": len(selected),
                         "pairs": [{"other_body": names[i], "position_m": readout.contact_pos_w[i].tolist(),
                                    "force_N": readout.contact_force_w[i].tolist()} for i in selected]})
        done.append(bool(env.done.numpy()[0] or env.truncation.numpy()[0]))
        if step % 50 == 0:
            print(f"CPU replay {step + 1}/{setup.reference.num_frames}", flush=True)
    poses = np.asarray(poses)
    reference = reference_object_pose(setup.reference, 0, len(poses))
    np.savez_compressed(output / "replay.npz", body_pose_xyzw=poses, initial_body_pose_xyzw=initial,
                        object_body_ids=object_ids, reference_pose_xyzw=reference,
                        time_s=np.arange(len(poses)) / env_config.sim.fps, done=done)
    write(output / "contacts.json", contacts)
    comparison = None
    if args.official:
        official = pq.read_table(args.official).sort_by([("step", "ascending"), ("object_body_id", "ascending")])
        official_hash = (pq.read_schema(args.official).metadata or {}).get(b"flash_chord.checkpoint_sha256", b"").decode()
        if official_hash != digest(args.checkpoint):
            raise ValueError("Official export is from a different checkpoint")
        if (set(official["environment_id"].to_pylist()) != {0}
                or not np.array_equal(official["step"].to_numpy(), np.repeat(np.arange(len(poses)), len(object_ids)))):
            raise ValueError("Official export does not match the replay's one-world frame clock")
        official_reference = np.column_stack([official[f"reference_position_{a}"].to_numpy() for a in "xyz"])
        if not np.allclose(official_reference.reshape(reference[..., :3].shape), reference[..., :3], atol=1e-6, rtol=0):
            raise ValueError("Official export and replay have different references")
        xyz = np.column_stack([official[f"achieved_position_{a}"].to_numpy() for a in "xyz"]).reshape(len(poses), len(object_ids), 3)
        difference = np.linalg.norm(poses[:, object_ids, :3] - xyz, axis=-1)
        comparison = dict(mean_m=float(difference.mean()), max_m=float(difference.max()),
                          close_within_1mm=bool(difference.max() <= .001),
                          definition="Unaligned Euclidean position difference, same post-control-step samples")
    inputs = [args.checkpoint, task, Path(__file__), FC / "src/flash_chord/evaluation/harness.py",
              FC / "src/flash_chord/evaluation/rollout.py"] + ([args.official] if args.official else [])
    summary = dict(diagnostic_only=True, scored=False, packed=False, device="cpu", worlds=1,
                   reset="explicit frame 0", assistance=False, checkpoint=str(args.checkpoint),
                   original_task=original_task, task=str(task), relocation=relocation,
                   frames=len(poses), control_fps=env_config.sim.fps, elapsed_s=time.monotonic()-started,
                   body_labels=labels, meshes=mesh_info, geometry="Sharpa visual meshes, vertex-clustered at 1.5 mm",
                   official_comparison=comparison, completion_first_step=next((i for i,v in enumerate(done) if v), None),
                   contact_caveat="Raw solver contact points/forces at final physics substep; counts are not grasp quality.",
                   inputs={str(p.relative_to(ROOT)): digest(p) for p in inputs})
    write(output / "summary.json", summary)
    write(output / "status.json", {"status": "completed", "pid": os.getpid(), "diagnostic_only": True})
    print(json.dumps(comparison), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL,
                        help="Matching official export; use --no-official for a new policy")
    parser.add_argument("--no-official", action="store_true")
    args = parser.parse_args()
    if args.no_official:
        args.official = None
    args.checkpoint = args.checkpoint.resolve()
    if args.official:
        args.official = args.official.resolve()
    try:
        replay(args)
    except Exception as error:
        if args.output.resolve().is_relative_to(ROOT / "data/outputs/visuals") and args.output.is_dir():
            write(args.output / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
