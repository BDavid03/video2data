"""Own task admission, upstream subprocesses, GPU lease and immutable trial records."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from track3.gpu import GpuLease
from track3.paths import output_path
from track3.provenance import run_fields, sha256_file, write_json
from track3.runs import ROOT, heavy_execution, recorded, runtime_flag

FEATURE = Path(__file__).parent
OUTPUT = ROOT / "outputs/dynamic_feasibility"
FC = ROOT / ".runtime/flashchord/video_to_data/robotic_grounding/flash_chord"
PYTHON = ROOT / ".runtime/flashchord/venv/bin/python"
EVALUATOR = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py"
KIT_METRIC = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/AUC.py"


def environment():
    """Set the existing runtime and all writable caches before heavy imports."""
    runtime = runtime_flag(FEATURE, "run_authorized", "Dynamic feasibility")
    heavy_execution(runtime)
    folders = {"WARP_CACHE_PATH": ".cache/warp", "JAX_COMPILATION_CACHE_DIR": ".cache/jax",
               "FLASH_CHORD_CACHE_DIR": ".cache/flash-chord"}
    for key, folder in folders.items():
        path = output_path(ROOT, folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(FC / "src")]),
                      PYTHONDONTWRITEBYTECODE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      JAX_ENABLE_COMPILATION_CACHE="true", JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
                      WANDB_MODE="disabled", HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1")
    return os.environ.copy()


def entry(path):
    """Hash a lexical project input, including read-only runtime symlinks."""
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def load_task(queue):
    """Read the one fixed example and its own mesh, without opening challenge data."""
    import numpy as np
    import pyarrow.parquet as pq
    import trimesh
    from flash_chord.assets.registry import resolve_asset_path, support_usda_for_parquet

    paths = list((ROOT / queue["task_directory"]).glob("*.parquet"))
    if len(paths) != 1 or "sequence_id=tissue_box_simple" not in str(paths[0]):
        raise ValueError("Expected the single shipped tissue-box task")
    task_path = paths[0]
    task = pq.read_table(task_path).to_pylist()[0]
    positions, quaternions = (np.asarray(task[key], dtype=np.float64)
                              for key in ("object_body_position", "object_body_wxyz"))
    reference = np.concatenate([positions, quaternions[..., 1:], quaternions[..., :1]], axis=-1)[:, None]
    if task["object_body_names"] != ["tissue_box_simple"] or len(reference) != 390:
        raise ValueError("Unexpected example task identity or frame count")
    mesh_path = Path(resolve_asset_path(task["object_mesh_paths"][0], reference_path=task_path))
    urdf_path = Path(resolve_asset_path(task["object_urdf_paths"][0], reference_path=task_path))
    support_path = support_usda_for_parquet(str(task_path), robot_name=task["robot_name"])
    if support_path is None:
        raise ValueError("The unchanged task requires its shipped support stage")
    mesh = trimesh.load(mesh_path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, queue["surface_point_count"], seed=queue["surface_seed"])
    inventory = [task_path, mesh_path, urdf_path, support_path, EVALUATOR, KIT_METRIC,
                 FC / "scripts/train_flash_sac.py"]
    for node in ET.parse(urdf_path).iter("mesh"):
        inventory.append(urdf_path.parent / node.attrib["filename"])
    inventory.extend(sorted((FC / "src/flash_chord/configs").rglob("*.yaml")))
    inventory.extend(sorted((FC / "src/flash_chord/evaluation").glob("*.py")))
    # Asset resolution returns physical paths; retain their lexical runtime prefix in the record.
    runtime_root = (ROOT / ".runtime/flashchord/video_to_data").resolve()
    inventory = [ROOT / ".runtime/flashchord/video_to_data" / p.relative_to(runtime_root)
                 if p.is_relative_to(runtime_root) else p for p in inventory]
    return {"path": task_path, "fps": float(task["fps"]), "reference": reference,
            "vertices": np.asarray(points)[None],
            "inputs": [entry(path) for path in dict.fromkeys(inventory)]}


def prepare_arrays(task):
    """Persist the admitted reference and seeded mesh samples for CPU measurement."""
    import numpy as np
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "task.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["reference"], task["reference"]) or not np.array_equal(saved["vertices"], task["vertices"]):
                raise ValueError("Prepared task arrays changed")
    else:
        with path.open("xb") as stream:
            np.savez_compressed(stream, reference=task["reference"], vertices=task["vertices"], fps=task["fps"])


def measure(directory=None):
    """Run kit measurement on CPU in the existing perception environment."""
    runtime = runtime_flag(FEATURE, "measure_authorized", "Dynamic feasibility measurement")
    target = directory / "metrics.json" if directory else OUTPUT / "comparator_S.json"
    command = [str(ROOT / runtime["measurement_python"]), "-B", "-m",
               "track3.features.dynamic_feasibility.measure", "--task", str(OUTPUT / "task.npz"),
               "--out", str(target)]
    if directory:
        command += ["--directory", str(directory)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with target.with_suffix(".log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return json.loads(target.read_text())


def training_command(request, task, directory):
    """Apply the frozen budget to the released recipe without changing task data."""
    b = request.budget
    values = {"task.parquet": str(task["path"]), "task.motion_speed": 1.,
              "task.source_frame_playback": False, "task.motion_start_frame": 0, "task.motion_end_frame": -1,
              "sim.fps": task["fps"], "scene.world_count": b["world_count"], "reset.seed": request.seed,
              "training.total_environment_steps": b["total_environment_steps"],
              "training.updates_per_collection": b["updates_per_collection"],
              "training.replay.minimum_size": b["minimum_size"], "training.replay.capacity": b["capacity"],
              "training.replay.observation_storage_dtype": b["observation_storage_dtype"],
              "training.checkpoint.save_interval_environment_steps": b["checkpoint_interval"],
              "training.checkpoint.save_replay": b["save_replay"], "logging.mode": "disabled",
              "output_dir": str(directory / "training"), "hydra.run.dir": str(directory / "train_hydra")}
    if "curriculum_thresholds" in b:
        values["curriculum.thresholds"] = b["curriculum_thresholds"]
    overrides = [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in values.items()]
    return [str(PYTHON), "-B", "-m", "track3.features.dynamic_feasibility._worker",
            str(FC / "scripts/train_flash_sac.py"), *overrides]


def execute(command, log, env, deadline, record, label):
    """Run an owned child that exits cooperatively at its deadline; never send a signal."""
    env = {**env, "DYNAMIC_FEASIBILITY_DEADLINE": str(deadline)}
    record.setdefault("commands", []).append(command)
    started = time.monotonic()
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record[f"{label}_pid"] = child.pid
        write_json(log.parent / "run.json", record)
        code = child.wait()
    record[f"{label}_seconds"] = time.monotonic() - started
    if code:
        raise RuntimeError(f"{label} exited {code}; see {log.relative_to(ROOT)}")


def checkpoint_counts(checkpoint, state, arm):
    """Read actual optimizer counters and reject nonfinite training state."""
    import numpy as np
    from safetensors import safe_open
    from flash_chord.training.checkpoint import read_checkpoint_metadata

    with safe_open(str(checkpoint), framework="numpy") as policy:
        actor_leaves = len(json.loads(policy.metadata()["manifest"]))
        metadata = read_checkpoint_metadata(checkpoint)
    with safe_open(str(state), framework="numpy") as saved:
        manifest = json.loads(saved.metadata()["manifest"])
        updates = int(saved.get_tensor(manifest[-1]["key"]).item())
        actor_updates = int(saved.get_tensor(manifest[actor_leaves]["key"]).item())
        finite = all(np.isfinite(saved.get_tensor(key)).all() for key in saved.keys())
    if not finite or (arm == "Z" and updates != 0) or (arm != "Z" and min(updates, actor_updates) < 1):
        raise ValueError(f"Invalid {arm} training state: finite={finite}, updates={updates}/{actor_updates}")
    if arm != "Z" and float(metadata["voc_scale"]) != 0.:
        raise ValueError("Learned arm did not finish with zero object assistance")
    return {"learner_updates": updates, "actor_adam_steps": actor_updates,
            "checkpoint_metadata": metadata}


def run(request, queue, task, campaign_identity):
    """Train, evaluate and record one fixed arm/seed, with no retry or episode access."""
    env = environment()
    directory = OUTPUT / "runs" / f"{request.arm}-seed{request.seed}"
    record = run_fields(ROOT, "dynamic_feasibility", "run", asdict(request),
                        inputs=[FEATURE / "queue.json", FEATURE / "runtime.json", ROOT / queue["protocol"]],
                        provenance="public_asset_diagnostic")
    record.update(campaign_identity=campaign_identity, pid=os.getpid(), task_inputs=task["inputs"],
                  diagnostic_only=True, challenge_episode_access=False, fps=task["fps"])
    with recorded(directory, record):
        main_root = next(parent for parent in ROOT.parents if (parent / ".git").is_dir())
        if main_root != ROOT and (main_root / ".runtime/gpu-stage.lock").exists():
            raise RuntimeError("Main-checkout GPU lease already exists")
        started = time.monotonic()
        deadline = time.time() + request.maximum_seconds
        with GpuLease(ROOT, queue["headroom_mib"], queue["admission_allowance_mib"],
                      stage="dynamic_feasibility") as lease:
            record["gpu_started_epoch"] = time.time()
            write_json(directory / "run.json", record)
            try:
                execute(training_command(request, task, directory), directory / "training.log", env,
                        deadline - queue["evaluation_reserve_seconds"], record, "training")
                steps = request.budget["total_environment_steps"]
                checkpoint = directory / f"training/policy_{steps}.safetensors"
                state = directory / f"training/state_{steps}.safetensors"
                record.update(checkpoint_counts(checkpoint, state, request.arm))
                command = [str(PYTHON), "-B", "-m", "track3.features.dynamic_feasibility._worker", str(EVALUATOR),
                           f"evaluation.checkpoint={checkpoint}", "evaluation.world_count=1",
                           "evaluation.reset_mode=explicit", "evaluation.start_frame=0",
                           f"evaluation.object_trajectories_output={directory / 'trajectory.parquet'}",
                           f"evaluation.metrics_output={directory / 'official_metrics.json'}",
                           f"hydra.run.dir={directory / 'evaluate_hydra'}", "logging.mode=disabled"]
                execute(command, directory / "evaluation.log", env, deadline, record, "evaluation")
                lease.require_headroom()
            finally:
                record.update(gpu=lease.report(), gpu_seconds=time.monotonic() - started)
        record["metrics"] = measure(directory)
        write_json(directory / "metrics.json", record["metrics"])
    return directory / "run.json"
