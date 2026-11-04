"""Train through upstream FlashSAC and export through the unchanged official evaluator."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
from src.gpu import GpuLease
from src.paths import output_path
from src.provenance import run_fields, write_json, sha256_file
from src.runs import (ROOT, runtime_flag, authorize_episode, heavy_execution, new_run_id,
                         producer, artifact, recorded)
from .scene import FEATURE, FC, KIT, build


def runtime_environment(runtime):
    heavy_execution(runtime)
    locations = {"WARP_CACHE_PATH":"data/.cache/warp", "JAX_COMPILATION_CACHE_DIR":"data/.cache/jax",
                 "TMPDIR":"data/.runtime/tmp/ctx03", "TEMP":"data/.runtime/tmp/ctx03", "TMP":"data/.runtime/tmp/ctx03"}
    for key, value in locations.items():
        path = output_path(ROOT, value)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(FC / "src")]),
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="true",
                      JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0", WANDB_MODE="disabled",
                      HYDRA_FULL_ERROR="1", PYTHONUNBUFFERED="1")
    return os.environ.copy()


def execute(command, log, env, record):
    record.setdefault("commands", []).append(command)
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def training_command(request, task, directory):
    budget = request.budget
    overrides = dict(
        **{"task.parquet":str(task), "task.motion_speed":1., "task.source_frame_playback":False,
           "task.motion_start_frame":0,"task.motion_end_frame":-1,"sim.fps":20.,
           "scene.world_count":budget["worlds"], "scene.support":False,"scene.decompose_objects":False,
           "training.total_environment_steps":budget["environment_steps"],
           "training.updates_per_collection":1., "training.replay.minimum_size":budget["replay_minimum"],
           "training.replay.capacity":64,"training.replay.batch_size":8,"training.replay.n_step":3,
           "training.network.actor_hidden_dim":32,"training.network.critic_hidden_dim":64,
           "training.network.actor_blocks":1,"training.network.critic_blocks":1,
           "training.network.expansion":2,"training.mixed_precision":False,"training.compute_dtype":"float32",
           "training.checkpoint.save_interval_environment_steps":budget["environment_steps"],
           "training.checkpoint.save_replay":False,"logging.mode":"disabled","logging.log_interval":250,
           "reset.always_reset_to_first_frame":True,"voc_scale_override":0.,
           "output_dir":str(directory / "training"), "hydra.run.dir":str(directory / "train_hydra")})
    return [str(ROOT / "data/.runtime/flashchord/venv/bin/python"), str(FC / "scripts/train_flash_sac.py"),
            *[f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k,v in overrides.items()]]


def inspect_export(path, task_record):
    import numpy as np
    import pyarrow.parquet as pq
    from ._scene import pose_statistics, initial_pose_error
    df = pq.read_table(path).to_pylist()
    df.sort(key=lambda r:(r["reference_frame"], r["environment_id"], r["object_body_id"]))
    columns = [f"achieved_position_{s}" for s in "xyz"] + [f"achieved_quaternion_{s}" for s in "xyzw"]
    names = task_record["object_names"]
    expected = {(0, f, b) for f in task_record["roster_frames"] for b in range(len(names))}
    actual = {(r["environment_id"], r["reference_frame"], r["object_body_id"]) for r in df}
    if len(actual) != len(df):
        raise ValueError("Duplicate exported pose keys")
    if not expected.issubset(actual) or not all(r["environment_valid"] for r in df):
        raise ValueError("Export lacks roster keys or contains invalid environments")
    if any(r["object_body_name"] != names[r["object_body_id"]] for r in df):
        raise ValueError("Official slot order mismatch")
    poses = np.array([[r[c] for c in columns] for r in df]).reshape(-1, 1, len(names), 7)
    stats = pose_statistics(poses)
    if not stats["finite"] or not stats["unit_quaternions"]:
        raise ValueError("Nonfinite pose or nonunit quaternion")
    cm, deg = stats["drift_cm"], stats["drift_deg"]
    if cm > 1. or deg > 2.:
        raise ValueError(f"Static drift exceeds 1 cm / 2 degrees: {cm}, {deg}")
    intended = np.array([row["position"] + [*row["quaternion_wxyz"][1:], row["quaternion_wxyz"][0]]
                         for row in task_record["scene"]])
    first_cm, first_deg = initial_pose_error(poses[0, 0], intended)
    if first_cm > 1. or first_deg > 2.:
        raise ValueError(f"First pose differs from intended by more than 1 cm / 2 degrees: {first_cm}, {first_deg}")
    return {"drift_cm":cm,"drift_deg":deg,"roster_keys":len(expected),"covered_keys":len(expected & actual),
            "first_pose_cm":first_cm,"first_pose_deg":first_deg,
            "roster_frames":len(task_record["roster_frames"]),"export_frames":len(df)//len(names),"names":names}


def run(request):
    runtime = runtime_flag(FEATURE, "run_authorized", "Insurance policy run")
    authorize_episode(runtime, request, "insurance policy run")
    env = runtime_environment(runtime)
    task_run = build(request)
    # build() resets common cache variables; restore this slice's narrower temporary directory.
    env = runtime_environment(runtime)
    task = producer(task_run, "pose_insurance.build", request)
    record = run_fields(ROOT, "pose_insurance", "run", asdict(request), parents=[task_run],
                        inputs=[FEATURE / "runtime.json", "data/outputs/pose_insurance/runtime_inventory.json", KIT / "flash_chord/scripts/evaluate_policy.py"])
    record.update(split=request.source_split, episode_index=request.episode_index,
                  dataset_revision=task[1]["dataset_revision"], scene=task[1]["scene"])
    version = "v2" if task[1].get("placement_candidate") == "K1a" else "v1"
    directory = output_path(ROOT, f"data/outputs/pose_insurance/run-{version}/{new_run_id()}/episode_{request.episode_index:06d}")
    with recorded(directory, record):
        task_path = artifact(task, "task.parquet", record)
        # The main checkout and this worktree share a GPU; do not enter alongside its owner.
        main_lock = ROOT.parents[2] / "data/.runtime/gpu-stage.lock"
        if main_lock.exists():
            raise RuntimeError(f"Main-checkout GPU lease exists: {main_lock}")
        started = time.perf_counter()
        with GpuLease(ROOT, 2048, 1024, stage="pose_insurance") as lease:
            try:
                execute(training_command(request, task_path, directory), directory / "training.log", env, record)
                steps = request.budget["environment_steps"]
                checkpoint = directory / f"training/policy_{steps}.safetensors"
                state = directory / f"training/state_{steps}.safetensors"
                from safetensors import safe_open
                import numpy as np
                with safe_open(str(checkpoint), framework="numpy") as policy:
                    actor_leaf_count = len(json.loads(policy.metadata()["manifest"]))
                with safe_open(str(state), framework="numpy") as f:
                    manifest = json.loads(f.metadata()["manifest"])
                    updates = int(f.get_tensor(manifest[-1]["key"]).item())
                    # ActorInferenceState has params + batch_stats; the next state leaf is Adam count.
                    actor_steps = f.get_tensor(manifest[actor_leaf_count]["key"])
                    if actor_steps.size != 1 or actor_steps.dtype.kind != "i":
                        raise ValueError("Unexpected actor Adam-state layout")
                    actor_steps = int(actor_steps.item())
                    if not all(np.isfinite(f.get_tensor(key)).all() for key in f.keys()):
                        raise ValueError("Training state contains nonfinite tensors")
                record.update(optimizer_updates=updates, actor_adam_steps=actor_steps)
                if not request.contract_only and (updates < 1 or actor_steps < 1):
                    raise ValueError("Learned trajectory requires actual optimizer updates")
                if request.contract_only and updates != 0:
                    raise ValueError("Contract-only budget unexpectedly learned")
                command = [str(ROOT / "data/.runtime/flashchord/venv/bin/python"),
                           str(KIT / "flash_chord/scripts/evaluate_policy.py"),
                           f"evaluation.checkpoint={checkpoint}","evaluation.world_count=1",
                           "evaluation.reset_mode=explicit","evaluation.start_frame=0",
                           f"evaluation.object_trajectories_output={directory / 'trajectory.parquet'}",
                           f"evaluation.metrics_output={directory / 'official_metrics.json'}",
                           f"hydra.run.dir={directory / 'evaluate_hydra'}","logging.mode=disabled"]
                execute(command, directory / "evaluation.log", env, record)
                lease.require_headroom()
            finally:
                record["gpu"] = lease.report()
        record["policy_seconds"] = time.perf_counter() - started
        record["checkpoint"] = str(checkpoint.relative_to(ROOT))
        record["checkpoint_sha256"] = sha256_file(checkpoint)
        record["checks"] = inspect_export(directory / "trajectory.parquet", task[1])
        write_json(directory / "result.json", {k:record[k] for k in ("optimizer_updates","actor_adam_steps","checks","policy_seconds","checkpoint","gpu")})
    return directory / "run.json"
