#!/usr/bin/env python3
"""Run the exact kit evaluator serially, or re-pack the exact stored CD-O meshes."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import uuid


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_copies(root: Path) -> None:
    files = read(root / "copy-hashes.json")
    failed = [name for name, row in files.items() if sha(root / name) != row["sha256"]]
    if failed:
        raise ValueError(f"staged hash mismatches: {failed}")
    print(f"PASS all {len(files)} copied files match their source SHA-256", flush=True)
    if (root / "generated-hashes.json").exists():
        generated = read(root / "generated-hashes.json")
        bad = [name for name, expected in generated.items() if sha(root / name) != expected]
        if bad:
            raise ValueError(f"generated plumbing hash mismatches: {bad}")
        print(f"PASS all {len(generated)} generated plumbing hashes", flush=True)


def pack_mesh(root: Path, manifest: dict, entry: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    packed = output / Path(entry["file"]).name
    kit = root / entry["kit"]
    url = entry.get("code_commit_url", manifest["code"]["commit_url"])
    subprocess.run([sys.executable, str(kit / "tools/pack_reconstruction.py"), "track3-meshes",
                    "--meshes", str(root / entry["meshes"]), "--sample",
                    str(kit / "data/track_3_mesh_sample_submission.parquet"), "--commit", url,
                    "--out", str(packed)], cwd=kit, check=True)
    return packed


def same_submission(root: Path, original: Path, new: Path) -> None:
    sys.path.insert(0, str(root / "verifier_kit/tools"))
    from verify_container import read_submission, same_content
    equal, detail = same_content(read_submission(original), read_submission(new))
    if not equal:
        raise ValueError(f"re-pack differs: {detail}")
    print(f"PASS rows, geometry and provenance equal; byte-identical={sha(original) == sha(new)}", flush=True)


def pack_pose(root: Path, entry: dict, manifest: dict, output: Path,
              rollouts: Path, adoptions: Path | None = None) -> Path:
    """Use the release packer's command on achieved exports, preserving moving adoption."""
    sys.path.insert(0, str(root / manifest["code"]["path"]))
    settings = entry["packer"]
    packer = importlib.import_module(settings["module"])
    if settings.get("adoptions"):
        adoptions = adoptions or root / settings["adoptions"]
        release = read(root / "provenance/track3_combined_v5.manifest.json")
        if sha(adoptions) != release["adoptions"]["sha256"]:
            raise ValueError("moving adoptions differ from frozen release")
        adoption, = read(adoptions)["adoptions"]
        admitted = release["admission"]
        if adoption["episode"] != admitted["episode"]:
            raise ValueError("moving adoption episode differs from frozen admission")
        for key in ("checkpoint", "trajectory", "reference", "admission"):
            if adoption[key] != admitted[key]["path"]:
                raise ValueError(f"moving adoption {key} differs from frozen admission")
        checkpoint = root / entry["checkpoints"][str(adoption["episode"])]
        if sha(checkpoint) != admitted["checkpoint"]["sha256"]:
            raise ValueError("moving checkpoint differs from adopted policy")
        run_command = packer.base.command
    else:
        if adoptions is not None:
            raise ValueError("static release cannot accept moving adoptions")
        run_command = packer.command
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    packed = output / Path(entry["file"]).name
    kit = root / entry["kit"]
    command = [sys.executable, kit / "tools/pack_submission.py", "--track", "track_3",
               "--results", rollouts, "--meshes", root / entry["meshes"],
               "--code-commit", manifest["code"]["commit_url"], "--eval-script",
               root / entry["evaluator"]["script"], "--skip-crosscheck", "-o", packed]
    # The release helper records log paths relative to its code checkout.
    source_log = root / manifest["code"]["path"] / "data/outputs/container" / f"pack-{uuid.uuid4().hex}.log"
    source_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_command(command, source_log)
    finally:
        if source_log.exists():
            shutil.copy2(source_log, logs / "pack.log")
    print(f"packed {packed} via {settings['module']}; adoptions={adoptions}", flush=True)
    return packed


def reproduce_pose(root: Path, entry: dict, manifest: dict, output: Path, episodes: str, adoptions: Path | None = None) -> None:
    """Reuse the repository lease and official evaluator, one policy at a time."""
    if root != Path("/submission"):
        raise SystemExit("GPU reproduction is image-only; host CPU preparation never launches a policy")
    sys.path.insert(0, str(root / "code"))
    from src.gpu import GpuLease
    trace = {row["episode"]: row for row in read(root / "provenance/reproduction-trace.json")}
    wanted = sorted(int(e) for e in entry["checkpoints"])
    if episodes:
        selected = [int(e) for e in episodes.split(",")]
        if len(set(selected)) != len(selected) or not set(selected).issubset(wanted):
            raise ValueError("--episodes must select unique scored episodes in the frozen manifest")
        wanted = selected
    runtime = root / "runtime/flashchord"
    script = root / entry["evaluator"]["script"]
    rollouts, logs = output / "rollouts", output / "logs"
    rollouts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(runtime / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", HYDRA_FULL_ERROR="1", WANDB_MODE="disabled",
               MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0",
               WARP_CACHE_PATH=str(output / "cache/warp"), JAX_COMPILATION_CACHE_DIR=str(output / "cache/jax"),
               XDG_CACHE_HOME=str(output / "cache"), TMPDIR=str(output / "tmp"))
    (output / "tmp").mkdir(parents=True, exist_ok=True)
    # This is a container-local lease under /output, never the host checkout's lease.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with GpuLease(output, 2048, 1024, stage="pose_insurance.container_reproduce") as lease:
        for ep in wanted:
            frozen = trace[ep]
            saved = Path(frozen["saved_task"])
            if not saved.is_file() or sha(saved) != frozen["task_sha256"]:
                raise ValueError(f"frozen task alias missing or wrong for {ep}")
            stem = f"episode_{ep:06d}"
            command = [os.environ["EVAL_PYTHON"], str(script),
                       f"evaluation.checkpoint={root / entry['checkpoints'][str(ep)]}",
                       "evaluation.world_count=1", "evaluation.reset_mode=explicit", "evaluation.start_frame=0",
                       f"evaluation.object_trajectories_output={rollouts / (stem + '.parquet')}",
                       f"evaluation.metrics_output={rollouts / (stem + '.json')}",
                       f"hydra.run.dir={logs / stem}", "logging.mode=disabled"]
            print(f"evaluating {ep}: {logs / (stem + '.log')}", flush=True)
            with (logs / (stem + ".log")).open("w") as stream:
                subprocess.run(command, cwd=runtime, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            lease.require_headroom()
    if episodes:
        return
    packed = pack_pose(root, entry, manifest, output, rollouts, adoptions)
    print(f"compare {packed} with run.sh verify --reproduced", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reproduce", "pack", "verify-mesh", "verify-copies"))
    parser.add_argument("--manifest", type=Path, default=Path("/submission/submission.json"))
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/data"))
    parser.add_argument("--episodes", default="")
    parser.add_argument("--adoptions", type=Path, help="moving adoptions; defaults to the staged release file")
    parser.add_argument("--rollouts", type=Path, help="CPU pack only: achieved exports to pack")
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    if args.command == "verify-copies":
        check_copies(root)
        return
    manifest = read(args.manifest)
    if args.command == "verify-mesh":
        entry, = [e for e in manifest["submissions"] if e["track"] == "track_3_mesh"]
    else:
        if not args.label:
            parser.error("--label is required")
        entry, = [e for e in manifest["submissions"] if e["label"] == args.label]
    output = args.output.resolve() / entry["label"]
    if entry["track"] == "track_3_mesh":
        packed = pack_mesh(root, manifest, entry, output)
        same_submission(root, root / entry["file"], packed)
    elif args.command == "pack":
        if args.rollouts is None:
            parser.error("pose pack requires --rollouts from the official evaluator")
        packed = pack_pose(root, entry, manifest, output, args.rollouts.resolve(), args.adoptions)
        same_submission(root, root / entry["file"], packed)
    else:
        reproduce_pose(root, entry, manifest, output, args.episodes, args.adoptions)


if __name__ == "__main__":
    main()
