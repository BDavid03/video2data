#!/usr/bin/env python3
"""Stage a manifest-selected frozen release without changing its producers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tarfile
import xml.etree.ElementTree as ET

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/outputs/container"
STAGE = OUT / "submission"
KIT = ROOT / "data/.runtime/kit-20261102/v2d_submission_kit"
OLD = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"
RUNTIME = ROOT / "data/.runtime/flashchord"
FC = RUNTIME / "video_to_data/robotic_grounding/flash_chord"
POSE_LABEL = "poses-combined-v5"
MESH_LABEL = "meshes-d1-v5"
FORBIDDEN = {2, 13, 22, 24, 27, 41, 43, 44}
BACKUP = OUT / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        backup(path)
    path.write_text(content)


def backup(path: Path) -> None:
    target = BACKUP / path.relative_to(OUT)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(value: str) -> Path:
    """Resolve the pre-migration output spelling only inside this repository."""
    p = Path(value)
    if not p.is_absolute():
        if p.parts[0] in {"outputs", ".runtime", ".cache"}:
            p = Path("data") / p
        elif p.parts[0] == "submissions":
            p = Path("submission", *p.parts[1:])
        p = ROOT / p
    p.resolve().relative_to(ROOT)
    return p


class Copier:
    """Hash both ends; leave existing destinations alone when they already match."""
    def __init__(self) -> None:
        self.files: dict[str, dict] = {}

    def file(self, src: Path, relative: str | Path, expected: str | None = None) -> Path:
        dst = STAGE / relative
        dst.relative_to(STAGE)
        actual = sha(src)
        if expected is not None and actual != expected:
            raise ValueError(f"source hash mismatch: {src}: {actual} != {expected}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha(dst) != actual:
            if dst.exists():
                backup(dst)
            shutil.copy2(src, dst, follow_symlinks=True)
        if sha(dst) != actual:
            raise ValueError(f"copy hash mismatch: {dst}")
        self.files[str(relative)] = dict(source=str(src.relative_to(ROOT)), sha256=actual, bytes=dst.stat().st_size)
        return dst

    def tree(self, src: Path, relative: str, skip: set[str] | None = None) -> None:
        skip = skip or set()
        for folder, dirs, files in os.walk(src, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in skip)
            for name in sorted(files):
                p = Path(folder) / name
                if p.is_symlink():
                    p.resolve().relative_to(ROOT)
                self.file(p, Path(relative) / p.relative_to(src))
            for name in dirs:
                p = Path(folder) / name
                if p.is_symlink():
                    raise ValueError(f"unhandled directory symlink: {p}")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


def checkout(relative: str, url: str) -> dict:
    """Copy Git objects and export a clean tree using read-only Git commands."""
    dst = STAGE / relative
    commit = url.rsplit("/", 1)[1]
    if dst.exists() and git("rev-parse", "HEAD", cwd=dst) != commit:
        previous = BACKUP / relative
        previous.parent.mkdir(parents=True, exist_ok=True)
        dst.rename(previous)
    if not dst.exists():
        dst.mkdir(parents=True)
        archive = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(dst, filter="data")
        common = Path(git("rev-parse", "--git-common-dir"))
        common = common if common.is_absolute() else ROOT / common
        metadata = dst / ".git"
        metadata.mkdir()
        shutil.copytree(common / "objects", metadata / "objects")
        shutil.copytree(common / "refs", metadata / "refs")
        if (common / "packed-refs").exists():
            shutil.copy2(common / "packed-refs", metadata / "packed-refs")
        (metadata / "HEAD").write_text(commit + "\n")
        (metadata / "config").write_text("[core]\nrepositoryformatversion = 0\nbare = false\nfilemode = true\n")
        # A v2 index records the exported tree; no git checkout/read-tree/index mutation.
        tree = subprocess.check_output(["git", "ls-tree", "-rz", commit], cwd=ROOT).split(b"\0")[:-1]
        records = []
        for row in tree:
            identity, name = row.split(b"\t", 1)
            mode, kind, oid = identity.split()
            if kind != b"blob":
                raise ValueError("code snapshot does not support submodules")
            st = (dst / os.fsdecode(name)).lstat()
            fields = (int(st.st_ctime), st.st_ctime_ns % 10**9, int(st.st_mtime), st.st_mtime_ns % 10**9,
                      st.st_dev, st.st_ino, int(mode, 8), st.st_uid, st.st_gid, st.st_size)
            record = struct.pack(">10I20sH", *(v & 0xffffffff for v in fields), bytes.fromhex(oid.decode()), min(len(name), 0xfff))
            record += name + b"\0"
            records.append(record + b"\0" * (-len(record) % 8))
        index = struct.pack(">4sII", b"DIRC", 2, len(records)) + b"".join(records)
        (metadata / "index").write_bytes(index + hashlib.sha1(index).digest())
    if git("rev-parse", "HEAD", cwd=dst) != commit or git("status", "--porcelain", "--untracked-files=no", cwd=dst):
        raise ValueError(f"existing snapshot is not clean at {commit}: {dst}")
    if (dst / ".git/objects/info/alternates").exists():
        raise ValueError("snapshot must be independent of the local object store")
    return dict(path=relative, commit=commit, shallow=git("rev-parse", "--is-shallow-repository", cwd=dst),
                commits=git("rev-list", "--all", "--count", cwd=dst), objects=git("count-objects", "-v", cwd=dst))


def environment(copier: Copier, env: Path, label: str) -> dict:
    """Copy installed bytes; generate only venv location plumbing, never install."""
    home = next(line.split("=", 1)[1].strip() for line in (env / "pyvenv.cfg").read_text().splitlines()
                if line.startswith("home ="))
    python_root = Path(home).resolve().parent
    version = next((env / "lib").glob("python*" )).name
    base = f"runtime/python-{label}"
    copier.tree(python_root, base, {"__pycache__"})
    copier.tree(env / "lib", f"runtime/{label}-env/lib", {"__pycache__"})
    dst = STAGE / f"runtime/{label}-env"
    (dst / "bin").mkdir(parents=True, exist_ok=True)
    (dst / "pyvenv.cfg").write_text(
        f"home = /submission/{base}/bin\ninclude-system-site-packages = false\nversion = {version[6:]}\n")
    for name, target in {"python": f"../../python-{label}/bin/{version}",
                         "python3": "python", version: "python", "../lib64": "lib"}.items():
        p = dst / "bin" / name if name != "../lib64" else dst / "lib64"
        if not p.is_symlink():
            p.symlink_to(target)
    # The editable package's source is unchanged. Its location is supplied by PYTHONPATH.
    pth = dst / "lib" / version / "site-packages/_editable_impl_flash_chord.pth"
    if pth.exists():
        pth.write_text("/submission/runtime/flashchord/src\n")
        copier.files.pop(str(pth.relative_to(STAGE)), None)
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions(
        path=[str(env / "lib" / version / "site-packages")]))
    return dict(python_source=str(python_root.relative_to(ROOT)), packages=packages)


def manifests(combined: dict, d1: dict, checkpoints: dict, counted_set: str) -> dict:
    pose = dict(label=POSE_LABEL, track="track_3", file="submitted/track3_combined_v5.parquet", kit="kit",
                kaggle_submission_ids={}, evaluator=dict(script="kit/flash_chord/scripts/evaluate_policy.py",
                evaluation_package="runtime/flashchord/src/flash_chord/evaluation"), checkpoints=checkpoints,
                outputs=f"outputs/{POSE_LABEL}/rollouts", meshes=f"meshes/{POSE_LABEL}")
    pose["packer"] = dict(module="src.features.pose_insurance.pack_combined_moving"
                          if "adoptions" in combined else "src.features.pose_insurance.pack_combined")
    if "adoptions" in combined:
        pose["packer"]["adoptions"] = "provenance/adoptions.json"
    mesh = dict(label=MESH_LABEL, track="track_3_mesh", file="submitted/track3_meshes_v5_final.parquet", kit="kit-old",
                kaggle_submission_ids={}, meshes=f"meshes/{MESH_LABEL}",
                code_commit_url=d1["code_commit_url"], code_path="code-mesh",
                repack_note="verify_container.py has no object_mesh repack branch; run.sh verify-mesh covers it")
    manifest = dict(schema="v2d_container_v1", team="anonymous", code=dict(path="code", commit_url=combined["code_commit_url"]),
                    kit="kit", submissions=[pose, mesh] if counted_set == "split" else [pose])
    write(STAGE / "split.draft.json", dict(manifest, submissions=[dict(pose), dict(mesh)]))
    ids_path = OUT / "kaggle-ids.json"
    ids = read(ids_path) if ids_path.exists() else {}
    required = {POSE_LABEL: {"add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm"}, MESH_LABEL: {"cd_o_cm"}}
    if counted_set == "single":
        required[POSE_LABEL].add("cd_o_cm")
    missing = []
    for entry in manifest["submissions"]:
        label = entry["label"]
        entry["kaggle_submission_ids"] = ids.get(label, {})
        values = entry["kaggle_submission_ids"]
        if set(values) != required[label] or any(type(v) is not int or v <= 0 for v in values.values()):
            missing.append(label)
    write(STAGE / "submission.draft.json", manifest)
    for entry in (pose, mesh):
        single = dict(manifest, submissions=[entry], code=dict(
            path=entry.get("code_path", "code"), commit_url=entry.get("code_commit_url", combined["code_commit_url"])))
        write(STAGE / f"{entry['label']}.draft.json", single)
    if missing:
        if (STAGE / "submission.json").exists():
            raise ValueError("IDs are incomplete but a previous final exists: refusing stale final")
        print(f"DRAFT ONLY: missing positive integer Kaggle IDs for {missing}", flush=True)
    else:
        write(STAGE / "submission.json", manifest)
        for entry in manifest["submissions"]:
            write(STAGE / f"{entry['label']}.json", read(STAGE / f"{entry['label']}.draft.json"))
    return dict(missing_ids=missing, final_written=not missing, counted_set=counted_set,
                verifier_limitation="One global code commit: the historical split set fails; the current single-file set avoids this conflict.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true", help="write manifests only; require all six Kaggle IDs")
    parser.add_argument("--combined-manifest", type=Path, help="explicit static/moving release manifest; default: newest local moving release")
    parser.add_argument("--counted-set", choices=("single", "split"), default="single",
                        help="current owner direction: one combined file for all six boards; split is diagnostic")
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        plan = read(STAGE / "stage-plan.json")
        combined, d1 = read(STAGE / "provenance/track3_combined_v5.manifest.json"), read(STAGE / "provenance/track3_meshes_v5_final.manifest.json")
        old = read(STAGE / "submission.draft.json")
        state = manifests(combined, d1, old["submissions"][0]["checkpoints"], plan["counted_set"])
        receipts = Copier()
        d1_path = source(next(p["path"] for p in combined["parents"] if "mesh" in Path(p["path"]).name))
        for folder in (ROOT / plan["combined_manifest"]).parent, d1_path.parent:
            for receipt in folder.glob("*.kaggle-receipt.json"):
                receipts.file(receipt, Path("submitted") / receipt.name)
        hashes = read(STAGE / "copy-hashes.json")
        hashes.update(receipts.files)
        write(STAGE / "copy-hashes.json", hashes)
        generated = read(STAGE / "generated-hashes.json")
        generated["copy-hashes.json"] = sha(STAGE / "copy-hashes.json")
        write(STAGE / "generated-hashes.json", generated)
        write(OUT / "ids-status.json", state)
        return 2 if state["missing_ids"] else 0
    combined_path = args.combined_manifest
    if combined_path is None:
        candidates = sorted((ROOT / "submission").glob("*-d2-moving-v5/track3_combined_v5.manifest.json"))
        if not candidates:
            parser.error("no moving release found; supply --combined-manifest")
        combined_path = candidates[-1]
    combined_path = combined_path.resolve()
    combined_path.relative_to(ROOT)
    combined = read(combined_path)
    parents = {Path(p["path"]).name: source(p["path"]) for p in combined["parents"]}
    d2_path = parents["track3_poses_v5_final.manifest.json"]
    d1_path = parents["track3_meshes_v5_final.manifest.json"]
    d1 = read(d1_path)
    if combined.get("kind") not in {"pose_insurance.combined_pack", "pose_insurance.combined_moving_pack"}:
        raise ValueError("expected a static or moving combined release manifest")
    moving = combined["kind"] == "pose_insurance.combined_moving_pack"
    if moving != ("adoptions" in combined):
        raise ValueError("moving manifest/adoptions mismatch")
    for parent in combined["parents"]:
        if sha(source(parent["path"])) != parent["sha256"]:
            raise ValueError(f"frozen parent hash mismatch: {parent['path']}")
    print(f"Staging {combined_path.relative_to(ROOT)}: {combined['file']['sha256']}, counted set={args.counted_set}", flush=True)
    if (STAGE / "stage-plan.json").exists() and read(STAGE / "stage-plan.json")["manifest_sha256"] != sha(combined_path):
        for relative in (f"meshes/{POSE_LABEL}",):
            old_tree = STAGE / relative
            if old_tree.exists():
                target = BACKUP / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                old_tree.rename(target)
    write(STAGE / "stage-plan.json", dict(combined_manifest=str(combined_path.relative_to(ROOT)),
          manifest_sha256=sha(combined_path), counted_set=args.counted_set, code_commit_url=combined["code_commit_url"], kind=combined["kind"]))
    copier = Copier()
    for manifest in (combined_path, d2_path, d1_path):
        copier.file(manifest, Path("provenance") / manifest.name)
    for manifest, name in ((combined, "track3_combined_v5.parquet"), (d1, "track3_meshes_v5_final.parquet")):
        path = combined_path.parent / name if manifest is combined else d1_path.parent / name
        copier.file(path, Path("submitted") / name, manifest["file"]["sha256"])
        for receipt in path.parent.glob("*.kaggle-receipt.json"):
            copier.file(receipt, Path("submitted") / receipt.name)
    derived = combined["cd_o"]["file"]
    copier.file(source(derived["path"]), Path("submitted") / Path(derived["path"]).name, derived["sha256"])
    for row in combined["kit_files"]:
        if sha(source(row["path"])) != row["sha256"]:
            raise ValueError(f"combined kit has drifted: {row['path']}")
    print("Copying both immutable kit versions and verifier", flush=True)
    for src, dst in ((KIT, "kit"), (OLD, "kit-old"), (KIT, "verifier_kit")):
        copier.tree(src, dst, {"__pycache__", ".git"})
    clones = [checkout("code", combined["code_commit_url"]), checkout("code-mesh", d1["code_commit_url"])]
    for entry in combined["sources"]:
        if sha(STAGE / "code" / entry["path"]) != entry["sha256"]:
            raise ValueError(f"combined source hash disagrees with code commit: {entry['path']}")
    checkpoints, traces, aliases = {}, [], {}
    import pyarrow.parquet as pq
    frames = read(source(combined["body_frames"]["path"]))
    if sha(source(combined["body_frames"]["path"])) != combined["body_frames"]["sha256"]:
        raise ValueError("body frame trace hash mismatch")
    copier.file(source(combined["body_frames"]["path"]), "provenance/body-frames.json", combined["body_frames"]["sha256"])
    for frozen in combined["exports"]:
        ep = int(Path(frozen["path"]).stem.split("_")[1])
        if ep in FORBIDDEN or str(ep) in checkpoints:
            raise ValueError(f"forbidden or duplicate export: {ep}")
        evidence = [r for r in frames["rows"] if r["episode"] == ep]
        adopted = moving and ep == combined["admission"]["episode"]
        if adopted:
            admission = combined["admission"]
            policy = admission["policy"]
            cp_entry, task_entry = admission["checkpoint"], policy["task"]
            run_entry = policy["evaluator_run"]
            if frozen["sha256"] != admission["trajectory"]["sha256"]:
                raise ValueError(f"moving export identity mismatch: {ep}")
            asset_rows = [dict(sim_mesh=combined["moving_mesh"]["source_mesh"], urdf=combined["moving_mesh"]["urdf"])]
            metrics = policy["official_metrics"]
            learning = policy["learning"]
            train = read(source(learning["train_run"]["path"]))
            budget = learning["logged_steps"]
            seconds = train.get("finished_unix", 0) - train.get("started_unix", 0)
            for key in ("train_run", "commands", "stage0"):
                item = learning[key]
                copier.file(source(item["path"]), f"provenance/moving/{key}.json", item["sha256"])
        else:
            run_entry = evidence[0]["rollout"]
            run = read(source(run_entry["path"]))
            cp_entry = dict(path=run["checkpoint"], sha256=run["checkpoint_sha256"])
            task_entry = evidence[0]["task"]
            recorded_task, = [i for i in run["inputs"] if i["path"].endswith("/task.parquet")]
            if (recorded_task["sha256"] != task_entry["sha256"]
                    or source(recorded_task["path"]) != source(task_entry["path"])):
                raise ValueError(f"task identity differs from frozen run: {ep}")
            asset_rows = evidence
            metric_hash = run["artifacts"]["official_metrics.json"]
            metrics = dict(path=str(source(run_entry["path"]).parent / "official_metrics.json"),
                           sha256=metric_hash if isinstance(metric_hash, str) else metric_hash["sha256"])
            budget, seconds = run["request"]["budget"]["environment_steps"], run["policy_seconds"]
        cp, task = source(cp_entry["path"]), source(task_entry["path"])
        relative = f"weights/{POSE_LABEL}/episode_{ep:06d}{cp.suffix}"
        copier.file(cp, relative, cp_entry["sha256"])
        checkpoints[str(ep)] = relative
        copier.file(source(frozen["path"]), f"outputs/{POSE_LABEL}/rollouts/episode_{ep:06d}.parquet", frozen["sha256"])
        copier.file(source(metrics["path"]), f"outputs/{POSE_LABEL}/rollouts/episode_{ep:06d}.json", metrics["sha256"])
        copier.file(source(run_entry["path"]), f"provenance/policies/episode_{ep:06d}.json", run_entry["sha256"])
        metadata = subprocess.check_output([str(RUNTIME / "venv/bin/python"), "-B", "-c",
            "from safetensors import safe_open; import sys; "
            "f=safe_open(sys.argv[1], framework='numpy'); print(f.metadata()['user.resolved_config_json'])", str(cp)], text=True)
        config = json.loads(metadata)
        saved_task = config["task"]["parquet"]
        task_target = f"tasks/episode_{ep:06d}/task.parquet"
        copier.file(task, task_target, task_entry["sha256"])
        aliases[saved_task] = task_target
        row = pq.read_table(task).to_pylist()[0]
        assets = []
        for key, frame_key in (("object_mesh_paths", "sim_mesh"), ("object_urdf_paths", "urdf")):
            for index, saved in enumerate(row[key]):
                entry = asset_rows[index][frame_key]
                p = source(entry["path"])
                target = f"tasks/episode_{ep:06d}/{p.name}"
                copier.file(p, target, entry["sha256"])
                aliases[saved] = target
                assets.append(dict(saved=saved, staged=target, sha256=entry["sha256"]))
                if key == "object_urdf_paths":
                    for mesh in ET.parse(p).getroot().iter("mesh"):
                        name = mesh.get("filename")
                        if name and name.startswith("/"):
                            match = next(a for a in assets if Path(a["saved"]).name == Path(name).name)
                            aliases[name] = match["staged"]
        traces.append(dict(episode=ep, checkpoint=relative, saved_task=saved_task, task=task_target,
                           task_sha256=task_entry["sha256"], assets=assets,
                           training_device=config["training"]["device"], environment_steps=budget,
                           policy_kind="moving" if adopted else "static",
                           historical_train_plus_eval_seconds=None if adopted else seconds,
                           historical_resumed_train_seconds=seconds if adopted else None))
    if len(checkpoints) != 20 or len(combined["meshes"]) != 36:
        raise ValueError("expected twenty policies and 36 combined meshes")
    mesh_root = Path(os.path.commonpath([str(source(e["path"]).parent) for e in combined["meshes"]]))
    for entry in combined["meshes"]:
        rel = source(entry["path"]).relative_to(mesh_root)
        copier.file(source(entry["path"]), Path("meshes") / POSE_LABEL / rel, entry["sha256"])
    if moving:
        item = combined["adoptions"]
        copier.file(source(item["path"]), "provenance/adoptions.json", item["sha256"])
    production_path = source(d1["production"]["path"])
    copier.file(production_path, "provenance/d1-production.json", d1["production"]["sha256"])
    for entry in read(production_path)["pairs"]:
        if int(entry["episode"]) in FORBIDDEN:
            raise ValueError("forbidden mesh episode")
        copier.file(source(entry["file"]), f"meshes/{MESH_LABEL}/episode_{int(entry['episode']):06d}/{Path(entry['file']).name}", entry["sha256"])
    print("Copying the evaluated native runtime and both Python environments (about 11 GB)", flush=True)
    copier.tree(FC, "runtime/flashchord", {"__pycache__", ".git", "tests"})
    environments = {"eval": environment(copier, RUNTIME / "venv", "eval"),
                    "kit": environment(copier, ROOT / "data/.runtime/kit-env", "kit")}
    copier.file(ROOT / "data/outputs/pose_insurance/runtime_inventory.json", "provenance/runtime-inventory.json")
    write(STAGE / "path-aliases.json", aliases)
    write(STAGE / "provenance/reproduction-trace.json", traces)
    write(STAGE / "provenance/environments.json", environments)
    for name in ("run.sh", "reproduce.py", "image_setup.py", "README.md"):
        copier.file(ROOT / "scripts/container" / name, name)
    copier.file(KIT / "examples/container/run.sh", "kit-example-run.sh")
    (STAGE / "run.sh").chmod(0o755)
    state = manifests(combined, d1, checkpoints, args.counted_set)
    example = {POSE_LABEL: dict.fromkeys(("add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm")),
               MESH_LABEL: {"cd_o_cm": None}}
    if args.counted_set == "single":
        example = {POSE_LABEL: dict(example[POSE_LABEL], cd_o_cm=None)}
    write(OUT / "kaggle-ids.example.json", example)
    write(STAGE / "copy-hashes.json", copier.files)
    generated = ["stage-plan.json", "path-aliases.json", "provenance/reproduction-trace.json",
                 "provenance/environments.json", "runtime/eval-env/pyvenv.cfg", "runtime/kit-env/pyvenv.cfg",
                 "runtime/eval-env/lib/python3.11/site-packages/_editable_impl_flash_chord.pth", "copy-hashes.json"]
    write(STAGE / "generated-hashes.json", {name: sha(STAGE / name) for name in generated})
    (OUT / ".dockerignore").write_text("*\n!submission/\n!submission/**\n")
    sizes = {}
    for p in sorted(STAGE.iterdir()):
        if p.is_dir():
            sizes[p.name] = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    write(OUT / "stage-summary.json", dict(state, clones=clones, sizes_bytes=sizes,
          copied_files=len(copier.files), copied_bytes=sum(f["bytes"] for f in copier.files.values()),
          aliases=len(aliases), checkpoints=len(checkpoints), combined_meshes=len(combined["meshes"]),
          runtime_commit=git("rev-parse", "HEAD", cwd=RUNTIME / "video_to_data")))
    print(json.dumps(read(OUT / "stage-summary.json"), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
