"""`migrate.py --verify`: one PASS/FAIL line per check, CPU only. Exit status is the number of failures."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

FORBIDDEN = ("episode_000041", "episode_000043", "episode_000044", "e000041", "e000043", "e000044")
EXPECTED_TOP = {".agents", ".claude", ".git", ".gitignore", "AGENTS.md", "CLAUDE.md", "LICENSE", "README.md",
                "data", "docs", "models", "scripts", "src", "submission", "upstream", "skills-lock.json"}
failures = []


def line(ok, label, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(label)


def sh(root, argv, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recorded_paths(root, limit=1500):
    """(old-or-new name, sha256 or None, record) from run.json files and manifests; forbidden episodes skipped."""
    records = sorted((root / "submission").glob("*/*manifest*.json"))
    records += sorted((root / "data/outputs/pose_insurance").rglob("run.json"))[:400]
    records += sorted((root / "data/outputs/mesh_generative").rglob("run.json"))[:200]
    rows = []

    def walk(value, record):
        if isinstance(value, dict):
            name = value.get("path")
            if isinstance(name, str) and "sha256" in value:
                rows.append((name, value["sha256"], record))
            for child in value.values():
                walk(child, record)
        elif isinstance(value, list):
            for child in value:
                walk(child, record)

    for record in records:
        if any(f in record.as_posix() for f in FORBIDDEN):
            continue
        try:
            walk(json.loads(record.read_text(encoding="utf-8-sig")), record.relative_to(root).as_posix())
        except (OSError, ValueError):
            continue
        if len(rows) >= limit:
            break
    return [r for r in rows if not any(f in r[0] for f in FORBIDDEN)]


def run(root, journal):
    root = Path(root)
    py = str(root / "data/.runtime/linux-cuda/bin/python")
    sys.path.insert(0, str(root))
    # 1 compile
    bad = []
    for base in ("src", "scripts"):
        for path in (root / base).rglob("*.py"):
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except (SyntaxError, ValueError) as error:
                bad.append(f"{path.name}: {error}")
    line(not bad, "py_compile over src/ and scripts/", "; ".join(bad[:3]))
    # 2 slice imports
    slices = sorted(p.name for p in (root / "src/features").iterdir() if p.is_dir() and (p / "__init__.py").exists())
    mods = [f"src.features.{s}" for s in slices] + ["src.features.dynamic_feasibility.m0c", "src.features.moving_policy",
                                                     "src.paths", "src.runs", "src.provenance", "src.gpu", "src.historical"]
    for mod in mods:
        code, tail = sh(root, [py, "-B", "-c", f"import {mod}"])
        line(code == 0, f"CPU import {mod}", tail[0] if code else "")
    # 3 environments
    envs = [("linux-cuda", "data/.runtime/linux-cuda/bin/python", ["torch"]),
            ("flashchord", "data/.runtime/flashchord/venv/bin/python", ["jax", "warp", "flash_chord"]),
            ("kit-env", "data/.runtime/kit-env/bin/python", ["pandas", "trimesh"])]
    for name, exe, modules in envs:
        for module in modules:
            code, tail = sh(root, [str(root / exe), "-B", "-c", f"import {module}"])
            line(code == 0, f"{name} imports {module}", tail[0] if code else "")
    fp = root / "data/.runtime/foundationpose"
    candidates = sorted(fp.glob("env/bin/python")) + sorted(fp.glob("*/bin/python"))
    if candidates:
        probe = ("import glob, sys; sys.path[:0] = glob.glob('data/.runtime/foundationpose/sources/FoundationPose-*/mycpp/build'); "
                 "import mycpp")
        code, tail = sh(root, [str(candidates[0]), "-B", "-c", probe])
        line(code == 0, "foundationpose imports mycpp (GPU hidden)", tail[0] if code else "")
    else:
        line(False, "foundationpose interpreter present")
    # 4 record resolution
    from src.paths import current_name, project_path
    rows = recorded_paths(root)
    resolved = mismatched = missing = checked = 0
    for name, digest, _ in rows:
        try:
            target = project_path(root, name)
        except ValueError:
            continue
        if not target.is_file():
            missing += 1
            continue
        resolved += 1
        if "/sources/" in name or name.endswith((".py", ".sh")) or name.startswith(("src/", "track3/", "scripts/", "docs/", "AGENTS.md")):
            continue  # sources legitimately drift; the historical mode covers them
        checked += 1
        mismatched += sha(target) != digest
    line(len(rows) >= 500 and resolved >= 500 and mismatched == 0,
         "old/new record resolution", f"{len(rows)} recorded paths, {resolved} resolve, {checked} artifact digests compared, "
         f"{mismatched} differ, {missing} missing (run-local or deleted scratch)")
    equal = all(current_name(current_name(n)) == current_name(n) for n, _, _ in rows)
    line(equal, "mapping is idempotent on recorded names")
    # 5 inventory
    inv = root / "data/layout/inventory-before.json"
    if inv.exists():
        before = json.loads(inv.read_text())
        diff = [n for n, d in before.items() if not (root / current_name(n)).is_file() or sha(root / current_name(n)) != d]
        line(not diff, "pack and record bytes unchanged", f"{len(before)} files, {len(diff)} differ {diff[:3]}")
    else:
        line(False, "inventory-before.json present")
    # 6 verifiers and release
    for label, manifest in (("D1 v5", "submission/20261103T2058Z-d1-v5/track3_meshes_v5_final.manifest.json"),
                            ("D2 v5", "submission/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json")):
        code, tail = sh(root, ["bash", "scripts/reproduce/verify.sh", "--historical", manifest])
        line(code == 0, f"verify.sh {label}", tail[0])
    code, tail = sh(root, ["bash", "scripts/reproduce/d2_poses_v5.sh", "--historical"])
    line(code == 0, "d2_poses_v5.sh --historical", tail[0] if code else "")
    r = subprocess.run(["bash", "scripts/reproduce/check_release.sh"], cwd=root, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    out = r.stdout.strip().splitlines()
    line(r.returncode == 0 and out[-1:] == ["READY"], "check_release.sh ends READY", "\n".join(l for l in out if l.startswith("FAIL"))[:300])
    # 7 top level
    top = {p.name for p in root.iterdir()}
    extra = sorted(top - EXPECTED_TOP)
    line(not extra, "top-level listing matches the target", f"unexpected {extra}" if extra else "")
    print(f"{len(failures)} failure(s)")
    return len(failures)
