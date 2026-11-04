#!/usr/bin/env python3
"""Check the staged dependency closure on CPU without creating host path aliases."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/outputs/container"
STAGE = OUT / "submission"


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import pyarrow.parquet as pq
    records = []
    for module in ("stage", "reproduce", "image_setup"):
        spec = importlib.util.spec_from_file_location(module, ROOT / f"scripts/container/{module}.py")
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    records.append(dict(check="container scripts CPU import", status="PASS"))
    aliases = read(STAGE / "path-aliases.json")
    traces = read(STAGE / "provenance/reproduction-trace.json")
    for row in traces:
        assert row["episode"] not in {2, 13, 22, 24, 27, 41, 43, 44}
        assert aliases[row["saved_task"]] == row["task"]
        assert sha(STAGE / row["task"]) == row["task_sha256"]
        task = pq.read_table(STAGE / row["task"]).to_pylist()[0]
        for key in ("object_mesh_paths", "object_urdf_paths"):
            for saved in task[key]:
                assert saved in aliases and (STAGE / aliases[saved]).is_file(), saved
        for asset in row["assets"]:
            target = STAGE / asset["staged"]
            assert sha(target) == asset["sha256"]
            if target.suffix == ".urdf":
                for mesh in ET.parse(target).getroot().iter("mesh"):
                    saved = mesh.get("filename")
                    candidate = STAGE / aliases[saved] if saved.startswith("/") else target.parent / saved
                    assert candidate.is_file(), saved
    records.append(dict(check="twenty task and object asset closures; no excluded episode", status="PASS",
                        tasks=len(traces), aliases=len(aliases), simulated_assets=sum(len(t["assets"]) for t in traces)))
    assets = STAGE / "runtime/flashchord/src/flash_chord/assets"
    robot_meshes = 0
    for side in ("left", "right"):
        urdf = assets / f"urdfs/sharpawave/{side}_sharpawave.urdf"
        for mesh in ET.parse(urdf).getroot().iter("mesh"):
            name = mesh.get("filename")
            assert (urdf.parent / name).is_file(), name
            robot_meshes += 1
    records.append(dict(check="SharpaWave robot URDF mesh closure", status="PASS", mesh_references=robot_meshes))
    for label, version, imports in (("kit", "3.13", "numpy,pandas,pyarrow,trimesh,fast_simplification"),
                                   ("eval", "3.11", "numpy,pyarrow,hydra,safetensors,jax")):
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   PYTHONPATH=str(STAGE / f"runtime/{label}-env/lib/python{version}/site-packages"))
        command = [str(STAGE / f"runtime/python-{label}/bin/python{version}"), "-B", "-c",
                   f"import {imports}; print('copied Python and installed packages imported on CPU')"]
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        records.append(dict(check=f"copied {label} base interpreter and package payload CPU import", status="PASS",
                            detail=result.stdout.strip(), caveat="host import via explicit PYTHONPATH; image venv relocation remains untested"))
    output = dict(results=records, gpu_launched=False, dataset_read=False)
    (OUT / "dependency-checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
