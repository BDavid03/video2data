"""Local CD-O: score candidate object meshes against the public scans with the kit's own code.

Run with the kit environment (it has fast_simplification):

    .runtime/kit-env/bin/python track3/features/mesh_eval/local_cdo.py --candidates <dir> --out <json>
    .runtime/kit-env/bin/python track3/features/mesh_eval/local_cdo.py --primitive sphere --out <json>

<dir> holds one mesh per object, named <object>.{glb,obj,ply,stl,off}. Both sides go through the
host's `budget_mesh` (4,096 faces / vertices) and then the kit scorer's `_object_chamfer_cm`, which
is imported from metric_code/track_3/CD-O.py, not copied. Seeds follow `_object_seed(episode, 0)`
over --episodes, so a score is the mean over those seeds.

The public scans are references for measurement only. The challenge FAQ says Track 3 entries must
submit their own reconstructions, so nothing a scan holds may flow into a submitted mesh.
`--primitive ... --oracle-proportions` scales a primitive by the scan's own PCA extents: that is an
analysis of what proportions alone are worth, never a submission candidate.
"""

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit"
SCANS = ROOT / ("datasets--nvidia--video_to_data_challenge/snapshots/"
                "5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh")
BUDGET = 4096
SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")

sys.path.insert(0, str(KIT))
from v2dlb.mesh_budget import budget_mesh  # noqa: E402

_spec = importlib.util.spec_from_file_location("kit_cd_o", KIT / "metric_code/track_3/CD-O.py")
CDO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CDO)


def scan_path(name):
    folder = SCANS / name
    for candidate in (folder / f"{name}.glb", folder / f"{name}_visual.glb"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no public scan for {name}")


def budgeted(path):
    vertices, faces = budget_mesh(path, BUDGET, BUDGET)
    return vertices, faces


def primitive(kind, extents=None):
    shapes = {
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=4),
        "cube": lambda: trimesh.creation.box(extents=(2, 2, 2)),
        "cylinder": lambda: trimesh.creation.cylinder(radius=1, height=2, sections=64),
    }
    mesh = shapes[kind]()
    if extents is not None:
        mesh.apply_scale(np.asarray(extents) / 2.0)
    return mesh


def pca_extents(path):
    """Extents of the scan along its own principal axes (analysis only)."""
    vertices, faces = budgeted(path)
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    points = mesh.sample(20000, seed=0) if hasattr(mesh, "sample") else mesh.vertices
    centred = points - points.mean(axis=0)
    _, axes = np.linalg.eigh(centred.T @ centred)
    projected = centred @ axes[:, ::-1]
    return projected.max(axis=0) - projected.min(axis=0)


def score(reference, candidate, episodes):
    values = [CDO._object_chamfer_cm(reference, candidate, CDO._object_seed(e, 0)) for e in episodes]
    return float(np.mean(values)), float(np.std(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", type=Path, help="directory of <object>.<ext> meshes")
    source.add_argument("--primitive", choices=("sphere", "cube", "cylinder", "scan"))
    parser.add_argument("--oracle-proportions", action="store_true",
                        help="scale the primitive by the scan's PCA extents (analysis only)")
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=[1, 3, 5],
                        help="episode numbers whose kit seeds are averaged")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = args.objects or sorted(p.name for p in SCANS.iterdir() if p.is_dir())
    rows = []
    with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as scratch:
        for name in names:
            reference_path = scan_path(name)
            reference = budgeted(reference_path)
            if args.candidates:
                matches = [args.candidates / f"{name}{s}" for s in SUFFIXES if (args.candidates / f"{name}{s}").is_file()]
                if not matches:
                    rows.append({"object": name, "status": "missing"})
                    continue
                candidate_path = matches[0]
            elif args.primitive == "scan":
                candidate_path = reference_path  # plumbing check: same seed both sides, so exactly 0
            else:
                extents = pca_extents(reference_path) if args.oracle_proportions else None
                candidate_path = Path(scratch) / f"{name}-{args.primitive}.ply"
                primitive(args.primitive, extents).export(candidate_path)
            candidate = budgeted(candidate_path)
            mean, spread = score(reference, candidate, args.episodes)
            rows.append({"object": name, "status": "scored", "cd_o_cm": round(mean, 4),
                         "seed_std_cm": round(spread, 4), "candidate": str(candidate_path),
                         "candidate_faces": int(len(candidate[1]))})
            print(f"{name:20s} {mean:7.3f} cm  (+/- {spread:.3f} over {len(args.episodes)} seeds)", flush=True)
    scored = [r["cd_o_cm"] for r in rows if r["status"] == "scored"]
    summary = {"source": str(args.candidates) if args.candidates else args.primitive,
               "oracle_proportions": bool(args.oracle_proportions), "episodes_for_seeds": args.episodes,
               "mean_cd_o_cm": round(float(np.mean(scored)), 4) if scored else None,
               "reference": "public scans through budget_mesh; measurement only", "objects": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"mean {summary['mean_cd_o_cm']} cm over {len(scored)} objects -> {args.out}")


if __name__ == "__main__":
    main()
