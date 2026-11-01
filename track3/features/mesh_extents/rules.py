"""Run and resume the three fixed-protocol mesh shape rules on CPU."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from track3.paths import output_path, project_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, new_run_id, recorded, runtime_flag

from . import _shapes
from ._rule_scores import CATEGORIES, FOLDS, NAMES, select, shape_for

FEATURE = Path(__file__).parent
BUILD = Path("outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff")
PROTOCOL = Path("docs/chain/mesh-selection-protocol.md")
OUTPUT = Path("outputs/mesh_eval/rules")
SEEDS = [1, 3, 5]


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_queue(path: Path) -> list[dict]:
    """Admit only the three predeclared rule families with one parameter apiece."""
    rules = json.loads(path.read_text())["rules"]
    expected = [
        ("R1 vessel-cylinder", None, ("cylinder", "box", "box")),
        ("R2 vessel-handle", "aspect_threshold", ("vessel_with_handle", "box", "box")),
        ("R3 thin-tool-head", "head_length_fraction", ("box", "box", "head_and_handle")),
    ]
    if len(rules) != 3:
        raise ValueError("The fixed protocol permits exactly the three requested rules")
    for rule, (name, parameter, shapes) in zip(rules, expected):
        if (rule["name"], rule["parameter"], rule["shapes"]) != (name, parameter, dict(zip(CATEGORIES, shapes))):
            raise ValueError(f"Rule family or category mapping changed: {name}")
        grid = rule["grid"]
        if not grid or len(set(grid)) != len(grid):
            raise ValueError("A parameter grid must be nonempty without duplicates")
        if parameter is None and grid != [None]:
            raise ValueError("R1 has no free parameter")
        if parameter is not None:
            bounds = (1, float("inf")) if parameter == "aspect_threshold" else (0, 1)
            if not all(isinstance(p, (int, float)) and bounds[0] < p < bounds[1] for p in grid):
                raise ValueError(f"Invalid grid for {parameter}")
        if name.startswith("R2") and rule.get("handle_height_fraction") != 0.6:
            raise ValueError("The R2 handle height fraction is fixed at 0.6")
    return rules


def append_ledger(path: Path, row: dict) -> None:
    """Append a durable result, separating any incomplete line left by interruption."""
    with path.open("a+b") as stream:
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write((json.dumps(row, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def build_candidate(rule: dict, parameter, name: str, dims, original: Path, directory: Path) -> tuple:
    """Build solely from video extents, keeping existing box files byte-for-byte."""
    shape = shape_for(rule, name, dims, parameter)
    if shape == "box":
        return shape, original, None
    if shape == "cylinder":
        mesh = _shapes.cylinder(dims)
    elif shape == "vessel_with_handle":
        mesh = _shapes.vessel_with_handle(dims, rule["handle_height_fraction"])
    else:
        mesh = _shapes.head_and_handle(dims, parameter)
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError(f"{name}: shape must have closed, consistently oriented components")
    data = mesh.export(file_type="ply")
    target = directory / (hashlib.sha256(data).hexdigest() + ".ply")
    if not target.exists():
        target.write_bytes(data)
    return shape, target, {"watertight": bool(mesh.is_watertight), "faces_before_budget": len(mesh.faces),
                           "closed_components": len(mesh.split()), "extents": mesh.extents.tolist()}


def evaluate_object(rule: dict, parameter, name: str, context: dict) -> dict:
    """Cache one rule/parameter/object trial and reuse identical mesh evaluations."""
    from scripts.track3 import local_cdo

    output = context["output"]
    row = context["objects"][name]
    shape, candidate, geometry = build_candidate(rule, parameter, name, row["extents_m"],
                                                ROOT / BUILD / row["file"], output / "meshes")
    score_inputs = {"candidate_sha256": sha256_file(candidate), "reference": context["references"][name],
                    "scoring": context["scoring"]}
    score_id = identity(score_inputs)
    request = {"rule": {key: value for key, value in rule.items() if key != "grid"},
               "parameter": parameter, "object": name, "extents_m": row["extents_m"], "shape": shape,
               "score_id": score_id, "sources": context["sources"], "protocol": context["protocol"]}
    trial_id = identity(request)
    cached = output / "cache" / f"{trial_id}.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        if trial_id not in context["logged"]:
            append_ledger(output / "ledger.jsonl", result)
            context["logged"].add(trial_id)
        context["resumed"] += 1
        return result
    record = run_fields(ROOT, "mesh_extents", "rules", request,
                        inputs=[context["queue"], PROTOCOL, candidate, local_cdo.scan_path(name)],
                        parents=[BUILD / "run.json"], provenance="public_asset_diagnostic")
    directory = output / "trials" / trial_id / new_run_id()
    score_cache = output / "scores" / f"{score_id}.json"
    with recorded(directory, record):
        if score_cache.exists():
            measurement = json.loads(score_cache.read_text())
            context["reused_scores"] += 1
        else:
            reference = context["reference_arrays"].setdefault(name, None)
            if reference is None:
                reference = local_cdo.budgeted(local_cdo.scan_path(name))
                context["reference_arrays"][name] = reference
            vertices, faces = local_cdo.budgeted(candidate)
            # The kit samples by face area; padded degenerate faces have probability zero.
            samples = local_cdo.CDO._sample_mesh_surface(vertices, faces, 1000, local_cdo.CDO._object_seed(1, 0))
            import numpy as np

            if not np.isfinite(samples).all():
                raise ValueError("The budgeted candidate produced nonfinite surface samples")
            mean, spread = local_cdo.score(reference, (vertices, faces), SEEDS)
            measurement = {"score_id": score_id, "inputs": score_inputs, "cd_o_cm": mean,
                           "seed_std_cm": spread, "candidate_faces": len(faces), "candidate_vertices": len(vertices),
                           "budget_surface_samples_finite": True}
            write_json(score_cache, measurement)
            context["new_scores"] += 1
        result = {"identity": trial_id, "rule": rule["name"], "parameter": parameter, "object": name,
                  "shape": shape, "candidate": candidate.relative_to(ROOT).as_posix(), "geometry": geometry,
                  "run": (directory / "run.json").relative_to(ROOT).as_posix(), **measurement}
        write_json(directory / "result.json", result)
    write_json(cached, result)
    append_ledger(output / "ledger.jsonl", result)
    context["logged"].add(trial_id)
    print(f"{rule['name']:20s} {str(parameter):5s} {name:20s} {result['cd_o_cm']:.5f} cm", flush=True)
    return result


def ranked_table(summary: dict) -> str:
    header = ("| Rule | Chosen on A → scored B | Chosen on B → scored A | Cross-fitted mean (cm) | "
              "In-sample mean (cm) | Pair-weighted mean (cm) | Worst regression (cm) | Gain / object / pair gates | Adopt |\n"
              "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    rows = [header]
    for row in summary["ranking"]:
        gates = " / ".join("pass" if value else "fail" for value in row["gates"].values())
        rows.append(f"| {row['rule']} | {row['chosen_on_fold']['A']} | {row['chosen_on_fold']['B']} | "
                    f"{row['cross_fitted_mean_cm']:.5f} | {row['in_sample_mean_cm']:.5f} | "
                    f"{row['pair_weighted_mean_cm']:.5f} | {row['worst_held_out_regression_cm']:+.5f} | "
                    f"{gates} | {'yes' if row['adopt'] else 'no'} |")
    return "\n".join(rows).replace("| None |", "| — |")


def plot(summary_path: str, target: str) -> None:
    """Render the completed measurements using the existing matplotlib environment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(Path(summary_path).read_text())
    x = np.arange(len(NAMES))
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    series = [("Box v1", summary["baseline_per_object_cm"])]
    series += [(row["rule"], row["per_object_cm"]) for row in summary["ranking"]]
    for i, (label, values) in enumerate(series):
        ax.bar(x + (i - 1.5) * 0.2, [values[n] for n in NAMES], 0.2, label=label)
    ax.set(xticks=x, xticklabels=[name.replace("_", " ") for name in NAMES],
           xlabel="Object (public scan)", ylabel="CD-O (cm; lower is better)",
           title="Mesh rules: cross-fitted development estimates against public scans")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def write_report(summary: dict, directory: Path) -> None:
    """Write the ranked and per-object tables with an independently runnable figure."""
    write_json(directory / "summary.json", summary)
    baseline = summary["baseline_mean_cm"]
    text = ["# Mesh shape rules", "", "Development estimates against the 10 public scans; scans are measurement-only.",
            f"Box v1: **{baseline:.5f} cm** per object; **{summary['baseline_pair_mean_cm']:.5f} cm** "
            "over the 26 scanned pairs. Kit seed episodes: 1, 3, 5.", "", ranked_table(summary), "",
            "Parameters are selected on the named fold and scored on the opposite fold. In-sample means "
            "select a single parameter on all 10 objects (optimistic). Ties use the original queue order. "
            "Pair weighting uses each object's count in the v1 roster, with its cross-fitted score.", "",
            "R2 tunes only the aspect threshold; handle height is fixed at 0.6, width at 0.15 of the short "
            "extent. This tests when the measured long-axis excess should become a handle. "
            "R3 uses a box head and handle width 0.2 of the short extent; the height remains measured.", "",
            "| Object | Fold | Pairs | Box v1 (cm) | " + " | ".join(r["rule"] + " (cm)" for r in summary["ranking"]) + " |",
            "| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in summary["ranking"]) + " |"]
    for name in NAMES:
        fold = next(f for f, names in FOLDS.items() if name in names)
        values = [summary["baseline_per_object_cm"][name], *(r["per_object_cm"][name] for r in summary["ranking"])]
        text.append(f"| {name} | {fold} | {summary['pair_counts'][name]} | " +
                    " | ".join(f"{value:.5f}" for value in values) + " |")
    text += ["", "![Per-object CD-O](rules-cdo.png)", "",
             "Figure: box v1 versus each rule's cross-fitted per-object CD-O (cm). "
             "These are development estimates against the public scans, not an independent final assessment.", "",
             "Concatenated components remain individually closed; overlaps leave internal faces. "
             "The real run checks the kit's 4096-row budget and finite area-weighted sampling, "
             "not Boolean union or physical simulation validity. The unscanned spoon is not scored or changed.", "",
             f"This invocation: {summary['new_scores']} new mesh scores, {summary['reused_scores']} reused mesh scores, "
             f"{summary['resumed']} completed object trials resumed.", "",
             "Every parameter/object result, including losers, is in `ledger.jsonl`; full trial inputs, "
             "source hashes and provenance are in `trials/<identity>/<run-id>/run.json`."]
    (directory / "summary.md").write_text("\n".join(text) + "\n")
    subprocess.run([str(ROOT / ".runtime/linux-cuda/bin/python"), "-B", "-c",
                    "import sys; from track3.features.mesh_extents.rules import plot; plot(*sys.argv[1:])",
                    str(directory / "summary.json"), str(directory / "rules-cdo.png")], check=True, cwd=ROOT)


def run(queue_path: Path) -> Path:
    """Run the editable queue, recording every object trial before fold selection."""
    runtime_flag(FEATURE, "build_authorized", "Mesh rule build and evaluation")
    queue_path = project_path(ROOT, queue_path)
    rules = read_queue(queue_path)
    output = output_path(ROOT, OUTPUT)
    for name in ("cache", "scores", "meshes", "trials"):
        (output / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(MPLCONFIGDIR=str(output_path(ROOT, ".cache/matplotlib")),
                      XDG_CACHE_HOME=str(output_path(ROOT, ".cache/xdg")),
                      TMPDIR=str(output_path(ROOT, ".runtime/tmp")), PYTHONDONTWRITEBYTECODE="1")
    from scripts.track3 import local_cdo
    import numpy as np

    record = run_fields(ROOT, "mesh_extents", "rules", {"queue": queue_path.relative_to(ROOT).as_posix()},
                        inputs=[queue_path, PROTOCOL], parents=[BUILD / "run.json"], provenance="public_asset_diagnostic")
    parent = read_run(ROOT, BUILD / "run.json", kind="mesh_extents.build", allowed=("video_only",))
    pooled = json.loads(artifact((ROOT / BUILD / "run.json", parent), "objects.json", record).read_text())
    objects = pooled["objects"]
    for name in NAMES:
        row = objects[name]
        if any(ep["episode_index"] in (41, 43, 44) for ep in row["episodes"]):
            raise ValueError("Excluded episode in pooled measurements")
        dims = np.asarray(row["extents_m"])
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0) or dims[0] < dims[1]:
            raise ValueError(f"Invalid video extents: {name}")
        artifact((ROOT / BUILD / "run.json", parent), row["file"], record)
    pairs = [pair.split(":") for pair in pooled["pairs_written"]]
    if any(int(episode) in (41, 43, 44) for episode, _ in pairs):
        raise ValueError("Excluded episode in v1 roster")
    counts = Counter(name for _, name in pairs if name in NAMES)
    if sum(counts.values()) != 26 or set(counts) != set(NAMES):
        raise ValueError("Expected the fixed 26 scanned pairs")
    scoring_paths = [Path(local_cdo.__file__), local_cdo.KIT / "v2dlb/mesh_budget.py",
                     local_cdo.KIT / "metric_code/track_3/CD-O.py"]
    sources = [input_file(ROOT, p) for p in [Path(__file__), FEATURE / "_shapes.py", FEATURE / "_rule_scores.py"]]
    scoring = {"sources": [input_file(ROOT, p) for p in scoring_paths], "seed_episodes": SEEDS,
               "budget": local_cdo.BUDGET, "packages": {p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "trimesh", "fast-simplification")}}
    logged = set()
    ledger = output / "ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                logged.add(json.loads(line)["identity"])
            except (json.JSONDecodeError, KeyError):
                print("Ignoring interrupted ledger line; completed cache remains authoritative", file=sys.stderr)
    context = {"output": output, "objects": objects, "queue": queue_path, "sources": sources,
               "protocol": input_file(ROOT, PROTOCOL), "scoring": scoring, "logged": logged,
               "references": {n: input_file(ROOT, local_cdo.scan_path(n)) for n in NAMES},
               "reference_arrays": {}, "new_scores": 0, "reused_scores": 0, "resumed": 0}
    record.update(rule_sources=sources, scoring=scoring, references=context["references"],
                  protocol=context["protocol"], categories=CATEGORIES, folds=FOLDS)
    directory = output / "runs" / new_run_id()
    with recorded(directory, record):
        baseline_rule = {"name": "box-v1 baseline", "parameter": None, "grid": [None],
                         "shapes": dict.fromkeys(CATEGORIES, "box")}
        baseline_rows = [evaluate_object(baseline_rule, None, name, context) for name in NAMES]
        baseline = {r["object"]: r["cd_o_cm"] for r in baseline_rows}
        historical = json.loads((ROOT / "outputs/mesh_eval/d1-ctx02/v1-box.json").read_text())
        if any(abs(baseline[r["object"]] - r["cd_o_cm"]) > 0.000051 for r in historical["objects"]):
            raise ValueError("Recomputed box differs from the fixed v1 baseline beyond JSON rounding")
        ranking = []
        for rule in rules:
            trials = [{"parameter": p, "objects": [evaluate_object(rule, p, name, context) for name in NAMES]}
                      for p in rule["grid"]]
            ranking.append(select(rule, trials, baseline, counts))
        ranking.sort(key=lambda row: row["cross_fitted_mean_cm"])
        summary = {"ranking": ranking, "baseline_per_object_cm": baseline,
                   "baseline_mean_cm": float(np.mean(list(baseline.values()))),
                   "baseline_pair_mean_cm": float(np.average([baseline[n] for n in NAMES], weights=[counts[n] for n in NAMES])),
                   "pair_counts": dict(counts), "protocol": context["protocol"], "queue": input_file(ROOT, queue_path),
                   "new_scores": context["new_scores"], "reused_scores": context["reused_scores"],
                   "resumed": context["resumed"], "baseline_matches_v1_rounding": True}
        write_report(summary, directory)
    backup = ROOT / ".runtime/tmp/backup" / new_run_id("mesh-rules")
    for name in ("summary.json", "summary.md", "rules-cdo.png"):
        target = output / name
        if target.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup / name)
        shutil.copy2(directory / name, target)
    print(ranked_table(summary), flush=True)
    print(f"Summary: {output / 'summary.md'}", flush=True)
    return directory / "run.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    run(args.queue)


if __name__ == "__main__":
    main()
