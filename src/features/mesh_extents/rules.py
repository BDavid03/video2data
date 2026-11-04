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
import tempfile
import time

from src.paths import output_path, project_path
from src.provenance import input_file, read_run, run_fields, sha256_file, write_json
from src.runs import ROOT, artifact, new_run_id, recorded, runtime_flag

from ._render import plot, render_candidate, render_video_comparison
from ._rule_scores import CATEGORIES, FOLDS, NAMES, select, shape_for
from ._union import mesh_for

FEATURE = Path(__file__).parent
BUILD = Path("data/outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff")
PROTOCOL = Path("docs/chain/mesh-selection-protocol.md")
OUTPUT = Path("data/outputs/mesh_eval/rules")
SEEDS = [1, 3, 5]
ROUND3_QUEUE_SHA256 = "5644a29dd2432feea4fb1bda6b4f156b3559e193d84d289cb97fdb602456edb3"
ROUND2_QUEUE_SHA256 = "3b21c00b7e215c5af027ffe2e09f28d7674fc1e48d48926c5751f9698fea05aa"
ROUND4_QUEUE_SHA256 = "2aa73fd03ef2d9199daa954a4442bd3b04b2f16ae9e1b9a49a3ba7b1ced38776"


ROUND_TEXT = {
    1: {
        "rules": "R2 tunes only the aspect threshold; handle height is fixed at 0.6, width at 0.15 of the short extent. R3 uses a box head and handle width 0.2 of the short extent; height remains measured.",
        "adoption": "",
        "spoon": "The unscanned spoon is not scored or changed.",
        "verdict": "Adopt",
    },
    2: {
        "rules": "R4 moves only video-confirmed rim-flat handles. R5 adds open vessel shells, and R6 adds video-counted rack pegs. R5/R6 tune wall/radius over [0.02, 0.05, 0.10]; threshold stays 1.3.",
        "adoption": "Choosing among three results on ten reused scans is optimistic. Development eligibility is provisional: production build/pack and the queue's exact packed veto must pass before adoption; no retuning or rule switching after the veto.",
        "spoon": "The unscanned spoon is not scored or changed.",
        "verdict": "Development eligible (veto pending)",
    },
    3: {
        "rules": "R7 rebuilds the rack; R8 applies one measured head/grip policy to all thin tools; R9 uses both exact components. There are no fitted parameters. R9 requires independently eligible R8.",
        "adoption": "Development eligibility is provisional until the queue's exact packed veto passes; no retuning or rule switching after the veto. Round 3 follows two adaptive rounds on the same ten scans and prior scan-render exposure. Cross-fitting covers declared parameters only; a third round raises selection optimism further. All results, including the packed veto, are development evidence, not an independent estimate.",
        "spoon": "The unscanned spoon is not scored; its profile changes only through independently eligible R8.",
        "verdict": "Development eligible (veto pending)",
    },
    4: {
        "rules": "C11 = v4 + R11: exact exterior union of the R7 rack boxes; every other object retains v4 geometry. No fitted parameter. R10 failed its video-only kill test; C10/C12 were dropped before scoring.",
        "adoption": "Fourth adaptive round on the same ten public scans: selection optimism is unknown. Development eligibility is provisional until the single-winner packed-v4 veto; failure keeps v4 without retuning or runner-up substitution.",
        "spoon": "The unscanned spoon retains v4 bytes and is not scored.",
        "verdict": "Development eligible (veto pending)",
    },
}


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_queue(path: Path) -> dict:
    """Read one admitted queue, checking frozen round-2/3/4 bytes exactly."""
    data = path.read_bytes()
    queue = json.loads(data)
    rules = queue["rules"]
    if queue.get("round") == 4:
        if hashlib.sha256(data).hexdigest() != ROUND4_QUEUE_SHA256:
            raise ValueError("Round-4 queue differs from its frozen hash")
        return queue
    if queue.get("round") == 3:
        if hashlib.sha256(data).hexdigest() != ROUND3_QUEUE_SHA256:
            raise ValueError("Round-3 queue differs from its frozen hash")
        return queue
    if queue.get("round") == 2:
        if hashlib.sha256(data).hexdigest() != ROUND2_QUEUE_SHA256:
            raise ValueError("Round-2 queue differs from its committed frozen hash")
        return queue
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
    return queue


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


def check_mesh(mesh, dims) -> dict:
    """Check component closure, positive volumes, budget and video extents."""
    import numpy as np

    parts = mesh.split(only_watertight=False)
    if not parts or not all(part.is_watertight and part.is_volume for part in parts):
        raise ValueError("Each mesh component must be watertight and a positive volume")
    if len(mesh.vertices) >= 4096 or len(mesh.faces) >= 4096:
        raise ValueError("Candidate reaches the kit simplification budget")
    if not np.allclose(mesh.extents, dims, rtol=1e-6, atol=0):
        raise ValueError(f"Candidate extents {mesh.extents} differ from video extents {dims}")
    return {"watertight": bool(mesh.is_watertight), "volume": bool(mesh.is_volume),
            "vertices": len(mesh.vertices), "faces_before_budget": len(mesh.faces),
            "closed_components": len(parts), "components_watertight_volume": True,
            "extents": mesh.extents.tolist(),
            "max_relative_extent_error": float(np.max(np.abs(mesh.extents / np.asarray(dims) - 1)))}


def build_candidate(rule: dict, parameter, name: str, dims, original: Path, directory: Path,
                    attributes: dict | None = None) -> tuple:
    """Build solely from video extents, keeping existing box files byte-for-byte."""
    shape = shape_for(rule, name, dims, parameter, attributes)
    if shape == "box":
        return shape, original, None
    mesh = mesh_for(shape, rule, name, dims, parameter, attributes or {})
    geometry = check_mesh(mesh, dims)
    data = mesh.export(file_type="ply")
    target = directory / (hashlib.sha256(data).hexdigest() + ".ply")
    if not target.exists():
        target.write_bytes(data)
    return shape, target, geometry


def evaluate_object(rule: dict, parameter, name: str, context: dict) -> dict:
    """Cache one rule/parameter/object trial and reuse identical mesh evaluations."""
    from src.features.mesh_eval import local_cdo

    output = context["output"]
    row = context["objects"][name]
    shape, candidate, geometry = build_candidate(rule, parameter, name, row["extents_m"],
                                                ROOT / context["build"] / row["file"], output / "meshes",
                                                context.get("attributes"))
    mesh_hash = sha256_file(candidate)
    if context.get("frozen_meshes") and mesh_hash != context["frozen_meshes"][name]:
        raise ValueError(f"Scoring mesh differs from frozen geometry: {name}")
    score_inputs = {"candidate_sha256": mesh_hash, "reference": context["references"][name],
                    "scoring": context["scoring"]}
    score_id = identity(score_inputs)
    request = {"rule": {key: value for key, value in rule.items() if key != "grid"},
               "parameter": parameter, "object": name, "extents_m": row["extents_m"], "shape": shape,
               "score_id": score_id, "sources": context["sources"], "protocol": context["protocol"]}
    if context.get("round2_inputs"):
        request["round2_inputs"] = context["round2_inputs"]
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
                        inputs=[context["queue"], PROTOCOL, candidate, local_cdo.scan_path(name),
                                *context.get("extra_inputs", [])],
                        parents=[context["build"] / "run.json"], provenance="public_asset_diagnostic")
    if time.monotonic() > context["deadline"]:
        raise TimeoutError("Queue time budget exhausted; resume the same command")
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
    """Name the estimation method and every applicable development gate explicitly."""
    ranking = summary["ranking"]
    fixed = all(row["parameter"] is None for row in ranking)
    score_label = "Fixed-rule mean (cm; all ten objects)" if fixed else "Development mean (cm)"
    gate_keys = list(dict.fromkeys(key for row in ranking for key in row["gates"]))
    gate_labels = []
    for key in gate_keys:
        if key.startswith("gain_gt_"):
            label = "Gain > " + key.removeprefix("gain_gt_").removesuffix("_cm") + " cm"
        elif key.startswith("worst_regression_le_"):
            label = "Worst object regression <= " + key.removeprefix("worst_regression_le_").removesuffix("_cm") + " cm"
        else:
            label = {"pair_mean_no_worse": "Pair mean no worse",
                     "shared_tool_rule_independently_eligible": "Spoon safeguard (R8 independently eligible)"}[key]
        gate_labels.append(label)
    headers = ["Rule", "Chosen on A → scored B", "Chosen on B → scored A", "Estimate",
               score_label, "In-sample mean (cm)", "Pair-weighted mean (cm)", "Worst regression (cm)",
               *gate_labels, ROUND_TEXT[summary.get("round", 1)]["verdict"]]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in ranking:
        no_parameter = row["parameter"] is None
        folds = ["fixed (no parameter)"] * 2 if no_parameter else [str(row["chosen_on_fold"][f]) for f in ("A", "B")]
        values = [row["rule"], *folds, "fixed-rule, all ten objects" if no_parameter else "cross-fitted",
                  f"{row['cross_fitted_mean_cm']:.5f}", f"{row['in_sample_mean_cm']:.5f}",
                  f"{row['pair_weighted_mean_cm']:.5f}", f"{row['worst_held_out_regression_cm']:+.5f}"]
        values += [("pass" if row["gates"][k] else "fail") if k in row["gates"] else "not applicable" for k in gate_keys]
        values.append("yes" if row["adopt"] else "no")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save_figure(figure, target: Path, dpi: int) -> None:
    """Persist and release a figure owned by this handler."""
    import matplotlib.pyplot as plt

    try:
        figure.savefig(target, dpi=dpi)
    finally:
        plt.close(figure)


def _plot_report(summary_path: str, target: str) -> None:
    """Load the report and save its figure in the plotting interpreter."""
    import matplotlib

    matplotlib.use("Agg")
    _save_figure(plot(json.loads(Path(summary_path).read_text())), Path(target), 160)


def write_report(summary: dict, directory: Path) -> None:
    """Write the ranked and per-object tables with an independently runnable figure."""
    write_json(directory / "summary.json", summary)
    baseline = summary["baseline_mean_cm"]
    prose = ROUND_TEXT[summary.get("round", 1)]
    text = ["# Mesh shape rules", "", "Development estimates against the 10 public scans; scans are measurement-only.",
            f"{summary.get('baseline_label', 'Box v1')}: **{baseline:.5f} cm** per object; **{summary['baseline_pair_mean_cm']:.5f} cm** "
            "over the 26 scanned pairs. Kit seed episodes: 1, 3, 5.", "", ranked_table(summary), "",
            "Rules with a declared parameter select on the named fold and score the opposite fold. "
            "Fixed rules have no fitted parameter and use all ten objects. In-sample means select a single "
            "parameter on all ten objects where applicable. Ties use queue order; pair weights use roster counts.", "",
            prose["rules"], "",
            "| Object | Fold | Pairs | " + summary.get("baseline_label", "Box v1") + " (cm) | " + " | ".join(r["rule"] + " (cm)" for r in summary["ranking"]) + " |",
            "| --- | --- | ---: | ---: | " + " | ".join("---:" for _ in summary["ranking"]) + " |"]
    for name in NAMES:
        fold = next(f for f, names in FOLDS.items() if name in names)
        values = [summary["baseline_per_object_cm"][name], *(r["per_object_cm"][name] for r in summary["ranking"])]
        text.append(f"| {name} | {fold} | {summary['pair_counts'][name]} | " +
                    " | ".join(f"{value:.5f}" for value in values) + " |")
    text += ["", "Seed spread: population standard deviation across kit seeds 1/3/5, per object (cm).",
             "| Rule | Mean seed SD (cm) | Maximum seed SD (cm) |", "| --- | ---: | ---: |"]
    for row in summary["ranking"]:
        spread = row["seed_spread_cm"]
        text.append(f"| {row['rule']} | {spread['mean']:.8f} | {spread['max']:.8f} |")
    if prose["adoption"]:
        text += ["", prose["adoption"]]
    text += ["", "![Per-object CD-O](rules-cdo.png)", "",
             f"Figure: {summary.get('baseline_label', 'Box v1')} versus each rule's development per-object CD-O (cm). "
             "These are development estimates against the public scans, not an independent final assessment.", "",
             ("R11 removes internal rack faces; other concatenated shapes can retain overlaps. "
              if summary.get("round") == 4 else "Concatenated components remain individually closed; overlaps leave internal faces. ") +
             "The real run checks the kit's 4096-row budget and finite area-weighted sampling, " +
             "not Boolean union or physical simulation validity. " + prose["spoon"], "",
             f"This invocation: {summary['new_scores']} new mesh scores, {summary['reused_scores']} reused mesh scores, "
             f"{summary['resumed']} completed object trials resumed.", "",
             "Every parameter/object result, including losers, is in `ledger.jsonl`; full trial inputs, "
             "source hashes and provenance are in `trials/<identity>/<run-id>/run.json`."]
    (directory / "summary.md").write_text("\n".join(text) + "\n")
    subprocess.run([str(ROOT / "data/.runtime/linux-cuda/bin/python"), "-B", "-c",
                    "import sys; from src.features.mesh_extents.rules import _plot_report; _plot_report(*sys.argv[1:])",
                    str(directory / "summary.json"), str(directory / "rules-cdo.png")], check=True, cwd=ROOT)


def incumbent_scores(queue: dict) -> tuple[dict, list]:
    """Read the exact fixed-threshold incumbent from its completed ledger rows."""
    incumbent = queue["incumbent"]
    if queue.get("round") in (3, 4):
        return incumbent["per_object_cm"], [{"object": n, "seed_std_cm": v}
                                           for n, v in incumbent["seed_std_cm"].items()]
    rows = {}
    path = project_path(ROOT, incumbent["ledger"])
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["rule"] == incumbent["rule"] and row["parameter"] == incumbent["parameter"]:
            if row["object"] in rows and rows[row["object"]] != row:
                raise ValueError("Conflicting incumbent ledger entries")
            rows[row["object"]] = row
    if set(rows) != set(NAMES):
        raise ValueError("Incumbent must cover exactly ten scanned objects")
    if any(row["inputs"]["scoring"]["seed_episodes"] != SEEDS for row in rows.values()):
        raise ValueError("Incumbent seeds must be 1/3/5")
    return {name: row["cd_o_cm"] for name, row in rows.items()}, list(rows.values())


def frozen_inputs_for(queue: dict) -> tuple[dict, dict, list]:
    """Read frozen video attributes and the matched incumbent inputs."""
    if queue.get("round") not in (2, 3, 4):
        return {}, {}, []
    path = project_path(ROOT, queue["attributes"])
    attributes = json.loads(path.read_text())
    if attributes["provenance"] != "video_only":
        raise ValueError("Attributes must have video-only provenance")
    for row in attributes["objects"].values():
        for evidence in row["evidence"]:
            if evidence["episode"] in (41, 43, 44):
                raise ValueError("Excluded episode in video attributes")
    baseline, _ = incumbent_scores(queue)
    if queue.get("round") in (3, 4):
        base_path = project_path(ROOT, queue["base_attributes"])
        return attributes["objects"], {"attributes": input_file(ROOT, path),
                                        "base_attributes": input_file(ROOT, base_path),
                                        "incumbent": queue["incumbent"]}, [path, base_path]
    inputs = [path, project_path(ROOT, queue["incumbent"]["ledger"])]
    hashes = {"attributes": input_file(ROOT, path), "incumbent": queue["incumbent"],
              "incumbent_scores": baseline, "incumbent_ledger": input_file(ROOT, inputs[1])}
    return attributes["objects"], hashes, inputs


def production_dry_builds(queue_path: Path, queue: dict, parent: dict, sources: dict, results: list, output: Path) -> list:
    """Resume production builds and compare their bytes with the dry meshes."""
    from .build import BuildRequest, build

    entries = []
    folder = output / "production"
    folder.mkdir(exist_ok=True)
    for rule in queue["rules"]:
        request = {**parent["request"], "rule": rule["name"], "rule_parameter": rule["grid"][0],
                   "attributes_file": queue["attributes"], "rule_queue": queue_path.relative_to(ROOT).as_posix()}
        key = identity({"request": request, "sources": sources, "builder": sha256_file(FEATURE / "build.py"),
                        "attributes": sha256_file(ROOT / queue["attributes"]), "protocol": sha256_file(ROOT / PROTOCOL)})
        cache = folder / (key + ".json")
        if cache.exists():
            entry = json.loads(cache.read_text())
            run = ROOT / entry["build_run"]
        else:
            run = build(BuildRequest(**request))
        record = read_run(ROOT, run, kind="mesh_extents.build", allowed=("video_only",))
        objects = json.loads(artifact((run, record), "objects.json", {"inputs": []}).read_text())["objects"]
        checks = [r for r in results if r["rule"] == rule["name"]]
        if any(objects[r["object"]]["sha256"] != r["mesh_sha256"] for r in checks):
            raise ValueError("Production and dry candidate bytes differ")
        entry = {"rule": rule["name"], "build_run": run.relative_to(ROOT).as_posix(),
                 "object_count": len(objects), "pairs_written": record["pairs_written"],
                 "all_dry_meshes_identical": True,
                 "unchanged_objects": [r["object"] for r in checks if not r["changed_from_incumbent"]]}
        if not cache.exists():
            write_json(cache, entry)
        entries.append(entry)
    return entries


def build_dry_trial(trial: dict, attributes: dict, output: Path) -> dict:
    """Build, check and persist one video-only candidate and its three-view render."""
    rule, parameter, name, dims = (trial[k] for k in ("rule", "parameter", "object", "extents_m"))
    shape = shape_for(rule, name, dims, parameter, attributes)
    mesh = mesh_for(shape, rule, name, dims, parameter, attributes)
    geometry = check_mesh(mesh, dims)
    data = mesh.export(file_type="ply")
    mesh_hash = hashlib.sha256(data).hexdigest()
    target = output / "meshes" / (mesh_hash + ".ply")
    if not target.exists():
        target.write_bytes(data)
    changed = mesh_hash != trial["incumbent_mesh_sha256"]
    render_id = identity({"mesh": mesh_hash, "renderer": trial["sources"]["_render.py"]})
    render = output / "renders" / (render_id + ".png")
    if changed and not render.exists():
        import matplotlib

        matplotlib.use("Agg")
        label = f"{name} / {shape}" + (f" / wall-radius {parameter}" if shape == "vessel_shell" else "")
        _save_figure(render_candidate(mesh, label), render, 140)
    comparison = None
    attrs = attributes.get(name, {})
    profile = attrs.get("rack_profile", attrs.get("profile"))
    if changed and profile:
        evidence = next(e for e in attrs["evidence"] if e["attribute"] == ("rack_profile" if "rack_profile" in attrs else "profile"))
        comparison = output / "renders" / (render_id + "-video.png")
        if not comparison.exists():
            from PIL import Image

            with Image.open(render) as candidate, Image.open(ROOT / evidence["path"]) as video:
                candidate.load()
                video.load()
                figure = render_video_comparison(candidate, video, name, profile.get("display_crop_px"))
                _save_figure(figure, comparison, 140)
    result = {"identity": identity(trial), "rule": rule["name"], "parameter": parameter, "object": name,
              "shape": shape, **geometry, "changed_from_incumbent": changed,
              "mesh": target.relative_to(ROOT).as_posix(), "mesh_sha256": mesh_hash,
              "render": render.relative_to(ROOT).as_posix() if changed else None,
              "video_comparison": comparison.relative_to(ROOT).as_posix() if comparison else None,
              "inputs": trial, "provenance": "video_only", "scoring": False}
    return result


def dry_build(queue_path: Path, output_override: Path | None = None) -> Path:
    """Resume all candidate builds and render changed meshes without scan access."""
    runtime_flag(FEATURE, "build_authorized", "Mesh rule dry build")
    queue_path = project_path(ROOT, queue_path)
    queue = read_queue(queue_path)
    rules = queue["rules"]
    if queue.get("round") not in (2, 3, 4):
        raise ValueError("Dry build requires a frozen round-2/3/4 queue")
    output = output_path(ROOT, Path(output_override or queue["output"]) / "dry")
    for folder in ("meshes", "renders", "trials", "runs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    os.environ.update(MPLCONFIGDIR=str(output), TMPDIR=str(output))
    tempfile.tempdir = str(output)
    attributes, hashes, extra_inputs = frozen_inputs_for(queue)
    build = Path(queue["build"])
    parent = read_run(ROOT, build / "run.json", kind="mesh_extents.build", allowed=("video_only",))
    request = {"queue": input_file(ROOT, queue_path), "round2_inputs": hashes, "dry_build": True}
    record = run_fields(ROOT, "mesh_extents", "rules_dry", request,
                        inputs=[queue_path, PROTOCOL, *extra_inputs], parents=[build / "run.json"])
    pooled = json.loads(artifact((ROOT / build / "run.json", parent), "objects.json", record).read_text())
    objects = pooled["objects"]
    if set(objects) != set(NAMES) | {"wooden_spoon"}:
        raise ValueError("Dry build must cover all eleven production objects")
    total = sum(len(rule["grid"]) for rule in rules) * len(objects)
    if total > queue["budget"]["max_candidate_builds"]:
        raise ValueError("Dry-build candidate count exceeds queue budget")
    if any(ep["episode_index"] in (41, 43, 44) for row in objects.values() for ep in row["episodes"]):
        raise ValueError("Excluded episode in video measurements")
    sources = {p.name: sha256_file(p) for p in (Path(__file__), FEATURE / "_shapes.py", FEATURE / "_rule_scores.py", FEATURE / "build.py", FEATURE / "_render.py")}
    if queue["round"] == 4:
        sources["_union.py"] = sha256_file(FEATURE / "_union.py")
    deadline = time.monotonic() + queue["budget"]["max_seconds"]
    results, resumed = [], 0
    ledger = output / "ledger.jsonl"
    logged = set()
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                logged.add(json.loads(line)["identity"])
            except (json.JSONDecodeError, KeyError):
                pass
    directory = output / "runs" / new_run_id()
    with recorded(directory, record):
        for rule in rules:
            for parameter in rule["grid"]:
                for name, row in objects.items():
                    import numpy as np

                    dims = np.asarray(row["extents_m"])
                    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0) or dims[0] < dims[1]:
                        raise ValueError(f"Invalid video extents: {name}")
                    trial = {"rule": rule, "parameter": parameter, "object": name, "extents_m": dims.tolist(),
                             "inputs": request, "sources": sources, "protocol": input_file(ROOT, PROTOCOL),
                             "incumbent_mesh_sha256": row["sha256"]}
                    trial_id = identity(trial)
                    cached = output / "trials" / f"{trial_id}.json"
                    if cached.exists():
                        result = json.loads(cached.read_text())
                        if not all((ROOT / result[key]).exists() for key in ("mesh", "render") if result.get(key)):
                            raise ValueError("Dry-build cache has missing artifacts")
                        resumed += 1
                    else:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Dry-build time budget exhausted; resume the same command")
                        result = build_dry_trial(trial, attributes, output)
                        write_json(cached, result)
                    if queue.get("candidate_mesh_sha256") and result["mesh_sha256"] != queue["candidate_mesh_sha256"][name]:
                        raise ValueError(f"Dry mesh differs from frozen geometry: {name}")
                    if trial_id not in logged:
                        append_ledger(ledger, result)
                        logged.add(trial_id)
                    results.append(result)
        production = production_dry_builds(queue_path, queue, parent, sources, results, output) if queue["round"] in (3, 4) else []
        write_json(directory / "production-builds.json", production)
        summary = {"production_builds": production, "scoring": False, "scan_access": False, "inputs": request, "results": results,
                   "candidate_count": len(results), "resumed": resumed, "new_builds": len(results) - resumed,
                   "unique_meshes": len({r["mesh_sha256"] for r in results}),
                   "unique_renders": len({r["render"] for r in results if r["render"]})}
        write_json(directory / "summary.json", summary)
        lines = [f"# Round-{queue['round']} candidate-only dry build", "", "No scan files or scoring code accessed.",
                 f"{len(results)} candidates; {resumed} resumed; {summary['unique_meshes']} distinct meshes; "
                 f"{summary['unique_renders']} three-view renders. Seed spread: not measured (no scoring).", "",
                 "| Rule | Parameter | Object | Shape | Vertices | Faces | Watertight / volume | Components | Max extent relative error | Render |",
                 "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |"]
        for r in results:
            render = f"[three views]({os.path.relpath(ROOT / r['render'], directory)})" if r["render"] else "unchanged"
            if r.get("video_comparison"):
                render += f" / [video]({os.path.relpath(ROOT / r['video_comparison'], directory)})"
            lines.append(f"| {r['rule']} | {r['parameter']} | {r['object']} | {r['shape']} | {r['vertices']} | "
                         f"{r['faces_before_budget']} | {r['watertight']} / {r['volume']} | {r['closed_components']} | "
                         f"{r['max_relative_extent_error']:.3g} | {render} |")
        (directory / "summary.md").write_text("\n".join(lines) + "\n")
    for name in ("summary.json", "summary.md"):
        shutil.copy2(directory / name, output / name)
    top_summary = output / "summary.md"
    text = (directory / "summary.md").read_text()
    top_summary.write_text(text.replace("../../renders/", "renders/"))
    print(f"Dry build: {len(results)} candidates, {resumed} resumed; {output / 'summary.md'}", flush=True)
    return directory / "run.json"


def run(queue_path: Path, output_override: Path | None = None) -> Path:
    """Run the editable queue, recording every object trial before fold selection."""
    runtime_flag(FEATURE, "build_authorized", "Mesh rule build and evaluation")
    queue_path = project_path(ROOT, queue_path)
    queue = read_queue(queue_path)
    rules = queue["rules"]
    build = Path(queue.get("build", BUILD))
    output = output_path(ROOT, output_override or queue.get("output", OUTPUT))
    attributes, round2_inputs, extra_inputs = frozen_inputs_for(queue)
    baseline_label = queue.get("incumbent", {}).get("label", "Box v1")
    for name in ("cache", "scores", "meshes", "trials"):
        (output / name).mkdir(parents=True, exist_ok=True)
    scratch = output / "runtime"
    scratch.mkdir(exist_ok=True)
    os.environ.update(MPLCONFIGDIR=str(scratch), XDG_CACHE_HOME=str(scratch),
                      TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1")
    tempfile.tempdir = str(scratch)
    from src.features.mesh_eval import local_cdo
    import numpy as np

    record = run_fields(ROOT, "mesh_extents", "rules", {"queue": queue_path.relative_to(ROOT).as_posix()},
                        inputs=[queue_path, PROTOCOL, *extra_inputs, *([queue["baseline_check"]] if "baseline_check" in queue else [])],
                        parents=[build / "run.json"], provenance="public_asset_diagnostic")
    parent = read_run(ROOT, build / "run.json", kind="mesh_extents.build", allowed=("video_only",))
    pooled = json.loads(artifact((ROOT / build / "run.json", parent), "objects.json", record).read_text())
    objects = pooled["objects"]
    for name in NAMES:
        row = objects[name]
        if any(ep["episode_index"] in (41, 43, 44) for ep in row["episodes"]):
            raise ValueError("Excluded episode in pooled measurements")
        dims = np.asarray(row["extents_m"])
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0) or dims[0] < dims[1]:
            raise ValueError(f"Invalid video extents: {name}")
        artifact((ROOT / build / "run.json", parent), row["file"], record)
    pairs = [pair.split(":") for pair in pooled["pairs_written"]]
    if any(int(episode) in (41, 43, 44) for episode, _ in pairs):
        raise ValueError("Excluded episode in v1 roster")
    counts = Counter(name for _, name in pairs if name in NAMES)
    if sum(counts.values()) != 26 or set(counts) != set(NAMES):
        raise ValueError("Expected the fixed 26 scanned pairs")
    scoring_paths = [Path(local_cdo.__file__), local_cdo.KIT / "v2dlb/mesh_budget.py",
                     local_cdo.KIT / "metric_code/track_3/CD-O.py"]
    sources = [input_file(ROOT, p) for p in [Path(__file__), FEATURE / "_shapes.py", FEATURE / "_rule_scores.py"]]
    if queue.get("round") == 4:
        sources.append(input_file(ROOT, FEATURE / "_union.py"))
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
               "reference_arrays": {}, "new_scores": 0, "reused_scores": 0, "resumed": 0, "build": build, "attributes": attributes,
               "round2_inputs": round2_inputs, "extra_inputs": extra_inputs,
               "frozen_meshes": queue.get("candidate_mesh_sha256", {}),
               "deadline": time.monotonic() + queue.get("budget", {}).get("max_seconds", 7200)}
    record.update(rule_sources=sources, scoring=scoring, references=context["references"],
                  protocol=context["protocol"], categories=CATEGORIES, folds=FOLDS, round2_inputs=round2_inputs)
    directory = output / "runs" / new_run_id()
    with recorded(directory, record):
        if queue.get("round") in (2, 3, 4):
            baseline, baseline_rows = incumbent_scores(queue)
        else:
            baseline_rule = {"name": "box-v1 baseline", "parameter": None, "grid": [None],
                             "shapes": dict.fromkeys(CATEGORIES, "box")}
            baseline_rows = [evaluate_object(baseline_rule, None, name, context) for name in NAMES]
            baseline = {r["object"]: r["cd_o_cm"] for r in baseline_rows}
            historical = json.loads(project_path(ROOT, queue["baseline_check"]).read_text())
            if any(abs(baseline[r["object"]] - r["cd_o_cm"]) > 0.000051 for r in historical["objects"]):
                raise ValueError("Recomputed box differs from the fixed v1 baseline beyond JSON rounding")
        ranking = []
        for rule in rules:
            trials = [{"parameter": p, "objects": [evaluate_object(rule, p, name, context) for name in NAMES]}
                      for p in rule["grid"]]
            result = select(rule, trials, baseline, counts, queue.get("gates"))
            if queue.get("round") == 4:
                result["gates"]["gain_gt_0.03_cm"] = result["cross_fitted_mean_cm"] < queue["gates"]["per_object_mean_strict_upper_cm"]
                result["gates"]["pair_mean_no_worse"] = result["pair_weighted_mean_cm"] <= queue["incumbent"]["pair_weighted_mean_cm"]
                result["adopt"] = all(result["gates"].values())
            spreads = {row["object"]: row["seed_std_cm"] for trial in trials for row in trial["objects"]
                       if trial["parameter"] == result["chosen_on_fold"]["B" if row["object"] in FOLDS["A"] else "A"]}
            result["seed_spread_cm"] = {"per_object": spreads, "mean": float(np.mean(list(spreads.values()))),
                                         "max": max(spreads.values())}
            ranking.append(result)
        if queue.get("eligibility"):
            safeguard = queue["eligibility"]
            eligible = next(r for r in ranking if r["rule"] == safeguard["requires_rule"])["adopt"]
            combined = next(r for r in ranking if r["rule"] == safeguard["combined_rule"])
            combined["gates"]["shared_tool_rule_independently_eligible"] = eligible
            combined["adopt"] = all(combined["gates"].values())
        ranking.sort(key=lambda row: row["cross_fitted_mean_cm"])
        summary = {"ranking": ranking, "baseline_per_object_cm": baseline,
                   "baseline_mean_cm": float(np.mean(list(baseline.values()))),
                   "baseline_pair_mean_cm": float(np.average([baseline[n] for n in NAMES], weights=[counts[n] for n in NAMES])),
                   "pair_counts": dict(counts), "protocol": context["protocol"], "queue": input_file(ROOT, queue_path),
                   "new_scores": context["new_scores"], "reused_scores": context["reused_scores"],
                   "resumed": context["resumed"], "baseline_matches_v1_rounding": queue.get("round", 1) == 1,
                   "baseline_label": baseline_label, "round": queue.get("round", 1),
                   "round2_inputs": round2_inputs,
                   "winner_pending_exact_veto": next((r["rule"] for r in ranking if r["adopt"]), None),
                   "baseline_seed_spread_cm": {r["object"]: r["seed_std_cm"] for r in baseline_rows}}
        write_report(summary, directory)
    backup = output / "backups" / new_run_id("mesh-rules")
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
    parser.add_argument("--output", type=Path, help="Write an isolated run and cache copy here")
    parser.add_argument("--dry-build", action="store_true", help="Build/check/render candidates without importing scoring code")
    args = parser.parse_args()
    if args.dry_build:
        dry_build(args.queue, args.output)
    else:
        run(args.queue, args.output)


if __name__ == "__main__":
    main()
