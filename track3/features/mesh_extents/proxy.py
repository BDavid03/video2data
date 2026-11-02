"""Frozen, resumable video-only silhouette diagnostic; dry mode never projects candidates."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from track3.paths import output_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, authorize_episode, new_run_id, read_json, recorded, runtime_flag, source_hashes
from .rules import append_ledger, identity

FEATURE = Path(__file__).parent
PROTOCOL = "docs/chain/mesh-selection-protocol.md"
BASE = "outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff"
ELLIPSOIDS = "outputs/mesh_extents/build/20261101T103839.520208Z-c31a2a92"
LEDGERS = ("outputs/mesh_eval/rules/ledger.jsonl", "outputs/mesh_eval/rules-round2/ledger.jsonl",
           "outputs/mesh_extents/rules-round3/ledger.jsonl")
ELLIPSOID_SCORES = "outputs/mesh_eval/d1-ctx02/v1-ellipsoid.json"
SPOON_BUILDS = {"v3_box": "outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295",
                "v4_bowl": "outputs/mesh_extents/build/20261102T051416.149421Z-f377e0a1"}


def _input(path, inputs, expected=None):
    """Admit one explicit local input without following reference-scan metadata."""
    path = str(path)
    if Path(path).is_absolute() or ".." in Path(path).parts or "/mesh/" in path:
        raise ValueError(f"Not a scan-free relative input: {path}")
    if path not in inputs:
        inputs[path] = input_file(ROOT, path)["sha256"]
    if expected is not None and inputs[path] != expected:
        raise ValueError(f"Input differs from its producer: {path}")
    return inputs[path]


def _build_candidate(build, name, inputs):
    """Read a recorded video-only build and admit just the requested object artifact."""
    _input(f"{build}/run.json", inputs)
    record = read_run(ROOT, f"{build}/run.json", kind="mesh_extents.build", allowed=("video_only",))
    _input(f"{build}/objects.json", inputs, record["artifacts"]["objects.json"])
    row = read_json(f"{build}/objects.json")["objects"][name]
    path = f"{build}/{row['file']}"
    digest = _input(path, inputs, row["sha256"])
    return {"candidate": path, "sha256": digest, "shape": row["shape"]}


def _freeze_panel(inputs):
    """Collect distinct recorded meshes and their existing CD-O scalars."""
    candidates = {}
    for ledger in LEDGERS:
        _input(ledger, inputs)
        for line, row in enumerate((ROOT / ledger).read_text().splitlines(), 1):
            row = json.loads(row)
            key = (row["object"], row["inputs"]["candidate_sha256"])
            digest = _input(row["candidate"], inputs, key[1])
            candidate = candidates.setdefault(key, {"object": key[0], "sha256": digest,
                "candidate": row["candidate"], "shape": row["shape"], "cd_o_cm": row["cd_o_cm"], "aliases": []})
            if candidate["cd_o_cm"] != row["cd_o_cm"]:
                raise ValueError("Identical candidate has inconsistent recorded CD-O")
            candidate["aliases"].append({"ledger": ledger, "line": line,
                                         "rule": row["rule"], "parameter": row["parameter"]})
    _input(ELLIPSOID_SCORES, inputs)
    for row in read_json(ELLIPSOID_SCORES)["objects"]:
        candidate = _build_candidate(ELLIPSOIDS, row["object"], inputs)
        candidates[(row["object"], candidate["sha256"])] = {**candidate, "object": row["object"],
            "cd_o_cm": row["cd_o_cm"], "aliases": [{"ledger": ELLIPSOID_SCORES, "rule": "v1 ellipsoid", "parameter": None}]}
    return list(candidates.values())


def _freeze_views(names, inputs):
    """Use the first measured frame of each assigned camera-a observation."""
    _input(f"{BASE}/run.json", inputs)
    views, missing_views = [], []
    for measure_path in read_json(f"{BASE}/run.json")["request"]["measure_runs"]:
        measure = read_run(ROOT, measure_path, kind="mesh_extents.measure", allowed=("video_only",))
        split, episode = measure["split"], measure["episode_index"]
        if episode in (41, 43, 44):
            raise PermissionError("Excluded episode in source manifest")
        objects = sorted(names.intersection(measure["request"]["objects"]))
        if split == "evaluation" and episode in (5, 15, 16):
            objects.append("wooden_spoon")
        if not objects:
            continue
        _input(measure_path, inputs)
        frame = measure["frames"][0]
        if frame["status"] != "completed":
            missing_views.extend({"object": name, "split": split, "episode": episode, "frame": frame["frame_index"],
                "measure_run": measure_path, "reason": "First assigned measurement is not completed; no substitution permitted"}
                for name in objects)
            continue
        index = frame["frame_index"]
        directory = Path(measure_path).parent.as_posix()
        labels, overlay = f"{directory}/labels_{index:06d}.npz", f"{directory}/overlay_{index:06d}.png"
        for artifact in (labels, overlay):
            _input(artifact, inputs, measure["artifacts"][Path(artifact).name])
        depth_path = measure["request"]["depth_run"]
        depth = read_run(ROOT, depth_path, kind="depth.generate", allowed=("video_only",))
        if (depth["split"], depth["episode_index"]) != (split, episode) or "ego_cam_a" not in depth["source_video"]:
            raise ValueError("Depth must match the frozen episode and camera a")
        _input(depth_path, inputs)
        source = next(f for f in depth["frames"] if f["frame_index"] == index and f["status"] == "completed")
        depth_dir = Path(depth_path).parent.as_posix()
        depth_npz, preprocessing = f"{depth_dir}/{source['artifact']}", f"{depth_dir}/preprocessing.npz"
        _input(depth_npz, inputs, source["sha256"])
        _input(preprocessing, inputs, depth["preprocessing_artifact_sha256"])
        for name in objects:
            views.append({"id": f"{split}-{episode:06d}-{name}", "object": name,
                "split": split, "episode": episode, "frame": index, "camera": "ego_cam_a",
                "measure_run": measure_path, "depth_run": depth_path, "depth_npz": depth_npz,
                "preprocessing": preprocessing, "labels": labels, "overlay": overlay,
                "cluster_ids": measure["request"]["objects"][name], "global_plane": frame["plane"]})
    return views, missing_views


def load_candidate(candidate):
    """Load a mesh and require x/y footprint ordering before any candidate projection."""
    import numpy as np
    import trimesh

    mesh = trimesh.load(ROOT / candidate["candidate"], force="mesh", process=False)
    x, y, z = mesh.extents
    if not np.isfinite((x, y, z)).all() or min(x, y, z) <= 0 or x < y * (1 - 1e-6):
        raise ValueError(f"Candidate violates x-long/y-short/z-up construction convention: {candidate['candidate']}")
    return mesh


def freeze_request(path):
    """Expand the pre-registered panel and hash its inputs, without projecting or scoring."""
    inputs = {}
    candidates = _freeze_panel(inputs)
    views, missing_views = _freeze_views({c["object"] for c in candidates}, inputs)
    spoon = [{**_build_candidate(build, "wooden_spoon", inputs), "object": "wooden_spoon", "label": label}
             for label, build in SPOON_BUILDS.items()]
    history = [{**_build_candidate(build, "wooden_spoon", inputs), "label": label, "reference_only": True}
               for label, build in (("primitive_box", BASE), ("primitive_ellipsoid", ELLIPSOIDS))]
    counts = Counter(c["object"] for c in candidates)
    view_counts = Counter(v["object"] for v in views)
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "opencv-python", "trimesh")}
    sources = source_hashes(FEATURE)
    sources.update({PROTOCOL: sha256_file(ROOT / PROTOCOL),
                    "track3/features/mesh_extents/runtime.json": sha256_file(FEATURE / "runtime.json")})
    request = {"version": 1, "output": "outputs/mesh_extents/spoon-proxy",
        "dry_output": "outputs/mesh_extents/spoon-proxy-dry", "inputs": inputs, "sources": sources,
        "packages": packages, "candidates": candidates, "spoon": spoon,
        "spoon_history": history, "views": views, "missing_views": missing_views,
        "axis_convention": [{"object": c["object"], "sha256": c["sha256"], "candidate": c["candidate"],
                             "extents_xyz": load_candidate(c).extents.tolist()} for c in [*candidates, *spoon]],
        "gates": {"mean_tau_b": 0.4, "rho": 0.6, "passing_objects": 6},
        "budget": {"max_trials": 149, "max_seconds": 7200},
        "panel": [{"object": n, "candidates": count, "views": view_counts[n], "trials": count * view_counts[n],
                   "missing_views": sum(v["object"] == n for v in missing_views)}
                  for n, count in sorted({**counts, "wooden_spoon": 2}.items())]}
    write_json(output_path(ROOT, path), request)
    return request


@dataclass(frozen=True)
class ProxyRequest:
    path: str
    data: dict

    @classmethod
    def from_file(cls, path):
        path = (ROOT / path).absolute().relative_to(ROOT).as_posix()
        return cls(path, read_json(path))


def _admit(request, dry_run):
    """Check frozen bytes and episode permissions, and enforce the pre-score commit."""
    runtime = runtime_flag(FEATURE, "proxy_authorized", "Spoon proxy diagnostic")
    data = request.data
    for path, digest in {**data["inputs"], **data["sources"]}.items():
        _input(path, {}, digest)
    for package, version in data["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen package version differs: {package}")
    for view in data["views"]:
        if view["episode"] in (41, 43, 44) or view["camera"] != "ego_cam_a":
            raise PermissionError("View outside the frozen diagnostic")
        measure = read_run(ROOT, view["measure_run"], kind="mesh_extents.measure", allowed=("video_only",))
        if view["frame"] != measure["frames"][0]["frame_index"]:
            raise PermissionError("Proxy view must use the first frame of its assigned measure run")
        from .measure import MeasureRequest

        authorize_episode(runtime, MeasureRequest(**measure["request"]), "proxy")
    if not dry_run:
        for path in [request.path, *data["sources"]]:
            committed = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != sha256_file(ROOT / path):
                raise PermissionError(f"Commit frozen request/protocol/code before scoring: {path}")
    return identity({"request": data, "request_sha256": sha256_file(ROOT / request.path)})


def load_view(view):
    """Read frozen depth/mask inputs and reuse the measure slice's support-plane refit."""
    import cv2
    import numpy as np
    from ._geometry import _local_plane, camera_points
    from ._proxy import placement

    measure = read_json(view["measure_run"])
    with np.load(ROOT / view["preprocessing"], allow_pickle=False) as arrays:
        k, input_valid = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].astype(bool)
    with np.load(ROOT / view["depth_npz"], allow_pickle=False) as arrays:
        z = arrays["depth_m"].astype(np.float64)
        valid = arrays["valid"] & arrays["input_valid"] & input_valid
    with np.load(ROOT / view["labels"], allow_pickle=False) as arrays:
        target = np.isin(arrays["label"], [i + 1 for i in view["cluster_ids"]]) & input_valid
    if not target.any():
        raise ValueError(f"Empty frozen target: {view['id']}")
    points = camera_points(z, k)
    ok = valid & np.isfinite(points).all(-1) & (z > 0)
    plane = view["global_plane"]
    normal, offset = np.asarray(plane["normal"]), plane["offset_m"]
    table = ok & (np.abs(points @ normal + offset) < measure["request"]["table_band_m"])
    count, labels = cv2.connectedComponents(table.astype(np.uint8), connectivity=4)
    sizes = np.bincount(labels[table], minlength=count)
    sizes[0] = 0
    table = labels == sizes.argmax()
    _, normal, offset, local = _local_plane(points, table, target, normal, offset, measure["request"]["ring_pixels"])
    pose = placement(points[target], normal, offset)
    pose_record = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in pose.items()}
    pose_record.update(normal=normal.tolist(), offset_m=float(offset), local_plane=local)
    return {"target": target, "input_valid": input_valid, "intrinsics": k, "pose": pose,
            "placement": pose_record}


def _overlay(view, loaded, path, projected=None):
    """Write target-only dry overlays or target/projected diagnostic overlays."""
    import cv2

    image = cv2.imread(str(ROOT / view["overlay"]))
    if image is None:
        raise ValueError("Missing recorded opening-frame overlay")
    image = image[:, :loaded["target"].shape[1]].copy()
    target = loaded["target"]
    image[target] = (image[target] * 0.5 + (0, 127, 0)).astype("uint8")
    if projected is not None:
        contours, _ = cv2.findContours(projected.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 0, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def _completed_result(run_path):
    """Read result bytes admitted by the completed trial, rather than trusting ledger copies."""
    completed = read_run(ROOT, run_path, kind="mesh_extents.proxy_trial")
    parent, admission = (ROOT / run_path, completed), {"inputs": []}
    result = read_json(artifact(parent, "result.json", admission))
    if result["identity"] != completed["request"]["identity"]:
        raise ValueError("Completed result identity differs from its request")
    artifact(parent, "overlay.png", admission)
    return result


def _trial(candidate, view, loaded, request, frozen_id, directory):
    """Resume or measure one candidate-view trial through a recorded slice operation."""
    key = identity({"frozen": frozen_id, "candidate": candidate, "view": view})
    cache = directory / "cache" / f"{key}.json"
    if cache.exists():
        result = _completed_result(json.loads(cache.read_text())["run"])
        if result["identity"] != key:
            raise ValueError("Cached trial identity differs from request")
        return result
    from ._proxy import compare

    inputs = [request.path, candidate["candidate"], view["labels"], view["depth_npz"], view["preprocessing"], view["overlay"]]
    record = run_fields(ROOT, "mesh_extents", "proxy_trial", {"identity": key, "candidate": candidate, "view": view},
                        inputs=inputs, parents=[view["measure_run"], view["depth_run"]], provenance="public_asset_diagnostic")
    cached = directory / "trials" / key / new_run_id()
    with recorded(cached, record):
        mesh = load_candidate(candidate)
        values, mask = compare(mesh.vertices, mesh.faces, loaded["pose"], loaded["intrinsics"],
                               loaded["input_valid"], loaded["target"])
        overlay = cached / "overlay.png"
        _overlay(view, loaded, overlay, mask)
        result = {"identity": key, "object": candidate["object"], "sha256": candidate["sha256"],
                  "view": view["id"], "run": (cached / "run.json").relative_to(ROOT).as_posix(),
                  "placement": loaded["placement"], "overlay": overlay.relative_to(ROOT).as_posix(),
                  "overlay_sha256": sha256_file(overlay), **values}
        write_json(cached / "result.json", result)
    write_json(cache, result)
    return result


def _summary(data, results):
    from ._proxy import rank_panel
    import numpy as np

    by_candidate = defaultdict(list)
    for row in results:
        by_candidate[(row["object"], row["sha256"])].append(row)
    objects = defaultdict(list)
    for candidate in data["candidates"]:
        rows = by_candidate[(candidate["object"], candidate["sha256"])]
        objects[candidate["object"]].append({**candidate,
            "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else None})
    summary = rank_panel(objects, data["gates"])
    spoon = []
    for view in (v for v in data["views"] if v["object"] == "wooden_spoon"):
        values = {c["label"]: next(r for r in by_candidate[("wooden_spoon", c["sha256"])] if r["view"] == view["id"])
                  for c in data["spoon"]}
        spoon.append({"view": view["id"], "v3_box": values["v3_box"], "v4_bowl": values["v4_bowl"],
                      "v4_minus_v3_iou": values["v4_bowl"]["iou"] - values["v3_box"]["iou"]})
    summary["gates"]["complete_inputs"] = not data["missing_views"]
    if data["missing_views"]:
        summary["validated"] = False
    summary.update(spoon=spoon, spoon_mean_difference=float(np.mean([r["v4_minus_v3_iou"] for r in spoon])),
                   spoon_history_reference_only=data["spoon_history"], panel=data["panel"], trials=results,
                   missing_views=data["missing_views"],
                   verdict="PANEL CONCORDANCE PASSED" if summary["validated"] else "PROXY NOT VALIDATED FOR SPOON CD-O INFERENCE",
                   interpretation="No spoon CD-O gain or regression is established; packed D1 v4 is unchanged.",
                   limitations="Reused development objects and construction views; small n, silhouette ties, "
                               "MoGe depth/mask distortion and placement error; gates are descriptive.")
    return summary


def _write_summary(summary, directory):
    write_json(directory / "summary.json", summary)
    lines = ["# Spoon proxy diagnostic", "", summary["verdict"], "", summary["interpretation"], "",
             summary["limitations"], "", f"Mean tau-b: {summary['mean_tau_b']}; median qualifying rho: "
             f"{summary['qualifying_median_rho']}; passing: {summary['qualifying_passing']}/8.", "",
             "| Object | Candidates | Rho | Tau-b | Proxy ties |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["objects"]:
        lines.append(f"| {row['object']} | {row['candidates']} | {row['rho']} | {row['tau_b']} | {row['proxy_ties']} |")
    lines += ["", "Descriptive gates: " + json.dumps(summary["gates"]), "", "R8 versus box, report only:"]
    lines += ["- " + json.dumps(row) for row in summary["tool_r8_vs_box_report_only"]]
    lines += ["", "| Spoon view | v3 IoU | v4 IoU | Difference |", "| --- | ---: | ---: | ---: |"]
    for row in summary["spoon"]:
        lines.append(f"| {row['view']} | {row['v3_box']['iou']} | {row['v4_bowl']['iou']} | {row['v4_minus_v3_iou']} |")
    lines += ["", f"Spoon equal-view mean difference: {summary['spoon_mean_difference']}", "",
              "Both yaws, target/projected areas, placements, rankings and overlay paths are in summary.json.",
              "Missing views: " + json.dumps(summary["missing_views"])]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")


def proxy(request, *, dry_run=False):
    """Run either target-only preparation or the committed, frozen scoring queue."""
    frozen_id = _admit(request, dry_run)
    data = request.data
    directory = output_path(ROOT, data["dry_output"] if dry_run else data["output"])
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / "ledger.jsonl"
    logged = {}
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted append is retained, never rewritten.
            if row["frozen_identity"] != frozen_id:
                raise ValueError("Output belongs to another frozen diagnostic")
            logged[row["identity"]] = row
    record = run_fields(ROOT, "mesh_extents", "proxy_dry" if dry_run else "proxy", {"manifest": request.path,
                        "frozen_identity": frozen_id}, inputs=[request.path, *data["inputs"]],
                        provenance="public_asset_diagnostic")
    run_directory = directory / "runs" / new_run_id()
    started, results, masks = time.monotonic(), [], []
    with recorded(run_directory, record):
        for view in data["views"]:
            loaded = load_view(view)
            if dry_run:
                path = directory / "overlays" / f"{view['id']}.png"
                _overlay(view, loaded, path)
                masks.append({"view": view["id"], "object": view["object"], "frame": view["frame"],
                              "mask_area_pixels": int(loaded["target"].sum()), "placement": loaded["placement"],
                              "overlay": path.relative_to(ROOT).as_posix()})
                continue
            candidates = data["spoon"] if view["object"] == "wooden_spoon" else data["candidates"]
            for candidate in (c for c in candidates if c["object"] == view["object"]):
                if len(results) >= data["budget"]["max_trials"] or time.monotonic() - started > data["budget"]["max_seconds"]:
                    raise TimeoutError("Frozen budget exhausted; resume the same request")
                key = identity({"frozen": frozen_id, "candidate": candidate, "view": view})
                if key in logged:
                    row = {"frozen_identity": frozen_id, **_completed_result(logged[key]["run"])}
                    if row["identity"] != key:
                        raise ValueError("Ledger trial identity differs from request")
                else:
                    row = {"frozen_identity": frozen_id, **_trial(candidate, view, loaded, request, frozen_id, directory)}
                    append_ledger(ledger_path, row)
                results.append(row)
        if dry_run:
            summary = {"frozen_identity": frozen_id, "panel": data["panel"], "views": masks,
                       "missing_views": data["missing_views"],
                       "candidate_projections": 0, "proxy_values_computed": False, "correlations_computed": False}
            write_json(run_directory / "summary.json", summary)
            write_json(directory / "summary.json", summary)
            text = ["# Spoon proxy target-only dry run", "", "No candidate projected; no IoU or correlation computed.", "",
                    "| View | Frame | Target pixels |", "| --- | ---: | ---: |"]
            text += [f"| {row['view']} | {row['frame']} | {row['mask_area_pixels']} |" for row in masks]
            text += ["", "Missing views: " + json.dumps(data["missing_views"])]
            (directory / "summary.md").write_text("\n".join(text) + "\n")
        else:
            expected = sum(row["trials"] for row in data["panel"])
            if len(results) != expected:
                raise ValueError("Incomplete frozen panel; no concordance summary permitted")
            summary = _summary(data, results)
            _write_summary(summary, run_directory)
            _write_summary(summary, directory)
    return run_directory / "run.json"
