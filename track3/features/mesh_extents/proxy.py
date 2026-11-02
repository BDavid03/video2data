"""Frozen, resumable video-only silhouette diagnostic; dry mode never projects candidates."""

from collections import Counter
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import re
import time

from track3.paths import output_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, authorize_episode, checked_entry, new_run_id, read_json, recorded, runtime_flag, source_hashes
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
            if row["inputs"]["scoring"]["seed_episodes"] != [1, 3, 5]:
                raise ValueError(f"Recorded score seeds differ from 1/3/5: {ledger}:{line}")
            key = (row["object"], row["inputs"]["candidate_sha256"])
            digest = _input(row["candidate"], inputs, key[1])
            candidate = candidates.setdefault(key, {"object": key[0], "sha256": digest,
                "candidate": row["candidate"], "shape": row["shape"], "cd_o_cm": row["cd_o_cm"], "aliases": []})
            if candidate["cd_o_cm"] != row["cd_o_cm"]:
                raise ValueError("Identical candidate has inconsistent recorded CD-O")
            candidate["aliases"].append({"ledger": ledger, "line": line,
                                         "rule": row["rule"], "parameter": row["parameter"]})
    _input(ELLIPSOID_SCORES, inputs)
    ellipsoids = read_json(ELLIPSOID_SCORES)
    if ellipsoids["episodes_for_seeds"] != [1, 3, 5] or ellipsoids["oracle_proportions"] is not False:
        raise ValueError("Ellipsoid scores require non-oracle seed episodes 1/3/5")
    for row in ellipsoids["objects"]:
        candidate = _build_candidate(ELLIPSOIDS, row["object"], inputs)
        if row["status"] != "scored" or row["candidate"] != candidate["candidate"]:
            raise ValueError(f"Ellipsoid score does not name the scored frozen mesh: {row['object']}")
        _input(row["candidate"], inputs, candidate["sha256"])
        for key in ("candidate_sha256", "sha256"):
            if key in row and row[key] != candidate["sha256"]:
                raise ValueError(f"Ellipsoid recorded SHA differs from its build: {row['object']}")
        merged = candidates.setdefault((row["object"], candidate["sha256"]), {**candidate,
            "object": row["object"], "cd_o_cm": row["cd_o_cm"], "aliases": []})
        if merged["cd_o_cm"] != row["cd_o_cm"]:
            raise ValueError("Identical ellipsoid and ledger mesh have inconsistent recorded CD-O")
        merged["aliases"].append({"ledger": ELLIPSOID_SCORES, "rule": "v1 ellipsoid", "parameter": None})
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


def freeze_request(path, *, baseline=None):
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
    if baseline is not None:
        request["post_result_engineering"] = baseline
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


def _post_result_scope(request):
    """Check an engineering request against the hash-pinned completed baseline artifacts."""
    pointer, admission = request.data["post_result_engineering"], {"inputs": []}
    run_path = checked_entry(pointer["baseline_run"], admission)
    completed = read_run(ROOT, run_path, kind="mesh_extents.proxy")
    recorded_inputs = {entry["path"]: entry["sha256"] for entry in completed["inputs"]}
    if pointer["baseline_request"]["sha256"] != recorded_inputs[completed["request"]["manifest"]]:
        raise PermissionError("Baseline request does not match the completed run's recorded input hash")
    baseline = read_json(checked_entry(pointer["baseline_request"], admission))
    if any(recorded_inputs.get(path) != digest for path, digest in request.data["inputs"].items()):
        raise PermissionError("Engineering inputs differ from the completed baseline run")
    for key in ("inputs", "packages", "candidates", "spoon", "spoon_history", "views", "missing_views",
                "gates", "budget", "panel", "axis_convention"):
        if request.data[key] != baseline[key]:
            raise PermissionError(f"Post-result engineering changed the scientific definition: {key}")
    allowed = {f"track3/features/mesh_extents/{name}.py" for name in ("_proxy", "proxy", "_geometry")}
    if set(request.data["sources"]) != set(baseline["sources"]) or any(
            digest != baseline["sources"][path] for path, digest in request.data["sources"].items() if path not in allowed):
        raise PermissionError("Post-result source changes exceed the authorized scope")


def _admit(request):
    """Check frozen bytes, episode permissions and any recorded engineering baseline."""
    runtime = runtime_flag(FEATURE, "proxy_authorized", "Spoon proxy diagnostic")
    data = request.data
    missing = []
    for path, digest in {**data["inputs"], **data["sources"]}.items():
        try:
            _input(path, {}, digest)
        except FileNotFoundError:
            missing.append({"path": path, "reason": "Frozen input file is missing; diagnostic not executed"})
    for package, version in data["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen package version differs: {package}")
    for view in data["views"]:
        if view["episode"] in (41, 43, 44) or view["camera"] != "ego_cam_a":
            raise PermissionError("View outside the frozen diagnostic")
        from .measure import MeasureRequest

        authorize_episode(runtime, MeasureRequest(source_split=view["split"], episode_index=view["episode"]), "proxy")
        if any(row.get("path") == view["measure_run"] for row in missing):
            continue
        measure = read_run(ROOT, view["measure_run"], kind="mesh_extents.measure", allowed=("video_only",))
        if view["frame"] != measure["frames"][0]["frame_index"]:
            raise PermissionError("Proxy view must use the first frame of its assigned measure run")
    if data.get("post_result_engineering"):
        try:
            _post_result_scope(request)
        except FileNotFoundError as error:
            missing.append({"reason": f"Recorded baseline input is missing: {error}"})
    return identity({"request": data, "request_sha256": sha256_file(ROOT / request.path)}), missing


def load_view(view):
    """Read frozen depth/mask inputs and reuse the measure slice's support-plane refit."""
    import numpy as np
    from ._proxy import prepare_view

    measure = read_json(view["measure_run"])
    with np.load(ROOT / view["preprocessing"], allow_pickle=False) as arrays:
        k, input_valid = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].astype(bool)
    with np.load(ROOT / view["depth_npz"], allow_pickle=False) as arrays:
        z = arrays["depth_m"].astype(np.float64)
        valid = arrays["valid"] & arrays["input_valid"] & input_valid
    with np.load(ROOT / view["labels"], allow_pickle=False) as arrays:
        labels = arrays["label"].copy()
    return prepare_view(z, valid, k, input_valid, labels, view["cluster_ids"], view["global_plane"], measure["request"])


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


def _summary(data, results, missing=()):
    from ._proxy import average_candidates, average_spoon, rank_panel

    expected = {row["object"]: row["views"] for row in data["panel"]}
    summary = rank_panel(average_candidates(data["candidates"], results, expected), data["gates"])
    spoon, mean, reason = average_spoon(data["spoon"], data["views"], results, expected["wooden_spoon"])
    complete = not (missing or data["missing_views"]) and len(results) == sum(row["trials"] for row in data["panel"])
    summary["gates"]["complete_inputs"] = complete
    if not complete:
        summary["validated"] = False
    summary.update(spoon=spoon, spoon_mean_difference=mean, spoon_undefined_reason=reason,
                   gate_thresholds=data["gates"],
                   spoon_history_reference_only=data["spoon_history"], panel=data["panel"], trials=results,
                   missing_views=data["missing_views"], missing_inputs=list(missing),
                   verdict="PANEL CONCORDANCE PASSED" if summary["validated"] else "PROXY NOT VALIDATED FOR SPOON CD-O INFERENCE",
                   interpretation="No spoon CD-O gain or regression is established; packed D1 v4 is unchanged.",
                   incomplete_panel_reason=None if complete else "Missing frozen inputs, views or completed trials",
                   limitations="Reused development objects and construction views; small n, silhouette ties, "
                               "MoGe depth/mask distortion and placement error; gates are descriptive.")
    return summary


def _format(value):
    return "null" if value is None else f"{value:.3f}"


def _write_summary(summary, directory):
    write_json(directory / "summary.json", summary)
    gates, thresholds = summary["gates"], summary["gate_thresholds"]
    tools = [r for r in summary["objects"] if r["object"] in ("mini_dust_pan", "mini_sweeper")]
    passing_tools = sum(r["rho"] is not None and r["rho"] >= thresholds["rho"] for r in tools)
    gate_rows = [
        ("Mean tau-b", _format(summary["mean_tau_b"]), f">={_format(thresholds['mean_tau_b'])}", gates["mean_tau_b"]),
        ("Qualifying median rho", _format(summary["qualifying_median_rho"]), f">={_format(thresholds['rho'])}", gates["median_rho"]),
        ("Qualifying objects passing rho", f"{summary['qualifying_passing']}/{summary['qualifying_objects']}",
         f">={thresholds['passing_objects']}", gates["qualifying_count"]),
        ("Both scanned tools", f"{passing_tools}/{len(tools)}", "all", gates["both_tools"]),
        ("Required correlations defined", "all" if gates["all_required_correlations_defined"] else "incomplete",
         "all", gates["all_required_correlations_defined"]),
        ("Inputs complete", "yes" if gates["complete_inputs"] else "no", "yes", gates["complete_inputs"])]
    gate_rows += [(r["object"] + " rho", _format(r["rho"]), f">={_format(thresholds['rho'])}",
                   r["rho"] is not None and r["rho"] >= thresholds["rho"]) for r in tools]
    lines = ["# Spoon proxy diagnostic", "", summary["verdict"], "", summary["interpretation"], "",
             "## Pre-registered gates", "", "| Gate | Value | Requirement | Result |", "| --- | ---: | ---: | --- |"]
    lines += [f"| {name} | {value} | {requirement} | {'pass' if passed else 'fail'} |"
              for name, value, requirement, passed in gate_rows]
    lines += ["", "## Per-object concordance", "",
              "| Object | Candidates | Rho | Tau-b | Proxy ties | Undefined reason |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in summary["objects"]:
        lines.append(f"| {row['object']} | {row['candidates']} | {_format(row['rho'])} | "
                     f"{_format(row['tau_b'])} | {row['proxy_ties']} | {row['undefined_reason'] or ''} |")
    lines += ["", "## R8 versus box (report only)", "",
              "| Object | R8 minus box IoU | R8 above box | Tie | Undefined reason |", "| --- | ---: | --- | --- | --- |"]
    for row in summary["tool_r8_vs_box_report_only"]:
        above = "null" if row["r8_ranked_above_box"] is None else ("yes" if row["r8_ranked_above_box"] else "no")
        tie = "null" if row["tie"] is None else ("yes" if row["tie"] else "no")
        lines.append(f"| {row['object']} | {_format(row['r8_minus_box_iou'])} | {above} | {tie} | {row['undefined_reason'] or ''} |")
    lines += ["", "## Spoon silhouettes", "", "| Spoon view | v3 IoU | v4 IoU | Difference |", "| --- | ---: | ---: | ---: |"]
    for row in summary["spoon"]:
        lines.append(f"| {row['view']} | {_format(row['v3_box']['iou'])} | {_format(row['v4_bowl']['iou'])} | "
                     f"{_format(row['v4_minus_v3_iou'])} |")
    lines += ["", f"Spoon equal-view mean difference: {_format(summary['spoon_mean_difference'])}", "",
              "Full precision, both yaws, areas, placements, rankings and overlay paths are in summary.json.", "",
              "## Post-result reporting (not pre-registered)", ""]
    post = summary["post_result_reporting"]
    lines += [f"Pre-registered mean tau-b: {_format(summary['mean_tau_b'])}. Mean over the {post['objects']} objects "
              f"with at least four variants: {_format(post['mean_tau_b_ge_four_variants'])}; this adds no gate."]
    two_variant = [r for r in summary["objects"] if r["candidates"] == 2]
    lines += ["Two-variant objects: " + ", ".join(f"{r['object']} tau-b {_format(r['tau_b'])}" for r in two_variant) +
              ". With two untied variants, tau-b can only be +1 or -1."]
    pan = next((r for r in tools if r["object"] == "mini_dust_pan"), None)
    if pan is not None and pan["rho"] == thresholds["rho"]:
        lines += [f"Dust pan passes at exactly rho = {_format(pan['rho'])}, the frozen threshold."]
    lines += ["Hypothesis: " + post["hypothesis"], "", summary["limitations"]]
    reasons = [r["reason"] for r in [*summary["missing_views"], *summary["missing_inputs"]]]
    reasons += [r for r in [*summary["undefined_reasons"].values(), summary["spoon_undefined_reason"]] if r]
    if reasons:
        lines += ["", "## Unavailable inputs or values", "", *("- " + reason for reason in reasons)]
    if not summary["validated"]:
        lines += ["", "Proxy not validated for spoon CD-O inference; proxy says nothing about spoon CD-O; v4 unchanged."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")


def read_ledger(path):
    """Read complete rows; quarantine only a truncated final append before resuming."""
    rows, offset = [], 0
    lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            text = line.decode().rstrip()
            suffix = text[error.pos:]
            partial_literal = error.msg == "Expecting value" and any(token.startswith(suffix) for token in ("true", "false", "null"))
            partial_number = error.msg == "Expecting ',' delimiter" and re.search(r"(?:^|[:\[,]\s*)-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?)?$", text)
            truncated = error.pos >= len(text) or error.msg.startswith("Unterminated string") or partial_literal or partial_number
            if index != len(lines) - 1 or line.endswith((b"\n", b"\r")) or not text.startswith("{") or not truncated:
                raise ValueError(f"{path}:{index + 1}: undecodable ledger row") from error
            fragments = path.parent / "ledger-fragments"
            fragments.mkdir(exist_ok=True)
            (fragments / f"{new_run_id()}.bin").write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(offset)
        offset += len(line)
    return rows


def proxy(request, *, dry_run=False):
    """Run either target-only preparation or the committed, frozen scoring queue."""
    frozen_id, missing = _admit(request)
    data = request.data
    directory = output_path(ROOT, data["dry_output"] if dry_run else data["output"])
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / "ledger.jsonl"
    logged = {row["identity"]: row for row in read_ledger(ledger_path) if row["frozen_identity"] == frozen_id}
    record = run_fields(ROOT, "mesh_extents", "proxy_dry" if dry_run else "proxy", {"manifest": request.path,
                        "frozen_identity": frozen_id}, inputs=[request.path, *(p for p in data["inputs"]
                        if not any(row.get("path") == p for row in missing))],
                        provenance="public_asset_diagnostic")
    run_directory = directory / "runs" / new_run_id()
    started, results, masks = time.monotonic(), [], []
    with recorded(run_directory, record):
        for view in ([] if missing else data["views"]):
            try:
                loaded = load_view(view)
            except FileNotFoundError as error:
                missing.append({"view": view["id"], "reason": f"Frozen view input became unavailable: {error}"})
                continue
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
                    try:
                        row = {"frozen_identity": frozen_id, **_trial(candidate, view, loaded, request, frozen_id, directory)}
                    except FileNotFoundError as error:
                        missing.append({"view": view["id"], "candidate": candidate["sha256"],
                                        "reason": f"Frozen candidate input became unavailable: {error}"})
                        continue
                    append_ledger(ledger_path, row)
                results.append(row)
        if dry_run and not missing:
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
            summary = _summary(data, results, missing)
            _write_summary(summary, run_directory)
            _write_summary(summary, directory)
    return run_directory / "run.json"
