"""Frozen CPU replay loop and one descriptive pair-analysis pass."""

from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path

from src.paths import output_path
from src.provenance import read_run, run_fields, sha256_file, write_json
from src.runs import (ROOT, artifact, authorize_episode, heavy_execution, new_run_id,
                         read_json, recorded, runtime_flag, save_npz)
from .measure import MeasureRequest
from .proxy import read_ledger

FEATURE = Path(__file__).parent


@dataclass(frozen=True)
class InsightsRequest:
    """Hold the pinned diagnostic manifest and its project-relative location."""
    path: str
    data: dict

    @classmethod
    def from_file(cls, path: str | Path) -> "InsightsRequest":
        """Read the diagnostic request within the project root."""
        path = (ROOT / path).absolute().relative_to(ROOT).as_posix()
        return cls(path, read_json(path))


def authorize(request: InsightsRequest) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Admit pinned inputs, baseline provenance, complete candidate-view sets and CPU execution."""
    runtime = runtime_flag(FEATURE, "proxy_insights_authorized", "Proxy insights")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("Proxy insights requires CUDA_VISIBLE_DEVICES=''")
    heavy_execution(runtime)
    data = request.data
    for path, digest in {**data["inputs"], **data["sources"]}.items():
        if Path(path).is_absolute() or ".." in Path(path).parts or "/mesh/" in path:
            raise ValueError(f"Not a scan-free relative input: {path}")
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Frozen input or source changed: {path}")
    if any(importlib.metadata.version(p) != v for p, v in data["packages"].items()):
        raise ValueError("Frozen package version changed")
    baseline, summary = read_json(data["baseline_request"]), read_json(data["baseline_summary"])
    producer = read_run(ROOT, data["baseline_run"], kind="mesh_extents.proxy")
    manifest_sha = next((r["sha256"] for r in producer["inputs"] if r["path"] == producer["request"]["manifest"]), None)
    if manifest_sha is None:
        raise ValueError("Baseline producer inputs do not record its request manifest")
    if manifest_sha != data["inputs"][data["baseline_request"]] or baseline["packages"] != data["packages"]:
        raise ValueError("Baseline request conflicts with the original producer")
    if producer["artifacts"]["summary.json"] != data["inputs"][data["baseline_summary"]]:
        raise ValueError("Baseline summary conflicts with the original producer")
    ledger = read_ledger(ROOT / data["first_run_ledger"])
    if ledger != summary["trials"]:
        raise ValueError("First-run ledger conflicts with the original summary")
    names = set(data["groups"]["failed"] + data["groups"]["contrast"])
    candidates = [c for c in baseline["candidates"] if c["object"] in names]
    views = [v for v in baseline["views"] if v["object"] in names]
    trials = [r for r in ledger if r["object"] in names]
    expected = {(c["object"], c["sha256"], v["id"]) for c in candidates for v in views if c["object"] == v["object"]}
    if len(trials) != len(expected) or {(r["object"], r["sha256"], r["view"]) for r in trials} != expected:
        raise ValueError("Incomplete or conflicting candidate-view replay panel")
    for c in candidates:
        if data["inputs"][c["candidate"]] != c["sha256"]:
            raise ValueError("Candidate differs from the baseline mesh")
    for view in views:
        if view["camera"] != "ego_cam_a" or view["episode"] in (41, 43, 44):
            raise PermissionError("View outside the fixed diagnostic")
        authorize_episode(runtime, MeasureRequest(source_split=view["split"], episode_index=view["episode"]), "proxy insights")
        for key in ("labels", "preprocessing", "depth_npz"):
            if data["inputs"][view[key]] != baseline["inputs"][view[key]]:
                raise ValueError("View arrays differ from the baseline")
    return baseline, candidates, views, trials


def replay_trial(original: dict, candidate: dict, view: dict, request: InsightsRequest,
                 key: str, directory: Path, previous: dict | None) -> tuple[dict, object]:
    """Admit a completed replay or save one selected-yaw mask and its equality-checked measurements."""
    import numpy as np
    import trimesh
    from ._insights import replay, require_replay

    if previous:
        run_path = ROOT / previous["run"]
        completed = read_run(ROOT, run_path, kind="mesh_extents.proxy_insights_replay")
        admission, parent = {"inputs": []}, (run_path, completed)
        result = read_json(artifact(parent, "result.json", admission))
        with np.load(artifact(parent, "mask.npz", admission), allow_pickle=False) as arrays:
            mask = arrays["mask"]
        if result["identity"] != key or completed["request"]["identity"] != key:
            raise ValueError("Cached replay identity conflicts with its request")
        require_replay(original, result)
        return result, mask
    record = run_fields(ROOT, "mesh_extents", "proxy_insights_replay", {"identity": key, "original": original},
                        inputs=[request.path, candidate["candidate"], view["preprocessing"], view["labels"]],
                        provenance="public_asset_diagnostic")
    replay_directory = directory / "replays" / key
    with recorded(replay_directory, record):
        mesh = trimesh.load(ROOT / candidate["candidate"], force="mesh", process=False)
        with (np.load(ROOT / view["preprocessing"], allow_pickle=False) as prep,
              np.load(ROOT / view["labels"], allow_pickle=False) as labels):
            values, mask = replay(mesh.vertices, mesh.faces, original["placement"], original["selected_yaw_degrees"],
                                  prep["input_intrinsics_pixels"], prep["input_valid"], labels["label"], view["cluster_ids"])
        require_replay(original, values)
        result = {"identity": key, "object": original["object"], "sha256": original["sha256"],
                  "view": original["view"], "original_trial_identity": original["identity"],
                  "selected_yaw_degrees": original["selected_yaw_degrees"], "placement": original["placement"],
                  "run": (replay_directory / "run.json").relative_to(ROOT).as_posix(), **values}
        save_npz(replay_directory / "mask.npz", np, mask=mask)
        write_json(replay_directory / "result.json", result)
    return result, mask


def proxy_insights(request: InsightsRequest) -> Path:
    """Replay the complete frozen panel before computing any conditional pair summary."""
    _, candidates, views, trials = authorize(request)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ._insights import analyse_object, assemble, pair_markdown, plot_pairs, replay_summary, summary_markdown
    from .rules import append_ledger, identity

    data = request.data
    frozen_id = identity(data)
    output = output_path(ROOT, data["output"])
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "replay-ledger.jsonl"
    logged = {r["identity"]: r for r in read_ledger(ledger_path) if r["frozen_identity"] == frozen_id}
    record = run_fields(ROOT, "mesh_extents", "proxy_insights", {"manifest": request.path, "frozen_identity": frozen_id},
                        inputs=[request.path, *data["inputs"]], provenance="public_asset_diagnostic")
    directory = output / new_run_id()
    meshes = {(c["object"], c["sha256"]): c for c in candidates}
    cameras = {v["id"]: v for v in views}
    replayed, masks, new = [], {}, 0
    with recorded(directory, record):
        for original in trials:
            key = identity({"frozen": frozen_id, "original": original["identity"]})
            result, mask = replay_trial(original, meshes[(original["object"], original["sha256"])],
                                        cameras[original["view"]], request, key, directory, logged.get(key))
            masks[(original["sha256"], original["view"])] = mask
            if key not in logged:
                append_ledger(ledger_path, {"frozen_identity": frozen_id, **result})
                new += 1
            replayed.append(result)
        objects, pairs = [], []
        for name in sorted(set(data["groups"]["failed"] + data["groups"]["contrast"])):
            panel = [c for c in candidates if c["object"] == name]
            frame_ids = [v["id"] for v in views if v["object"] == name]
            scores = {(r["sha256"], r["view"]): r["iou"] for r in trials if r["object"] == name}
            metadata = [{"sha256": c["sha256"], "cd_o_cm": c["cd_o_cm"],
                         "label": c["shape"] + " " + c["sha256"][:8]} for c in panel]
            q = np.array([[scores[(c["sha256"], v)] for v in frame_ids] for c in panel])
            projections = [np.array([masks[(c["sha256"], v)] for c in panel]) for v in frame_ids]
            row, comparisons = analyse_object(name, metadata, frame_ids, q, projections)
            objects.append(row)
            pairs.extend(comparisons)
        summary = assemble(objects, pairs, data["groups"])
        summary.update(replay=replay_summary(trials, replayed, new),
                       run=(directory / "run.json").relative_to(ROOT).as_posix(),
                       frozen_identity=frozen_id, replay_records=replayed)
        figure = plot_pairs(pairs, {(c["object"], c["sha256"]): c["cd_o_cm"] for c in candidates}, data["groups"]["failed"])
        summary["figures"] = {suffix: (directory / f"pair-orders.{suffix}").relative_to(ROOT).as_posix() for suffix in ("png", "pdf")}
        for suffix in summary["figures"]:
            figure.savefig(directory / f"pair-orders.{suffix}", dpi=160)
        plt.close(figure)
        write_json(directory / "pairs.json", pairs)
        (directory / "pairs.md").write_text(pair_markdown(pairs))
        write_json(directory / "summary.json", summary)
        (directory / "summary.md").write_text(summary_markdown(summary))
    # Stable summaries are mirrors of the completed run.
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(summary_markdown(summary))
    return directory / "run.json"
