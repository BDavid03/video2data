"""File admission, recorded trials and resumable orchestration for the frozen proxy panel."""

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

from track3.paths import output_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json
from track3.runs import ROOT, artifact, authorize_episode, checked_entry, new_run_id, read_json, recorded, runtime_flag
from . import _proxy
from .measure import MeasureRequest
from .rules import append_ledger, identity

FEATURE = Path(__file__).parent
if TYPE_CHECKING:
    from trimesh import Trimesh
    from numpy import ndarray


@dataclass(frozen=True)
class ProxyRequest:
    """Hold the frozen manifest and its project-relative location."""
    path: str
    data: dict

    @classmethod
    def from_file(cls, path: str | Path) -> "ProxyRequest":
        """Read the frozen request within the project root."""
        path = (ROOT / path).absolute().relative_to(ROOT).as_posix()
        return cls(path, read_json(path))


def freeze_request(path: str) -> dict:
    """Refresh only source hashes while preserving the authenticated frozen scientific inputs."""
    request = ProxyRequest.from_file(path)
    _post_result_scope(request)
    request.data["sources"] = {name: sha256_file(ROOT / name) for name in request.data["sources"]}
    write_json(output_path(ROOT, path), request.data)
    return request.data


def _post_result_scope(request: ProxyRequest) -> None:
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
    allowed = {f"track3/features/mesh_extents/{name}" for name in ("_proxy.py", "proxy.py", "_geometry.py", "__main__.py", "runtime.json")}
    if set(request.data["sources"]) != set(baseline["sources"]) or any(
            digest != baseline["sources"][path] for path, digest in request.data["sources"].items() if path not in allowed):
        raise PermissionError("Post-result source changes exceed the authorized scope")


def _admit(request: ProxyRequest) -> tuple[str, list[dict]]:
    """Check frozen hashes, package versions, episode permissions and baseline scope."""
    runtime = runtime_flag(FEATURE, "proxy_authorized", "Spoon proxy diagnostic")
    data, missing = request.data, []
    for path, digest in {**data["inputs"], **data["sources"]}.items():
        if Path(path).is_absolute() or ".." in Path(path).parts or "/mesh/" in path:
            raise ValueError(f"Not a scan-free relative input: {path}")
        try:
            if input_file(ROOT, path)["sha256"] != digest:
                raise ValueError(f"Input differs from its producer: {path}")
        except FileNotFoundError:
            missing.append({"path": path, "reason": "Frozen input file is missing; diagnostic not executed"})
    for package, version in data["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen package version differs: {package}")
    for view in data["views"]:
        if view["episode"] in (41, 43, 44) or view["camera"] != "ego_cam_a":
            raise PermissionError("View outside the frozen diagnostic")
        authorize_episode(runtime, MeasureRequest(source_split=view["split"], episode_index=view["episode"]), "proxy")
    try:
        _post_result_scope(request)
    except FileNotFoundError as error:
        missing.append({"reason": f"Recorded baseline input is missing: {error}"})
    return identity({"request": data, "request_sha256": sha256_file(ROOT / request.path)}), missing


def load_candidate(candidate: dict) -> "Trimesh":
    """Load a frozen mesh and enforce its construction axis convention."""
    import trimesh

    mesh = trimesh.load(ROOT / candidate["candidate"], force="mesh", process=False)
    if not _proxy.valid_extents(mesh.extents):
        raise ValueError(f"Candidate violates x-long/y-short/z-up construction convention: {candidate['candidate']}")
    return mesh


def load_view(view: dict) -> dict:
    """Read frozen depth/mask inputs and reuse the measure slice's support-plane refit."""
    import numpy as np

    measure = read_json(view["measure_run"])
    with (np.load(ROOT / view["preprocessing"], allow_pickle=False) as prep,
          np.load(ROOT / view["depth_npz"], allow_pickle=False) as depth,
          np.load(ROOT / view["labels"], allow_pickle=False) as labels):
        return _proxy.prepare_view(depth["depth_m"], depth["valid"], prep["input_intrinsics_pixels"],
                                   prep["input_valid"], labels["label"], view["cluster_ids"],
                                   view["global_plane"], measure["request"], depth["input_valid"])


def _overlay(view: dict, loaded: dict, path: Path, projected: "ndarray | None" = None) -> None:
    """Read the recorded frame and persist its target/silhouette overlay."""
    import cv2

    image = cv2.imread(str(ROOT / view["overlay"]))
    if image is None:
        raise ValueError("Missing recorded opening-frame overlay")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), _proxy.overlay_image(image, loaded["target"], projected)):
        raise OSError(f"Could not write {path}")


def _trial(candidate: dict, view: dict, loaded: dict, request: ProxyRequest,
           key: str, directory: Path, previous: dict | None) -> dict:
    """Admit a completed trial or record one candidate-view projection."""
    cache = directory / "cache" / f"{key}.json"
    previous = previous or (read_json(cache) if cache.exists() else None)
    if previous:
        run_path = ROOT / previous["run"]
        completed = read_run(ROOT, run_path, kind="mesh_extents.proxy_trial")
        parent, admission = (run_path, completed), {"inputs": []}
        result = read_json(artifact(parent, "result.json", admission))
        artifact(parent, "overlay.png", admission)
        if result["identity"] != key or result["identity"] != completed["request"]["identity"]:
            raise ValueError("Completed trial identity differs from request")
        return result
    inputs = [request.path, candidate["candidate"], view["labels"], view["depth_npz"], view["preprocessing"], view["overlay"]]
    record = run_fields(ROOT, "mesh_extents", "proxy_trial", {"identity": key, "candidate": candidate, "view": view},
                        inputs=inputs, parents=[view["measure_run"], view["depth_run"]], provenance="public_asset_diagnostic")
    cached = directory / "trials" / key / new_run_id()
    with recorded(cached, record):
        mesh = load_candidate(candidate)
        values, mask = _proxy.compare(mesh.vertices, mesh.faces, loaded["pose"], loaded["intrinsics"],
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


def read_ledger(path: Path) -> list[dict]:
    """Read complete rows and quarantine only a truncated final append."""
    rows, offset = [], 0
    lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index != len(lines) - 1 or line.endswith((b"\n", b"\r")) or not _proxy.truncated_tail(line.decode(), error):
                raise ValueError(f"{path}:{index + 1}: undecodable ledger row") from error
            fragments = path.parent / "ledger-fragments"
            fragments.mkdir(exist_ok=True)
            (fragments / f"{new_run_id()}.bin").write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(offset)
        offset += len(line)
    return rows


def proxy(request: ProxyRequest, *, dry_run: bool = False) -> Path:
    """Run the frozen candidate queue or target-only preparation and write its report."""
    frozen_id, missing = _admit(request)
    data = request.data
    directory = output_path(ROOT, data["dry_output"] if dry_run else data["output"])
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / "ledger.jsonl"
    logged = {row["identity"]: row for row in read_ledger(ledger_path) if row["frozen_identity"] == frozen_id}
    record = run_fields(ROOT, "mesh_extents", "proxy_dry" if dry_run else "proxy", {"manifest": request.path,
                        "frozen_identity": frozen_id}, inputs=[request.path, *(p for p in data["inputs"]
                        if not any(row.get("path") == p for row in missing))], provenance="public_asset_diagnostic")
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
                              "mask_area_pixels": loaded["mask_area_pixels"], "placement": loaded["placement"],
                              "overlay": path.relative_to(ROOT).as_posix()})
                continue
            candidates = data["spoon"] if view["object"] == "wooden_spoon" else data["candidates"]
            for candidate in (c for c in candidates if c["object"] == view["object"]):
                if len(results) >= data["budget"]["max_trials"] or time.monotonic() - started > data["budget"]["max_seconds"]:
                    raise TimeoutError("Frozen budget exhausted; resume the same request")
                key = identity({"frozen": frozen_id, "candidate": candidate, "view": view})
                try:
                    row = {"frozen_identity": frozen_id,
                           **_trial(candidate, view, loaded, request, key, directory, logged.get(key))}
                except FileNotFoundError as error:
                    missing.append({"view": view["id"], "candidate": candidate["sha256"],
                                    "reason": f"Frozen candidate input became unavailable: {error}"})
                    continue
                if key not in logged:
                    append_ledger(ledger_path, row)
                results.append(row)
        if dry_run and not missing:
            summary, markdown = _proxy.dry_summary(data, masks, frozen_id)
        else:
            summary = _proxy.summarize(data, results, missing)
            markdown = _proxy.summary_markdown(summary)
        write_json(run_directory / "summary.json", summary)
        write_json(directory / "summary.json", summary)
        (directory / "summary.md").write_text(markdown)
        if not dry_run or missing:
            (run_directory / "summary.md").write_text(markdown)
    return run_directory / "run.json"
