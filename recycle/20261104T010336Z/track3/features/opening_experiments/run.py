"""Owned stereo, camera and optional observed-object opening experiments."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

from track3.paths import interpreter, permitted_root, output_path, project_path
from track3.provenance import input_file, read_run, run_fields, write_json

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parent
STAGES = ("rectify", "stereo", "anchor", "camera", "clusters", "hull", "tsdf")


def _json(path):
    return json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig"))


def _now():
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class OpeningExperimentRequest:
    depth_run: str | None = None
    frame_count: int = 20
    mesh_experiments: bool = True

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if self.depth_run is not None and (not isinstance(self.depth_run, str) or not self.depth_run.strip()):
            raise ValueError("depth_run must name an explicit completed teacher run.json")
        if type(self.frame_count) is not int or not 1 <= self.frame_count <= 20:
            raise ValueError("frame_count must be an integer in [1,20]")
        if type(self.mesh_experiments) is not bool:
            raise ValueError("mesh_experiments must be boolean")


def _teacher(request):
    """Resolve the completed contiguous public episode-39 teacher run."""
    if request.depth_run is None:
        raise ValueError("Supply the completed contiguous episode39 teacher run; no teacher is generated here")
    path = project_path(ROOT, request.depth_run)
    depth = read_run(ROOT, path, kind="depth.generate", allowed=("video_only",))
    selected = [row for row in depth.get("frames", []) if row.get("frame_index") in range(request.frame_count)]
    if (path.name != "run.json" or depth.get("episode_index") != 39 or depth.get("split") != "public"
            or depth.get("camera") != "ego_cam_a" or len(selected) != request.frame_count
            or sorted(row["frame_index"] for row in selected) != list(range(request.frame_count))
            or any(row.get("status") != "completed" for row in selected)):
        raise ValueError("Expected completed contiguous public episode39 RGB depth frames from zero")
    return path


def _redirect_temporary_files():
    tmp = output_path(ROOT, ".runtime/opening_experiments/tmp")
    cache = output_path(ROOT, ".cache/opening_experiments/pycache")
    tmp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(tmp), TMP=str(tmp), TMPDIR=str(tmp), PYTHONPYCACHEPREFIX=str(cache))
    tempfile.tempdir, sys.pycache_prefix = str(tmp), str(cache)


def _failed_child(child_root, before, kind, child_request):
    """The unique new failed producer run matching this request, if any (older handlers do not report it)."""
    expected = json.loads(json.dumps(asdict(child_request))) if child_request is not None else None
    matches = [path for path in set(child_root.glob("*/run.json")) - before
               if (child := _json(path)).get("kind") == kind and child.get("request") == expected
               and child.get("status") in ("failed", "interrupted")]
    return matches[0] if len(matches) == 1 else None


def run_opening_experiments(request: OpeningExperimentRequest):
    runtime_path = FEATURE / "runtime.json"
    runtime = _json(runtime_path)
    if runtime["execution"].get("run_authorized") is not True:
        raise PermissionError("Opening experiments remain disabled")
    permitted_root()
    if Path(sys.executable).resolve() != interpreter(ROOT, runtime["python"]):
        raise RuntimeError("Select the approved project interpreter (runtime.json python for this platform)")
    depth_path = _teacher(request)
    _redirect_temporary_files()
    from track3.features.stereo_depth import StereoDepthRequest, rectify, generate, anchor
    from track3.features.camera_motion import EstimateCameraRequest, estimate
    from track3.features.object_masks import OpeningClustersRequest, opening_clusters
    from track3.features.object_mesh import ObjectMeshRequest, reconstruct

    run_id = _now().strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/opening_experiments/{run_id}")
    record = run_fields(ROOT, "opening_experiments", "run", asdict(request),
                        inputs=[runtime_path], parents=[depth_path], allowed=("video_only",))
    record.update(schema_version=1, run_id=run_id, episode_index=39, split="public", artifacts={},
                  started_utc=_now().isoformat(), tests_run=False, validation_run=False,
                  model_execution=False, official_evaluation=False, research_only=True, submission_candidate=False,
                  stages={name: {"status": "unattempted"} for name in STAGES})
    stages = record["stages"]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def persist():
        write_json(directory / "run.json", record)

    def stage_path(name):
        return stages[name]["run"]["path"]

    def attempt(name, operation, build_request, dependencies, feature, kind):
        stage = stages[name]
        blocked = [parent for parent in dependencies if stages[parent]["status"] != "completed"]
        stage["parents"] = [input_file(ROOT, stage_path(parent)) for parent in dependencies if parent not in blocked]
        stage["parents"].append(input_file(ROOT, depth_path))
        if blocked:
            stage.update(status="blocked", reason=f"Required stages did not complete: {', '.join(blocked)}")
            persist()
            return
        child_root = ROOT / "outputs" / feature / "public/episode_000039"
        before = set(child_root.glob("*/run.json"))
        stage.update(status="running", started_utc=_now().isoformat())
        persist()
        stage_started = time.perf_counter()
        child_request = None
        try:
            child_request = build_request()
            print(f"Opening experiment: {name}", flush=True)
            path = operation(child_request)
            child = read_run(ROOT, path, kind=kind, allowed=("video_only",))
            stage.update(status="completed", run=input_file(ROOT, path), child_timing=child.get("timing"))
            record["inputs"].append(stage["run"])
        except Exception as error:
            stage.update(status="failed", error=f"{type(error).__name__}: {error}")
            failed = _failed_child(child_root, before, kind, child_request)
            if failed is not None:
                stage["failed_run"] = input_file(ROOT, failed)
                record["inputs"].append(stage["failed_run"])
        finally:
            stage["wall_seconds"] = time.perf_counter() - stage_started
            persist()

    persist()
    try:
        base = {"source_split": "public", "episode_index": 39, "start_frame": 0, "stop_frame": request.frame_count}
        attempt("rectify", rectify, lambda: StereoDepthRequest(**base, max_image_side=768), [], "stereo_depth", "stereo_depth.rectify")
        attempt("stereo", generate, lambda: StereoDepthRequest(**base, depth_run=request.depth_run,
            rectification_run=stage_path("rectify")), ["rectify"], "stereo_depth", "stereo_depth.generate")
        attempt("anchor", anchor, lambda: StereoDepthRequest(**base, depth_run=request.depth_run,
            stereo_run=stage_path("stereo"), masks_run=None), ["stereo"], "stereo_depth", "stereo_depth.anchor")
        attempt("camera", estimate, lambda: EstimateCameraRequest(depth_run=request.depth_run,
            stereo_run=stage_path("stereo"), anchor_run=stage_path("anchor"), opening_frame_count=request.frame_count),
            ["stereo", "anchor"], "camera_motion", "camera_motion.estimate")
        if request.mesh_experiments:
            attempt("clusters", opening_clusters, lambda: OpeningClustersRequest(depth_run=request.depth_run,
                anchor_run=stage_path("anchor"), camera_run=stage_path("camera"), stop_frame=request.frame_count),
                ["anchor", "camera"], "object_masks", "object_masks.opening_clusters")
            clusters = _json(stage_path("clusters")) if stages["clusters"]["status"] == "completed" else {}
            labels = clusters.get("instances", [])
            if clusters.get("object_names") == ["mini_dust_pan"] and len(labels) == 1 and clusters.get("opening_stop_frame", 0) > 0:
                record["cluster_assignment"] = {"object": "mini_dust_pan", "instance": labels[0], "index": 0}
                for name, method in (("hull", "opening_hull"), ("tsdf", "opening_tsdf")):
                    attempt(name, reconstruct, lambda method=method: ObjectMeshRequest(depth_run=request.depth_run,
                        masks_run=stage_path("clusters"), anchor_run=stage_path("anchor"), camera_run=stage_path("camera"),
                        method=method, objects=["mini_dust_pan"], cluster_assignments={"mini_dust_pan": 0}),
                        ["clusters", "anchor", "camera"], "object_mesh", "object_mesh.reconstruct")
            else:
                for name in ("hull", "tsdf"):
                    stages[name].update(status="blocked", reason="No completed persistent single-component opening with the exact mini_dust_pan roster")
        else:
            for name in ("clusters", "hull", "tsdf"):
                stages[name].update(reason="Mesh experiments were not requested")
        counts = Counter(stage["status"] for stage in stages.values())
        record.update(status="completed", completed_experiments=counts["completed"],
                      failed_experiments=counts["failed"], blocked_experiments=counts["blocked"])
        write_json(directory / "result.json", {key: record[key] for key in
            ("stages", "completed_experiments", "failed_experiments", "blocked_experiments")})
        record["artifacts"]["result.json"] = input_file(ROOT, directory / "result.json")["sha256"]
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        for stage in stages.values():
            if stage["status"] == "running":
                stage.update(status=record["status"], error=record["error"])
        raise
    finally:
        record.update(timing={"wall_seconds": time.perf_counter() - started}, finished_utc=_now().isoformat())
        persist()
    return directory / "result.json"
