"""Feature-owned artifact index/export and optional Viser review, with no estimation."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from functools import cache
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
import time
from uuid import uuid4

from .annotations import admit as admit_annotations, overlay as annotation_overlay

from track3.paths import output_path, project_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
EPISODES = (0, 11, 12, 18, 21, 23, 26, 31, 39, 42)
KINDS = {"preview_run": "hand_motion.preview", "depth_run": "depth.generate",
         "masks_run": "object_masks.generate", "hands_run": "hand_motion.lift",
         "camera_run": "camera_motion.estimate", "scene_run": "scene_assembly.assemble",
         "stereo_run": "stereo_depth.generate", "tracks_run": "object_masks.correspondences"}
# (consumer, producer, message): the consumer run must list the selected producer run among its inputs.
LINEAGE = (("masks_run", "depth_run", "Masks require the exact selected depth producer"),
           ("tracks_run", "masks_run", "Point tracks require the exact selected mask producer"),
           ("camera_run", "depth_run", "Camera review requires the exact selected depth producer"))


def _json(path):
    return json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig"))


def _gate(operation, episode):
    runtime = _json(FEATURE / "runtime.json")
    if runtime.get(operation + "_authorized") is not True:
        raise PermissionError(f"Inspection {operation} remains disabled")
    if type(episode) is not int or episode not in EPISODES:
        raise PermissionError("Inspection admits only frozen public training/tuning episodes; 41/43/44 excluded")
    if episode not in runtime["authorized_episode_indices"]:
        raise PermissionError("Inspection episode requires explicit runtime admission")
    output_path(ROOT, "outputs/inspection")  # Canonical-root guard; no output created.
    if operation == "serve" and runtime.get("compatible_environment_authorized") is not True:
        raise PermissionError("Serving requires a separately approved compatible existing environment")


@dataclass(frozen=True)
class InspectionRequest:
    episode_index: int
    candidates: list[dict]
    frame_indices: list[int]
    depth_limits_m: list[float]
    manual_events: list[dict] = field(default_factory=list)

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if type(self.episode_index) is not int or self.episode_index not in EPISODES:
            raise PermissionError("Only public training/tuning episodes are admitted")
        if (not 1 <= len(self.frame_indices) <= 256 or
                any(type(i) is not int or i < 0 for i in self.frame_indices) or
                self.frame_indices != sorted(set(self.frame_indices))):
            raise ValueError("Select 1–256 ascending distinct source frames")
        if (len(self.depth_limits_m) != 2 or any(type(x) not in (int, float) or not math.isfinite(x)
                for x in self.depth_limits_m) or not 0 < self.depth_limits_m[0] < self.depth_limits_m[1]):
            raise ValueError("Declare shared finite positive depth limits before viewing")
        if not 1 <= len(self.candidates) <= 2:
            raise ValueError("Supply one or two explicit candidate mappings")
        for candidate, label in zip(self.candidates, ("A", "B")):
            if set(candidate) - {"label", "annotations", *KINDS} or candidate.get("label") != label:
                raise ValueError("Candidate labels must be A then B, with only supported producer keys")
            if not candidate.get("preview_run"):
                raise ValueError("Each candidate needs completed hand_motion.preview or depth.preview calibrated RGB artifacts")
        if len(self.manual_events) > 128:
            raise ValueError("At most 128 manual events are admitted")
        for event in self.manual_events:
            if (set(event) != {"kind", "start_frame", "stop_frame", "provenance", "reviewer", "notes"}
                    or event["kind"] not in ("grasp", "lift", "release", "full_occlusion", "reacquisition")
                    or event["provenance"] != "manual_image_review"
                    or not isinstance(event["reviewer"], str) or not event["reviewer"].strip()
                    or not isinstance(event["notes"], str)
                    or any(type(event[key]) is not int for key in ("start_frame", "stop_frame"))
                    or not 0 <= event["start_frame"] < event["stop_frame"]
                    or event["stop_frame"] - event["start_frame"] > 256
                    or not set(range(event["start_frame"], event["stop_frame"])).issubset(self.frame_indices)):
                raise ValueError("Manual image-review events require reviewer, notes and fully selected half-open frame windows")


def _artifact(run_path, name, digest):
    path = project_path(ROOT, run_path.parent / name)
    if not path.is_relative_to(run_path.parent):
        raise ValueError("Artifact escapes producer directory")
    entry = input_file(ROOT, path)
    if entry["sha256"] != digest:
        raise ValueError("Artifact hash differs from producer")
    return entry


def _verified(entry, message):
    """Return the local path of an exported artifact entry whose hash is unchanged."""
    path = project_path(ROOT, entry["path"])
    if input_file(ROOT, path) != entry:
        raise ValueError(message)
    return path


def _confine_caches():
    """Confine optional decoder/viewer caches before their lazy imports."""
    cache_paths = {"TEMP": ".runtime/inspection/tmp", "TMP": ".runtime/inspection/tmp",
                   "TMPDIR": ".runtime/inspection/tmp", "XDG_CACHE_HOME": ".cache/inspection/xdg",
                   "PYTHONPYCACHEPREFIX": ".cache/inspection/pycache", "MPLCONFIGDIR": ".cache/inspection/matplotlib"}
    for name, relative in cache_paths.items():
        destination = output_path(ROOT, relative)
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(destination)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    tempfile.tempdir = os.environ["TMPDIR"]


def _source_clock(request, revision, clock_path):
    inventory = _json(clock_path)
    clocks = [row for row in inventory["videos"] if row["split"] == "public" and
              row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if (inventory["status"] != "completed" or inventory["revision"] != revision or
            len(clocks) != 1 or clocks[0]["status"] != "ok"):
        raise ValueError("Completed pinned RGB clock is required")
    clock = clocks[0]
    pts = clock["pts_integer"]
    if (len(pts) != clock["decoded_frame_count"] or not pts or
            any(type(value) is not int for value in pts) or
            any(b <= a for a, b in zip(pts, pts[1:])) or request.frame_indices[-1] >= len(pts)):
        raise ValueError("Source PTS inventory is incomplete or selected frame is out of range")
    rates = clock.get("time_base_and_frame_rate", [])
    if len(rates) != 1 or len(rates[0]) != 2:
        raise ValueError("One unambiguous source clock time base is required")
    time_base = Fraction(rates[0][0])
    if time_base <= 0:
        raise ValueError("Source time base must be positive")
    return clock, pts, time_base


def _producers(candidate, episode, revision):
    """Admit a candidate's producer runs from their feature-owned episode directories; return sources and grids."""
    sources, grids = {}, set()
    episode_dir = f"episode_{episode:06d}"
    for role, kind in KINDS.items():
        if not candidate.get(role):
            continue
        path = project_path(ROOT, candidate[role])
        expected = ROOT / "outputs" / kind.split(".")[0] / "public" / episode_dir
        alternative = ROOT / "outputs/depth/public" / episode_dir
        if not (path.is_relative_to(expected) or role == "preview_run" and path.is_relative_to(alternative)):
            raise PermissionError("Producer path must belong to the admitted public episode before it is opened")
        source = read_run(ROOT, path, kind=None if role == "preview_run" else kind, allowed=("video_only",))
        if role == "preview_run" and source["kind"] not in ("hand_motion.preview", "depth.preview"):
            raise ValueError("Preview must be hand_motion.preview or depth.preview")
        if role == "preview_run" and not path.is_relative_to(ROOT / "outputs" / source["kind"].split(".")[0] / "public" / episode_dir):
            raise ValueError("Preview producer kind differs from its feature-owned directory")
        if (source.get("split") != "public" or source.get("episode_index") != episode or
                source.get("dataset_revision") != revision):
            raise ValueError("Producer split, episode or revision differs")
        grid = source.get("grid") or source.get("preprocessing_artifact_sha256")
        if not isinstance(grid, str) or len(grid) != 64:
            raise ValueError("Producer must name its preprocessing grid hash")
        grids.add(grid)
        sources[role] = (path, source)
    return sources, grids


def _consumed(sources, consumer, producer):
    return (consumer in sources and producer in sources
            and input_file(ROOT, sources[producer][0]) in sources[consumer][1]["inputs"])


def _check_lineage(sources):
    """Exact parent identity, not merely matching episode/grid, binds overlays to RGB."""
    preview = sources["preview_run"][1]
    # An RGB-only depth preview does not claim to have rendered hands.
    roles = ("depth_run",) if preview["kind"] == "depth.preview" else ("depth_run", "hands_run")
    if any(role in sources and not _consumed(sources, "preview_run", role) for role in roles):
        raise ValueError("Selected preview did not consume the selected producer")
    if preview["kind"] == "depth.preview" and "depth_run" not in sources:
        raise ValueError("RGB-only preview requires its exact selected depth parent")
    for consumer, producer, message in LINEAGE:
        if consumer in sources and not _consumed(sources, consumer, producer):
            raise ValueError(message)
    if "stereo_run" in sources and not (_consumed(sources, "camera_run", "stereo_run")
                                        and _consumed(sources, "stereo_run", "depth_run")):
        raise ValueError("Stereo point clouds require the exact selected camera and depth lineage")
    if "scene_run" in sources and not _consumed(sources, "scene_run", "camera_run"):
        raise ValueError("Scene review requires the exact selected camera producer")


def _scene_shift(scene_path, scene_source):
    shift = _json(_artifact(scene_path, "scene.json", scene_source["artifacts"]["scene.json"])["path"])["world"]["assembly_z_shift_m"]
    if type(shift) not in (int, float) or not math.isfinite(shift):
        raise ValueError("Scene world shift must be explicit and finite")
    return shift


def _check_world(loaded, scene_shifts):
    """A/B world channels must share one camera-motion producer and one assembled world."""
    world_parents = []
    for _, sources in loaded:
        if "hands_run" in sources:
            entries = [entry for entry in sources["hands_run"][1]["inputs"]
                       if entry["path"].startswith("outputs/camera_motion/") and entry["path"].endswith("/run.json")]
            if len(entries) != 1:
                raise ValueError("World hand review requires one explicit camera-motion parent")
            world_parents.append(entries[0])
    if world_parents and any(entry != world_parents[0] for entry in world_parents):
        raise ValueError("A/B world hands require the exact same camera-motion parent")
    camera_parents = [input_file(ROOT, sources["camera_run"][0]) for _, sources in loaded if "camera_run" in sources]
    if camera_parents and any(entry != camera_parents[0] for entry in [*camera_parents, *world_parents]):
        raise ValueError("All world channels require the same camera-motion producer")
    if scene_shifts and (len(scene_shifts) != len(loaded) or any(shift != scene_shifts[0] for shift in scene_shifts)):
        raise ValueError("Scene A/B review requires both candidates in the identical assembled world")


def _channel(role, path, source, row, sources, producer, artifact, read):
    """Available channel for one completed producer frame: hashed artifacts the viewer may open."""
    channel = {"status": "available", "producer": producer, "artifacts": {}}
    artifacts = channel["artifacts"]
    if role == "preview_run":
        for name in ("rgb", "overlay") if source["kind"] == "hand_motion.preview" else ("rgb",):
            artifacts[name] = artifact(path, row[name]["file"], row[name]["sha256"])
        channel["preview_kind"] = source["kind"]
    elif role == "hands_run":
        if source.get("units") != "metres" or source.get("joint_order") != "MANO21":
            raise ValueError("World hands require metric MANO21 producer")
        artifacts["hands"] = artifact(path, "hands.npz", source["artifacts"]["hands.npz"])
    elif role == "camera_run":
        artifacts["trajectory"] = artifact(path, "trajectory.npz", source["artifacts"]["trajectory.npz"])
        depth_path, depth_source = sources["depth_run"]
        artifacts["preprocessing"] = artifact(depth_path, "preprocessing.npz", depth_source["preprocessing_artifact_sha256"])
    elif role == "scene_run":
        artifacts["trajectory"] = artifact(path, "trajectories.npz", source["artifacts"]["trajectories.npz"])
        artifacts["scene"] = artifact(path, "scene.json", source["artifacts"]["scene.json"])
        channel["objects"] = read(artifacts["scene"]["path"])["objects"]
        for slot, item in enumerate(channel["objects"]):
            for geometry, key in (("visual", "file"), ("collision", "collision_file")):
                relative = project_path(ROOT, item[key]).relative_to(path.parent).as_posix()
                artifacts[f"{slot}_{geometry}"] = artifact(path, relative, source["artifacts"][relative])
    elif role == "stereo_run":
        artifacts["frame"] = artifact(path, row["artifact"], row["sha256"])
        camera_path, camera_source = sources["camera_run"]
        artifacts["camera"] = artifact(camera_path, "trajectory.npz", camera_source["artifacts"]["trajectory.npz"])
        depth_path, depth_source = sources["depth_run"]
        artifacts["preprocessing"] = artifact(depth_path, "preprocessing.npz", depth_source["preprocessing_artifact_sha256"])
    else:
        artifacts["frame"] = artifact(path, row["artifact"], row["sha256"])
    return channel


def _frame(index, pts, time_base, loaded, producers, grid, events, artifact, read):
    """Index every candidate channel at one source frame; record producer rows and gaps as events."""
    frame = {"frame_index": index, "source_pts": pts[index],
             "source_seconds": float(pts[index] * time_base), "candidates": {}}
    for label, sources in loaded:
        channels = {}
        for role, (path, source) in sources.items():
            producer = producers[label][role]
            rows = [row for row in source.get("frames", []) if row.get("frame_index") == index]
            if len(rows) != 1 or rows[0].get("status") != "completed":
                missing = {"status": "missing", "reason": "no unique completed source frame",
                           "producer": producer, "producer_frames": rows}
                channels[role] = missing
                events.append({"frame_index": index, "source_pts": pts[index], "candidate": label,
                               "producer": producer, "kind": "missing_producer_frame",
                               "status": "missing", "details": missing})
                continue
            row = rows[0]
            if row.get("video_pts") != pts[index]:
                raise ValueError("Producer timestamp differs from source clock")
            channel = _channel(role, path, source, row, sources, producer, artifact, read)
            if role == "tracks_run":
                tracks = read(channel["artifacts"]["frame"]["path"])
                if (tracks.get("schema_version") != 1 or tracks.get("grid") != grid
                        or tracks.get("frame_index") != index or tracks.get("video_pts") != pts[index]):
                    raise ValueError("Point-track internal grid/frame/PTS differs")
                channel["identity_states"] = tracks["identities"]
                for state in tracks["identities"]:
                    if state.get("gap_frames", 0) or state.get("rejection_reason") or not state.get("identity_supported"):
                        events.append({"frame_index": index, "source_pts": pts[index], "candidate": label,
                                       "producer": producer, "kind": "producer_identity_support", "details": state})
            channel["producer_frame"] = row
            if role == "masks_run":
                channel["instances"] = source["instances"]
            channels[role] = channel
            # Preserve producer observations, including tracking/support failures, without inventing contact labels.
            events.append({"frame_index": index, "source_pts": pts[index], "candidate": label,
                           "producer": producer, "kind": "producer_frame", "details": row})
        frame["candidates"][label] = channels
    return frame


def _scorecard(request, document, annotations, loaded):
    return {"schema_version": 1, "episode_index": request.episode_index, "selection": "request-selected frames",
            "frame_count": len(document["frames"]), "quality": None, "quality_reason": "not measured",
            "inspection_latency_seconds": None, "inspection_peak_memory_bytes": None,
            "producers": document["candidates"],
            "annotation_coverage": [{"candidate": label, "frame_index": index,
                                     "status": "available" if index in annotations.get(label, {}).get("frames", {}) else "not_supplied"}
                                    for label, _ in loaded for index in request.frame_indices],
            "coverage": [{"frame_index": f["frame_index"], "source_pts": f["source_pts"],
                          "channels": {label: {role: channels.get(role, {"status": "not_supplied"})["status"] for role in KINDS}
                                       for label, channels in f["candidates"].items()}}
                         for f in document["frames"]]}


def export(request):
    """Export an index, coverage scorecard and event-reel manifest, never rendered metrics."""
    _gate("export", request.episode_index)
    revision = _json("configs/sources.lock.json")["dataset"]["revision"]
    clock_path = project_path(ROOT, "reports/data-audit/videos.json")
    clock, pts, time_base = _source_clock(request, revision, clock_path)
    parents, loaded, grids, scene_shifts = [], [], set(), []
    for candidate in request.candidates:
        sources, candidate_grids = _producers(candidate, request.episode_index, revision)
        grids |= candidate_grids
        parents.extend(path for path, _ in sources.values())
        _check_lineage(sources)
        if "scene_run" in sources:
            scene_shifts.append(_scene_shift(*sources["scene_run"]))
        loaded.append((candidate["label"], sources))
    if len(grids) != 1:
        raise ValueError("A/B channels require the same calibrated camera grid")
    grid = next(iter(grids))
    _check_world(loaded, scene_shifts)
    record = run_fields(ROOT, "inspection", "export", asdict(request), parents=parents,
                        allowed=("video_only",), inputs=[FEATURE / "runtime.json", clock_path])
    annotations = {}
    for candidate, (label, sources) in zip(request.candidates, loaded):
        if candidate.get("annotations"):
            entry, data = admit_annotations(ROOT, candidate["annotations"], sources, grid, pts, request.frame_indices)
            record["inputs"].append(entry)
            annotations[label] = data
    record.update(split="public", episode_index=request.episode_index, dataset_revision=revision,
                  grid=grid, review_only=True, quality_approved=False, artifacts={})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/inspection/public/episode_{request.episode_index:06d}/{run_id}")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record["started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "run.json", record)
    try:
        # Each producer file is hashed (and each JSON read) once per export.
        artifact, read = cache(_artifact), cache(_json)
        producers = {label: {role: input_file(ROOT, path) for role, (path, _) in sources.items()} for label, sources in loaded}
        document = {"schema_version": 1, "clock": clock, "depth_limits_m": request.depth_limits_m,
                    "source_time_base": str(time_base), "annotations": annotations,
                    "assembly_z_shift_m": scene_shifts[0] if scene_shifts else 0.0,
                    "manual_events": request.manual_events,
                    "grid": grid, "frames": [], "events": [],
                    "candidates": [{"label": label, "producers": producers[label]} for label, _ in loaded],
                    "quality_approved": False,
                    "unsupported": ["calibrated_uncertainty", "automatic_semantic_events"]}
        document["frames"] = [_frame(index, pts, time_base, loaded, producers, grid, document["events"], artifact, read)
                              for index in request.frame_indices]
        for frame in document["frames"]:
            for channels in frame["candidates"].values():
                for channel in channels.values():
                    for entry in channel.get("artifacts", {}).values():
                        if entry not in record["inputs"]:
                            record["inputs"].append(entry)
        reels = {"schema_version": 1, "status": "manifest_only", "events": document["events"],
                 "manual_windows": [{**event, "frames": [{"frame_index": i, "source_pts": pts[i]}
                                                         for i in range(event["start_frame"], event["stop_frame"])]}
                                    for event in request.manual_events],
                 "event_classes_pending": ["grasp", "lift", "release", "full_occlusion", "reacquisition"],
                 "reason": "Producer rows and explicit manual image-review windows retained; encoder is separately gated"}
        for name, value in (("inspection.json", document), ("scorecard.json", _scorecard(request, document, annotations, loaded)),
                            ("reel-manifest.json", reels)):
            write_json(directory / name, value)
            record["artifacts"][name] = sha256_file(directory / name)
        record["status"] = "completed"
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        record.update(status=status, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["timing"] = {"wall_seconds": time.perf_counter() - started}
        write_json(directory / "run.json", record)
    return directory / "run.json"


class _Viewer:
    """Loopback Viser review of one inspection export; each role renders saved producer artifacts only."""

    def __init__(self, server, document, cv2, np, SO3):
        self.server, self.document, self.cv2, self.np, self.SO3 = server, document, cv2, np, SO3
        self.frames = document["frames"]
        self.slider = server.gui.add_slider("Selected source frame", min=0, max=max(1, len(self.frames) - 1),
                                            step=1, initial_value=0)
        self.details = server.gui.add_markdown("")
        self.dynamic = []
        self.lock = Lock()
        self.renderers = {"preview_run": self._preview, "depth_run": self._depth, "masks_run": self._masks,
                          "tracks_run": self._tracks, "hands_run": self._hands, "camera_run": self._camera,
                          "stereo_run": self._stereo, "scene_run": self._scene}

    def load(self, entry, image=False):
        artifact = _verified(entry, "Viewer artifact changed since export")
        if image:
            pixels = self.cv2.imread(str(artifact))
            if pixels is None:
                raise ValueError("Could not decode saved preview image")
            return self.cv2.cvtColor(pixels, self.cv2.COLOR_BGR2RGB)
        with self.np.load(artifact, allow_pickle=False) as arrays:
            return {key: arrays[key].copy() for key in arrays.files}

    def add(self, handle):
        self.dynamic.append(handle)

    def note(self, text):
        self.add(self.server.gui.add_markdown(text))

    def clear(self):
        for handle in self.dynamic:
            handle.remove()
        self.dynamic.clear()

    def selected(self):
        return min(int(self.slider.value), len(self.frames) - 1)

    def draw(self, _=None):
        with self.lock:
            self.clear()
            frame = self.frames[self.selected()]
            seconds = frame.get("source_seconds")
            self.details.content = f"Frame {frame['frame_index']} · source PTS {frame['source_pts']} · source seconds {seconds}\n\nShared camera grid and fixed depth limits. Magenta depth pixels are invalid. Review only."
            for label, channels in frame["candidates"].items():
                for role in KINDS:
                    channel = channels.get(role, {"status": "not_supplied"})
                    if channel["status"] != "available":
                        self.note(f"{label} {role}: {channel['status']}")
                        continue
                    self.renderers[role](label, channel, channels, frame)
                self._annotations(label, channels, frame)
            self.note("Pending adapters: " + ", ".join(self.document["unsupported"]))

    def safe_draw(self, event=None):
        try:
            self.draw(event)
        except Exception as error:
            with self.lock:
                self.clear()
                self.details.content = f"Frame rendering failed: {type(error).__name__}: {error}. No successful rendering is claimed."

    def _preview(self, label, channel, channels, frame):
        for name in ("rgb", "overlay"):
            if name in channel["artifacts"]:
                self.add(self.server.gui.add_image(self.load(channel["artifacts"][name], True), label=f"{label} {name}"))

    def _depth(self, label, channel, channels, frame):
        np = self.np
        arrays = self.load(channel["artifacts"]["frame"])
        z, valid = arrays["depth_m"], arrays["valid"]
        low, high = self.document["depth_limits_m"]
        admitted = valid & np.isfinite(z) & (z > 0)
        gray = np.where(admitted, np.clip((z - low) / (high - low), 0, 1), 0)
        image = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        image[~admitted] = (255, 0, 255)
        self.add(self.server.gui.add_image(image, label=f"{label} depth {low}–{high} m (clipped)"))

    def _masks(self, label, channel, channels, frame):
        arrays = self.load(channel["artifacts"]["frame"])
        for slot, identity in enumerate(channel["instances"]):
            image = self.np.repeat((arrays["masks"][slot].astype(self.np.uint8) * 255)[..., None], 3, axis=2)
            self.add(self.server.gui.add_image(image, label=f"{label} producer identity {identity}"))

    def _tracks(self, label, channel, channels, frame):
        entry = channel["artifacts"]["frame"]
        _verified(entry, "Point-track artifact changed since export")
        tracks = _json(entry["path"])
        if tracks["frame_index"] != frame["frame_index"] or tracks["video_pts"] != frame["source_pts"]:
            raise ValueError("Point-track internal source clock differs")
        image = self.load(channels["preview_run"]["artifacts"]["rgb"], True).copy()
        for track in tracks["tracks"]:
            if track["visible"] and track["geometric_accepted"] and track["xy"] is not None:
                xy = track["xy"]
                if (len(xy) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in xy)
                        or not 0 <= xy[0] < image.shape[1] or not 0 <= xy[1] < image.shape[0]):
                    raise ValueError("Admitted point track falls outside the shared RGB grid")
                self.cv2.circle(image, tuple(round(v) for v in xy), 3, (30, 240, 130), 1)
        self.add(self.server.gui.add_image(image, label=f"{label} admitted visible producer tracks"))
        self.note(f"{label} identity visibility/discontinuities (producer assertions)\n\n"
                  + "```json\n" + json.dumps(tracks["identities"], indent=2) + "\n```")

    def _hands(self, label, channel, channels, frame):
        np = self.np
        arrays = self.load(channel["artifacts"]["hands"])
        i = frame["frame_index"]
        if int(arrays["pts"][i]) != frame["source_pts"]:
            raise ValueError("Lifted hands timestamp differs")
        # Same world coordinates and viewer camera for both candidates. No candidate offset or rescaling.
        for side, valid in enumerate(arrays["world_valid"][i]):
            if valid:
                points = arrays["joints_world"][i, side]
                if points.shape != (21, 3) or not np.isfinite(points).all():
                    raise ValueError("Invalid admitted world hand joints")
                color = (30, 200, 240) if label == "A" else (255, 150, 40)
                points = points.copy()
                points[:, 2] += self.document["assembly_z_shift_m"]
                self.add(self.server.scene.add_point_cloud(f"/{label}/hand{side}", points=points,
                                                           colors=np.tile(color, (21, 1)).astype(np.uint8), point_size=0.006))

    def _camera(self, label, channel, channels, frame):
        np = self.np
        arrays = self.load(channel["artifacts"]["trajectory"])
        i = frame["frame_index"]
        if int(arrays["pts"][i]) != frame["source_pts"]:
            raise ValueError("Camera timestamp differs")
        if not arrays["valid"][i]:
            self.note(f"{label} camera: unavailable world transform")
            return
        pose = arrays["world_T_camA"][i].copy()
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid admitted camera transform")
        pose[2, 3] += self.document["assembly_z_shift_m"]
        quaternion = self.SO3.from_matrix(pose[:3, :3]).wxyz
        self.add(self.server.scene.add_frame(f"/{label}/camera_axes", wxyz=quaternion,
                                             position=pose[:3, 3], axes_length=0.1, axes_radius=0.003))
        preprocessing = self.load(channel["artifacts"]["preprocessing"])
        k = preprocessing["input_intrinsics_pixels"]
        height, width = preprocessing["input_valid"].shape
        # Calibrated off-centre principal points cannot be represented by Viser's symmetric frustum.
        if abs(k[0, 2] - (width - 1) / 2) <= 1 and abs(k[1, 2] - (height - 1) / 2) <= 1 and abs(k[0, 0] - k[1, 1]) <= 1:
            self.add(self.server.scene.add_camera_frustum(f"/{label}/camera_frustum",
                                                          fov=2 * math.atan(height / (2 * k[1, 1])), aspect=width / height,
                                                          scale=0.15, wxyz=quaternion, position=pose[:3, 3]))
        else:
            self.note(f"{label} frustum omitted: off-centre principal point or unequal focal lengths")

    def _stereo(self, label, channel, channels, frame):
        np = self.np
        stereo = self.load(channel["artifacts"]["frame"])
        camera = self.load(channel["artifacts"]["camera"])
        i = frame["frame_index"]
        if (int(stereo["frame_index"]) != i or int(stereo["pts"]) != frame["source_pts"]
                or int(camera["frame_index"][i]) != i or int(camera["pts"][i]) != frame["source_pts"]):
            raise ValueError("Stereo/camera internal frame or timestamp differs")
        if not camera["valid"][i]:
            self.note(f"{label} stereo cloud omitted: unavailable camera transform")
            return
        z, valid, k = stereo["rgb_depth_m"], stereo["rgb_valid"], stereo["rgb_intrinsics_pixels"]
        preprocessing = self.load(channel["artifacts"]["preprocessing"])
        if (z.ndim != 2 or valid.shape != z.shape or valid.dtype != np.dtype("bool")
                or k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("Stereo cloud requires a valid calibrated RGB depth grid")
        if z.shape != preprocessing["input_valid"].shape or not np.allclose(k, preprocessing["input_intrinsics_pixels"], atol=1e-5):
            raise ValueError("Stereo cloud intrinsics differ from the exact shared preprocessing grid")
        selected = np.flatnonzero(valid & preprocessing["input_valid"] & np.isfinite(z) & (z > 0))
        if len(selected) > 20_000:
            selected = selected[np.linspace(0, len(selected) - 1, 20_000, dtype=np.int64)]
        if not len(selected):
            self.note(f"{label} stereo cloud: no admitted pixels")
            return
        v, u = np.divmod(selected, z.shape[1])
        camera_points = np.column_stack((u, v, np.ones(len(u)))) @ np.linalg.inv(k).T * z[v, u, None]
        pose = camera["world_T_camA"][i]
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid stereo cloud camera transform")
        points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        points[:, 2] += self.document["assembly_z_shift_m"]
        # RGB is optional: geometry remains stereo-observed and is never completed from MoGe.
        rgb_channel = channels.get("preview_run", {})
        if rgb_channel.get("status") == "available":
            rgb = self.load(rgb_channel["artifacts"]["rgb"], True)
            if rgb.shape[:2] != z.shape:
                raise ValueError("Stereo colour differs from the shared RGB grid")
            colors = rgb[v, u]
        else:
            colors = np.tile((100, 190, 220), (len(points), 1)).astype(np.uint8)
        self.add(self.server.scene.add_point_cloud(f"/{label}/observed_stereo", points=points, colors=colors, point_size=0.003))

    def _scene(self, label, channel, channels, frame):
        import trimesh

        np = self.np
        arrays = self.load(channel["artifacts"]["trajectory"])
        i = frame["frame_index"]
        if int(arrays["pts"][i]) != frame["source_pts"]:
            raise ValueError("Assembled scene timestamp differs")
        # Existing scene geometry is shown as producer geometry, never labelled fully observed.
        for slot, item in enumerate(channel["objects"]):
            pose = arrays["world_T_object"][i, slot]
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Invalid assembled object transform")
            for geometry in ("visual", "collision"):
                asset = _verified(channel["artifacts"][f"{slot}_{geometry}"], "Scene geometry changed since export")
                mesh = trimesh.load(asset, force="mesh", process=False)
                if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) > 1_000_000:
                    raise ValueError("Scene mesh is unsupported or exceeds bounded viewer vertex budget")
                vertices = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
                self.add(self.server.scene.add_mesh_simple(f"/{label}/object{slot}/{geometry}",
                                                           vertices=vertices, faces=np.asarray(mesh.faces),
                                                           color=(80, 170, 220) if geometry == "visual" else (240, 150, 40),
                                                           opacity=0.45 if geometry == "visual" else 0.15))
        contact = arrays["contact_body"][i] > 0
        points = arrays["contact_position"][i][contact]
        if len(points):
            self.add(self.server.scene.add_point_cloud(f"/{label}/contact_proxy", points=points,
                                                       colors=np.tile((255, 60, 80), (len(points), 1)).astype(np.uint8), point_size=0.008))
        timeline = {"object_state": arrays["object_state"][i].tolist(),
                    "object_filled": arrays["object_filled"][i].tolist(),
                    "hand_source": arrays["hand_source"][i].tolist(),
                    "contact_body": arrays["contact_body"][i].tolist()}
        self.note(f"{label} scene support/contact timeline at selected PTS\n\n"
                  "Geometric contact proxy; object_state 0=resting, 1=moving. Filled poses and completed hands are inferred.\n\n"
                  + "```json\n" + json.dumps(timeline, indent=2) + "\n```")
        selected = np.asarray([f["frame_index"] for f in self.frames], dtype=np.int64)
        if (not np.array_equal(arrays["frame_index"][selected], selected)
                or not np.array_equal(arrays["pts"][selected], [f["source_pts"] for f in self.frames])):
            raise ValueError("Scene timeline differs from selected source PTS sequence")
        states = np.concatenate((arrays["object_state"][selected],
                                 arrays["object_filled"][selected].astype(np.uint8),
                                 arrays["hand_source"][selected],
                                 (arrays["contact_body"][selected] > 0).any(axis=2).astype(np.uint8)), axis=1).T
        if not np.isin(states, [0, 1, 2]).all():
            raise ValueError("Unknown scene timeline state requires a producer adapter")
        palette = np.asarray([(45, 65, 90), (240, 165, 45), (190, 80, 170)], dtype=np.uint8)
        strip = np.repeat(np.repeat(palette[states], 10, axis=0), 4, axis=1)
        marker = self.selected() * 4
        strip[:, marker:marker + 1] = (255, 255, 255)
        self.add(self.server.gui.add_image(strip, label=f"{label} selected-PTS states; white=current"))
        self.note("Rows: object state (one per object), filled (one per object), "
                  "hand source (left/right), contact presence (left/right). Fixed colours: 0=blue, 1=orange, 2=purple. "
                  "Hand source 0=observed, 1=gap completion, 2=parked. Columns follow selected PTS order; gaps are not constant-rate playback.")

    def _annotations(self, label, channels, frame):
        annotation = self.document.get("annotations", {}).get(label)
        row = annotation.get("frames", {}).get(str(frame["frame_index"])) if annotation else None
        if row is None:
            self.note(f"{label} optional annotation channels: not supplied for this source PTS")
            return
        rgb_channel = channels.get("preview_run", {})
        if rgb_channel.get("status") == "available":
            rgb = self.load(rgb_channel["artifacts"]["rgb"], True)
            self.add(self.server.gui.add_image(annotation_overlay(rgb, row, annotation["image_size"], self.cv2),
                                               label=f"{label} producer tracks and reprojection arrows (predicted to observed)"))
        self.note(f"{label} supplied visibility/support/discontinuity and surface provenance\n\n"
                  "Null uncertainty means missing. Member disagreement and scores are uncalibrated.\n\n"
                  + "```json\n" + json.dumps({"states": row["states"], "surfaces": row["surfaces"]}, indent=2) + "\n```")


def _jump(slider, index):
    def jump(_):
        slider.value = index
    return jump


def serve(value):
    """Optional loopback viewer; never installs or launches an estimator."""
    # Admission happens before opening an arbitrary producer record.
    runtime = _json(FEATURE / "runtime.json")
    if runtime.get("serve_authorized") is not True:
        raise PermissionError("Inspection serving remains disabled")
    path = project_path(ROOT, value)
    if not path.is_relative_to(ROOT / "outputs/inspection/public"):
        raise ValueError("Only feature-owned inspection exports may be served")
    relative = path.relative_to(ROOT / "outputs/inspection/public")
    if len(relative.parts) != 3 or relative.parts[0] not in {f"episode_{i:06d}" for i in EPISODES}:
        raise PermissionError("Inspection path must identify an admitted episode before opening")
    run = read_run(ROOT, path, kind="inspection.export", allowed=("video_only",))
    _gate("serve", run["episode_index"])
    if importlib.metadata.version("viser") != "1.0.30":
        raise RuntimeError("Use an existing separately approved Viser 1.0.30 environment; no installation is performed")
    document = _json(_artifact(path, "inspection.json", run["artifacts"]["inspection.json"])["path"])
    # Configure runtime-owned caches before importing optional viewer/decoder modules.
    _confine_caches()
    import cv2
    import numpy as np
    import viser

    server = viser.ViserServer(host="127.0.0.1")
    from viser.transforms import SO3
    viewer = _Viewer(server, document, cv2, np, SO3)
    viewer.slider.on_update(viewer.safe_draw)
    position = {frame["frame_index"]: i for i, frame in enumerate(viewer.frames)}
    for event in document["events"]:
        button = server.gui.add_button(f"{event['candidate']} frame {event['frame_index']} · {event['producer']['path']}")
        button.on_click(_jump(viewer.slider, position[event["frame_index"]]))
    for event in document["manual_events"]:
        button = server.gui.add_button(f"Manual {event['kind']} [{event['start_frame']}, {event['stop_frame']}) · {event['reviewer']}")
        button.on_click(_jump(viewer.slider, position[event["start_frame"]]))
    viewer.safe_draw()
    return server
