"""CPU analytic intersections with original procedural geometry and materials."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

from track3.paths import interpreter, permitted_root, output_path, project_path
from track3.provenance import input_file, read_run, run_fields, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parent
TRAIN = {0, 11, 12, 18, 21, 23, 26, 31}
REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
HEIGHT, WIDTH = 96, 128
PLACEMENT_ATTEMPTS = 60


def _json(path):
    return json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class SyntheticDepthRequest:
    reference_dataset: str = "outputs/scratch_depth/prepare/20261018T113402.984050Z-d446c33e/dataset.json"
    sample_count: int = 128
    seed: int = 31

    @classmethod
    def from_file(cls, path):
        return cls(**_json(path))

    def __post_init__(self):
        if not isinstance(self.reference_dataset, str) or not self.reference_dataset.strip():
            raise ValueError("Supply the explicit train-only reference dataset.json")
        if type(self.sample_count) is not int or not 1 <= self.sample_count <= 512:
            raise ValueError("sample_count must be an integer in [1,512]")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed must be an integer in [0,2**31-1]")


def _sphere_hit(rays, center, radius):
    """Nearest positive ray parameter and surface normal for a sphere (inf where missed)."""
    import numpy as np
    a, b, c = (rays * rays).sum(-1), -2 * (rays @ center), float(center @ center - radius**2)
    discriminant = b * b - 4 * a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    near, far = (-b - root) / (2 * a), (-b + root) / (2 * a)
    hit = np.where(near > 1e-8, near, far)
    hit = np.where((discriminant >= 0) & (hit > 1e-8), hit, np.inf)
    normal = (rays * np.where(np.isfinite(hit), hit, 0)[..., None] - center) / radius
    return hit, normal


def _box_hit(rays, center, half, axes):
    """Slab-test ray parameter and face normal for an oriented box (inf where missed)."""
    import numpy as np
    direction, local_origin = rays @ axes, -center @ axes
    parallel = abs(direction) < 1e-10
    safe = np.where(parallel, 1., direction)
    t1, t2 = (-half - local_origin) / safe, (half - local_origin) / safe
    low = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(-1)
    high = np.where(parallel, np.inf, np.maximum(t1, t2)).min(-1)
    outside = (parallel & (abs(local_origin) > half)).any(-1)
    hit = np.where(low > 1e-8, low, high)
    hit = np.where(~outside & (high >= low) & (hit > 1e-8), hit, np.inf)
    local_hit = local_origin + direction * np.where(np.isfinite(hit), hit, 0)[..., None]
    face = np.argmax(abs(local_hit / half), axis=-1)
    local_normal = np.zeros_like(rays)
    np.put_along_axis(local_normal, face[..., None], np.take_along_axis(np.sign(local_hit), face[..., None], -1), -1)
    return hit, local_normal @ axes.T


def _render(rays, rng, minimum, maximum):
    """Ray parameter is camera z; nearest positive analytic hit owns each pixel."""
    import numpy as np
    up = np.array([0., -0.8, -0.6])
    u, v = np.array([1., 0., 0.]), np.array([0., -0.6, 0.8])
    offset = float(rng.uniform(0.65, 0.95))
    origin = np.array([0., 0., offset / 0.6])
    denominator = rays @ up
    z = np.full(rays.shape[:2], np.inf)
    np.divide(-offset, denominator, out=z, where=denominator < -1e-10)
    points = rays * np.where(np.isfinite(z), z, 0)[..., None]
    tx, ty = (points - origin) @ u, (points - origin) @ v
    texture = 0.85 + 0.10 * np.sin(tx * rng.uniform(35, 90) + 2 * np.sin(ty * 13))
    texture += rng.normal(0, 0.025, z.shape)
    material = texture[..., None] * rng.uniform(0.25, 0.8, 3)
    normals = np.broadcast_to(up, rays.shape).copy()
    instance_id = np.zeros(z.shape, dtype=np.int16)
    objects, footprints = [], []
    desired = int(rng.integers(1, 4))
    for _ in range(PLACEMENT_ATTEMPTS):
        if len(objects) == desired:
            break
        sphere = bool(rng.integers(0, 2))
        radius = float(rng.uniform(0.055, 0.13))
        half = rng.uniform([0.045, 0.045, 0.04], [0.12, 0.12, 0.13])
        footprint = radius if sphere else float(np.linalg.norm(half[:2]))
        xy = rng.uniform([-0.4, -0.3], [0.4, 0.3])
        if any(np.linalg.norm(xy - position) < footprint + size + 0.025 for position, size in footprints):
            continue
        footprints.append((xy, footprint))
        center = origin + xy[0] * u + xy[1] * v + (radius if sphere else half[2]) * up
        color = rng.uniform(0.08, 0.95, 3)
        if sphere:
            hit, normal = _sphere_hit(rays, center, radius)
            item = {"shape": "sphere", "center_camera_m": center.tolist(), "radius_m": radius}
        else:
            yaw = float(rng.uniform(-np.pi, np.pi))
            axes = np.column_stack((np.cos(yaw) * u + np.sin(yaw) * v, -np.sin(yaw) * u + np.cos(yaw) * v, up))
            hit, normal = _box_hit(rays, center, half, axes)
            item = {"shape": "oriented_box", "center_camera_m": center.tolist(),
                    "half_extents_m": half.tolist(), "axes_camera": axes.tolist()}
        visible = hit < z
        z = np.where(visible, hit, z)
        instance_id = np.where(visible, len(objects) + 1, instance_id).astype(np.int16)
        normals = np.where(visible[..., None], normal, normals)
        material = np.where(visible[..., None], color, material)
        objects.append({**item, "instance_id": len(objects) + 1, "diffuse_rgb": color.tolist(),
                        "footprint_center_table_m": xy.tolist()})
    if not objects:
        raise RuntimeError("Bounded placement produced no objects")
    light = np.array([rng.uniform(-0.6, 0.6), -0.8, -1.0])
    light /= np.linalg.norm(light)
    ambient = float(rng.uniform(0.2, 0.4))
    shade = ambient + (1 - ambient) * np.maximum(normals @ light, 0)
    color = material * shade[..., None] + rng.normal(0, 0.008, rays.shape)
    depth = z.astype(np.float32)
    stored = depth.astype(np.float64)
    valid = np.isfinite(stored) & (stored > 0) & (stored >= minimum) & (stored <= maximum)
    background = rng.uniform(0.04, 0.12, 3)
    rgb = (np.clip(np.where(valid[..., None], color, background), 0, 1) * 255).round().astype(np.uint8)
    scene = {"plane_up_camera": up.tolist(), "plane_offset_m": offset, "objects": objects,
             "light_towards_source_camera": light.tolist(), "ambient_fraction": ambient,
             "placement_attempt_limit": PLACEMENT_ATTEMPTS, "requested_object_count": desired,
             "placed_object_count": len(objects)}
    return (rgb, np.where(valid, depth, np.nan).astype(np.float32), valid,
            np.where(valid, instance_id, 0).astype(np.int16), scene)


def _reference(request):
    """Validated reference metadata, its producer run and its paths."""
    path = project_path(ROOT, request.reference_dataset)
    parent_path = path.parent / "run.json"
    parent = read_run(ROOT, parent_path, kind="scratch_depth.prepare", allowed=("video_only",))
    reference = _json(path)
    identity = input_file(ROOT, path)
    samples = reference.get("samples")
    if (path.name != "dataset.json" or reference.get("producer_run") != parent_path.relative_to(ROOT).as_posix()
            or reference.get("kind") != "scratch_depth_training_pairs" or reference.get("status") != "completed"
            or reference.get("provenance") != parent["provenance"] or reference.get("source_split") != "public"
            or reference.get("subset") != "development" or reference.get("camera") != "ego_cam_a"
            or reference.get("dataset_revision") != REVISION or reference.get("target_kind") != "moge2_pseudo_depth"
            or reference.get("geometry", {}).get("output_size_width_height") != [WIDTH, HEIGHT]
            or parent.get("dataset_sha256") != identity["sha256"]
            or parent.get("artifacts", {}).get("dataset.json") != identity["sha256"]
            or not isinstance(samples, list) or not samples
            or any(not isinstance(row, dict) or type(row.get("episode_index")) is not int
                   or row["episode_index"] not in TRAIN for row in samples)
            or {row["episode_index"] for row in samples} != TRAIN):
        raise ValueError("Reference must be the completed 128x96 train-eight-only public preparation metadata")
    return path, parent_path, reference, identity


def _redirect_temporary_files():
    temporary_root = output_path(ROOT, ".runtime/synthetic_depth/tmp")
    bytecode_root = output_path(ROOT, ".cache/synthetic_depth/pycache")
    temporary_root.mkdir(parents=True, exist_ok=True)
    bytecode_root.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temporary_root), TMP=str(temporary_root), TMPDIR=str(temporary_root))
    tempfile.tempdir = str(temporary_root)
    os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix = str(bytecode_root)


def _camera_rays(reference):
    """Unit-z pixel-centre rays from the reference intrinsics, after checking intrinsics and depth bounds."""
    import numpy as np
    k = np.asarray(reference["geometry"]["input_intrinsics_pixels"], dtype=np.float64)
    minimum, maximum = reference["minimum_depth_m"], reference["maximum_depth_m"]
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0
            or type(minimum) not in (int, float) or type(maximum) not in (int, float)
            or not np.isfinite([minimum, maximum]).all() or not 0 < minimum < maximum):
        raise ValueError("Reference needs finite canonical pinhole intrinsics and positive ordered depth bounds")
    y, x = np.indices((HEIGHT, WIDTH))
    return k, np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)), -1)


def _save_sample(destination, k, rgb, depth, valid, instance_id):
    import numpy as np
    temporary = destination.with_suffix(".npz.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, rgb=rgb, depth_m=depth, valid=valid, instance_id=instance_id,
                                input_intrinsics_pixels=k.astype(np.float32))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _save_preview(path, previews):
    """Four-column RGB montage of the first samples."""
    import cv2
    import numpy as np
    montage = np.zeros((((len(previews) + 3) // 4) * HEIGHT, 4 * WIDTH, 3), dtype=np.uint8)
    for index, rgb in enumerate(previews):
        row, column = divmod(index, 4)
        montage[row * HEIGHT:(row + 1) * HEIGHT, column * WIDTH:(column + 1) * WIDTH] = rgb
    if not cv2.imwrite(str(path), montage[..., ::-1]):
        raise RuntimeError("Could not save synthetic preview montage")


def generate_synthetic_depth(request: SyntheticDepthRequest):
    runtime_path = FEATURE / "runtime.json"
    runtime = _json(runtime_path)
    if runtime["execution"].get("generate_authorized") is not True:
        raise PermissionError("Synthetic depth generation remains disabled")
    permitted_root()
    if Path(sys.executable).resolve() != interpreter(ROOT, runtime["python"]):
        raise RuntimeError("Select the approved project interpreter (runtime.json python for this platform)")
    path, parent_path, reference, identity = _reference(request)
    _redirect_temporary_files()
    import numpy as np

    k, rays = _camera_rays(reference)
    minimum, maximum = reference["minimum_depth_m"], reference["maximum_depth_m"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    directory = output_path(ROOT, f"outputs/synthetic_depth/{run_id}")
    record = run_fields(ROOT, "synthetic_depth", "generate", asdict(request), inputs=[path, runtime_path],
                        parents=[parent_path], allowed=("video_only",))
    parameter_hash = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode()).hexdigest()
    record.update(schema_version=1, run_id=run_id, artifacts={}, samples=[], data_origin="procedural_synthetic",
                  training_only=True, tests_run=False, validation_run=False, official_evaluation=False,
                  model_execution=False, simulation_executed=False, research_only=True, submission_candidate=False,
                  parameter_sha256=parameter_hash, started_utc=datetime.now(timezone.utc).isoformat())
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "samples").mkdir()
    started, previews = time.perf_counter(), []
    write_json(directory / "run.json", record)
    try:
        for index in range(request.sample_count):
            sample_seed = request.seed + index
            rgb, depth, valid, instance_id, scene = _render(rays, np.random.default_rng(sample_seed), minimum, maximum)
            name = f"samples/synthetic_{index:06d}.npz"
            _save_sample(directory / name, k, rgb, depth, valid, instance_id)
            digest = sha256_file(directory / name)
            record["artifacts"][name] = digest
            record["samples"].append({"sample_id": f"synthetic_{index:06d}", "seed": sample_seed,
                                      "artifact": name, "sha256": digest, "valid_pixels": int(valid.sum()), "scene": scene})
            if index < 16:
                previews.append(rgb)
            write_json(directory / "run.json", record)
        _save_preview(directory / "preview.png", previews)
        document = {"schema_version": 1, "kind": "synthetic_depth_training_pairs", "status": "completed",
                    "data_origin": "procedural_synthetic", "training_only": True, "provenance": record["provenance"],
                    "producer_run": (directory / "run.json").relative_to(ROOT).as_posix(),
                    "reference_camera_source": identity, "geometry": reference["geometry"],
                    "minimum_depth_m": minimum, "maximum_depth_m": maximum, "target_units": "analytic optical-axis metres",
                    "target_kind": "analytic_ray_intersection", "samples": record["samples"],
                    "instance_labels": "Per-sample int16: 0 table/background; 1..3 procedural objects; valid distinguishes measured table from background",
                    "parameter_sha256": parameter_hash, "sources": record["sources"],
                    "render_limits": "Procedural diffuse colors; no cast shadows, photorealism, physics or sensor-noise validation"}
        write_json(directory / "dataset.json", document)
        record["artifacts"].update({name: sha256_file(directory / name) for name in ("dataset.json", "preview.png")})
        record.update(status="completed", generated_samples=len(record["samples"]))
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(timing={"wall_seconds": time.perf_counter() - started}, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / "run.json", record)
    return directory / "dataset.json"
