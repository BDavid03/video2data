"""Recorded kit pose metrics for the two authorised public tuning episodes."""
import argparse
from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
from track3.paths import output_path
from track3.provenance import run_fields, write_json
from track3.runs import (ROOT, read_json, producer, artifact, runtime_flag, authorize_episode,
                         heavy_execution, recorded, new_run_id)

FEATURE = Path(__file__).parent


@dataclass(frozen=True)
class DiagnosticRequest:
    episode_index: int
    policy_run: str
    source_split: str = "public"

    def __post_init__(self):
        if self.source_split != "public" or self.episode_index not in (39, 42):
            raise PermissionError("Pose diagnostics are restricted to public tuning episodes 39/42")


def diagnose(request):
    runtime = runtime_flag(FEATURE, "diagnostic_authorized", "Public pose diagnostic")
    authorize_episode(runtime, request, "public pose diagnostic")
    heavy_execution(dict(runtime, python=runtime["diagnostic_python"]))
    policy = producer(request.policy_run, "pose_insurance.run", request)
    sampling_path = FEATURE / "diagnostic_sampling.json"
    sampling = read_json(sampling_path)
    baseline_path = ROOT / "scripts/track3/static_pose_baseline.py"
    metric_path = ROOT / "upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/AUC.py"
    dataset = next(ROOT.glob("datasets--*/snapshots/*/track_3/public"))
    reference = dataset / f"data/chunk-000/episode_{request.episode_index:06d}.parquet"
    mesh_paths = []
    for row in policy[1]["scene"]:
        name = row["name"]
        path = dataset / f"mesh/{name}/{name}.glb"
        mesh_paths.append(path if path.exists() else path.with_name(f"{name}_visual.glb"))
    record = run_fields(ROOT, "pose_insurance", "diagnostic", asdict(request), parents=[policy[0]],
                        inputs=[reference, *mesh_paths, baseline_path, metric_path, sampling_path,
                                FEATURE / "runtime.json", "outputs/pose_eval/static-baseline.json"],
                        provenance="public_label_diagnostic")
    record.update(split="public", episode_index=request.episode_index, sampling=sampling)
    directory = output_path(ROOT, f"outputs/pose_insurance/diagnostic-v1/{new_run_id()}")
    with recorded(directory, record):
        trajectory = artifact(policy, "trajectory.parquet", record)
        import numpy as np
        import pandas as pd
        import trimesh
        spec = importlib.util.spec_from_file_location("insurance_static_metric", baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        names, reference_poses = baseline.load_reference(request.episode_index)
        df = pd.read_parquet(trajectory)
        if set(names) != set(df.object_body_name.unique()):
            raise ValueError("Export and reference object names differ")
        cols = [f"achieved_position_{s}" for s in "xyz"] + [f"achieved_quaternion_{s}" for s in "xyzw"]
        poses = np.stack([df[df.object_body_name == name].sort_values("reference_frame")[cols].to_numpy()
                          for name in names], axis=1)[:,None]
        if poses.shape != reference_poses.shape:
            raise ValueError("Export and reference timelines differ")
        vertices = []
        for name in names:
            path = next(p for p in mesh_paths if p.parent.name == name)
            mesh = trimesh.load(path, force="mesh")
            points, _ = trimesh.sample.sample_surface(mesh, sampling["surface_points_per_object"], seed=sampling["seed"])
            vertices.append(points)
        vertices = np.stack(vertices)
        kit = baseline.load_kit()
        oracle = np.repeat(reference_poses[:1], len(reference_poses), axis=0)
        existing = read_json("outputs/pose_eval/static-baseline.json")
        metrics = {"learned":baseline.score(kit, poses, reference_poses, vertices),
                   "oracle_static":baseline.score(kit, oracle, reference_poses, vertices),
                   "oracle_17_episode_mean":existing["means"]["static"],
                   "note":"Kit pose functions; slot-0 alignment; xyzw export; sampled public surfaces approximate host vertices."}
        record["metrics"] = metrics
        write_json(directory / "metrics.json", metrics)
    return directory / "run.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(diagnose(DiagnosticRequest(args.episode, args.run)))
