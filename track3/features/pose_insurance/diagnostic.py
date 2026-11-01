"""Local kit metrics on the two authorised tuning episodes, after policy export."""
import importlib.util
import numpy as np
from track3.runs import ROOT, read_json


def score_episode(episode, trajectory):
    if episode not in (39, 42):
        raise PermissionError("Pose-label diagnostics are restricted to tuning episodes 39/42")
    import pandas as pd
    spec = importlib.util.spec_from_file_location("insurance_static_metric", ROOT / "scripts/track3/static_pose_baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    names, reference = baseline.load_reference(episode)
    df = pd.read_parquet(trajectory)
    exported_names = list(df.sort_values("object_body_id").object_body_name.unique())
    if set(names) != set(exported_names):
        raise ValueError(f"Tuning task lacks public objects: {exported_names} versus {names}")
    cols = [f"achieved_position_{s}" for s in "xyz"] + [f"achieved_quaternion_{s}" for s in "xyzw"]
    # Reorder only for this diagnostic to match the baseline's explicitly sorted reference names.
    poses = np.stack([df[df.object_body_name == name].sort_values("reference_frame")[cols].to_numpy() for name in names], axis=1)[:,None]
    if poses.shape != reference.shape:
        raise ValueError(f"Export/reference shapes differ: {poses.shape}, {reference.shape}")
    kit = baseline.load_kit()
    vertices = np.stack([baseline.vertices(name) for name in names])
    oracle = np.repeat(reference[:1], len(reference), axis=0)
    existing = read_json("outputs/pose_eval/static-baseline.json")
    return {"provenance":"public_label_diagnostic", "learned":baseline.score(kit, poses, reference, vertices),
            "oracle_static":baseline.score(kit, oracle, reference, vertices),
            "recorded_oracle_static":existing["episodes"][str(episode)]["static"],
            "oracle_17_episode_mean":existing["means"]["static"],
            "note":"kit pose functions; slot-0-only alignment; 1000 public-mesh surface points, seed=0, approximates host vertices; xyzw export"}


if __name__ == "__main__":
    import argparse
    from track3.provenance import write_json
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(args.out, score_episode(args.episode, args.trajectory))
