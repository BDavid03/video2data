"""Static-objects pose baseline scored with the kit's own Track 3 metric functions."""
import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import trimesh

EPISODES = [0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 42, 45]
KIT = "upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/AUC.py"
ROOT = glob.glob("datasets--nvidia--video_to_data_challenge/snapshots/*/track_3/public")[0]


def load_kit():
    spec = importlib.util.spec_from_file_location("kit_auc", KIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vertices(name):
    path = f"{ROOT}/mesh/{name}/{name}.glb"
    if not os.path.exists(path):
        path = f"{ROOT}/mesh/{name}/{name}_visual.glb"
    scene = trimesh.load(path, force="mesh")
    points, _ = trimesh.sample.sample_surface(scene, 1000, seed=0)
    return np.asarray(points, dtype=np.float64)


def load_reference(ep):
    df = pd.read_parquet(f"{ROOT}/data/chunk-000/episode_{ep:06d}.parquet", columns=["observation.objects"])
    names, poses, vis = None, [], []
    for row in df["observation.objects"]:
        row = sorted(list(row), key=lambda o: o["name"])
        names = [o["name"] for o in row]
        poses.append([np.asarray(o["pose"], float) for o in row])
        vis.append([bool(o["visible"]) for o in row])
    poses = np.array(poses)  # T,B,7 as [x, y, z, qw, qx, qy, qz]
    vis = np.array(vis)
    for b in range(poses.shape[1]):
        idx = np.flatnonzero(vis[:, b])
        if len(idx) == 0:
            raise ValueError(f"episode {ep} object {names[b]} never visible")
        for t in range(poses.shape[0]):
            if not vis[t, b]:
                poses[t, b] = poses[idx[np.abs(idx - t).argmin()], b]
    # Dataset pose is [x, y, z, qw, qx, qy, qz] (track_3/README.md); the kit wants xyz + xyzw.
    out = np.concatenate([poses[..., 0:3], poses[..., 4:7], poses[..., 3:4]], axis=-1)
    out[..., 3:] /= np.linalg.norm(out[..., 3:], axis=-1, keepdims=True)
    return names, out[:, None]  # T,1,B,7


def score(k, ach, ref, verts):
    ach = k._warp_to_reference_start(ach, ref)
    ids = np.arange(ref.shape[2])
    return {
        "AUC": k._add_auc(k._object_add(ach, ref, verts)),
        "SP_SR": float(np.mean(k._spider_success(ach, ref))),
        "MP_SR": float(np.mean(k._maniptrans_success(ach, ref, ids))),
        "RPE_cm": k._relative_position_error_cm(ach, ref),
        "MPPE_cm": k._mean_per_point_error_cm(ach, ref),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    k = load_kit()
    meta = {}
    for line in open(f"{ROOT}/meta/episodes_metadata.jsonl"):
        d = json.loads(line)
        ei = d.get("episode_index")
        if ei in EPISODES:
            meta[ei] = d
    vcache = {}
    res = {}
    for ep in EPISODES:
        names, ref = load_reference(ep)
        for n in names:
            vcache.setdefault(n, vertices(n))
        verts = np.stack([vcache[n] for n in names])
        static = np.repeat(ref[:1], ref.shape[0], axis=0)
        mis = static.copy()
        if mis.shape[2] > 1:
            mis[:, :, 1, 0] += 0.05
        disp = np.linalg.norm(ref[..., :3] - ref[:1, ..., :3], axis=-1)
        rot = np.degrees(k._quaternion_geodesic_angle(ref[..., 3:], ref[:1, ..., 3:]))
        res[str(ep)] = {
            "objects": names,
            "n_objects": len(names),
            "frames": int(ref.shape[0]),
            "task": {a: b for a, b in meta.get(ep, {}).items() if a != "episode_index"},
            "mean_displacement_cm": float(disp.mean() * 100),
            "max_displacement_cm": float(disp.max() * 100),
            "max_rotation_deg": float(rot.max()),
            "perfect": score(k, ref.copy(), ref, verts),
            "static": score(k, static, ref, verts),
            "static_slot1_off_5cm": score(k, mis, ref, verts),
        }
        print(ep, res[str(ep)]["static"], flush=True)
    summary = {}
    for v in ("perfect", "static", "static_slot1_off_5cm"):
        summary[v] = {m: float(np.mean([r[v][m] for r in res.values()])) for m in res[str(EPISODES[0])][v]}
    summary["static_SP_SR_pass_episodes"] = [int(e) for e, r in res.items() if r["static"]["SP_SR"] > 0]
    out = {
        "note": "AUC uses our own 1000-point surface samples (seed 0) of the public scans, approximating the host's vertices. "
                "Slot order = sorted by name. worlds=1. Episodes exclude 41/43/44.",
        "episodes": res,
        "means": summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    md = ["| variant | AUC | SP-SR | MP-SR | RPE cm | MPPE cm |", "|---|---|---|---|---|---|"]
    for v in ("perfect", "static", "static_slot1_off_5cm"):
        s = summary[v]
        md.append(f"| {v} | {s['AUC']:.3f} | {s['SP_SR']:.3f} | {s['MP_SR']:.3f} | {s['RPE_cm']:.2f} | {s['MPPE_cm']:.2f} |")
    md.append("")
    md.append("AUC uses our own vertex samples (approximation of the host's). Mean over 17 public episodes.")
    open(args.out[:-5] + ".md", "w").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
