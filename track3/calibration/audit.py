"""Reproduce numeric calibration evidence; optionally inspect sparse public frames.

Run: python -m track3.calibration.audit --snapshot PATH --output reports/calibration
Add --image-check to use already installed OpenCV (no installs performed).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .geometry import load_calibration, validate_calibration, matmul, rigid_inverse
from track3.paths import output_path
from track3.provenance import sha256_file

CAMERAS = ("ego_cam_a", "ego_cam_b", "ego_cam_c")


def _sample_frames(cv2, public, episode, output):
    """Grey first/middle frames per camera plus four review frames of ego_cam_a; writes the review images."""
    frames, counts, cells, review_indices = {}, {}, {}, []
    for camera in CAMERAS:
        cap = cv2.VideoCapture(str(public / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"),
                               cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        counts[camera] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames[camera] = {}
        indices = {0, counts[camera] // 2}
        if camera == "ego_cam_a":
            review_indices = [round(i * (counts[camera] - 1) / 3) for i in range(4)]
            indices.update(review_indices)
        for index in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode {camera} frame {index}")
            frames[camera][index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if camera == "ego_cam_a" and index in review_indices:
                cell = np.full((515, 640, 3), 255, dtype=np.uint8)
                cell[:480] = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                cv2.putText(cell, f"Episode {episode} | frame {index} | MP4 PTS {index/20:.2f}s",
                            (10, 503), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
                cells[index] = cell
            if index == 0:
                cv2.imwrite(str(output / f"episode_{episode:06d}_{camera}_frame000000.jpg"), frame)
        cap.release()
    sheet = np.vstack([np.hstack([cells[i] for i in review_indices[:2]]),
                       np.hstack([cells[i] for i in review_indices[2:]])])
    cv2.imwrite(str(output / f"episode_{episode:06d}_rgb_review_contact_sheet.jpg"), sheet)
    return frames, counts, review_indices


def _mutual_matches(matcher, da, db):
    def ratio(query, train):
        return [m for group in matcher.knnMatch(query, train, k=2) if len(group) == 2
                for m, n in [group] if m.distance < .7 * n.distance]
    backward = {(m.trainIdx, m.queryIdx) for m in ratio(db, da)}
    return [m for m in ratio(da, db) if (m.queryIdx, m.trainIdx) in backward]


def _hypothesis(cv2, p, q, a, b, essential, r, t, unrotate, rectify):
    """Sampson (and optionally rectified) residuals for one pixel-orientation hypothesis."""
    normalized = []
    for points, camera in ((p, a), (q, b)):
        pts = np.asarray(camera["resolution"]) - 1 - points if unrotate else points.copy()
        normalized.append(cv2.undistortPointsIter(
            pts.reshape(-1, 1, 2), np.asarray(camera["intrinsic_matrix"]),
            np.asarray(camera["distortion_coefficients"]), None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2))
    x, y = [np.column_stack([n, np.ones(len(n))]) for n in normalized]
    ex, ety = x @ essential.T, y @ essential
    numerator = np.abs(np.sum(y * ex, axis=1))
    denom = np.sqrt(ex[:, 0]**2 + ex[:, 1]**2 + ety[:, 0]**2 + ety[:, 1]**2)
    sampson = numerator / np.maximum(denom, 1e-20) * b["fx"]  # approximate stereo-B pixel units
    measure = {"sampson_stereo_pixel_equivalent_median": float(np.median(sampson)),
               "sampson_stereo_pixel_equivalent_p90": float(np.quantile(sampson, .9)),
               "fraction_below_2px_equivalent": float(np.mean(sampson < 2))}
    if rectify:
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            np.asarray(a["intrinsic_matrix"]), np.asarray(a["distortion_coefficients"]),
            np.asarray(b["intrinsic_matrix"]), np.asarray(b["distortion_coefficients"]),
            tuple(a["resolution"]), r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        rectified = []
        for pts, rr, pp in ((x, r1, p1), (y, r2, p2)):
            homogeneous = pts @ rr.T @ pp[:3, :3].T
            rectified.append(homogeneous[:, :2] / homogeneous[:, 2:])
        disparity = rectified[0][:, 0] - rectified[1][:, 0]
        vertical = np.abs(rectified[0][:, 1] - rectified[1][:, 1])
        measure.update(rectified_vertical_abs_px_median=float(np.median(vertical)),
                       rectified_vertical_abs_px_p90=float(np.quantile(vertical, .9)),
                       positive_disparity_fraction=float(np.mean(disparity > 0)))
    return measure


def image_check(snapshot, calibration, output):
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    public = snapshot / "track_3/public"
    episodes = [json.loads(line) for line in (public / "meta/episodes_metadata.jsonl").read_text().splitlines()]
    episode = min(row["episode_index"] for row in episodes if row["task"] == "Pan_Solo")
    frames, counts, review_indices = _sample_frames(cv2, public, episode, output)
    results = {"opencv_version": cv2.__version__, "episode_index": episode,
               "selection": "lowest public Pan_Solo id; calibration diagnostic only",
               "frame_counts_from_container": counts, "decoder_autorotation": False,
               "review_contact_sheet_frame_indices": review_indices,
               "review_sampling": "uniformly spaced review samples, not detected contact/release events",
               "method": "SIFT mutual Lowe-ratio 0.7 matches; no geometry-fitted inlier filter; full 14-coefficient iterative undistortion",
               "pairs": []}
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher()
    for first, second in (("ego_cam_c", "ego_cam_b"), ("ego_cam_a", "ego_cam_b")):
        a, b = calibration["cameras"][first], calibration["cameras"][second]
        transform = np.asarray(a["extrinsics_to_stereo_left"], dtype=float)
        r, t = transform[:3, :3], transform[:3, 3]
        essential = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ r
        for index in sorted(set(frames[first]) & set(frames[second])):
            ka, da = sift.detectAndCompute(frames[first][index], None)
            kb, db = sift.detectAndCompute(frames[second][index], None)
            if da is None or db is None:
                results["pairs"].append({"first": first, "second": second, "frame": index, "status": "no_descriptors"})
                continue
            matches = _mutual_matches(matcher, da, db)
            if not matches:
                raise RuntimeError("no mutual matches for diagnostic")
            p = np.array([ka[m.queryIdx].pt for m in matches], dtype=float)
            q = np.array([kb[m.trainIdx].pt for m in matches], dtype=float)
            results["pairs"].append({"first": first, "second": second, "frame": index, "mutual_matches": len(matches),
                                     "hypotheses": {name: _hypothesis(cv2, p, q, a, b, essential, r, t, unrotate,
                                                                      first == "ego_cam_c")
                                                    for name, unrotate in (("decoded_pixels_as_calibrated", False),
                                                                           ("undo_180_in_pixels", True))}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-check", action="store_true")
    args = parser.parse_args()
    args.output = output_path(Path(__file__).resolve().parents[2], args.output)
    started = time.perf_counter()
    path = args.snapshot / "track_3/public/meta/camera_calibration.json"
    calibration = load_calibration(path)
    results = validate_calibration(calibration)
    results["source"] = {"snapshot": args.snapshot.name, "calibration_sha256": sha256_file(path),
                         "readme_sha256": sha256_file(args.snapshot / "track_3/README.md")}
    cameras = calibration["cameras"]
    b_t_c = cameras["ego_cam_c"]["extrinsics_to_stereo_left"]
    b_t_a = cameras["ego_cam_a"]["extrinsics_to_stereo_left"]
    pair = next(p for p in calibration["stereo_pairs"] if p["camera"] == "ego")
    results["ego_transform_anchor_evidence"] = {
        "interpretation": "extrinsics_to_stereo_left uses historical CAM_B anchor, not rotated pair's named left CAM_C",
        "cam_c_extrinsic_vs_pair_left_to_right_max_abs": float(np.abs(np.subtract(b_t_c, pair["left_to_right"])).max()),
        "ego_cam_c_T_ego_cam_a": matmul(rigid_inverse(b_t_c), b_t_a),
        "status": "direction inferred from matrix equality/schema and supported by sparse image checks; RGB residual remains material"}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.image_check:
        results["sparse_image_check"] = image_check(args.snapshot, calibration, args.output)
    results["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "numeric-audit.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output / "numeric-audit.json"), "elapsed_seconds": results["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
