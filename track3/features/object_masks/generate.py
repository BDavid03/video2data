"""Named full-video masks from detector keyframes and bounded SAM2 propagation."""

import shutil

from track3.gpu import GpuLease
from track3.paths import project_path
from track3.provenance import input_file, sha256_file, write_json
from track3.runs import ROOT, artifact, producer, read_json, require_input
from ._generate_common import FEATURE, authorize, operation_run
from .opening_clusters import _object_names, _save
from ._identity import assign, suppress
from ._models import detect, detector, provisioned, segmenter


def _episode(request, record):
    """Ordered object roster and the pinned ego_cam_a RGB clock of the episode."""
    names = _object_names(request, record["dataset_revision"])
    clocks = read_json("reports/data-audit/videos.json")
    rows = [row for row in clocks["videos"] if row["split"] == request.source_split and
            row["episode_index"] == request.episode_index and row["camera"] == "ego_cam_a"]
    if (clocks["status"] != "completed" or clocks["revision"] != record["dataset_revision"] or
            len(rows) != 1 or rows[0]["status"] != "ok"):
        raise ValueError("A completed pinned RGB clock inventory is required")
    clock = rows[0]
    if clock["decoded_frame_count"] <= 0 or len(clock["pts_integer"]) != clock["decoded_frame_count"]:
        raise ValueError("Incomplete mask RGB clock")
    return names, clock


def _opening_identity(opening, clusters, names, pts, grid, record, np):
    if opening is None:
        return {}
    if opening[1].get("grid") != grid or clusters[1].get("grid") != grid:
        raise ValueError("Opening identity uses a different pixel grid")
    require_input(opening, clusters)
    document = read_json(artifact(opening, "opening.json", record))
    if set(document["object_assignments"]) != set(names):
        raise ValueError("Opening assignments differ from the episode object roster")
    rows = [row for row in clusters[1]["frames"] if row["frame_index"] == 0]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["video_pts"] != int(pts[0]):
        raise ValueError("Opening identity needs an observed frame-zero cluster mask")
    path = artifact(clusters[0], rows[0]["artifact"], record, rows[0]["sha256"])
    with np.load(path, allow_pickle=False) as arrays:
        masks = arrays["masks"].copy()
    if masks.ndim != 3 or masks.dtype != np.dtype("bool"):
        raise ValueError("Invalid opening identity mask arrays")
    previous = {}
    for slot, name in enumerate(names):
        layer = document["object_assignments"][name]
        if type(layer) is not int or not 0 <= layer < len(masks):
            raise ValueError("Invalid opening object-to-cluster assignment")
        y, x = np.nonzero(masks[layer])
        if not len(x):
            raise ValueError("Opening identity assigned an empty cluster")
        previous[slot] = {"frame": 0, "cluster": layer,
                          "box": np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=np.float64)}
    return previous


def _propagate(runtime, window, start, end, seeds, valid, request, lease, np):
    predictor, torch = runtime
    directions = []
    for reverse, anchor in ((False, start), (True, end)):
        if reverse and start == end:
            directions.append({})
            continue
        prompts = seeds[anchor]
        found = {}
        if not prompts:
            directions.append(found)
            continue
        state = None
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lease.require_headroom()
                state = predictor.init_state(str(window), offload_video_to_cpu=True,
                                             offload_state_to_cpu=True, async_loading_frames=False)
                for slot, prompt in prompts.items():
                    predictor.add_new_points_or_box(state, frame_idx=anchor - start, obj_id=slot + 1,
                                                    box=np.asarray(prompt["box"], dtype=np.float32))
                lost, visited = set(), set()
                for local_index, object_ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=anchor - start, max_frame_num_to_track=end - start + 1, reverse=reverse):
                    index = start + int(local_index)
                    if not start <= index <= end or index in visited or len(object_ids) != len(logits):
                        raise ValueError("SAM2 returned a frame/object grid outside its window")
                    visited.add(index)
                    if {int(value) - 1 for value in object_ids} != set(prompts) or len(object_ids) != len(prompts):
                        raise ValueError("SAM2 changed the prompted object roster")
                    lease.require_headroom()
                    for position, object_id in enumerate(object_ids):
                        slot = int(object_id) - 1
                        if slot not in prompts:
                            raise ValueError("SAM2 returned an unprompted instance")
                        if slot in lost:
                            continue
                        values = logits[position].detach().float().cpu().numpy()
                        if values.shape != (1, *valid.shape) or not np.isfinite(values).all():
                            raise ValueError("SAM2 returned malformed mask logits")
                        values = values[0]
                        mask = (values > 0) & valid
                        score = float(np.mean(1 / (1 + np.exp(-np.clip(values[mask], -60, 60))))) if mask.any() else 0.0
                        if int(mask.sum()) < request.minimum_mask_pixels or score < request.minimum_mask_score:
                            lost.add(slot)
                            continue
                        found[index, slot] = {"mask": mask, "score": score, "anchor": anchor,
                                              "detector_score": prompts[slot]["score"]}
                if visited != set(range(start, end + 1)):
                    raise ValueError("SAM2 did not return every frame in its propagation window")
        finally:
            if state is not None:
                predictor.reset_state(state)
                del state
        directions.append(found)
    return directions


def _select(directions, index, slot, request, np):
    candidates = [items[index, slot] for items in directions if (index, slot) in items]
    if len(candidates) == 2:
        a, b = [item["mask"] for item in candidates]
        union = int((a | b).sum())
        agreement = int((a & b).sum()) / union if union else 0
        if agreement < request.direction_agreement_iou:
            return None
    return min(candidates, key=lambda item: (abs(index - item["anchor"]), -item["score"])) if candidates else None


def _prompt_groups(names, previous):
    """Fixed prompt per instance slot, grouped so slots sharing a prompt are assigned jointly."""
    prompts = read_json(FEATURE / "prompts.json")
    texts = [prompts["objects"][name] for name in names] + [prompts["hand"]] * 2
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every mask instance needs an explicit fixed prompt")
    groups = {text: [slot for slot, value in enumerate(texts) if value == text] for text in dict.fromkeys(texts)}
    for group in groups.values():
        if len(group) > 1 and any(slot < len(names) for slot in group):
            if any(slot not in previous for slot in group) or len({previous[slot]["cluster"] for slot in group}) != len(group):
                raise ValueError("Indistinguishable named objects require distinct measured opening assignments")
    return texts, groups


def _source_video(depth, clock, record):
    source = read_json("configs/sources.lock.json")["dataset"]
    video = project_path(ROOT, depth[1]["source_video"])
    expected = project_path(ROOT, f"{source['cache_path']}/snapshots/{source['revision']}/{clock['path']}")
    entry = input_file(ROOT, video)
    if video != expected or entry["sha256"] != depth[1]["input_hashes"]["video_sha256"]:
        raise ValueError("Mask RGB differs from depth or clock inventory")
    record["inputs"].append(entry)
    return video


def _prepare_frames(video, count, depth, grid_maps, input_frames, directory, record, cv2):
    """Decode, undistort onto the depth grid and persist numbered JPEGs (SAM2's directory loader format)."""
    mx, my, valid = grid_maps
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open mask RGB video")
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        for index in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Mask RGB decode stopped at frame {index}")
            if list(bgr.shape[1::-1]) != depth[1]["preprocessing"]["source_size_width_height"]:
                raise ValueError("Mask RGB dimensions differ from depth preprocessing")
            bgr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            bgr[~valid] = 0
            path = input_frames / f"{index:06d}.jpg"
            if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError("Could not persist calibrated mask input frame")
            record["artifacts"][path.relative_to(directory).as_posix()] = sha256_file(path)
    finally:
        capture.release()


def _detect_keyframes(prepared, keyframes, input_frames, pts, groups, previous, names, shape,
                      request, lease, keyframe_log, cv2, np):
    """Detector boxes per prompt at each keyframe, assigned to identity slots as SAM2 seeds."""
    seeds = {}
    with detector(prepared) as runtime:
        for index in keyframes:
            bgr = cv2.imread(str(input_frames / f"{index:06d}.jpg"))
            if bgr is None:
                raise RuntimeError("Prepared mask keyframe could not be read")
            rgb, seeds[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {}
            row = {"frame_index": index, "video_pts": int(pts[index]), "detections": [], "assignments": []}
            keyframe_log["frames"].append(row)
            for text, group in groups.items():
                lease.require_headroom()
                boxes = suppress(detect(runtime, rgb, text, request, np), shape, request, np)
                row["detections"].append({"prompt": text, "boxes_xyxy_score": boxes.tolist()})
                assigned = assign(boxes, group, previous, index, shape,
                                  all(slot >= len(names) for slot in group), request, np)
                for slot, box in assigned.items():
                    seeds[index][slot] = {"box": box[:4].tolist(), "score": float(box[4])}
                    previous[slot] = {"frame": index, "box": box[:4].copy()}
                    row["assignments"].append({"slot": slot, **seeds[index][slot]})
    return seeds


def _write_frame(directory, index, directions, instances, valid, pts, record, request, np):
    """Persist one frame's selected masks and append its completed record row; returns presence."""
    slots = len(instances)
    masks = np.zeros((slots, *valid.shape), dtype=bool)
    score = np.full(slots, np.nan, dtype=np.float32)
    source_kind = np.zeros(slots, dtype=np.uint8)
    anchor_frame = np.full(slots, -1, dtype=np.int64)
    for slot in range(slots):
        selected = _select(directions, index, slot, request, np)
        if selected is None:
            continue
        masks[slot], score[slot] = selected["mask"], selected["score"]
        anchor_frame[slot] = selected["anchor"]
        source_kind[slot] = 1 if index == selected["anchor"] else 2
    present = masks.reshape(slots, -1).any(axis=1)
    path = directory / "frames" / f"frame_{index:06d}.npz"
    _save(path, np, masks=masks, present=present, score=score, source=source_kind,
          anchor_frame_index=anchor_frame, frame_index=np.int64(index), pts=pts[index])
    relative = path.relative_to(directory).as_posix()
    digest = sha256_file(path)
    record["artifacts"][relative] = digest
    record["frames"].append({"frame_index": index, "video_pts": int(pts[index]),
        "status": "completed", "artifact": relative, "sha256": digest,
        "present_instances": [name for name, keep in zip(instances, present) if keep]})
    write_json(directory / "run.json", record)
    return present


def generate(request):
    authorize(request)
    depth = producer(request.depth_run, "depth.generate", request)
    opening = producer(request.opening_run, "object_pose.initialise", request) if request.opening_run else None
    clusters = producer(request.opening_masks_run, "object_masks.opening_clusters", request) if request.opening_masks_run else None
    parents = [depth] + ([opening, clusters] if opening else [])
    if opening:
        require_input(opening, depth)
        require_input(clusters, depth)
    with operation_run(request, parents) as (directory, record):
        prepared = provisioned(record)
        import cv2
        import numpy as np

        names, clock = _episode(request, record)
        instances = [*names, "hand_0", "hand_1"]
        count = clock["decoded_frame_count"]
        pts = np.asarray(clock["pts_integer"], dtype=np.int64)
        if (np.diff(pts) <= 0).any():
            raise ValueError("Mask RGB clock must be strictly increasing")
        grid = depth[1]["preprocessing_artifact_sha256"]
        with np.load(artifact(depth[0], "preprocessing.npz", record, grid), allow_pickle=False) as arrays:
            mx, my, valid = arrays["source_x"].copy(), arrays["source_y"].copy(), arrays["input_valid"].copy()
        if valid.ndim != 2 or valid.dtype != np.dtype("bool") or mx.shape != valid.shape or my.shape != valid.shape:
            raise ValueError("Invalid mask pinhole grid")
        previous = _opening_identity(opening, clusters, names, pts, grid, record, np)
        texts, groups = _prompt_groups(names, previous)
        video = _source_video(depth, clock, record)
        record.update(grid=grid, instances=instances, object_names=names,
                      score_kind="mean_sigmoid_foreground_logits", absence_policy="stop_at_first_failed_mask_in_each_direction")
        keyframes = sorted({*range(0, count, request.keyframe_interval), count - 1})
        input_frames = directory / "input_frames"
        input_frames.mkdir()
        (directory / "frames").mkdir()
        lease = None
        keyframe_log = {"grid": grid, "instances": instances, "frames": [], "prompts": texts}
        seen = np.zeros(len(instances), dtype=bool)
        try:
            record["phase"] = "prepare_frames"
            _prepare_frames(video, count, depth, (mx, my, valid), input_frames, directory, record, cv2)
            with GpuLease(ROOT, request.gpu_headroom_mib, request.gpu_admission_mib, stage="object_masks.generate") as lease:
                record["phase"] = "detect_keyframes"
                seeds = _detect_keyframes(prepared, keyframes, input_frames, pts, groups, previous, names,
                                          valid.shape, request, lease, keyframe_log, cv2, np)
                import torch
                torch.cuda.empty_cache()
                record["phase"] = "propagate_masks"
                with segmenter(prepared) as segmentation_runtime:
                    windows = list(zip(keyframes[:-1], keyframes[1:])) if len(keyframes) > 1 else [(0, 0)]
                    for start, end in windows:
                        record["active_window"] = {"start_frame": start, "stop_frame": end + 1}
                        window = directory / "window_frames"
                        window.mkdir()
                        try:
                            for index in range(start, end + 1):
                                shutil.copyfile(input_frames / f"{index:06d}.jpg", window / f"{index-start:06d}.jpg")
                            directions = _propagate(segmentation_runtime, window, start, end, seeds,
                                                    valid, request, lease, np)
                            # Each window owns [start, end); the clip's last keyframe closes the final window.
                            for index in range(start, end + 1 if end == count - 1 else end):
                                seen |= _write_frame(directory, index, directions, instances, valid, pts,
                                                     record, request, np)
                            del directions
                        finally:
                            shutil.rmtree(window)
                del segmentation_runtime
            if [row["frame_index"] for row in record["frames"]] != list(range(count)):
                raise ValueError("Full-video mask output does not cover the exact clip frame grid")
            record.pop("active_window", None)
            missing = [name for name, found in zip(names, seen[:len(names)]) if not found]
            if missing:
                raise ValueError("instance_not_found: " + ", ".join(missing))
            record["phase"] = "completed"
        finally:
            if lease is not None:
                record["gpu"] = lease.report()
            record["ever_present"] = seen.tolist()
            write_json(directory / "keyframes.json", keyframe_log)
            record["artifacts"]["keyframes.json"] = sha256_file(directory / "keyframes.json")
    return directory / "run.json"
