"""Estimate observed per-frame and per-episode metric scale without filling gaps."""

from src.provenance import write_json
from ._common import (artifact, authorize, frame_file, operation_run, producer, same_revision, select_episode,
                      selected_indices)
from .support import PARTITION, episode_location, reserved_grid


def _producers(request):
    depth = producer(request.depth_run, None, request)
    if depth[1]["kind"] not in ("depth.generate", "depth.ingest_challenger"):
        raise ValueError("Expected calibrated depth or an admitted challenger")
    if depth[1]["kind"] == "depth.ingest_challenger" and request.holdback_modulus != 5:
        raise PermissionError("Challenger anchoring requires the disabled research mode and disjoint holdback")
    stereo = producer(request.stereo_run, "stereo_depth.generate", request)
    masks = None
    if request.masks_run is not None:
        masks = producer(request.masks_run, None, request)
        if masks[1]["kind"] not in ("object_masks.generate", "object_masks.opening_clusters"):
            raise ValueError("Expected a mask producer")
    return depth, stereo, masks


def _support(depth_parent, stereo_parent, masks, index, pts, k, grid_shape, record, np):
    """Monocular and stereo depth on the shared grid, with their joint positive support."""
    depth_record = depth_parent[1]
    with np.load(frame_file(*depth_parent, index, pts, record), allow_pickle=False) as arrays:
        key = "depth_relative" if depth_record.get("depth_kind") == "relative_optical_z" else "depth_m"
        depth, valid = arrays[key].copy(), arrays["valid"].astype(bool)
        depth_k = arrays["intrinsics_pixels" if depth_record["kind"] == "depth.ingest_challenger"
                         else "input_intrinsics_pixels"].copy()
    with np.load(frame_file(*stereo_parent, index, pts, record), allow_pickle=False) as arrays:
        stereo, stereo_valid = arrays["rgb_depth_m"].copy(), arrays["rgb_valid"].astype(bool)
        stereo_k = arrays["rgb_intrinsics_pixels"].copy()
    if (stereo.shape != grid_shape or depth.shape != grid_shape or valid.shape != grid_shape
            or stereo_valid.shape != grid_shape or not np.allclose(k, depth_k, atol=1e-5)
            or not np.allclose(k, stereo_k, atol=1e-5)):
        raise ValueError("Anchor depth dimensions or intrinsics disagree")
    valid &= stereo_valid & np.isfinite(depth) & np.isfinite(stereo) & (depth > 0) & (stereo > 0)
    if masks is not None:
        with np.load(frame_file(*masks, index, pts, record), allow_pickle=False) as arrays:
            layers = arrays["masks"].astype(bool)
            if layers.ndim != 3 or layers.shape[1:] != grid_shape:
                raise ValueError("Mask dimensions disagree with the depth grid")
            valid &= ~layers.any(axis=0)
    return depth, stereo, valid


def anchor(request):
    authorize("anchor", request)
    selection = select_episode(request)
    depth, stereo, masks = _producers(request)
    (depth_path, depth_record), (stereo_path, stereo_record) = depth, stereo
    parents = [depth_path, stereo_path] + ([masks[0]] if masks is not None else [])
    with operation_run("anchor", request, selection, parents) as (directory, record):
        import numpy as np

        grid = depth_record["preprocessing_artifact_sha256"]
        if stereo_record["grid"] != grid or (masks is not None and masks[1]["grid"] != grid):
            raise ValueError("Anchor inputs use different colour pixel grids")
        same_revision([depth, stereo], selection["revision"])
        grid_file = artifact(depth_path, "preprocessing.npz", record, grid)
        with np.load(grid_file, allow_pickle=False) as arrays:
            k, grid_shape = arrays["input_intrinsics_pixels"].copy(), arrays["input_valid"].shape
        record["grid"] = grid
        scales = []
        region = "outside_supplied_masks" if masks is not None else "unsegregated"
        for index in selected_indices(request, selection["stop"]):
            pts = selection["clocks"]["ego_cam_a"]["pts_integer"][index]
            row = {"frame_index": index, "video_pts": pts, "status": "running"}
            record["frames"].append(row)
            depth_values, stereo_values, valid = _support(depth, stereo, masks, index, pts, k, grid_shape, record, np)
            support = eligible_support = int(valid.sum())
            reserved_support = 0
            if request.holdback_modulus:
                heldback = reserved_grid(request.episode_index, index, grid_shape, np)
                reserved_support = int((valid & heldback).sum())
                valid &= ~heldback
                support = int(valid.sum())
            row.update(support_region=region, grid_pixels=int(valid.size),
                       support_share=float(support / valid.size), eligible_support_pixels=eligible_support,
                       reserved_support_pixels=reserved_support)
            if support < request.minimum_support_pixels:
                row.update(status="completed", scale_status="insufficient_support", scale=None,
                           support_pixels=support, median_absolute_deviation=None)
            else:
                ratios = stereo_values[valid].astype(np.float64) / depth_values[valid]
                scale = float(np.median(ratios))
                scales.append(scale)
                row.update(status="completed", scale_status="observed", scale=scale, support_pixels=support,
                           median_absolute_deviation=float(np.median(abs(ratios - scale))),
                           ratio_quantiles=np.quantile(ratios, [0.1, 0.5, 0.9]).tolist(),
                           log_ratio_mad=float(np.median(abs(np.log(ratios) - np.log(scale)))))
            write_json(directory / "run.json", record)
        if len(scales) / len(record["frames"]) < request.minimum_frame_share:
            raise ValueError("insufficient_support: too few frames have an observed metric scale")
        episode_scale = episode_location(scales, request.episode_aggregation, np)
        document = {"grid": grid, "split": request.source_split, "episode_index": request.episode_index,
                    "frames": record["frames"], "episode_scale": episode_scale,
                    "frames_used": len(scales), "provenance": record["provenance"],
                    "frames_requested": len(record["frames"]),
                    "frame_support_share": len(scales) / len(record["frames"]),
                    "episode_log_scale_mad": float(np.median(abs(np.log(scales) - np.log(episode_scale)))),
                    "support_region": region,
                    "holdback_partition": PARTITION if request.holdback_modulus else None,
                    "holdback_modulus": request.holdback_modulus,
                    "query_identity": "a:{row}:{column}",
                    "episode_aggregation": request.episode_aggregation,
                    "support_note": "Mask exclusion is not proof of static background; unsegregated support may include moving objects and hands.",
                    "method": "median_stereo_over_monocular", "masks_excluded": masks is not None}
        write_json(directory / "anchor.json", document)
    return directory / "run.json"
