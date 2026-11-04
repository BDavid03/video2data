# Object and hand masks

Finds each tracked object and hand in the colour video. `opening-clusters` uses geometry and no model.
`generate` follows instances through the clip with Grounding DINO and SAM2. `correspondences` tracks image points inside
those masks. `opening-clusters`, `generate` and `correspondences` are implemented but have never been run, all their
authorization flags in `runtime.json` are false, and the detector and segmenter environment under
`data/.runtime/object-masks` is not provisioned. Fixed prompts are initial choices, neither tuned nor validated.

A mask on the wrong object gives a confident wrong pose track. Slot 0 anchors the scorer's alignment, so swapping slots 0 and 1
corrupts every metric for that episode.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.object_masks opening-clusters --request <request.json>
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.object_masks generate --request src/features/object_masks/generate.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.object_masks correspondences --request <request.json>
```

Inputs: the `ego_cam_a` clip, the depth slice's `preprocessing.npz` (masks and depth share pixels), the kit's evaluation object
roster, `data/reports/data-audit/trajectories.json` for public object names, and `prompts.json`. No pose label or public Parquet is
opened. The order is the roster, never sorted, then `hand_0` and `hand_1`.

## opening-clusters

Before a hand arrives, objects stand apart on the table.

1. Take each opening frame's metric point cloud and the table plane from `world.json`.
2. Keep points in a height band above the table, inside its extent.
3. Group neighbours into clusters by 3D distance and drop small ones.
4. Write `cluster_0`, `cluster_1` and so on by size. This operation does not name objects.
5. Stop at the first frame where the count changes or a cluster enters from the border.

It fails with `opening_not_quiet` if a cluster touches the border in the first frame, and records
`fewer_clusters_than_objects` otherwise. Touching objects, such as a lid on a pot, form one cluster.

## generate

The handler decodes the clip once and applies the depth run's remap, then keeps JPEGs at quality 95 in `input_frames/`, hashed.
Detection runs per prompt at each keyframe (default every 10 frames). Hand slots are anonymous and are never called left or right
here. Two objects sharing a prompt, such as the wooden pieces, need `opening_run` and `opening_masks_run` as identity anchors; ambiguous
identities stay absent. SAM2 runs forward and backward between keyframes, with CPU offload. A small or low-score mask stops that
direction, and disagreement between directions marks the instance absent. No mask is carried across a gap.

`score` is the mean foreground sigmoid, not a calibrated probability. A named object with no accepted mask fails the episode with
`instance_not_found`. Partial runs keep their frames, and there is no automatic resume.

`backends.json` targets Transformers 4.56.2 Grounding DINO and a local SAM2 checkpoint, with remote code off. Install records must
pin the detector revision, SAM2 commit, packages, files and assets, and all of that is still null. Proposed weights, not installed:
Grounding DINO Swin-T 694 MB, SAM 2.1 Hiera-Large 898 MB, both Apache-2.0. The `transformers` Grounding DINO avoids a custom CUDA
operator whose recipe does not list the RTX 5080's architecture; this is unverified until a run. The longest scored episode has 421
frames, and memory and runtime are unmeasured.

## Output

`data/outputs/object_masks/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Meaning |
| --- | --- | --- |
| `run.json` | `instances`, `grid` | Names indexing every array; hash of `preprocessing.npz` |
| `frames/frame_XXXXXX.npz` | `masks` | bool KxHxW; layers may overlap |
| | `present`, `score` | Found this frame; heuristic score |
| | `source` | 0 absent, 1 keyframe, 2 propagated, 3 geometric cluster |
| | `anchor_frame_index`, `frame_index`, `pts` | Anchor (-1 when absent) and identity |
| `keyframes.json` | | Indices, boxes, detector scores, prompt |

## correspondences

`correspondences_authorized` is false. Requires completed video-only depth and mask parents with matching hashes. Admits public 0, 11,
12, 18, 21, 23, 26, 31, 39, 42 only. Seeds corners from exclusive masks at the first frame, then uses Lucas-Kanade flow, forward/backward
checks and RANSAC similarity (at least six queries). This is an image geometry baseline and not a 3D model; it can reject correct tracks
and cannot tell occlusion from leaving view. Limits: 384 frames, 256 queries per slot, 16 slots, 1800 s.

Frame JSON holds `tracks` (query IDs scoped to the run), `identities` (`observed` or `unknown`) and `corrections`. Optional corrections need a
hash-pinned file with `schema_version:1`, `provenance:"video_only"`, `masks_run_sha256`, `grid` and `frames`; the software cannot verify how an
operator reached their judgment. SAM3/3.1 and CoTracker were looked at separately as challengers. No completed result of theirs is ingested, and no reliability gain is claimed.

## Open questions

- Is a manual click or box on an evaluation video allowed when detection fails? Manual prompts stay off until the organisers answer.
- Do text prompts separate `wooden_piece_1` from `wooden_piece_2`, and `beige_cup` from `blue_cup`?
