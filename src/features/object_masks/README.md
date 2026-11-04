# Object and hand masks

Pipeline role: step 2 (shapes) helper. `mesh_extents` reuses this slice's table-cluster code to find objects in the
depth. The model-based mask tools here are not in the submitted entry; the episode 38 pan masks come from
`moving_reference`.

## Operations

| Operation | What it does | State |
| --- | --- | --- |
| `opening-clusters` | Finds objects as 3D clusters above the table before a hand arrives. No model. | On for public episode 39 only |
| `generate` | Tracks objects and hands through the clip with Grounding DINO and SAM2 | Off, never run, models not installed |
| `correspondences` | Tracks image points inside masks with optical flow | Off, never run |

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.object_masks opening-clusters --request <request.json>
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.object_masks generate --request src/features/object_masks/generate.json
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.object_masks correspondences --request <request.json>
```

Switches and episode lists are in `runtime.json`. Example requests: `opening-clusters.json`, `generate.json`,
`correspondences.json`. Inputs: the `ego_cam_a` clip, the depth slice's `preprocessing.npz` (masks and depth share
pixels), the kit's object list, `data/reports/data-audit/trajectories.json` for public object names, and `prompts.json`.
No pose label is read. Object order follows the kit's list, then `hand_0`, `hand_1`.

Why it matters: a mask on the wrong object gives a confident wrong pose. Slot 0 anchors the scorer's alignment, so
swapping slots 0 and 1 spoils every metric for that episode.

## opening-clusters

1. Take each opening frame's metric point cloud and the table plane from `world.json`.
2. Keep points in a height band above the table, inside its extent.
3. Group nearby points into clusters; drop small ones.
4. Name them `cluster_0`, `cluster_1`, ... by size. It does not name objects.
5. Stop at the first frame where the count changes or a cluster enters from the border.

It fails with `opening_not_quiet` if a cluster touches the border in the first frame. Touching objects (a lid on a pot)
form one cluster. The one recorded run (public 39, under `data/outputs/object_masks/public/`) has status `failed`.

## generate and correspondences (not run)

`generate` detects each prompt at keyframes (every 10 frames), then SAM2 propagates masks forward and backward.
Disagreement marks the object absent; no mask is carried across a gap. `score` is a heuristic, not a probability.
`backends.json` names the intended models (Grounding DINO Swin-T, SAM 2.1 Hiera-Large, both Apache-2.0); none is
installed under `data/.runtime/object-masks`.

`correspondences` seeds corners in the first-frame masks and follows them with Lucas-Kanade flow, forward-backward
checks and RANSAC. It is 2D only.

## Output

`data/outputs/object_masks/<split>/episode_XXXXXX/<run-id>/`

| File | Field | Meaning |
| --- | --- | --- |
| `run.json` | `instances`, `grid` | Names for every array layer; hash of `preprocessing.npz` |
| `frames/frame_XXXXXX.npz` | `masks` | bool KxHxW; layers may overlap |
| | `present`, `score` | Found this frame; heuristic score |
| | `source` | 0 absent, 1 keyframe, 2 propagated, 3 geometric cluster |
| | `anchor_frame_index`, `frame_index`, `pts` | Anchor (-1 when absent) and identity |
| `keyframes.json` | | Indices, boxes, detector scores, prompt |

## Open question

Is a manual click or box on an evaluation video allowed when detection fails? Manual prompts stay off.
