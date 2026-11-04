# Depth

Pipeline role: step 2 (shapes). It runs MoGe-2 on the video to get per-frame depth. The mesh_extents slice measures
object sizes from this depth.

MoGe-2 is a pretrained monocular depth model: it predicts depth from one image. Its scale is not anchored to a ruler.
Any depth score in this project compares against MoGe-2 pseudo-labels, not measured depth.

## Run

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.depth --request src/features/depth/generate.json
bash src/features/depth/depth_opening_frames.sh <split> <episode> [<episode> ...]
```

`source scripts/linux/use-linux-cuda.sh` sets up a manual shell with the MoGe source trees on the path.

A run needs three switches on: `execution.model_execution_authorized` in `data/configs/perception/moge/local.json`,
`execution.generate_authorized` in `runtime.json`, and the split and episode lists. With a switch off it refuses before
importing torch or opening a video. Episode 41 is refused; `episode41.json` is kept only as a record. Episodes 43 and 44
are held out.

`generate.json` picks camera `ego_cam_a`, frames 0 to 19 of public episode 39, longest side 768, 1,200 model tokens,
mixed precision. `start_frame` is inclusive, `stop_frame` exclusive (null means end of clip), `frame_stride` defaults
to 1. Other requests are in `requests/`. No pose labels, hand labels or object poses are opened.

## Geometry

All 14 lens-distortion terms go into one OpenCV remap to a centred pinhole image. Axes: x right, y down, z forward.
MoGe-2 gets the field of view from the output focal length. Its normalised intrinsics become pixels with
`[[W,0,-0.5],[0,H,-0.5],[0,0,1]] @ K_normalized`. Each frame keeps its index and its MP4 timestamp. Parquet timestamps are
not read; the clocks are not interchangeable.

## Output

Each run is a new folder under `data/outputs/depth/<split>/episode_XXXXXX/`. Nothing is overwritten.

- `run.json`: request, provenance, per-frame status, GPU use, timing, hashes. Kind `depth.generate`, provenance
  `video_only`. Consumers accept only completed runs with these tags.
- `source-files.json`: hashes of this slice and the pinned source trees.
- `preprocessing.npz`: output-to-source pixel maps, usable-input mask, output intrinsics.
- `frames/frame_000000.npz`:

| Field | Meaning |
| --- | --- |
| `depth_m` | HxW float32 depth along the camera axis, in MoGe-2's metres, NaN when invalid |
| `valid` | Model mask, usable pixel and positive finite depth |
| `model_mask`, `input_valid` | MoGe-2 mask; geometric sampling mask |
| `model_intrinsics_normalized`, `model_intrinsics_pixels`, `input_intrinsics_pixels` | 3x3 intrinsics |
| `normal_camera`, `normal_valid` | Optional normals |

The model mask is not a calibrated confidence. Nothing is smoothed or filled in.

## GPU

Uses the shared GPU lock at `data/.runtime/gpu-stage.lock` and never removes another live process's lock. The default
memory allowance is 4,096 MiB plus 2,048 MiB headroom, not measured. Caches stay in the checkout; the checkpoint must
already exist; Hugging Face access is offline.

## Unused tools (switched off, never run)

- `python -m src.features.depth.challenger --request <request.json>` imports existing ViPE, Depth Anything 3 or Video
  Depth Anything exports (`challenger_ingest_authorized: false`). It runs no model and fits no scale.
  `ingest-challenger.json` shows the request shape; the export schema is checked in `challenger.py`.
- `python -m src.features.depth.preview --request <request.json>` writes RGB PNGs from a finished run
  (`preview_authorized: false`). `preview.json` shows the request shape.
