# Depth

Generates per-frame monocular depth with MoGe2 on a calibrated pinhole image. Runs need
`execution.model_execution_authorized` in `data/configs/perception/moge/local.json`, `execution.generate_authorized` in
`runtime.json`, and the split and episode allowlists. A disabled flag refuses before importing torch or opening a video.
Episode 41 is refused; `episode41.json` is a retained request only.

```bash
PYTHONPATH=. data/.runtime/linux-cuda/bin/python -m src.features.depth --request src/features/depth/generate.json
bash src/features/depth/depth_opening_frames.sh <split> <episode> [<episode> ...]
```

`source scripts/linux/use-linux-cuda.sh` sets up a manual shell with the MoGe source trees on the path.

`generate.json` selects `ego_cam_a`, frames 0 to 19 of public 39, maximum side 768, 1,200 model tokens and mixed
precision. `start_frame` is inclusive, `stop_frame` exclusive (null means end of clip), and `frame_stride` defaults to 1.
No Parquet trajectories, hand labels or object poses are opened.

## Geometry

Decoded RGB gets no extra rotation. All 14 distortion coefficients go into one OpenCV remap to a centred pinhole image.
Optical axes are x right, y down, z forward. MoGe2 gets the field of view derived from the output focal length, and its
normalised K is converted with `[[W,0,-0.5],[0,H,-0.5],[0,0,1]] @ K_normalized`. This does not establish metric accuracy,
stereo agreement or an official Track 3 convention.

Each frame keeps its index, integer MP4 PTS and PTS in seconds. Parquet timestamps and capture times are not read. The
clocks are not interchangeable.

## Output

Each run is a new directory under `data/outputs/depth/<source_split>/episode_XXXXXX/`, and results are never overwritten.

- `run.json`: request, provenance, preprocessing, per-frame status, GPU telemetry, timing, hashes.
- `source-files.json`: hashes of the feature and pinned source trees.
- `preprocessing.npz`: output-to-source pixel maps, usable-input mask, output K.
- `frames/frame_000000.npz`:

| Field | Meaning |
| --- | --- |
| `depth_m` | HxW float32 camera-z depth, metres as estimated by MoGe2, NaN when invalid |
| `valid` | Model mask, usable sample and positive finite depth |
| `model_mask`, `input_valid` | MoGe2 binary mask; geometric sampling mask |
| `model_intrinsics_normalized`, `model_intrinsics_pixels`, `input_intrinsics_pixels` | 3x3 K values |
| `normal_camera`, `normal_valid` | Optional normals |

The model mask is not a calibrated confidence. Nothing is interpolated or smoothed. A completed run is not a completed
reconstruction. The run record uses `kind: depth.generate` and `provenance: video_only`; consumers require completed tagged
producers and refuse untagged legacy records.

## GPU and storage

Uses the cooperative lock at `data/.runtime/gpu-stage.lock` and never removes another process's lock. The default allowance
of 4,096 MiB plus 2,048 MiB headroom is provisional and unmeasured. Sampling via `nvidia-smi` can miss short peaks and cannot
interrupt a CUDA call. Host RAM is unmeasured. Caches stay in the repository, the checkpoint must already exist, and
offline Hub access is on.

## Offline challengers (never run)

```bash
python -m src.features.depth.challenger --request <request.json>
```

`challenger.py` ingests existing ViPE, Depth Anything 3 or Video Depth Anything exports, behind
`challenger_ingest_authorized` (false). It launches no model, downloads nothing and fits no scale. `ingest-challenger.json`
shows the request shape. Admits public 0, 11, 12, 18, 21, 23, 26, 31, 39, 42 only, at most 384 frames, maximum side 518.

Export schema 1 needs: `approach` (`vipe`, `da3`, `vda`); `status: completed`; `provenance: video_only`; `split: public`;
`episode_index`; `dataset_revision`; `camera: ego_cam_a`; `camera_frame`; `grid` (preprocessing hash);
`source_video_sha256`; `source_time_base_and_frame_rate`; `depth_kind` (`metric_optical_z_m` or `relative_optical_z`);
`provisioning` (commit, archive, weights and licence receipts as `{path, sha256}`, plus `license_review`); and `frames` with
`frame_index`, `video_pts`, `video_pts_seconds`, `path`, `sha256`, whose PTS must match the calibration producer.

Each frame NPZ has `depth`, `valid`, and `intrinsics_pixels` equal to the grid, with optional uncalibrated `confidence`.
ViPE and DA3 may add `extrinsic` and `pose_valid`, and the manifest then needs `pose_world`, `extrinsic_direction` and
`pose_translation_units`. VDA poses are refused. Output is neither aligned to the table world nor shown to be more accurate.
Ingested results never pose as `depth.generate` or `camera_motion.estimate`. Promotion and comparison are unimplemented.

## Preview (never run)

```bash
python -m src.features.depth.preview --request <request.json>
```

`preview.py` writes selected RGB PNGs from a completed depth run, for inspection without hand models, behind `preview_authorized`
(false). `preview.json` shows the request shape. It uses 1 to 64 increasing frames, invokes no model, and makes no quality
or accuracy claims.
