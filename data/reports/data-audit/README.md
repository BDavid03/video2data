# Track 3 local dataset audit

Pinned dataset revision: `5f68335f3acc802033d1e80728c1633197521de8`.
The dataset cache was opened read-only. All derived artifacts live under this
project. No downloads, installation, GPU processing, or evaluation labels were
used. Evaluation videos were decoded only for integrity and timing checks.

Run from the project root with the already installed Python environment:

```powershell
. ./scripts/use-local-artifacts.ps1
python -m track3.data.audit --decode
python -m unittest discover -s tests -p 'test_data*.py' -v
```

The audit uses Python standard-library hashing and discovery. The existing
`pyarrow` installation reads public parquet; the existing `imageio_ffmpeg`
FFmpeg 7.1 binary decodes videos with `-hwaccel none -noautorotate`, two CPU
decoder threads, `showinfo`, and null output. Raw frames are not persisted.
Missing capabilities and malformed required data produce a nonzero CLI exit.
CLI report destinations must stay within this project and outside immutable
dataset, upstream and Git storage.

## Results

- All **169 Track 3 files** match pinned tree sizes and hashes, totaling
  **1,469,035,134 bytes**. LFS files use content SHA-256; other files use Git blob
  SHA-1 including the `blob <size>\0` header.
- Explicit discovery finds **20 public episodes and 20 evaluation episodes**,
  each with cameras `ego_cam_a`, `ego_cam_b`, `ego_cam_c`. Public episode IDs are
  `0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45`.
  The metadata split string `0:46` is never expanded.
- All **120 videos fully decode**, representing **31,281 decoded camera
  frames**: 15,738 public and 15,543 evaluation. No decode errors were observed.
- Public parquet contains **5,246 rows and 8,492 object observations**. Object
  membership, names, asset files, URDF mesh references, episode/frame indices,
  finite timestamps, pose direction and quaternion norms pass. All observed
  public object poses are visible and valid. The adapter nevertheless rejects
  invisible, zero, nonfinite and nonunit poses, and does not interpolate them.
- All three streams in every episode have equal frame counts and integer PTS.
  Every public row count equals each associated camera's decoded frame count.
  All clips have 20 fps, time base 1/10240, initial PTS zero and 0.05-second
  intervals. No duplicate/backward timestamps or adjacent equal decoded-frame
  checksum candidates were found. No container rotation/display-matrix lines
  were reported; calibration rotation still applies separately.
- Public `timestamp` uses the metadata nominal rate of approximately 20.2541
  Hz, while MP4 PTS uses 20 Hz. The maximum public timestamp/PTS difference is
  **0.233375 seconds**. `capture_time` has its own origin and jitter. Keep all
  three time domains explicit and pair frames by verified row/frame index.

Container timestamps cannot establish physical exposure synchronization, nor
prove that acquisition dropped no frames before encoding. Physical timing and
calibration checks remain in `reports/calibration/`. Frame checksums are Adler-32
diagnostics from FFmpeg, not cryptographic integrity checks.

## Artifacts and adapter

- `manifests/dataset/inventory.json`: exact per-file expected and observed size,
  algorithm and hash, plus the pinned tree's SHA-256.
- `manifests/dataset/episodes.json`: explicit sparse episode discovery and
  snapshot-relative paths. Evaluation records have no trajectory or task label.
- `reports/data-audit/trajectories.json`: public identities, assets, raw pose
  metadata, validity counts and full public timestamp/capture-time arrays.
- `reports/data-audit/videos.json`: per-clip full-decode status, dimensions,
  original integer PTS and seconds, frame checksums and container metadata.
- `reports/data-audit/audit.json`: final failures, totals, elapsed CPU audit time
  and per-episode pairing comparisons. This file is the authoritative run result.

`track3.data.discover_episodes(snapshot)` returns serializable episode records.
`track3.data.iter_public_frames(snapshot, episode)` yields explicit frame-indexed
public diagnostic object poses in xyz+wxyz order with validity and reason. It
refuses evaluation episodes and requires a declared `world_T_object` convention.
The public metadata says object xy centering is baked into poses and the offset
is not recorded; the adapter preserves it and performs no coordinate alignment.
Norm tolerance is 1e-3, matching the metric protocol. Mesh geometry units and
physical properties are not independently validated by this file audit.

The first complete run took 98.701 seconds on CPU. Final rerun timing is recorded
in `audit.json`. Derived audit data is approximately 2.4 MB; no GPU memory was
allocated by this audit. Peak process RAM was not instrumented.
