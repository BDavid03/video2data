# Episode 38 figures: build notes

CPU only, with no scoring or packing. The generator takes an episode parameter for
`video_policy_chart` and `video_policy_footage`, which draw figures 20 and 21.

| Quantity | Value |
| --- | ---: |
| Midpoint mean position / rotation | 25.162517 cm / 74.558209 degrees |
| Final mean position / rotation | 1.422456 cm / 1.591761 degrees |
| Held still against video goal | 10.256631 cm / 50.123557 degrees |
| First pose, unaligned | 0.298645 cm / 0.266501 degrees |
| Reprojection median | 0.778234, blind to camera drift |
| Final official frames / resets | 263 / 0 (saved record) |
| CPU replay frames / wall time | 263 / 65.60 s |
| Replay vs official, unaligned mean / maximum | 0.631987 cm / 1.837995 cm |
| Figure 21 decoded video frames / bytes | 132 / 2,402,706 |
| Restored video-reference translation discrepancy | 2.98e-8 m |
| Leaderboard shape CD-O | 0.53562 cm, from the upload record |
| Leaderboard pose scores at build time | None; five rejected boards |

The final error curves were recomputed during the report build. The replay/official
distances and held-still means were recomputed separately with NumPy. No test suite ran.

The overlay shows a larger pan contour, a filled handle and floating robot hands that
sometimes leave the frame. This is not a recovered human grasp, and foreground occlusion
is ignored.

The stops and resumes, lost replay, no-reset result and check verdict come from the saved
outcome; this build did not rerun training, the scorer or the checks. Why the midpoint
failed is unknown. The video goal's world accuracy, camera drift, grip realism and
performance against the still-object export are unmeasured. Episode 38 has no labels and
none were read.

## Commands

CPU replay:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 data/.runtime/flashchord/venv/bin/python -u -B scripts/report/policy_hands.py --checkpoint data/outputs/moving_policy/ep38-v1/training/policy_66763776.safetensors --task data/outputs/moving_policy/ep38-v1/motion/sequence_id=episode_000038/robot_name=sharpa_wave/task.parquet --output data/outputs/visuals/policy-hands/ep38-video --official data/outputs/moving_policy/ep38-v1/score-66763776/trajectory.parquet > data/outputs/visuals/policy-hands/ep38-video/replay.log 2>&1
```

Rebuild:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. nice -n 19 data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py > data/outputs/visuals/episode38-report-build.log 2>&1
```

File lists: `data/outputs/visuals/episode38-report-files.txt` and `episode38-report-audit.json`.
