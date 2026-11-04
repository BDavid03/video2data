# Episode 38 report build

Status: complete, CPU only; no scoring or packing. The lead reviews and commits.

The generator now takes an episode parameter for `video_policy_chart` and
`video_policy_footage`. Figure 20 compares the failed saved midpoint with the
scheduled end against the policy's own video goal. Figure 21 draws the official
achieved pan and CPU replay hands through the frozen camera, beside the mask.
The report replaces the live-run placeholder with the completed outcome and
adds the shape leaderboard finding plus the old-scorer warning near the top.
`EP38_IN_ENTRY = False` and `POSE_SCORES = "none yet"` are near the generator top.

| Quantity | Recorded / recomputed |
| --- | ---: |
| Midpoint mean position / rotation | 25.162517 cm / 74.558209 degrees |
| Final mean position / rotation | 1.422456 cm / 1.591761 degrees |
| Held still against video goal | 10.256631 cm / 50.123557 degrees |
| First pose, unaligned | 0.298645 cm / 0.266501 degrees |
| Reprojection median | 0.778234, blind to camera drift |
| Final official frames / resets | 263 / 0 (saved record) |
| CPU replay frames / wall time | 263 / 65.60 seconds |
| Replay vs official, unaligned mean / maximum | 0.631987 cm / 1.837995 cm |
| Figure 21 decoded video frames / bytes | 132 / 2,402,706 |
| Restored video-reference translation discrepancy | 2.98e-8 m |
| Leaderboard shape CD-O | 0.53562 cm, from the upload record |
| Leaderboard pose scores | none yet; five rejected boards |

The report build independently recomputes the final error curves. A separate
NumPy calculation reproduced the replay/official distances from saved arrays.
The held-still position and angle means were independently recomputed too.
Syntax compilation and CPU import passed; old/new pure position and orientation
functions returned identical arrays on the final episode-38 export. No test
suite ran. `report-before-hashes.json` retains the baseline; the final audit
records all 24 pre-existing figure hashes unchanged and no missing files.

I inspected figure 20 and figure 21 as images, plus four decoded MP4 previews
in `review-episode38-on-real-footage.png`. The chart clearly separates the
failed midpoint from the final export. The overlay shows a larger pan contour,
a filled handle, and floating robot hands, sometimes extending off screen.
This visible mismatch is preserved and disclosed. It is not a recovered human
grasp. Foreground occlusion is ignored.

Only the saved outcome supports the stops/resumes, lost training memory,
no-reset result, final gate verdict and entry cut-off status; this job did not
rerun training, the scorer or those gates. The leaderboard fact comes from
`docs/chain/SUBMISSION.md`; no network lookup or upload was made. Why the
midpoint failed is unknown. The video goal's world accuracy, camera drift,
grip realism and performance against the still-object export are unmeasured.
No labels exist for 38 and none were requested or read.

## Commands run

All Python work ran at nice 19; BLAS threads were 1 and rendering set OpenCV
threads to 1. Existing interpreters only, with the GPU hidden.

CPU replay (one run):

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 data/.runtime/flashchord/venv/bin/python -u -B scripts/report/policy_hands.py --checkpoint data/outputs/moving_policy/ep38-v1/training/policy_66763776.safetensors --task data/outputs/moving_policy/ep38-v1/motion/sequence_id=episode_000038/robot_name=sharpa_wave/task.parquet --output data/outputs/visuals/policy-hands/ep38-video --official data/outputs/moving_policy/ep38-v1/score-66763776/trajectory.parquet > data/outputs/visuals/policy-hands/ep38-video/replay.log 2>&1
```

Default rebuild (run twice, the second after tightening axis/caption caveats):

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. nice -n 19 data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py > data/outputs/visuals/episode38-report-build.log 2>&1
```

The CPU import/compilation used `importlib.util.spec_from_file_location` and
`py_compile.compile`, writing only `data/outputs/visuals/nontechnical_figures.pyc`.
One-off CPU scripts recomputed errors and SHA-256 hashes. Read-only `git diff`
and `git status` inspected the allowed source/report paths. No install,
mutating Git operation, process stop, GPU operation, score, pack or upload ran.

## Files changed

See `data/outputs/visuals/episode38-report-files.txt` for every new or
content-changed file, including replay compiler caches. Existing presentation
files rewritten with identical contents are separately identified in
`episode38-report-audit.json`. Inputs, dataset, models, runtime source,
`src/features/pose_insurance/` and `scripts/reproduce/` were not edited.

No requested report work remains. Adoption, new-kit pose scoring, explaining
the midpoint failure and uploading remain outside this job.
