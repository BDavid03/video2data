# Plain-language report: how to rebuild it

`report.md` is our plain-language report with graphs and video overlays (README steps 3 and 4: static
poses and the episode 38 moving policy). An **episode** is one recorded video clip from the challenge
dataset. `scripts/report/nontechnical_figures.py` writes `report.md` and everything in `figures/`. Do not
edit `report.md` by hand.

## Rebuild

From the checkout root:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. nice -n 19 data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py
```

This runs on CPU from saved outputs. It reads cached camera frames and masks for episodes 39 and 38.
It reads no scans and no challenge labels. It never scores or packs. Videos must stay under 5 MB.

Short names used in the report:

- **M0c**: a two-handed hold on the kit's example task. A diagnostic only.
- **M2**: the dust-pan policy trained on episode 39 against the motion-capture path.
- **M2v**: the same setup, trained against the path we tracked from the video.
- **Candidate C**: the video tracker we used for that path.
- **AUC**: the main pose score, area under the success curve (0 to 1, higher is better).

## Replay a policy first (optional)

`--replay` reruns all four CPU policy replays before drawing. `--replay-pan` runs M2 only.
`--replay-video-policy` runs M2v on episode 39. `--replay-video-policy 38` runs episode 38 only.
Replays use the FlashCHORD interpreter. The M0c log is `data/outputs/visuals/policy-hands/m0c/replay.log`,
with `status.json` and `summary.json` beside it.

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py --replay-video-policy
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. nice -n 19 data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py --replay-video-policy 38
```

To replay one trained policy directly:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu data/.runtime/flashchord/venv/bin/python -u -B scripts/report/policy_hands.py \
  --checkpoint data/outputs/moving_policy/ep39-a1/training/policy_66763776.safetensors \
  --task data/outputs/moving_policy/ep39-a1/motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet \
  --output data/outputs/visuals/policy-hands/ep39 \
  --official data/outputs/moving_policy/ep39-a1/score-66763776/trajectory.parquet
```

For another checkpoint, pass its task, an output folder and its matching export, or `--no-official`.
The checkpoint must match the export's metadata. Only FlashSAC policies work. The replay changes only:
one world, CPU, deterministic actions, reset at frame zero, and the upstream evaluation setting that
removes assistance. A new replay must be added to the builder by hand.

Replay output: per-step body poses (xyzw), goal poses and timing; `geometry.npz` for drawing;
`contacts.json` with contact partners, positions and raw simulator forces. Contact counts do not show
a firm or natural grip.

Two constants near the top of the generator set the text: `EP38_IN_ENTRY` and `POSE_SCORES`.
Provenance: `data/outputs/visuals/inputs.json` (hashed inputs per figure) and
`data/outputs/visuals/build-summary.json` (numeric checks, video frames, paths).

## What the figures show, and what they do not

Policy pictures are drawings from a fixed camera, using recorded link positions and the scene's
meshes simplified to 1.5 mm. Blue dashed lines are the goal's bounding box, aligned to the mesh.
(Audit of an earlier, wrongly rotated box: `data/outputs/visuals/goal-box-angle-audit.json`.) Over 390
steps the angle between achieved and goal pose is 0.515 degrees mean, 1.932 maximum in CPU replay, and
0.539 / 1.977 in the official export.

| Figure | Shows | Limits |
| --- | --- | --- |
| M0c | Two-handed hold on the example task, then release. CPU replay differs from the GPU path by 0.097 cm mean, 0.365 cm maximum | Example task only. Not a challenge result, score or pack input |
| 13 pan masks | Saved pan masks on camera frames | Episode 38's first try picked the table; a smallest-box rule fixed it. Hands hide parts of the pan. At 10 s in 39 one hand mask marks a bystander's hand |
| 14 pan learning | Three M2 checkpoints: AUC 0.168, 0.951, 0.982. Holding still scores about 0.17 | Save times 16.73, 25.82, 57.62 min come from file times and include start-up |
| 15 pan policy hands | Robot hands and our pan mesh. CPU replay differs from official by 0.305 cm mean, 2.118 cm maximum | The hands hold the pan like a tray with open fingers. Contact on 177 of 180 raised steps does not show a firm grip. One seed, motion-capture goal, practice episode 39 |
| 16 tracked pan | Candidate C's tracked pan drawn on real frames, 39 and 38, 10 fps | Ignores hands in front of the pan. The pan's body axes were once read from the public scan |
| 17 video vs mocap | The one sealed comparison on 39: holding still 0.18, video path 0.83, motion-capture policy 0.98 | The three bars measure different things, and are labelled so |
| 18 policy on video path | M2v: AUC 0.774882445, aligned error 2.668767 cm, 8.312726 degrees | The first automatic verdict was NULL: it compared video-world and mocap-world poses directly. Aligning the first pose, with the same limits, gave GO. Old scorer |
| 19 policy on real footage | Simulated pan and replayed hands on 39. Replay differs by 0.409 cm mean, 2.102 cm maximum | The simulated pan is larger and thicker than the real one. Robot hands are not the person's hands. The 0.80 overlap used the smaller tracker mesh |
| 20 episode 38 goal | Midpoint at 32.81M steps: 25.16 cm, 74.56 degrees. End at 66.76M: 1.42 cm, 1.59 degrees. Holding still: 10.26 cm, 50.12 degrees | Against its own video goal. No labels, no challenge score. Grey lines are whole-clip means |
| 21 episode 38 on footage | Official pan through the fitted camera, replayed hands. Replay differs by 0.632 cm mean, 1.838 cm maximum over 263 samples | Same limits as 19. The camera can drift. The 0.778 overlap uses the smaller tracker mesh |

Points the report states plainly:

- The first video tracker failed. Candidate C failed its first frozen checks and the amendment. We
  continued under a rule written before the sealed comparison with labels. This is weaker evidence.
- The checks do not show that episode 38's tracked path is accurate.
- Episode 38's training was stopped and resumed twice and lost its replay buffer each time. Why it
  failed its midpoint check is not known.
- Every pose number from before 4 November uses the old scorer.

Other records: `data/outputs/visuals/episode39-setup.json` (older setup),
`data/outputs/moving_reference/ep39/results.md` (candidate A),
`data/outputs/visuals/pan-checkpoint-times.json`.
