# Rebuilding the non-technical report

From the checkout root, using existing environments only:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B scripts/report/nontechnical_figures.py
```

This rebuilds the report, figures, table and videos from saved outputs. It reads
the cached camera JPEGs and masks for 39 and 38, not scans or challenge labels.
The policy pictures use all recorded link transforms and the installed scene's
visual meshes, simplified by 1.5 mm vertex clustering for presentation. They are
orthographic mesh drawings, with a fixed camera and scale over the whole clip.
Blue dashed lines show the goal object's mesh-oriented bounding box, transformed
from the same body-local frame as the actual visual mesh. The mesh is tilted
within its body frame; using body-axis-aligned bounds previously made the goal
look rotated and too large. Figures 08–10 now use mesh-oriented bounds too and
retain demonstration hand guides, labelled accordingly.

The achieved-to-goal geodesic angle over all 390 steps is **0.515° mean,
1.932° maximum** in the CPU replay, and **0.539° mean, 1.977° maximum** in the
official export. Quaternion order is xyzw in both records. The per-step angle
arrays and bounds diagnosis are in `data/outputs/visuals/goal-box-angle-audit.json`.
The large apparent angle in the earlier drawing was a bounds-frame error.

To rerun both CPU policy diagnostics before rebuilding everything, add `--replay`.
Use `--replay-pan` to rerun only the finished M2 pan policy.
This launches the isolated FlashCHORD interpreter. It never scores or packs the
diagnostic and never accesses the GPU. Its log is
`data/outputs/visuals/policy-hands/m0c/replay.log`; completion is recorded in
`status.json`, with data and comparison details in `summary.json`.

To replay the finished episode-39 policy directly:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu data/.runtime/flashchord/venv/bin/python -u -B scripts/report/policy_hands.py \
  --checkpoint data/outputs/moving_policy/ep39-a1/training/policy_66763776.safetensors \
  --task data/outputs/moving_policy/ep39-a1/motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet \
  --output data/outputs/visuals/policy-hands/ep39 \
  --official data/outputs/moving_policy/ep39-a1/score-66763776/trajectory.parquet
```

For another checkpoint, pass its task, output directory and matching official
export; use `--no-official` if no matching export is available.
Checkpoint identity must match the export's metadata. Only FlashSAC is supported.
The script accepts arbitrary reference lengths and preserves the checkpoint's
frame window, speed, support, collision, timing and curriculum configuration.
Its explicit changes are one world, CPU, deterministic inference, frame-zero
reset and the upstream evaluation composition that removes assistance. CPU
execution uses the same uncaptured `env.step`; GPU graph capture is omitted.
The report renders the fixed M0c and M2 final artifacts. To show another saved
replay, add it deliberately to the report builder after inspecting it.

The replay arrays contain each body's position and xyzw orientation after each
control step, the initial reset state, object body IDs, goal poses and timing.
`geometry.npz` contains the drawing geometry in each body's local frame.
`contacts.json` records object-contact partner links, positions and solver force
vectors at the last physics substep. Forces are raw simulator estimates; contact
counts do not demonstrate force closure, grip strength or naturalness.

The M0c replay differs from the official GPU object path by **0.097 cm mean,
0.365 cm maximum**, over 390 matched post-control-step samples, without alignment.
Both hands have raw object contacts on all **162** steps with the box more than
5 cm above its initial replay height. The drawings show a two-handed hold at
opposite ends, followed by release. This is an example-task diagnostic, never a
challenge result, score or pack input.

`13-pan-masks.png` overlays saved binary masks on their matching resized camera
JPEGs. Amber marks pan pixels; teal and pink mark hand pixels. Episode 38's first
attempt selected the table. The fixed smallest-qualifying-box rule corrects that
selection and is applied to both clips. Occlusion and late handle-hole filling
remain. Episode 38's footage is public challenge data; its scored labels are
unavailable and were not used. Pixel masks do not establish world-space tracking.
At 10 seconds in 39, one hand mask marks a bystander's hand; hand identity is
therefore not reliable throughout this clip.

`14-pan-learning.png` shows the three fixed M2 checkpoint AUC measurements
(0.168, 0.951, 0.982), the held-still level of about 0.17, and the final official
goal/achieved height paths. Checkpoint save times (16.73, 25.82, 57.62 minutes)
come from retained file timestamps relative to the handler start, snapshotted in
`data/outputs/visuals/pan-checkpoint-times.json`. Times include startup; the full
run finished in 57.98 minutes. Evaluations ran afterwards. Lines connect the
measurements without smoothing; no intermediate evaluations are implied.

`15-pan-policy-hands.png` and `.mp4` show the actual policy links and our own pan
mesh in a CPU replay. The goal contours use the same mesh, scale and xyzw body
transform, with view-dependent silhouette and sharp feature edges. The CPU path
differs from the official achieved path by 0.305 cm mean and 2.118 cm maximum.
The hands form an unusual two-handed tray-like support with open fingers; the
model's circular handle is filled in. Both hands have raw contacts on 177 of 180
steps more than 5 cm above the starting origin height. This is not evidence of a
natural or secure grip. Contact forces are large and unverified.

M2 is one seed, trained by trial and error in simulation from nothing, with our
shape and a motion-capture goal on practice episode 39. It is the first moving
result on a challenge-type clip in this project. **Not a challenge result;
scored episodes have no motion capture. A video-derived path is now estimated,
but no policy has followed it yet.** No scan is read by the report rebuild or CPU
diagnostic; the displayed kit measurements were computed earlier by the slice.

`data/outputs/visuals/episode39-setup.json` retains the older dated setup/training
snapshot. The report now uses the completed M2 outcome and recorded metrics.

`16-tracked-pan.png` and `.mp4` project the frozen candidate-C references for
39 and 38 through their fitted cameras onto the saved rectified real frames.
Green is the model's rendered contour, including its handle hole, not the input
segmentation mask. The reference body pose is converted to the own asset frame
using the saved axis/origin constant, then transformed into the fitted camera.
That reconstructed asset pose agrees with the recorded tracking pose to
2.22e-16 numerical precision. No scan geometry is read or drawn. The saved
constant previously used public scan bounding axes; that use is disclosed.
The full outline ignores foreground hand occlusion, so it can cross the hand.
The shorter 38 clip holds its labelled final frame while 39 finishes. Both clips
are shown over their whole duration at 10 fps; no tracking model is rerun.

`17-video-versus-mocap.png` reuses the one existing sealed comparison on practice
39. Height and top-view paths use the saved first-pose rigid alignment; error
curves are the saved sealed errors. The mocap goal is read from the completed
M2 export's reference columns, agreeing with the sealed position errors to
within 1 micrometre. The scorer and sealed script are never executed. Bars
compare held still (0.18), the path alone (0.83), and the earlier policy with a
mocap goal (0.98), with their different meanings labelled.

The report discloses that the first tracker failed, C failed the first frozen
checks and the amendment, and the work proceeded under a conditional rule
written before the sealed label comparison. It explicitly asks the reader to
weigh that weaker evidence. The requested table-check outcome remains a
placeholder for the lead's reviewed finding. No full gate pass or adoption on
38 is claimed: 38 has no labels available to us and its world-path accuracy is
unknown. Candidate A's report is at `data/outputs/moving_reference/ep39/results.md`;
the requested `candidate-a/results.md` does not exist.

Provenance: `data/outputs/visuals/inputs.json` maps figures to hashed inputs;
`build-summary.json` records numeric checks, decoded video frames and file paths.
Review sheets include `review-policy-hands-video.png`. The video is decoded
completely during the rebuild and must remain below 5 MB. No test suite runs.
