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

To rerun the CPU policy diagnostic before rebuilding everything, add `--replay`.
This launches the isolated FlashCHORD interpreter. It never scores or packs the
diagnostic and never accesses the GPU. Its log is
`data/outputs/visuals/policy-hands/m0c/replay.log`; completion is recorded in
`status.json`, with data and comparison details in `summary.json`.

For a different finished checkpoint, including the episode-39 policy later:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu data/.runtime/flashchord/venv/bin/python -u -B scripts/report/policy_hands.py \
  --checkpoint data/outputs/moving_policy/ep39-a1/training/policy_STEP.safetensors \
  --task data/outputs/moving_policy/ep39-a1/motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet \
  --output data/outputs/visuals/policy-hands/ep39 \
  --no-official
```

Replace `STEP` with the finished checkpoint. If a matching official export is
available, replace `--no-official` with `--official PATH/trajectory.parquet`.
Checkpoint identity must match the export's metadata. Only FlashSAC is supported.
The script accepts arbitrary reference lengths and preserves the checkpoint's
frame window, speed, support, collision, timing and curriculum configuration.
Its explicit changes are one world, CPU, deterministic inference, frame-zero
reset and the upstream evaluation composition that removes assistance. CPU
execution uses the same uncaptured `env.step`; GPU graph capture is omitted.
The report currently renders the fixed M0c artifact. To show another saved
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

`data/outputs/visuals/episode39-setup.json` is a dated snapshot of setup evidence
and training state, with source hashes. It intentionally contains no training or
scoring outcome. The lead owns later outcome updates.

Provenance: `data/outputs/visuals/inputs.json` maps figures to hashed inputs;
`build-summary.json` records numeric checks, decoded video frames and file paths.
Review sheets include `review-policy-hands-video.png`. The video is decoded
completely during the rebuild and must remain below 5 MB. No test suite runs.
