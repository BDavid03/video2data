# Design decisions

The choices the code is built on. `data/configs/evaluator/contract.json` points here. The IDs D03,
D07, D10 and D17 are kept because other pages cite them.

## D03: follow the published rules, not only what the packer checks

Use real policy rollouts, metres, exact frame matching and recorded source hashes, even where the
kit's packer does not enforce them. The [contract report](../../data/reports/contract.md) separates
written rules from what the code checks. A permissive parser or an empty allowlist does not allow a
substitute result. We wrap the unchanged kit and check the file it writes.

## D07: metric scale needs a measured ruler

Calibrated stereo sets metric scale. Known public scan sizes are only a cross-check (see
[stereo depth](../../src/features/stereo_depth/README.md)). Geometry code uses direct stereo or the
frame's own measured scale. It never falls back to an episode median when a frame has no scale.

## D10: keep gaps until the completion step

Perception writes validity masks. A world-frame value is valid only when both the observation and
the camera pose are valid. Only the completion step fills gaps, and it flags what it filled. Filling
in place would hide which values were measured and which were made up.

## D17: compare a challenger on the same full scene

A trained or replayed candidate must stay in the working area and beat the resting fallback by the
recorded margin on the same scene. The default choice is the resting rollout. Manual overrides are
recorded separately.

## Architecture

| Choice | Why | Result |
|---|---|---|
| Vertical slices | A change stays inside one feature | Request, handler, adapters, files, flags and README live together |
| Explicit run paths and hashes | Reproduction needs the exact bytes used | No "latest run" lookups; a changed parent fails admission |
| Pins keep their role | Different passes serve different purposes | A later pin in another role does not erase an earlier one |
| Inherited provenance | Diagnostic data must not leak into production | Unknown or diagnostic lineage is never relabelled |
| Execution off by default | Runs need the owner's permission | Docs never enable a flag |
| Separate model environments | Native and model packages conflict | Each backend has its own install record |
| Generated files inside the checkout | One storage boundary | Caches, runtimes and outputs stay under the project root |
| Shared GPU lease | One GPU | `src/gpu.py` lock at `data/.runtime/gpu-stage.lock`; one GPU job at a time |

## Geometry and models

- **One colour pinhole grid.** The depth slice owns it; stereo, masks and geometry use it. Equal
  image size alone does not mean equal intrinsics.
- **Table-defined world.** Up is the table normal; x is the first camera heading projected on the
  table. This needs no labels but assumes a level table and a good plane fit.
- **Separate simulation and submission meshes.** Public scans keep the body frame for tracking.
  Every CD-O pair uses a mesh rebuilt from the same episode's video. Closed hulls suit collision but
  fill hollows. Reusing shapes across episodes is not allowed (open question Q06).
- **Anatomical MANO hand frames.** Hand rotations come from MANO kinematics with left and right kept
  apart. Parked hands use the model's rest pose, not another episode's hand.
- **Approximate contacts.** Sixteen upstream link slots use sampled geometric closeness. This is not
  measured contact force.
- **MoGe-2 depth.** Depth scores compare against MoGe-2 pseudo-labels, not measured depth.

## Policies and release

- Use the exact 20 Hz reference-to-output frame match, not a guess from the capture rate.
- Save the resolved training recipe and tie each checkpoint to it.
- Keep the submission plan fixed; write later steps as separate records.
- Check the exact packed bytes before upload. Upload permission covers named files only.
- Episode 41 and holdout episodes 43 and 44 are excluded.

When you change one of these, update the owning slice and every consumer, say what breaks, and keep
the old run records.
