# Mesh pilot G1: pretrained image-to-3D shapes for the costliest objects (context 19; written before any generated mesh exists; commit time is the record)

Runs only if the owner approves the Hunyuan3D-2mini install (`NEEDS-YOU.md`). Route:
`outputs/chain/codex/*route-mesh-19.answer.md`. Owner focus: `PROGRAM.md` section 0, to-do 3 and 4.

## Why

D1 v4 is measured primitives: CD-O 0.66292 cm pair-weighted, 0.65338 object-weighted, on the 26 public-scan pairs.
White pot (0.858 cm, 6 pairs), dish rack (0.865, 5) and water pitcher (1.487, 1) contribute 0.4216 cm, 64 % of the
pair-weighted mean. A pretrained shape prior may recover handles, curved bodies and compartments that primitives
cannot. The public scans only measure; no scan geometry, dimension or template enters any generated mesh.

## Disclosure

The scans and v4's per-object scores are known; the selection of these three objects used those scores. The pilot is
evidence about one pretrained prior on reused scans, not an untouched validation.

## Pilot (fixed now)

- Model: `tencent/Hunyuan3D-2mini`, subfolder `hunyuan3d-dit-v2-mini`, FP16, weight revision `bdaa271f`; code revision
  pinned and hashed at install; shape only (`mc_algo="mc"`, no texture), seed 12345, 30 steps, octree resolution 380.
- Inputs (video only, fixed before generation): rack public 12, pitcher public 21, pot tuning 42; the frame-0 camera-A
  RGB crop around the recorded `mesh_extents` mask (the assignments' cluster ids), with pixels outside the mask set to
  white, crop square with a 10 % margin. One mesh per object; no re-rolls, no prompt or crop changes after generation.
- Conversion to the submission budget exactly as v4 meshes are converted; scoring with the existing matched CD-O pair
  code against the same scan pairs and scoring seeds as v4.
- One resumable loop in its own slice; inputs, weights, code and outputs hashed in `run.json`; ledger and summary.

**Pilot continuation gate:** mean CD-O gain over v4 on the three objects > 0.10 cm, their pair-weighted contribution
improves, and no object regresses > 0.15 cm. Passing authorises only the full panel below; failing ends G1 and v4 stays.

## Full panel and adoption (fixed now; only after the pilot passes)

- Generate the same way for every public-scan pair of the pot, rack and pitcher (other objects keep v4), with a
  video-only admission rule fixed before scoring: the generated mesh, after one Sim(3) fit to the frame-0 observed
  depth points of the object, must leave a median point-to-surface residual <= 1.5 x that of the v4 mesh fitted the
  same way; otherwise that pair keeps v4.
- Adoption needs all of: equal-object mean < 0.62270304 cm and pair-weighted mean <= 0.66288090 cm against matched
  development v4 on the existing folds A/B (any adjustable choice cross-fitted), no object worse by > 0.15 cm, and the
  exact packed veto (object mean < 0.62338396, pair mean <= 0.66292148). Failure keeps v4, with no repair or
  runner-up switch. Spoon unchanged. Seed spread reported.
- A passing candidate is packed as D1 v5 by 12:00 UTC 4 Nov or not at all.

Episodes 41, 43, 44 are never read; no leaderboard use.
