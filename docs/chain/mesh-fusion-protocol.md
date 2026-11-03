# Mesh fusion F1: multi-frame stereo reconstruction of the pot and rack (context 20; written before any fused geometry or score exists; commit time is the record)

Owner focus `PROGRAM.md` section 0, to-do 3 (whole-clip reconstruction in place of primitives). Installed tools only
(no new package or weights). Route: `outputs/chain/codex/*route-fusion-20.answer.md`.

## Why

D1 v4 is measured primitives: CD-O 0.66292 cm pair / 0.65338 cm object on the 26 public-scan pairs. White pot
(0.858 cm, 6 pairs) and dish rack (0.865 cm, 5 pairs) contribute 0.364 cm, 55 % of the pair-weighted mean. Fusing the
calibrated stereo of several pre-contact frames could recover their real proportions and cavities instead of
constructed ones.

## Disclosure

The scans and v4's per-object scores are known, and choosing these two objects used those scores. Stage A opens no scan.
Stage B reuses the development scans of `docs/chain/mesh-selection-protocol.md`; it is evidence on reused scans, not an
untouched validation. Episodes 41/43/44 are never read; validation 2/13/22/24/27 is spent and not used.

## Stage A: geometry feasibility (no scan is opened)

- Episodes: tuning 42 (white pot, opening cluster 0) and train 12 (dish rack, opening clusters 0/2/3/4/5/6), the
  label files named in the route answer.
- Pre-contact interval: the frames before the first visible hand-object contact, located from camera-A RGB (every fifth
  frame, then refined) and recorded with a contact-sheet figure. Candidate frames are every fifth frame inside it.
- Each candidate frame: calibrated stereo depth by the existing `stereo_depth` code; camera pose relative to frame 0
  by background-feature PnP (object region masked out by the frame-0 3D object region projected into that frame),
  refined by ICP on background stereo points.
- **Frame acceptance:** >= 30 PnP inliers, median reprojection <= 1 px, and an independent background-correspondence
  disagreement <= 3 mm after registration. Rejected frames are listed with their reason.
- **Measurements per object:** accepted frame count; angular span of the unit vectors from the camera centres to the
  frame-0 object centre (degrees); additional supported surface: the fraction of 3 mm surface voxels of the fused
  object cloud not already supported by frame 0.
- **Stage A kill (per object):** fewer than 3 accepted frames, or angular span < 10 degrees and additional supported
  surface < 10 %. An object that is killed keeps v4. If both are killed, F1 ends with a recorded finding and v4 stands.

## Stage B: one frozen recipe, then CD-O (only for objects that pass Stage A)

- The recipe is fixed in writing (committed) after Stage A and before any scan is opened: the object region rule, the
  3 mm occupancy grid with unknown space kept distinct from observed free space, the surface by
  `trimesh.voxel.ops.multibox(..., remove_internal_faces=True)` or an equivalent exact boundary, any completion rule
  (for example rotational completion of the pot body that keeps inner and outer walls), and v4's 4096 vertex / face
  budgeting. One recipe, no grid, no re-rolls, no change after a score.
- Every development pair of the object is built from its own episode's video by that recipe; objects not changed keep
  v4 bytes. Scoring: `scripts/track3/local_cdo.py` matched seeds 1/3/5 against the same scans and folds as
  `mesh-selection-protocol.md`; per-object seed SD reported.
- **Development gate:** equal-object mean strictly below 0.62270304 cm, pair mean <= 0.66288090 cm, no object worse than
  v4 by more than 0.15 cm, and every changed object better than v4.
- **Packed veto:** the production build of all objects, packed with a placeholder URL and scored by
  `track3/features/mesh_eval` pairs: object mean < 0.62338396 cm, pair mean <= 0.66292148 cm, no object worse than
  packed v4 by more than 0.15 cm. Pass: repack as D1 v5 at a new real-URL commit with NEEDS-YOU upload commands by
  12:00 UTC 4 Nov. Fail: v4 stands; no runner-up, no repair, no retuning.

## Not in F1

Deforming v4 toward fused observations (the route's alternative, IDEAS 56) and post-contact fusion (needs object motion
compensation) are recorded but not run under this protocol.

## Amendment 1 (F1b), written after Stage A and before any rerun; commit time is the record

**Why, and what it is informed by.** Stage A killed both objects because no later frame passed acceptance
(`docs/chain/mesh-fusion-outcome.md`). The Claude critic (critic-20) read the run records and found that the
background-PnP poses alone had median reprojection 0.68 / 0.95 / 0.93 px (rack) and 0.82 / 0.92 / 0.94 px (pot), all
within the 1 px limit; the ICP refinement on background stereo raised them to 1.65-5.13 px. So the kill measured our ICP
step, not the clips. This amendment is informed by those Stage A numbers on train 12 and tuning 42 (choices on
train/tuning data); no scan has been opened and no CD-O computed.

**Change (only this):** Stage A is re-run with the camera pose taken from background PnP alone; the ICP refinement step is
removed. Everything else is unchanged and frozen: the episodes, objects, pre-contact intervals and candidate frames
(0/5/10/15), frame acceptance (>= 30 PnP inliers, median reprojection <= 1 px, independent background disagreement
<= 3 mm, now measured with the PnP pose), the measurements, and the kill rule (fewer than 3 accepted frames, or angular
span < 10 degrees and additional supported surface < 10 %). The first Stage A runs stay on record as F1. Stage B and its
gates are unchanged. If F1b also kills both objects, the fusion route ends for this deadline with no further amendment.
