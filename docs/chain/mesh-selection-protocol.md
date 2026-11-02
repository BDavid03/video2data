# Mesh selection protocol (fixed 2026-11-01, context 03, before any new mesh was scored)

D1 v1's box-versus-ellipsoid choice was made on the same 10 public scans that report its score.
From now on every mesh change follows this protocol, written down before scoring. The scans stay
measurement-only: no scan geometry, dimension or template enters a submitted mesh.

## Baseline

D1 v1 (one video-measured box per object, build run
`outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff`), scored with
`scripts/track3/local_cdo.py` over kit seeds of episodes 1, 3, 5 (per-object table in
`outputs/mesh_eval/d1-ctx02/v1-box.json`).

## Rules, not objects

A candidate is a **rule** that maps video measurements (and the object's name category, written
below before scoring) to a shape. Picking a shape per object by its scan score is not allowed.

Categories, fixed now from the object names and how they look in the video:

| Category | Objects |
| --- | --- |
| round vessel (revolved profile plausible) | beige_cup, blue_cup, white_pot, white_pot_lid, water_pitcher |
| flat or blocky | wooden_piece_1, wooden_piece_2, plastic_dish_rack |
| thin tool | mini_sweeper, mini_dust_pan, wooden_spoon (no scan) |

## Two-fold leave-objects-out

Folds, fixed now (each holds part of every category that has a scan):

- Fold A: beige_cup, white_pot_lid, wooden_piece_1, plastic_dish_rack, mini_dust_pan
- Fold B: blue_cup, white_pot, water_pitcher, wooden_piece_2, mini_sweeper

Any free parameter of a rule (a threshold, a handle fraction, a taper) is chosen on fold A and
scored on fold B, then chosen on fold B and scored on fold A. The **cross-fitted score** is the
mean of the held-out per-object scores over all 10 scanned objects. A rule with no free parameter
is scored once on all 10; that is its cross-fitted score.

## Adoption

A rule replaces the box in a repack only if all three hold:

1. its cross-fitted per-object mean beats the box's (1.008 cm) by more than 0.03 cm (several times
   the largest per-object seed spread in v1, 0.036 cm on the rack, divided by sqrt(10));
2. no held-out object gets worse by more than 0.15 cm;
3. the pair-weighted mean over the 26 scanned pairs does not get worse.

Report the cross-fitted numbers as the headline, beside the in-sample ones, and log every rule
tried (including losers) in `docs/chain/IDEAS.md`. At most three rules per context, to bound
selection on 10 objects.

## The spoon (no scan)

The spoon cannot be scored. A spoon shape changes only through a rule that also covers scanned
objects (for example "thin tool: ellipsoid head plus box handle, sizes from the measured extents")
and passes the adoption test on the scanned thin tools and the rest; or through a video-only proxy
(silhouette IoU of the projected shape against the opening-frame depth cluster) that has first
been shown to rank shapes the same way CD-O does on the 10 scanned objects (Spearman >= 0.6 over
box, ellipsoid and the candidate). Otherwise the spoon keeps its box.

## Repack

Only when a rule is adopted. The repack, its run records and the cross-fitted table go in
`docs/chain/SUBMISSION.md` (Packed files) before the file is listed for the owner.


## Amendment 1 (round 2), fixed before any round-2 score

Dated **2026-11-02**. Earlier protocol text is unchanged. This amendment and the video
attributes must be committed before scoring begins. Job A builds candidate geometry only;
it does not access scans or compute new CD-O measurements.

### Incumbent and adoption

The incumbent is shipped **R2 at the fixed aspect threshold 1.3**, with handle height 0.6.
Recomputed from `outputs/mesh_eval/rules/ledger.jsonl`, using the kit seeds of episodes 1/3/5:
**0.92506284 cm per object**, **1.08767842 cm pair-weighted** over 26 scanned pairs.
These fixed-threshold scores differ from round 1's cross-fitted threshold result.

| Object | Incumbent CD-O (cm) | Pairs |
| --- | ---: | ---: |
| beige_cup | 0.37907943 | 2 |
| blue_cup | 0.44187011 | 5 |
| mini_dust_pan | 1.06125693 | 2 |
| mini_sweeper | 0.72358521 | 2 |
| plastic_dish_rack | 2.11479209 | 5 |
| water_pitcher | 1.68285236 | 1 |
| white_pot | 1.32768803 | 6 |
| white_pot_lid | 0.67558238 | 1 |
| wooden_piece_1 | 0.44443347 | 1 |
| wooden_piece_2 | 0.39948844 | 1 |

A round-2 rule passes the development gates only if its cross-fitted per-object mean is
strictly less than incumbent minus 0.03 cm (**< 0.89506284 cm**, approximately < 0.89506),
no held-out object regresses by more than **0.15 cm** relative to its own fixed-threshold
incumbent, and its cross-fitted pair-weighted mean is no worse than **1.08767842 cm**
(approximately 1.08768). Full-precision ledger values define these comparisons.

There are exactly **three nested rules**, R4/R5/R6, all adoption candidates. Among rules
passing every development gate, the lowest cross-fitted mean wins; exact ties use queue order.
Free parameters minimize equal-object training-fold mean, ties use frozen grid order, and the
final production parameter minimizes the all-object mean with that same tie order.
**Choosing among three nested results on ten reused scans is optimistic**; thickness cross-fitting
does not cross-fit rule-family selection and these are not independent final estimates.

An **exact veto** then applies to that single selected candidate: build all **11 objects** with
the production builder and pack with the placeholder URL, then evaluate through
`track3/features/mesh_eval`, using each pair's own kit seed. Both means must be no worse than
packed v2: **1.0850 cm pair-weighted / 0.9257 cm per object** (recorded unrounded values
1.08502364 / 0.92572504). Failure retains v2. **No retuning after the veto**, including no
switch to another rule. Production changes, packing and all scoring belong to job B.

### Video attributes and geometry, frozen now

`track3/features/mesh_extents/attributes.json` records episode, camera, frame index,
overlay/frame path, SHA-256, observation and confidence for each attribute. Unknown or uncertain
required attributes preserve the object's complete R2 shape; only high-confidence evidence enables
changes. Geometry is derived from video extents and these attributes, never from scan geometry.
All other objects, including the spoon, retain R2. Category membership and folds remain unchanged.

- **R4 rim handle, no free parameter.** Round vessels above aspect 1.3 with `handle: "rim_flat"`
  retain the R2 circular body, but the bar has length long-axis excess plus R2 overlap (0.05 x
  short extent), width 0.15 x short extent, height 0.15 x vessel height and top flush with rim.
  `handle: "loop"` retains the centered R2 bar of height 0.6 x vessel height. Cups below 1.3
  retain elliptical cylinders and get no handles. The pot qualifies as rim-flat, pitcher as loop.
- **R5 hollow vessels = R4 plus video-confirmed `open_top: true` round vessels.** One closed,
  connected body shell has outer wall, inner wall, flat annular rim, outer bottom and inner
  floor, with no cap across the mouth. Wall/radius fraction grid **[0.02, 0.05, 0.10]**, in that
  order; floor thickness equals wall thickness. Body radius is half the short extent. Without
  a handle, the circular profile is scaled along the long axis to match the elliptical footprint;
  with a handle, the body remains circular as in R2. Profile revolution uses 64 sections, as R2.
  The handle remains attached as in R4 using overlapping closed components. Cups, pot and pitcher
  have visible open mouths; the lid is not open-topped. The reviewed pot opening frames are
  uncovered, so "usually lid-on" is not established. An open pot in any opening frame qualifies.
- **R6 peg rack = R5 with the same thickness grid plus video-counted pegs on flat/blocky objects.**
  The rack shows **two rows of six** upright tines in public 12 and unobstructed evaluation 15/16,
  frame 0, ego_cam_a. Base slab height is 0.25 x measured height. Pegs rise from base to full
  height; long-axis cell spacing is long extent / count, cell-centered, peg length 0.5 x spacing.
  Each peg width is 0.3 x total short extent; row centers are evenly spread across the short axis
  with half-peg-width margins (one row is centered). Pegs overlap the base by 0.01 x measured
  height to avoid coincident contact faces. This fixed simplification omits the visible utensil
  compartment and tray rim; no grid or scan-driven adjustment. An unclear count would preserve
  the R2 box and make R6 equal R5 for that object.

Boolean-free handle/base/peg concatenation leaves overlapping closed components with internal
faces, as permitted for R2. The vessel body itself is one connected shell. Every component must
be watertight and a positive volume, every mesh below 4096 vertices and 4096 faces; overall
extents must match video extents to 1e-6 relative. Candidate-only dry renders show three views
per changed object. No new score is computed as part of this amendment's preparation.

## Amendment 2 (round 3), fixed before any round-3 score

Dated **2026-11-02**. All earlier text is retained. The lead must commit this amendment,
`attributes-round3.json`, `rules-round3.json` and the geometry before Job B scores anything.
Job A reads video and candidate geometry only, apart from the explicitly allowed R6/0.10
incumbent ledger rows. It does not open public scans or call a CD-O scorer.

### Incumbent, gates and selection

The incumbent is **packed D1 v3**, R6 peg-rack with wall/radius **0.10**, Parquet SHA-256
`a61f02b12078a05c940037034e37ec4201e3e919a55e4bea9e0058dfa2f45f40`.
Matched development comparisons use kit seed episodes **1/3/5**, retaining the original folds.
The ten R6/0.10 rows in `outputs/mesh_eval/rules-round2/ledger.jsonl` give
**0.7731489706385066 cm per object / 0.8167541550356001 cm pair-weighted**.
The brief's 0.77314847 is a transcription discrepancy of about 0.00000050 cm;
full-precision ledger values, frozen in the round-3 queue, govern development gates.

| Object | R6/0.10 development CD-O (cm) | Scanned pairs |
| --- | ---: | ---: |
| beige_cup | 0.3207342096946351 | 2 |
| blue_cup | 0.380004128502687 | 5 |
| mini_dust_pan | 1.0612569261637064 | 2 |
| mini_sweeper | 0.723585214852021 | 2 |
| plastic_dish_rack | 1.3907362291207122 | 5 |
| water_pitcher | 1.472552596081469 | 1 |
| white_pot | 0.8631161086673155 | 6 |
| white_pot_lid | 0.6755823809179679 | 1 |
| wooden_piece_1 | 0.44443347270991 | 1 |
| wooden_piece_2 | 0.39948843967464204 | 1 |

A rule must have cross-fitted equal-object mean **strictly below incumbent minus 0.03 cm**
(**< 0.7431489706385066 cm**), no object regression **> 0.15 cm** relative to that object's
matched incumbent, and pair-weighted mean **<= 0.8167541550356001 cm**. Never compare a
three-seed development score with a packed-pair score. There are exactly three rules below,
with **no free parameter and no grids** (`grid: [null]` denotes one fixed construction);
wall/radius stays 0.10. A no-parameter rule's all-ten-object score is its cross-fitted score.
Among eligible rules, lowest cross-fitted equal-object mean wins; ties use **R7, R8, R9** order.
There is no scan-selected per-object mixture. R9 uses the exact R7 rack and exact R8 tools.

The spoon safeguard from the route review is fixed explicitly: **R8 must independently pass
all development gates before R9 is eligible**, so a rack improvement cannot subsidize an
unsupported spoon change. The unscanned spoon changes only if R8 or eligible R9 is adopted.

### Three video-only rules

The full annotations are in `track3/features/mesh_extents/attributes-round3.json`: original
frame coordinates, episode/camera/frame, decoded-image SHA-256, observation, confidence and
uncertainty. `attributes.json` and `rules-round2.json` remain byte-identical. Original vessel
attributes are inherited without modification. Measurements are approximate projected pixel
ratios, not calibrated 3-D metrology; their numerical precision does not imply that accuracy.
Tracking markers are ignored. All meshes keep the pooled video extents.

- **R7 compartment rack = v3 plus the rebuilt rack.** Evaluation 15/16 frame 0 `ego_cam_a`,
  with public 12 corroboration, show a full-footprint tray, open utensil compartment at one end,
  and two rows of six pegs lower than that compartment. The unseen floor/underside keeps the
  v3 slab (0.25 total height). Above-floor projected heights, approximately **160/80/40 px**
  for compartment/peg/rim in evaluation 15 and **150/75/38 px** in evaluation 16, freeze peg
  top at **0.625** and rim top at **0.4375** total height; compartment top is **1.0**.
  Its footprint length is **155/900 = 0.17222222** of long extent, full short extent, centered
  at **(0.08611111, 0.5)** in normalized footprint coordinates. Wall thickness is **10/230 =
  0.04347826** of short extent. Peg footprint is **40/900 = 0.04444444** long by **40/230 =
  0.17391304** short; long centers **[0.22, 0.32, 0.42, 0.52, 0.62, 0.72]**, short centers
  **[60/230, 150/230]**. Shapes are closed boxes, omitting small ridges and rounded corners;
  the compartment has four walls and the existing floor, no cap across its mouth. Contact
  overlap remains v3's **0.01 x height**, an inherited construction convention. Unknown or
  low-confidence rack components retain their corresponding v3 geometry, rather than gaining
  scan-inferred detail. Height-ratio uncertainty is about +/-0.10 of total height.
- **R8 thin-tool profiles = v3 plus one broad-head/measured-grip policy for all three tools.**
  Public 18 frame 0 `ego_cam_a/b/c` supplies pan and sweeper outlines; public 39 corroborates
  the pan (its frame-0 sweeper is absent). Evaluation 5/15/16 frame 0 supplies the spoon.
  The longitudinal axis follows the visible tool: in-plane long axis for pan/spoon, vertical
  for the upright sweeper. Pan: head length **210/300 = 0.70**, neck width **70/280 = 0.25**,
  elliptical solid grip width **154/280 = 0.55**, starting at **195/300 = 0.65**; tapered
  scoop floor thickness **6/42**, grip thickness **12/42**, side/back wall width **9/280**.
  Open front and low raised walls use the measured full height. Sweeper: solid tapered brush
  envelope length **70/140 = 0.50**, neck width **60/80 = 0.75**, grip width **56/80 = 0.70**,
  grip starting at **63/140 = 0.45**; grip thickness **26/40 = 0.65** of brush depth.
  Spoon: elliptical shallow bowl length **250/750 = 1/3**, neck **24/100 = 0.24**, narrow
  grip **22/100 = 0.22**, starting at **225/750 = 0.30**; inner bowl radius **0.90**, floor
  fraction **0.25**, grip thickness **0.50** from coarse visible edge ratios. The bowl is a
  closed shallow paraboloid shell with 48 angular sections and six radial intervals; the grip
  sits flush with its rim. Heights
  and thicknesses are visibly approximate (medium confidence), not recovered cavity geometry.
  Visible grip holes are omitted as authorized; no tracking markers or individual bristles.
- **R9 rack and tools = exactly R7 + R8**, with no extra components, measurements or parameters.
  Every other object retains v3 bytes; production building all eleven objects uses the same
  frozen queue and constructors as development building.

### Geometry checks and buried surfaces

Every component must be watertight with positive volume. Each mesh has **<4096 vertices and
<4096 faces**, and overall extents equal video extents within **1e-6 relative**. Untouched
objects must remain byte-identical to v3, and existing constructors must return identical
vertex and face arrays on the v3 objects. Candidate-only renders show three views per changed
object, plus a side-by-side video frame/candidate; no scans appear in these figures.

A geometry-only diagnostic reports, for both v3 and R7 racks, the surface-area fraction inside
another component: sample each component's surface with fixed seeds, use trimesh containment
against other components (all rack components are axis-aligned boxes, so
`trimesh.bounds.contains` is exact and needs no optional ray-index dependency), and weight each buried fraction by that component's area. Report
sampling resolution, uncertainty and boundary convention. Overlapping coincident boundary
faces are reported separately if necessary. CD-O samples area-weighted points and normalizes
by surface radius, so buried faces can influence it. **This diagnostic chooses nothing in
round 3** and never accesses a scan.

### Stricter packed veto and optimism

Only the selected development winner is built with the production builder (all **11** objects),
packed with the placeholder URL and scored through `track3/features/mesh_eval`, each pair using
its own kit seed. Apply the explicit brief's thresholds from `rules-round3.json`:
packed per-object mean **< 0.74107703 cm**, packed pair-weighted mean **<= 0.81149513 cm**,
and no object's packed mean worse than packed v3's by **> 0.15 cm**. Job B obtains matched v3
per-object packed means through that same pair evaluator; Job A does not read those ledgers.
Failure retains v3, **no runner-up, no retuning**. Packed-v3 unrounded means are
0.7710770332786878 per object / 0.8114951342417491 pair-weighted; the explicit veto thresholds
above are deliberately retained as written in the brief. Build, pack and veto follow the
pre-score commit; dry builds do not authorize adoption.

Round 3 follows two adaptive rounds on the same ten scans plus prior exposure to scan renders.
Cross-fitting covers declared parameters only. The results are development evidence, not an
independent estimate. A third round on the same ten scans raises selection optimism further.
New seeds and the packed veto do not remove that optimism, whose size is not known.

## Amendment 3: spoon proxy diagnostic, reported only
Dated **2026-11-02**; this amendment, request and code are committed before any proxy value.
Packed D1 v4 is unchanged either way; no adoption, repacking, scan access or new CD-O is permitted.
A = `outputs/mesh_eval/rules/ledger.jsonl`; B = `outputs/mesh_eval/rules-round2/ledger.jsonl`.
C = `outputs/mesh_extents/rules-round3/ledger.jsonl`; E = `outputs/mesh_eval/d1-ctx02/v1-ellipsoid.json`.
Use every candidate in A/B/C plus E with existing seed-1/3/5 means; deduplicate by object and mesh SHA-256.
Keep rule aliases and source rows. Counts: beige 7, blue 6, pan 6, sweeper 6, rack 4, pitcher 7,
pot 8, lid 4, each wood block 2 (52 candidates); exclude oracle, scan-self, unit and earlier proportion variants.
V = `outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff/run.json`.
Each view uses the first frame of its assigned measure run in V.request.measure_runs, camera `ego_cam_a`.
This is frame 0 except the lid at frame 110; fixed after the phase-1 inventory, before any candidate was projected.
There are 26 scanned views, 143 scanned trials plus six spoon trials; episodes 41/43/44 are excluded.
Spoon views are evaluation 5/15/16, first measured frame (0), camera a, from V.
Read each measure run, `labels_<frame>.npz`, overlay, matching depth parent/frame NPZ and `preprocessing.npz`.
Expanded paths and hashes, including saved global planes inside measure runs, are frozen in the request.
M is the union of assigned labels clipped to `input_valid`, without morphology, edits or view selection.
Reconstruct local support planes with `_geometry._local_plane`, saved global planes and recorded settings.
In-plane PCA uses cached assigned points, axes ordered by 2–98% extent, longest first.
Centre mesh footprint bounds on the PCA-bounds midpoint projected onto the local plane; base on plane; require x >= y*(1-1e-6).
One uniform scale matches long footprint extent, preserving proportions; try yaw 0/pi only and keep max IoU.
Record both yaw IoUs, both projected areas, selected yaw and target mask area for every candidate/view.
Project with `input_intrinsics_pixels` on the native depth grid; opaque triangle union via OpenCV `fillPoly`, shift=8.
Clip projection P to `input_valid`; IoU = intersection(M,P)/union(M,P); no continuous alignment search.
Average views equally per candidate/object before SciPy `spearmanr(IoU,-CD-O)` and `kendalltau` (tau-b).
Primary aggregate: equal-object mean tau-b across ten objects; report all per-object rho/tau, ties and counts.
Qualifying Spearman summary uses eight objects with >=4 variants: median rho and count with rho >=0.6.
Descriptive gates fixed now, before any value: mean tau-b >=0.4; qualifying median rho >=0.6;
>=6/8 qualifying objects pass rho >=0.6; dust pan and sweeper each pass rho >=0.6.
Missing inputs or undefined required correlations mean NOT VALIDATED; no silent exclusions or substitutions.
These replace the original three-shape Spearman test for this diagnostic only: n=3 makes rho >=0.6 a perfect ranking.
The original spoon clause still governs adoption; these descriptive gates authorize none and imply no significance.
Report, without another gate, whether R8 ranks above the box for each scanned thin tool; distinguish ties.
Spoon v3: `outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295/objects/wooden_spoon.ply`.
Spoon v4: `outputs/mesh_extents/build/20261102T051416.149421Z-f377e0a1/objects/wooden_spoon.ply` (packed production).
Record existing primitive-era spoon box/ellipsoid hashes as historical references only; do not add scored spoon trials.
Report v3/v4 IoUs, differences per view and equal-view mean, both yaws, areas and overlays; infer no centimetre gain.
On failure: "Proxy not validated for spoon CD-O inference; proxy says nothing about spoon CD-O; v4 unchanged."
Pure functions: `track3/features/mesh_extents/_proxy.py`; handler: `track3/features/mesh_extents/proxy.py`.
Operation: `mesh_extents proxy`; request: `track3/features/mesh_extents/proxy-spoon.json` with all input hashes.
Outputs: `outputs/mesh_extents/spoon-proxy/{ledger.jsonl,summary.json,summary.md}`, `trials/`, `runs/`, overlay PNGs.
Phase 1: hash inputs and target-only dry outputs in `outputs/mesh_extents/spoon-proxy-dry/`; no candidate projection or scores.
One synthetic unit cube outside the panel may check rendering; its artifact is never a summary input.
Phase 2: CPU only, <=149 candidate-view trials and 7200 seconds per invocation; identity-based resume, append-only ledger.
Require request/protocol/source bytes to match HEAD before phase-2 scoring; never modify historical trial records.
Report small n, silhouette blindness, mask/placement errors, reused development objects and construction-view optimism.
