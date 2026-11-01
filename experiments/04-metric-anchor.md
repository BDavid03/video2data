# 04 — Calibrated stereo and metric anchoring

Status: calibration reconciled, incumbent retained; reserved-support scale
challengers and comparison source implemented, unexecuted. Independent
accuracy/coverage adoption evidence blocked. Owners: stereo_depth,
depth, camera_motion. Source-only observability additions, no execution.

Reuse `configs/calibration/ego.json`, `reports/calibration/README.md`, depth's
saved pinhole grid, stereo's existing LR-consistent SGBM, median anchoring and
the completed episode-39 opening records in item 01. Historical episode-41
calibration evidence may be cited but episode 41 must never be reopened/run.

The rig is C-left/B-right. Recompute R1/R2/P1/P2/Q together using full
14-coefficient distortion and calibrated 75.47 mm translation rather than
mixing the nominal 75 mm baseline with these matrices. Preserve upright source
pixels and explicit resize/pixel-centre K; camera-z is not Euclidean ray range.
Transfer C-rectified → C-optical via R1 transpose → A-optical via inverse C_T_A,
then use saved A pinhole intrinsics and nearest-depth visibility. Require exact
frame index/PTS across all streams, not nominal frame-rate matching alone.
Public calibration's applicability to evaluation recordings remains unresolved.

Hypothesis: a robust episode-level static scale can reduce per-frame fitting
noise while retaining genuine drift evidence. Existing per-frame median already
offers robust fitting; no replacement is justified by the old teacher scores.
Compare current per-frame scale, episode median and robust log-ratio aggregation
as **separate** challengers on reserved independent support. Exclude hands/moving
objects from fit where admitted masks exist; report all three strata plus
unknown separately. A missing mask is unsegregated support, not static evidence.
Textureless/specular regions, occlusions and LR failures remain invalid, never
filled as measured stereo. Minimum support/count coverage must be visible.

`stereo_depth/anchor.py` now records raw support share, ratio p10/p50/p90,
log-ratio MAD, frame support share and episode log-scale MAD. Existing medians
and production scale application remain unchanged. Mask provenance is explicit;
mask exclusion alone does not prove static support. This improvement exposes
limits without selecting a new scale estimator on consumed holdout evidence.

The source-owned [comparison request](../track3/features/stereo_depth/anchor-comparison.json)
is explicitly **not executable**. It fixes episodes 39/42, ≤16 sample centres
each, side 384, 20 CPU minutes and 2 GiB outputs, fitting-support holdback,
coverage/strata, stopping rules and item-11 promotion criteria. Resolution differs
from the old learning comparison and results must not be mixed. Metric definitions
are in item 02. No new holdout, fitting, teacher inference or correspondence
measurement occurred.

Alternative calibration remedies: refine pair rotation from training-only static
matches if independent vertical residuals establish a material issue; investigate
a learned stereo matcher only after that comparison. Neither is adopted. Historical
~2 px vertical residual on two old episode-41 frames cannot establish a full-rig
error distribution or distinguish timing/rolling shutter from rotation. Rotation
refinement must not fit tuning reserved correspondences or change baseline
length/gauge; a learned matcher needs pinned source/weights, terms and explicit
dependency approval before integration.

Observed result: opening-39 rectification, stereo, anchor and camera stages
completed, but no independent accuracy comparison establishes better geometry.
Object clustering failed at the image border; meshes remain blocked. Do not
relax the quiet-opening guard merely to obtain a completed result.

Disposition: retain calibrated MoGe2/stereo, existing robust median, invalid
regions and strict producer identity. New scale methods and calibration remedies
remain designed/unassessed. Exact blockers: canonical writes unavailable;
reserved correspondences and admitted region masks absent; no-testing remains
active. Later model-family research is independently eligible, but its empirical
selection depends on this evidence. Next: item 05 challenger dispositions.

## Implemented research challengers, source only

`stereo_depth.anchor` now has explicit `holdback_modulus=5` support exclusion
using the same stable colour-pixel query identities as independent image
correspondences. The default zero retains historical fitting support. Episode
aggregation is selectable between incumbent `median`, `log_median` and bounded
Huber log-location; the default remains median and no result selects a challenger.
Research variants require a new false `anchor_research_authorized` flag as well
as ordinary anchor authorization. An admitted relative-depth challenger must use
holdback5 before anchoring; no silent conversion to metric metres occurs.

The separately disabled correspondence/report handlers give the comparison a
concrete independent-image producer and matched-support residual report. A
hash-bound frozen protocol names candidate/baseline/query producers, exact frames
and fixed metric scales. Native metric scale is exactly1; other scales must
match a completed disjoint episode anchor. Masks split hands/objects; overlaps,
mask complements and missing masks stay unknown rather than inferred static.
The report preserves numerator/denominator, own/intersection/union coverage,
missingness, per-query both-eye errors and support-rejected visibility counts.
Temporal static evidence is implemented for admitted camera motion and explicit
static regions; dynamic-object motion remains unavailable and explicitly counted.

All source is unexecuted. No tests/checks, correspondence extraction, model
execution, anchor fitting, new packages, runtime writes or holdout access ran.
There is no claimed improvement over the incumbent. Canonical writes and actual
admitted static/event producers remain blockers for empirical adoption. Rotation
refinement and learned stereo remain deferred pending independent residual
outcomes and, for learned stereo, source/weight/dependency approval.
