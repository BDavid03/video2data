# 02 — Independent evidence and error attribution

Status: specification, diagnostic corrections and feature-owned independent
geometry source implemented, unexecuted. No independent metric outcome exists.
Owners: public_diagnostics and each label-free producer.

The hypothesis is that independent geometry plus visible missing support can
detect errors that a lower teacher loss hides. Reuse the completed campaign and
the existing diagnostic substitutions; do not refit or repeat them here.

## Evidence channels

Every scorecard must distinguish these channels, with no combined headline score:

| Channel | What it measures | What it cannot establish |
|---|---|---|
| Teacher agreement | Existing equal-episode log-depth MAE against MoGe2 | Physical depth, independently correct teacher, achieved task success |
| Independent image geometry | Reserved stereo/image correspondences, unused-view reprojection, fixed-scale depth and rigidity evidence | Textureless/unseen accuracy or physical contact forces |
| Reference diagnostics | Pinned kit public-pose and scan comparisons; diagnostic fits | Production correction, exact hidden evaluator equivalence, untouched assessment after reuse |
| Achieved behaviour | Simulated achieved object poses under a fixed controller and pinned kit score | Real-world transfer; reconstruction target accuracy by itself |

Report source revision, request, producer/run/artifact hashes, exact frame index
and integer PTS/timebase, object roster, admitted support, rejected support and
reason, units, split and event in every row. Missing score is null with a reason,
never zero error. Separate observed, inferred, interpolated and generated data.

## Predeclared independent measurements

| Measurement | Definition, admission and coverage |
|---|---|
| Stereo correspondence | Before depth anchoring, reserve a deterministic disjoint set of static correspondences (hash of episode/frame/query ID mod 5 = 0). Report rectified vertical residual and candidate reprojection distance in both eyes in pixels; require bidirectional visibility and LR consistency. Anchor-fit pixels are excluded. Report admitted queries / eligible queries and frame support separately |
| Unused view | Project camera-A points into a calibrated view not used by fitting that candidate; compare to independently tracked image positions. Record actual withheld view identity. If every view contributed, name it fit residual, not unused-view evidence |
| Metric depth | Absolute log-depth residual and relative metre error on reserved reliable stereo support at the candidate's frozen metric scale; no per-frame fitting. Supplementary episode-scale-aligned scores must be separately named and fit only on fitting support |
| Scale drift | Log of per-frame admitted stereo/mono scale, fixed origin and episode scale; median, p90 spread, first-to-last change and gaps. Do not independently normalize predictions before reporting drift |
| Edges | Symmetric image/depth-edge distance in pixels within valid support, with RGB texture edges and occlusion boundaries reported separately. An image edge is a correspondence cue, not a depth discontinuity label |
| Temporal geometry | Back-project optical-z depth with saved K; transform by fixed camera poses (and accepted object SE(3) for dynamic instances), z-buffer into next frame, compare log-z only on forward/backward visible support. Occluded/out-of-FOV/untracked/dynamic-unmodelled pixels are excluded with counts. Report temporal distances and PTS delta. Raw depth-frame subtraction is inadmissible |
| Rigidity | Distances between persistent accepted point tracks in object coordinates, normalized by one fixed object scale; separate object identity, symmetry hypotheses, static support and tracking gaps |
| Surface coverage | Observed viewpoints and supported surface area/voxels versus inferred completion. Compare only matched observed support; no claim of full surface coverage from a visually complete hull |

A metric implemented later must retain numerator and denominator, conditional
error and coverage. Candidate intersection support is required for paired error
deltas, alongside each candidate's own support and union missingness. Neither a
candidate dropping hard pixels nor a model producing invalid arrays may appear
better by silent filtering.

## Comparisons and stopping rules

Baseline: calibrated MoGe2/stereo plus existing producers, and the frozen
teacher-imitation ensemble as a separate research baseline. Changed variable:
one named estimator/anchor/context/objective at a time. Training remains
0/11/12/18/21/23/26/31; tuning 39/42; 41 excluded; 43/44 consumed and unavailable
for new selection. All camera streams/derivatives stay grouped. No new holdout
is chosen here.

Before outcome access, fix 16 uniformly spaced sample centres per tuning episode
and deterministic whole-event windows around independently annotated grasp,
lift, release, full occlusion and reacquisition. Preserve negative/failure events.
Event annotations may use source images, never reserved pose labels. Specify
annotation provenance and freeze before comparisons. If no admissible event
annotation or contiguous frames exist, that metric is unavailable: the old
sparse 16-frame sample cannot manufacture temporal evidence.

Proposed first measurement ceiling: two episodes, at most 32 sample centres plus
five 16-frame event windows/episode, maximum side 384, CPU 20 minutes, at most
2 GiB new artifacts; no model fitting. Stop at the first lineage/grid/clock
failure; retain any completed rows with failures and missing denominators. This
is a research design, not a command or permission to run verification. Runtime
fit is unmeasured and canonical writes currently unavailable.

Use per-episode/per-object/event medians, p90, maximum errors, coverage and worst
failure identities. Whole-episode bootstrap is optional descriptive sensitivity
only: two tuning episodes do not support confident population intervals. Pixels
and successive frames are not independent trials. Freeze candidate ranking
rules in item 11 before measurements; do not choose favourable channels later.

## Reference-only attribution

Reuse `public_diagnostics` oracle's independent scale, anchor, body-frame,
integer-time and resting-drift substitutions. Report each against the same
unmodified trajectory with fit parameters, support and pinned kit metrics;
they are not sequential improvements and never export corrected production
poses. A fitted gain is diagnostic attribution, not an achievable production
gain. No label or reference scan enters estimation.

Source corrections in `public_diagnostics/diagnostics.py` and `advanced.py` now
preserve missing-spoon-scan coverage, null unsupported ADD/AUC, other metric
availability and per-metric aggregate denominators/missing-as-miss summaries.
`scene_assembly` now refuses irregular PTS before uniform 20 Hz export. These
are unexecuted source changes; no new scoring result or compatibility claim.

The shared `track3.paths.output_path` now refuses generated writes from any
checkout other than canonical Z:. This closes a storage gap in older slices
whose operation flags may still reflect prior authorized campaigns.

Legacy supplementary valid-only RPE is evaluated on retained complete frames;
across missing original frames its intervals differ from a complete trajectory.
Do not treat that conditional statistic as a timestamp-matched temporal result.
Missing-as-miss RPE stays null for incomplete trajectories.

Disposition: adopt the evidence specification and source admission/reporting
fixes. Runtime geometric reports and label diagnostics remain blocked by
canonical write access, missing contiguous/admitted producers, diagnostic gates,
environment provisioning and the no-testing restriction. Next independent work:
item 03 inspection implementation, then item 04 anchor design. Future source
changes do not turn planned measurements into completed outcomes.

## Implemented independent report, source only

The stereo-owned `correspondences` and `report` handlers now implement the first
sparse image-geometry path rather than only a specification. The image producer
uses fixed-budget reciprocal ORB matching and calibrated triangulation without
candidate depth values. The report uses reserved SHA256 query identities shared
with explicit anchor-fit exclusion, fixed episode metric scales, separate
unknown/static/hands/object strata, conditional error and coverage, and paired
candidate intersection/union missingness. It retains per-query signed depth and
both-eye reprojection residuals, frame maxima and independent vertical residuals.
Existing metric challenger ingests are admitted on their unchanged calibrated
grid; relative challengers require an explicit disjoint frozen metric anchor.

Motion-compensated static temporal source handles saved observed camera poses,
forward z-buffering and backward visibility. It requires admitted static regions;
semantic mask complements are unknown. Dynamic temporal, independent unused
views, edges, rigidity, object identities and event annotations remain unavailable
without their actual upstream observations. Correspondence visibility is an
explicit reciprocal-match/positive-triangulation proxy, not full occlusion proof.
The selection receipt must preserve these missing channels, never replace nulls
with success. Details and artifact fields are in the
[stereo slice](../track3/features/stereo_depth/README.md).

Disposition: source implementation delivered; unassessed. New operation flags are
false and incumbent flags unchanged. No tests, import checks, runtime artifacts,
measurements, dependencies or holdout access occurred. Canonical writes remain
unavailable and static/event/dynamic-motion producers remain exact blockers for
the corresponding channels. The frozen protocol binds complete selected frame,
query-producer and candidate identities before residual outcome access.

Integration: independent reports can explicitly admit the original learning
slice's per-episode tuning-only `scratch_depth.research_predict` optical-z
candidate outputs on the unchanged hash-bound calibration grid. Existing SGBM
source additionally owns continuous linear LR-residual matching quality,
transferred from the exact nearest depth point, with explicit uncalibrated
semantics for learning input admission. Historical archives gain no fields or
claims; this is unexecuted producer source, not independent accuracy evidence.
