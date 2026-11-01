# 07 — Moving observed fusion source build

Status: source implemented; unrun. Owner: `track3/features/object_mesh/`.

Hypothesis: compensating each masked metric observation into the measured
object body frame avoids static-scene smearing of a moving rigid object.
Baseline: existing bounded observed point accumulation. Candidate: bounded
TSDF extraction on identical admitted frames, with no unseen completion.
Reuses `_points.py`, `_fusion.py`, producer/hash admission and canonical run
persistence rather than creating a reconstruction framework.

Implemented `MovingFusionRequest`, independently disabled `fuse` handler/CLI
and unset-parent `fuse.json`. Requires exact completed track/mask/depth/anchor/
camera parents, proper rigid transforms, ordered frame/PTS clocks, same pixel
grid, observed scale and admitted pose support. Hand/object overlaps and missing
observations are excluded. Retains tracking-body origin and axes. Does not load
reference scans, diagnostic labels or weights. Per-frame exclusions persist
on failure. Mesh artifacts and observed point/TSDF counts have separate exact
SHA256 entries in the run and mesh records.

Bound: at most 120 samples, default stride five over frames 0–599; 500,000
observed point voxels, two million TSDF grid points, 200,000 faces per object,
2 mm spacing and four-voxel truncation. Request values are recorded before
execution. Exceeding bounds or insufficient views fails; no coarsening, hull
completion or implicit model is substituted. Runtime artifacts must remain in
canonical Z: storage; this implementation checkout is not an output workaround.

Partition: training 0/11/12/18/21/23/26/31 and tuning 39/42 only, grouping each
episode's cameras and frames. 41/43/44 and evaluation are refused. Existing
43/44 results are consumed and did not set any new threshold. Thresholds are
proposals requiring tuning evidence and an independently fixed future assessment.

Proposed matched comparison: observed point agreement, independently reserved
view agreement where available, metric dimensions, coverage/exclusions and
downstream contact sensitivity. Compare exact matching frames, pixels, depth
and poses. Frame counts are support, not independent coverage or calibrated
uncertainty. Closedness alone cannot promote collision geometry. The opening
hull remains the implementation baseline. The completed opening-39 campaign
failed clustering and left both hull and TSDF stages blocked; it did not produce
a completed mesh baseline.

No tests, checks, imports, models, fusion runs, dependency changes, simulator
runs or runtime outputs were executed. No experimental result or speed/accuracy
improvement is established. The new runtime flag remains false; historical
reconstruction authorization was preserved. Empty/insufficient observations,
pose ambiguity, repeated viewpoints, mask contamination, metric-depth bias and
thin structures remain material failure risks.

Disposition: simple observed fusion source implemented. Neural/generative
completion is deferred: it adds unmeasured geometry and unapproved dependencies;
metric collision accuracy needs independent evidence. FoundationPose and
BundleSDF retain the license/provisioning, dependency and admissible-input
blockers described in item 07. No scan-based alignment or label correction
was introduced.

Next eligible work: connect observation-aware hand/contact consumers and
scene-refinement proposals without executing them. Runtime comparison remains
blocked by disabled `fuse`, canonical storage access, missing admitted producer
runs and no-testing constraints. No eligible holdout assessment exists for
these new choices.
