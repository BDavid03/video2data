# Source buildout after “with subagents build each”

Date: 2026-10-18. Status: source implementations built by feature owners;
**unexecuted and unassessed**. This extends the earlier research/design pass.
It does not replace completed campaign records or grant new execution rights.

Each workstream read existing source/contracts before extending its vertical
slice. Independent agents owned disjoint feature directories; manual source
reviewers reported integration defects to their owners. No test, smoke check,
validation command, compiler/profiler command, project import, model operation,
metric calculation, simulation, dependency installation or runtime export ran.
New operation gates are false; existing historical authorization is unchanged.

## Built source and integration boundaries

| Backlog | Owning source | Implemented behaviour |
|---|---|---|
| 01/16 | `campaign/evidence.py` | Explicit completed/failed evidence inventory, immutable metadata catalog, channel/disposition report and conditionally admitted complete-chain freeze; blocked chains remain blocked |
| 02/04 | `stereo_depth/correspondences.py`, `report.py`, `support.py`, `anchor.py` | Reciprocal image correspondences and triangulation, deterministic fitting/assessment holdback, fixed-scale supported geometric reports, strata and explicit anchor aggregation alternatives |
| 03 | `depth/preview.py`, `inspection/operations.py`, `annotations.py`, `reels.py` | RGB-only early-geometry inspection without hand dependencies, actual correspondence adapter, supported annotations/residual overlays, visibility/discontinuity/support review, fixed-scale A/B and separately gated bounded manual-event reel encoding |
| 05 | `depth/challenger.py`, `camera_motion/challenger.py` | Receipt-bound offline ViPE/DA3/VDA outputs, calibrated grids/clocks, explicit optical-z units and rigid extrinsic direction; relative depth never silently becomes metres |
| 06 | `object_masks/correspondences.py` | Bounded optical flow, reciprocal/geometric rejection, persistent query IDs, roster support, loss/unknown/discontinuities and pinned image-only corrections |
| 07 pose | `object_pose/refine.py` | Bounded accepted-point reprojection/stereo refinement, explicit body-point origins and clocks, motion/loss/symmetry boundaries and retained originals |
| 07 surface | `object_mesh/fuse.py` | Moving-object point/observed-TSDF fusion in inherited body coordinates, support artifacts and explicit exclusions, no learned completion |
| 08 | `hand_motion/refine.py`, `contacts.py` | Depth-supported bounded wrist corrections, retained articulation, inferred-vs-observed joint masks, PTS-aware diagnostics and ambiguous multi-cue contact hypotheses |
| 09 | `scene_assembly/prepare_refinement.py`, `refine.py` | Concrete measured-correspondence adapter and bounded fixed-gauge scene solver with independent/sequential/joint arms, explicit supported residual families, clock/grid/lineage admission and retained corrections/rejections |
| 10 | `scratch_depth/research.py`, `_research_network.py` | Separate preparation/training/tuning-inference requests, matched direct/residual original CNN inputs, context and supported robust/boundary/normal/geometry-temporal objectives; original checkpoints unchanged |
| 11 | `learning_experiments/selection.py` | Full-query tuning receipts, error-versus-coverage and member complementarity, prospective geometry gates, separate candidate freeze and explicit consumed-holdout ineligibility |
| 12 | `retargeting/_evidence.py`, `operations.py` | Exact-clock contact support admission, unsupported-slot suppression and lineage sidecars; measured error envelopes remain proposals without invented probability/sampling |
| 12–14 | `policy_training/recipes.py`, `comparison.py`, `hybrid.py` | Matched motion-only/contact-aware objective and curriculum overrides, bounded immutable attempt plan/final-checkpoint rule, executed56D action contracts and separately gated original tiny policy factory |
| 13 | `policy_rollout/achieved_comparison.py`, `operations.py` | Frozen-protocol achieved-rollout comparison, complete failure/coverage roster, matched resources/seed lineage, per-object pinned-kit self-scores and diagnostic references |
| 15 | `track3/timing.py`, `runtime_measurement/` | Opt-in training phase spans, connected whole-arm accounting and bounded matched-workload receipts, distinct timing regimes/phases and memory fields; no automatic compiler or speedup claim |

Paths in the table are relative to `track3/features/`, except the explicitly
shared stable timing primitive. Public request classes, JSON templates, CLI
routes, runtime gates, persistence and documentation remain in each owner.
Templates containing null/placeholder producer paths intentionally cannot run.
The baseline producers and incumbent are not automatically replaced by these
research candidates.

The LR-consistent stereo producer now also exports an explicit continuous
matching-quality field from the same nearest-visible source point as its depth.
This is an uncalibrated matcher diagnostic, not binary validity or probability;
the learning adapter admits its exact field/semantics. New tuning inference
outputs retain a separately supplied calibrated ray grid, avoid target-mask use
and have an explicit independent-geometry report admission path.

Related build details: [moving fusion](07-moving-fusion-build.md),
[policy buildout](12-14-policy-build.md), [freeze](16-freeze-build.md), and each
numbered record. Exact exported schemas and source-identity admission are in the
owning feature README/source; these files are not generated runtime receipts.

## Review and limits

Manual source review corrected cross-slice producer-kind and ancestry mismatches,
retimed PTS admission, duplicate correspondence support, metric-scale admission,
contact intervals spanning gaps and immutable policy lineage. Corrections are
source changes, not verified runtime results. No test/check outcome is claimed.

The hand refinement preserves estimated articulation; full articulated MANO
optimization remains unimplemented. Contacts are inferred geometric/motion
hypotheses, never confirmed force/contact measurements. The strict downstream
adapter rejects unsupported contacts. Scene residual families require concrete
admitted measurements; absent anatomy/contact/silhouette evidence cannot be
manufactured by a solver. Matched temporal supervision in the learning slice is
geometry-consistency supervision, not a recurrent student model.

External SAM3/CoTracker/FoundationPose/BundleSDF/HaMeR and VLA/world-model stacks
remain provisioning/license/dependency blocked. No new model package was added.
Offline challenger ingestion is a source integration boundary, not model
execution or a claim that their outputs exist. Hybrid action contracts and a
tiny policy constructor do not supply executed robot data or a fitted controller.
Observed surface fusion does not establish hidden collision geometry.

Confidence/disagreement remains uncalibrated. Geometry selection requires its
declared evidence channels; missing channels retain the incumbent rather than
promote the least incomplete candidate. Runtime traces require matching admitted
schemas/workloads; incompatible or over-budget traces cannot prove a speedup.
An incomplete final chain cannot be called frozen merely because its individual
Python handlers now exist.

## Continuing work

Last completed: source buildouts across the eligible numbered work items, with
documented limitations above. Current first unresolved item: 03 runtime
inspection/admission (02/04 independent reports also remain unexecuted). Next
conditional work: supply canonical Z: write access and exact compatible producer
records/environment, then perform only separately eligible predeclared research.
Existing no-testing and simulation restrictions still apply.

Canonical storage is outside this session's writable roots. No output, cache or
temporary file may substitute in this checkout or `/tmp`; none was produced.
Episode41 remains excluded; all frames/cameras/derivatives stay episode-grouped.
Consumed43/44 were not accessed for new recipes, thresholds or synthetic mixtures.
The historical frozen research winner remains unchanged. No production promotion,
new untouched assessment, submission packaging or upload is justified here.
