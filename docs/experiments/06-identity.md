# 06 — Masks, correspondences and persistent identity

Status: existing implementation reconciled; research/design complete enough for a
future bounded proposal; event-specific evidence blocked. Owner: `object_masks/`.
Research accessed 2026-10-18; upstream `main` is mutable and is not a deployment pin.

## Reuse and hypothesis

`track3/features/object_masks/{generate.py,_models.py,_identity.py}` already own
GroundingDINO + SAM2.1 Hiera-Large integration, detector association, bounded
forward/backward windows, ambiguity rejection and anonymous hand slots.
`backends.json` still has null installation/source pins. Geometry-only
`opening_clusters.py` is the simpler opening baseline, not a full-video tracker.
Do not rebuild these adapters or claim an unexecuted implementation succeeded.

Hypothesis: admitted point correspondences and explicit visibility can reduce
identity switches beyond mask overlap/box displacement, particularly under overlap
and reacquisition. Semantic labels alone cannot establish persistent roster slots.

| Approach | Input/role and concrete gap | License/weights/dependency/resource evidence | Disposition |
|---|---|---|---|
| Existing detector + SAM2 | Text boxes and RGB frames produce anchored masklets. Current windows stop at unsupported masks and refuse ambiguous association; retain these semantics. | Offline hash-pinned installation/assets required but absent. SAM2 upstream requires Python≥3.10, Torch≥2.5.1; Apache code/checkpoints with component notices; optional CUDA postprocessing is disabled locally. Runtime/memory and event reliability unmeasured. | Implemented, unassessed; retain incumbent design pending provisioning. |
| SAM3 | RGB plus text/visual prompts detects, segments and tracks concept instances. Needs explicit model-ID→roster-ID association and same absent/ambiguous semantics. | 848M parameters; gated checkpoints and custom SAM License; Python≥3.12, Torch≥2.7, CUDA≥12.6. No local memory fit. | Investigated; defer new adapter/dependencies until matched evidence justifies replacing incumbent. |
| SAM3.1 | Shared-memory Object Multiplex extends SAM3 multi-object tracking. Same identity integration gap. | New checkpoints/code; ~7× upstream H100 speedup at128 objects does not establish benefit at tabletop object counts. Pin license/checkpoint separately before use. | Investigated; defer alongside SAM3; no inferred superiority. |
| CoTracker3 | Video plus query points/grid gives pixel trajectories and visibility, complementary to segmentation. Needs feature-owned correspondence artifact and geometric rejection. | Majority CC-BY-NC with MIT/Apache components. Online/offline pretrained variants; Torch Hub examples download implicitly and are unsuitable here. No local measured memory bound; package metadata alone is not a complete environment lock. | Investigated; defer provisioning and execution, retain as a complementary challenger. |

Sources: [SAM2](https://github.com/facebookresearch/sam2),
[SAM3](https://raw.githubusercontent.com/facebookresearch/sam3/main/README.md),
[SAM3.1](https://raw.githubusercontent.com/facebookresearch/sam3/main/RELEASE_SAM3p1.md),
[CoTracker3](https://github.com/facebookresearch/co-tracker).

## Artifact and correction design

Keep each roster slot stable, with original frame index/PTS, source grid, mask
validity, visibility state (`observed`, `occluded`, `out_of_view`, `unknown`),
association support, ambiguity/rejection reason and discontinuity ID. Visibility
and identity confidence are distinct from foreground-logit heuristic scores.
Future point tracks record query ID, originating frame/pixel, per-frame pixel,
visibility, geometric acceptance and source hashes. Track IDs are evidence, not
roster identities. No copying a confident identity across an unsupported gap.

Manual corrections must record source RGB hash, grid, frame/PTS, operator-supplied
points/boxes/masks or slot assignment, reason and revision. Reserved pose labels
cannot supply corrections; anonymous hands remain non-anatomical until estimation.

## Proposed bounded evidence

After items03–04, preselect six32-frame windows per tuning episode39/42 covering
hand/object overlap, similar objects, complete occlusion, exit/re-entry and mask
contamination. Freeze prompt/correction policy before comparison. Compare current
SAM2, SAM3/3.1 separately, then incumbent plus CoTracker; maximum384 frames/arm,
256 query points/object/window, 30 GPU minutes/arm, 12 GiB ceiling and2 GiB headroom.
Stop on resource limit, unsupported source grid or ambiguous mandatory identity;
record absence/failure rather than assigning by image order. No automatic retries
with favorable prompts or changed models.

Report identity switches, unsupported relabeling, reacquisition latency in source
seconds, valid/visible coverage and geometric correspondence rejection. Use
label-free manual image adjudication with recorded provenance where needed;
report unresolved ambiguity, matched-support mask agreement and worst events.
Episode41 excluded;43/44 consumed and unavailable for selection/new final evidence.

No new runtime outcomes exist. Exact blockers: synchronized event inspection and
admitted geometry from03–04 unavailable; canonical Z: read-only; backend install
records/weights unprovisioned; new production dependencies need confirmation.
No checks run. Next independent research:07; empirical06 stays open.

## Source build: correspondence and visibility artifact

Implemented (unrun) in feature-owned `object_masks/correspondences.py`, request
JSON and CLI operation. The new execution gate remains false. The handler reuses
the incumbent's hashed prepared RGB and exact depth/mask lineage; existing cv2
provides a bounded LK baseline with forward/backward, finite/bounds/displacement,
exclusive-mask and RANSAC image-similarity rejection. No existing correspondence
implementation was found in `track3`; this small baseline adds no production
dependency and does not duplicate the detector/SAM adapters.

Run kind `object_masks.correspondences` schema1 persists per-frame JSON query IDs,
roster IDs, source origin/frame/PTS, geometric acceptance and rejection, explicit
identity support/visibility/gaps/discontinuities, and RGB/grid provenance. Query
IDs are scoped to the run and never reused after loss. Initial roster association
inherits incumbent masks; full loss becomes unknown, with no automatic restart.
Optional strict image-only corrections bind operator/reason/revision/points to the
exact mask run, RGB hash, frame/PTS and grid; new corrections start new query IDs
and discontinuities. No diagnostic label producer is admitted. Hands stay
anonymous. No 3D body correspondences, metric rigidity or occlusion classification
are fabricated from 2D image support.

Budget is at most384 frames,256 queries per slot,sixteen slots and1800 wall seconds
checked between frames; defaults32 frames/128 queries. Public train/tuning only:
0/11/12/18/21/23/26/31/39/42, plus explicit runtime allowlisting;41/43/44 excluded.
Failure preserves an incomplete run and rejects downstream admission. No tests,
checks, optional imports, models or runtime artifact creation were executed.

Disposition update: incumbent SAM2 adapter retained; bounded correspondence
baseline **source implemented, unassessed**. SAM3/SAM3.1/CoTracker adapters and
stored-output ingestion remain deferred: no admitted pinned challenger artifacts
or event evidence justify their implementation/provisioning. Corrective points
are implemented; box/mask edits and cross-slot relabeling remain unsupported.
Exact blockers and proposed comparison protocol above remain unchanged. This
build establishes artifact plumbing, not an identity-switch reduction result.
