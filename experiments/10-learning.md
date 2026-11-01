# 10 — Learning beyond teacher imitation

Status: existing results reconciled; gated ablation preparation/training source
implemented; fitting/inference unrun and blocked. Owners: learning_experiments
and scratch_depth.

Reuse all completed ANN/CNN/transformer/boosting recipes and five scratch
variants from item 01. Do not relaunch their campaign. The existing original
`CourseworkDepthCNN` already has U-Net-style encoder/decoder skip connections;
adding a second generic U-Net would duplicate it. It predicts depth directly
from RGB and uses absolute log-depth MAE, with no scale alignment. Original
random initialization is explicit in `_network.py` and research `models.py`.

Hypothesis: admitted geometric supervision and local temporal context could
improve independently supported depth where teacher-only imitation fails.
Lower pseudo-target loss alone cannot promote a recipe.

## Ordered proposed ablations

Each row changes one factor from its declared baseline. Progress only when
independent evidence from item 02 is available; do not combine every new factor.

| Order / alternative | Comparison and disposition |
|---|---|
| A: residual CNN/U-Net | Implemented gated matched six-channel direct/residual wrappers reusing unchanged CourseworkDepthCNN, residual bound ±0.5 log metres and explicit centre support. Producer confidence is required, never replaced by validity. RGB incumbent retained. Unrun; confidence archive unavailable |
| B: temporal context | Implemented optional producer-warped three-frame channels with source-owned visibility/motion receipts, contiguous indices and actual PTS admission. No raw stacking across gaps. Temporal encoder deferred until channel context earns its complexity. Unrun; admitted clips absent |
| C: stereo-supported labels | Implemented source adapter distinguishing pseudo-labels and admitted training stereo. Raw dense stereo targets refused until a producer-owned reserved-support exclusion receipt exists. Sparse correspondence evidence cannot supply dense confidence. Unrun; support archive absent |
| D: robust log loss | Implemented absolute log baseline and Huber log residual with transition fixed 0.1; neither removes metric scale. Unrun; no lower loss or geometric gain claimed |
| E: boundary loss | Implemented masked valid-neighbour log-depth-gradient residual with fixed weight 0.1, separate from F/G. RGB edges never create labels. Unrun |
| F: surface normals | Implemented calibrated backprojection with bounded cosine residual, fixed weight 0.1, valid nondegenerate neighbourhoods and required producer boundary exclusion. Separate from E/G. Unrun; boundary support artifact absent |
| G: temporal objective | Implemented visibility-masked Huber log-z supervision against admitted motion-transformed neighbour optical depth, fixed weight 0.1, separate from E/F. Frozen geometric supervision does not prove student-to-student consistency. Differentiable prediction reprojection deferred pending geometry. Unrun; archive absent |
| H: diverse teachers | Only after measured complementary independent errors: compare one teacher replacement at a time with target-source/quality masks. ViPE/DA3/VDA family names do not establish complementary errors. Investigated; defer without paired evidence and approved weights |
| I: synthetic enrichment | Separate 128-sample pilot is training-only procedural data, never captured evidence. No mixture chosen: select any future ratio on real tuning geometry under a separately fixed budget, never historic holdout. Adapter currently admits real prepared pairs only. Investigated; defer mixing and extend provenance before use |

A proposed common fit ceiling is ten epochs, batch four, maximum side 128,
seed 31, 128 training clip centres with three contiguous frames when needed,
≤30 GPU minutes/arm, ≤12 GiB process device memory with 2 GiB free headroom,
≤4 GiB artifacts/arm. At most eight new fits (A–G, with at most two A controls)
and four GPU hours total; H/I require separate proposals, not extra hidden arms.
Stop each arm on invalid support/gradient objective, OOM or resource ceiling;
retain failed outcomes without silently changing the model. No gradient-check,
test or validation command is authorized. These are proposed budgets, not a
claim that fitting fits in memory/time.

Training episode grouping remains 0/11/12/18/21/23/26/31; tune 39/42; exclude
41 and never fit on 43/44. Freeze clip/event indices before fitting. Sample
uniform whole-clip centres plus image-annotated hard events; old sparse frames
are insufficient for contiguous motion loss. Fit preprocessing/normalization on
training episodes only. No fitting of preprocessing on tuning or test pixels.
Match labelled supports when comparing objectives and report native support too.

Primary outcome: item 02 fixed-scale reserved geometry, with temporal and
coverage gates per item 11; teacher MAE remains supplementary. Report equal
episode/object/event errors, worst events and resource counters. Training seed
replication is a later separately bounded design; one seed/two tuning episodes
cannot establish generalization. Pretrained challengers stay separate from the
original randomly initialized models.

Disposition: retain existing original models and frozen teacher ensemble as
research baselines, no new architecture/objective promoted. Exact blockers:
independent geometric evidence 02/04–06 unavailable, residual confidence and
contiguous admitted clip artifacts absent, canonical output not writable. New teachers
also need dependency/weight approval. No empirical ablation outcome is claimed.
Next independent item: 11 uncertainty/selection specification.

## Source implementation handoff

New feature-owned `scratch_depth/research.py` provides explicit preparation and
single-arm training handlers; `_research_network.py` reuses the unchanged
coursework CNN, preserving historical network hashes/checkpoint admission.
`research_prepare_authorized`/`research_train_authorized` are false; all legacy
flags stay unchanged. API/schema/support semantics are documented in the slice
README. Requests pin source protocol/dataset hashes. Arrays must be hash-owned
by completed video-only producers and bound to exact episode/frame/PTS/grid;
legacy multi-episode preparation requires its unique owning sample receipt.
Unknown producers need explicit per-artifact learning-input receipts. New
normalization uses training input depth only and is frozen before fitting.
No held-out sample or tuning loader is available to the trainer. The adapter
admits real captured arrays only; H/I remain deferred with their original reasons.

The optional disabled trace flag connects the new trainer to shared PhaseRecorder
and runtime_measurement grouped accounting, with first-32-example cold/warm
phases plus whole-arm timing, dataset/source identities and failure retention.
It is not a matched speedup experiment. Native objective support is recorded;
independent geometry remains the promotion requirement. Single-arm ceilings
are implemented cooperatively; an outer campaign still must enforce eight-arm/
four-hour totals. Production checkpoint promotion remains disabled
until independent evidence and a frozen admission protocol exist. A separately
disabled tuning-only `research_predict` handler now completes the source path
to independent comparison: exact safe-loaded checkpoint/dataset/config/source
admission, every declared sample of selected episode 39 or 42, no target masks,
calibration-grid binding and owned per-frame/nested episode artifacts. It uses
an explicit same-episode depth calibration run and is admitted by the geometry
report candidate reader. This establishes no evaluated gain or promotion.

The dense stereo producer source now supplies continuous left/right matching
quality as `rgb_match_confidence` with explicit uncalibrated semantics; the
learning adapter admits that exact named field and source declaration. Historical
archives still lack it and no new matching archives were generated. A source
route for A therefore exists without inventing confidence, while reserved
training-stereo targets and B/F/G geometry remain separately blocked.

No tests, validation, imports, model execution, training, dependency changes or
runtime artifacts were produced. New producer contracts are requirements, not
claims that matching archives exist. Exact blockers remain explicit confidence,
training-only reserved stereo exclusions, contiguous admitted warps/motion and
normal boundary masks, independent geometry evidence, and canonical Z: access.
Last completed: source build of A–G admission/objectives. Current: unrun evidence
assessment. Next eligible independent item: 11. Historic campaign results remain
unchanged; consumed 43/44 outcomes were not accessed for this implementation.
