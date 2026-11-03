# 11 — Uncertainty, ensembles and selection

Status: historical research candidate retained; prospective decision rule fixed
as a design before any new outcomes. No new weights, thresholds or candidates
were fitted or assessed. Owner: learning_experiments.

Reuse the item 01 frozen `top2_equal` ensemble, eager/BF16 original scratch CNNs
at weights 0.5/0.5. Both members have common architecture, training samples and
seed; different arithmetic alone is weak diversity evidence. Existing top-three
equal/inverse-MAE alternatives already have historical outcomes. Do not reselect
from consumed 43/44 or report a new untouched assessment. No production geometry
candidate is established by those teacher metrics.

Existing persisted ensemble admission requires exact completed member frame sets,
dataset/partition/grid/intrinsics/ray masks and artifact lineage. Missing whole
member/frame fails; invalid pixels exclude that member and renormalize positive
weights. All members invalid yields NaN depth and false validity; zero-weight
members add no support. No target-validity mask enters prediction.

Source-only addition in `ensemble_operations.py`: persist `member_count`,
`effective_member_count` and weighted `log_depth_std` beside unchanged geometric
mean predictions. The count covers supported positive-weight **direct parents**,
not statistically independent models; effective count is squared admitted total
weight divided by summed squared admitted weights. Standard deviation is NaN
with fewer than two supported parents; zero disagreement from identical parents
is not proof of correctness. It is never named calibrated uncertainty and never
used to modify frozen predictions or weights. Existing artifact bytes/hashes
remain historical; only a future authorized export could contain the new fields.

Investigated alternatives: equal weights (already assessed), inverse teacher-MAE
weights (already assessed), geometric-metric fitted weights (deferred without
matched support), threshold gating, complementarity-selected membership and
confidence calibration (designed, unassessed). Avoid adding members just to
increase count; paired per-event residuals and incremental resource cost must
show useful complementary failures first.

Proposed confidence study: fixed thresholds for log-depth standard deviation
0/0.025/0.05/0.1/0.2/∞, minimum direct support 2, report error versus coverage
on item 02 reserved geometry, per episode/object/event. Thresholds are research
diagnostic bins, not consumer trust levels. Tune any threshold on 39/42 only,
with train-only preprocessing. No calibrated probabilities/intervals without
an explicit calibration target, reliability evidence and episode dependence
limits. Current two tuning episodes are inadequate for strong calibration claims.

## Predeclared prospective choice rule

Apply in order on matched eligible tuning support; failure or unavailable
evidence means retain incumbent, not choose the next favourable metric:

1. Admit exact lineage, metric scale, clocks, observed support and original
   candidate definitions; no excluded/consumed episode outcomes for fitting.
2. Require at least 90% of the incumbent's supported queries in **each** tuning
   episode and each declared static/dynamic event stratum. Empty strata remain
   unavailable and block a claim for that stratum. Report absolute coverage too.
3. Primary: lower equal-episode median absolute log-depth error on reserved
   stereo-supported queries in both tuning episodes. Also report raw errors
   in metres. No per-frame or evaluation-support scale fitting.
4. Guardrails: p90 reprojection error, motion-compensated temporal residual and
   rigidity drift may worsen by at most 5% in either episode on matched support;
   predeclared catastrophic identity loss or new invalid event blocks promotion.
   Zero baseline error admits no positive degradation. Missing any required
   independent channel means insufficient evidence for promotion.
5. If multiple candidates satisfy all gates, choose lowest primary equal-episode
   error, then lower measured complete-operation time, then fewer members, then
   lexical frozen candidate ID. Teacher MAE is supplementary and cannot override
   geometry. This rule is an engineering decision, not a significance claim.

Proposed analysis ceiling: stored tuning predictions only, ≤8 fixed candidates,
32 sample centres plus declared event windows per item02, CPU 20 minutes/2 GiB
artifacts; no new inference/fitting. Stop on missing paired support or resource
failure and retain visible unavailable outcomes. Neither this design nor existing
runtime flags authorize checks or consumed-holdout reuse.

Before any eligible final assessment, persist exact model/checkpoint, weights,
thresholds, geometry/anchor settings, data/event indices, code/config identities,
decision rule and eligibility protocol. A new unbiased assessment needs an
independently fixed protocol; none is designated here. Runtime writes and
independent item07–10 evidence are absent, so only incumbent retention is
justified. Next: item12 retargeting designs; empirical11 remains blocked.

## Source build with subagents (2026-10-18, unrun)

Implemented `learning_experiments/selection.py` and two gated CLI operations:
`geometry-selection` and `geometry-freeze`. Both new authorization flags are false;
old operations, historical selection and geometric ensemble prediction semantics
are unchanged. No tests, imports, model execution, fitting or runtime writes ran.

The comparison implements the ordered 90% support, both-episode primary and 5%
guardrail gates above from exact hash-bound independent geometry receipts, not
teacher scores. It records full reserved/missing support per episode/object/event/
stratum, threshold error/coverage curves, direct-member missing/invalid support and
paired complementarity diagnostics. Predeclared definitions and exact protocol,
checkpoint, weight, threshold, source/data/config/anchor/event identities bind
comparison and freeze. Every parsed query is limited to tuning episodes 39/42.
The receipt schema is documented in the feature README and shared with the
independent-geometry producer. Its absent event/identity/temporal/rigidity fields
remain unavailable; source-report hashes cannot make those channels established.

The freeze handler requires a completed owning comparison, preserves the
historical campaign winner, and records holdout ineligibility because 43/44 were
consumed. No new candidate or threshold was selected. Exact blockers remain
canonical writable storage and admitted independent event/geometry receipts;
source implementation does not establish calibrated uncertainty or superiority.
Alternatives remain: equal/inverse teacher weights historically assessed;
geometric-weight fitting deferred; fixed threshold and complementarity reporting
implemented unrun; probabilistic confidence calibration deferred without a target
and adequate episode evidence. No new production dependencies were added.
