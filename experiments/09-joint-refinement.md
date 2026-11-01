# 09 — Bounded joint scene refinement

Status: concrete preparation/refinement source handlers implemented, disabled and
unrun; completed admitted residual inputs and independent evidence remain absent.
Owner: scene_assembly. See the build update below and source-owned
[design record](../track3/features/scene_assembly/refinement-design.json).
No dependency, run or test was introduced.

Reuse existing `assemble.py`, `_geometry.py`, `_hands.py`, `_contacts.py` and
source-owned producer contracts. Current assembly performs bounded gap fills,
rest/seating and contact queries; it is not a joint estimator. Preserve its
unmodified artifacts and explicit fill/observed distinctions as the baseline.

Hypothesis: short-window joint refinement can reduce conflicting camera,
object and hand observations, while exposing underconstrained motion. Compare
unchanged independent estimates, sequential camera→object→hand correction, then
joint correction. Do not adopt the more complex solver without a baseline gain.

Fix the original world frame, first camera transform, calibration/baseline,
metric scale and mesh body frames. Optimize SE(3) increments for supported
camera/rigid objects, articulated hand parameters and a stable episode hand
shape; never optimize global scale. Symmetries/alternative contact hypotheses
remain separate candidates. No confident filled pose or contact may substitute
for missing observation support.

Residuals: robust reprojection in pixels, optical-z depth in metres/log metres,
silhouette distance in pixels, accepted persistent correspondences, MANO anatomy,
contact distance/penetration/sliding, PTS-aware velocity/acceleration. Declare
noise scales and weights using training evidence before tuning; normalize by
admitted observation count rather than rewarding disappearing observations.
Use explicit static/dynamic, visibility, depth validity, joint observation and
occlusion masks for every residual. Priors and interpolations remain labelled
priors. Proximity alone is not a force/contact constraint.

Bounded proposal: episodes 39/42, four windows each, 16 frames/window with four
frames overlap, ≤256 admitted points/frame, ≤15 iterations/window, 20 CPU minutes
and 2 GiB artifacts total. Same budget for sequential/joint arms; no refit or
silent model/resolution changes on failure. Stop on missing residual identity,
gauge drift, invalid objective, resource ceiling or increasing fixed objective.
Resolve overlapping estimates by one declared boundary-anchored rule and retain
both pre-merge window results so seams cannot hide drift.

Independent assessment uses item 02 reserved correspondences and unchanged
measurement support, fixed-scale metric depth, silhouette/rigidity/hand/contact
diagnostics and coverage. Optimization residual reduction is a fit statistic,
not independent success. Ablations remove one residual family at a time under
the same support/budget. Report worst windows, ambiguity and lost support.

Disposition: retain independent estimates until the source implementation below
has admitted inputs and independent outcome evidence. Items05–08 lack completed
admitted geometry, identity and joint/contact evidence; canonical runtime storage
is unwritable and no-testing prevents correctness checks. The handler deliverable
now exists; the comparison and independent assessment remain unmet. Episode 41
excluded and 43/44 consumed. Independent item10 learning work can proceed.

## Subagent build update — 2026-10-18

Status: concrete source-only `scene_assembly.prepare_refinement` and `refine`
request/handlers implemented; all new runtime gates remain false. Earlier
statements that no handler exists describe the pre-build disposition and are
superseded by this source update. No optimization, import, test, validation,
model execution, installation or holdout access ran in this build.

The preparation handler reuses object_pose's measured-origin admission helper
for exact compose/camera/identity/stereo parents and a pinned stereo-origin pose
receipt. It checks accepted tracks, original metric body backprojections,
current same-pixel stereo depths and frame clocks, then emits hash-owned
reprojection/optical-z observation records. It excludes unresolved symmetry and
unobserved camera support. It cannot generate observations from absent geometry.
No generic framework or new dependency was introduced.

The refiner retains independent, sequential camera→object→hand and joint
candidates. Finite-difference robust damped Gauss–Newton fits fixed-gauge rigid
pose increments and explicitly observed hand joint displacements in overlapping
windows. Velocity and endpoint bone lengths are labelled priors. Explicit
source clocks, observation/occlusion/depth/identity support, calibrated intrinsics
and original metric/body frames are admitted before fitting. Rank deficiency
stops an arm with ambiguity retained; no damped arbitrary solution is promoted.
Trial objective increase/invalid geometry/resource ceilings stop it. Accepted and
rejected step vectors and both pre/post window states are persisted. Earlier
windows own overlap frames; the earliest supported camera per window is fixed.

Bounded default: episodes 39/42 separately, four 16-frame windows, four-frame
overlap, at most 256 admitted residual records/frame, 768 variables/window,
15 total iterations/window and 600 cooperative CPU seconds each sequential/joint
arm, 2 GiB total generated artifacts. Scalar residual counts differ from record
counts and are reported. One SciPy linear solve is not interruptible; the CPU
ceiling is cooperative. Defaults have no new calibration or empirical benefit
claim. All support is fixed before fitting; independent image evidence is not
used as an optimization residual or candidate selector.

Investigated/disposition: robust reprojection, optical-z depth, correspondence
and world-point residuals, explicit contact-distance/endpoint bone-length terms,
and PTS-aware velocity priors have source implementations. The preparation
adapter currently supplies only measured object reprojection/depth. MANO pose
and shape fitting, silhouette distance, signed penetration, contact sliding,
joint limits and acceleration remain absent and declared as such in output.
Current hand FK landmarks are inferred, not observed; current contact phase
hypotheses are not physically confirmed contact, so neither is promoted into
joint measurement support. Sparse hand intervals require an explicit aligned
adapter and are refused. Separate frozen inputs/runs are required for competing
symmetry/contact hypotheses; no hypothesis averaging is implemented.

Assembly admission now optionally accepts refined object/hand producer runs
while preserving exact compose/lift/camera lineage and existing metric MANO/grid,
body-frame/mesh contracts. Default inputs remain unchanged. Resting assembly
refuses refined-pose summaries because `opening.json` intentionally retains the
original compose values. Joint research outputs are not automatically assembled
or selected.

Results/failures: source implementation only; no new numerical result exists.
The runtime blocker is now missing completed admitted source receipts/runs and
canonical writable storage, with prohibited checks still preventing correctness
evidence. The adapter is concrete, but has not produced an artifact. The bounded
three-arm comparison, ablations and independent assessment are still unrun.
Consumed holdout episodes 43/44 cannot support new recipe selection or an
untouched final assessment; episode 41 remains excluded.

Source references: `track3/features/scene_assembly/prepare_refinement.py`,
`refine.py`, `prepare_refinement.json`, `refine.json`, `runtime.json` and the
feature README. Last completed item remains evidence reconciliation/source
contracts; item09 implementation advanced, empirical deliverable remains open.
Next independent work: item10/11 source integrations; resume item09 assessment
only after admitted inputs, canonical access and separately eligible protocol.
