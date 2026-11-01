# Error budget and evidence plan

This is a qualitative error budget, dated 5 October 2026. It identifies where
error enters and how it propagates. It is **not a measured allocation**, and it
sets no new acceptance thresholds. Current instructions prohibit collecting the
runtime evidence described here.

The pinned scoring interpretation is documented in the
[contract report](../../reports/contract.md) and
[object-pose feature](../../track3/features/object_pose/README.md). One initial
rigid alignment removes a world offset/rotation; it does not remove scale drift,
wrong body frames, temporal mismatch or subsequent trajectory error.

## Separate the sources of error

| Source | Propagation | Evidence that would distinguish it |
|---|---|---|
| Calibration/grid mismatch | Distorted rays affect depth, registration and projection throughout the chain | Exact calibration/grid identity and authorized geometric residual evidence |
| Metric scale | Relative distances and movement amplitudes are wrong | Observed stereo support, per-frame scale/spread and separate scan-dimension diagnostics |
| Camera motion | A stationary object's reconstructed world pose drifts with head tracking | Static landmark support, reprojection and invalid-frame distribution |
| Opening object frame | Initial orientation/layout errors affect later alignment and relative placement | Opening registration alternatives, ambiguity flags and body-frame provenance |
| Object tracking | Occlusion, drift or symmetry creates incorrect motion | Observation/recovery outcomes, residuals, surface support and gap distribution |
| Hand fit | Wrong side, depth or anatomical rotation changes retargeted actions | Camera/world validity, fit residuals and authorized selected-frame plus full-clip review |
| Scene completion | Smoothing/fills can suppress or invent motion | Filled shares, gap lengths, seating corrections and explicit hand-source flags |
| Retargeting/contact approximation | A plausible demonstration can become an infeasible robot reference | Robot reference inspection and contact/support diagnostics |
| Policy tracking | The simulated robot fails to realize the reference | Actual achieved motion compared with the same complete scene |
| Packaging/clock | Correct arrays can be assigned to the wrong rows or frames | Exact roster, 20 Hz correspondence and packed-file identity admission |

These contributions are coupled. For example, a wrong scale changes registration,
seating and grasp distances simultaneously. Adding their magnitudes as independent
Gaussian errors would be unjustified without evidence.

## Useful geometric sensitivity

For a small orientation error of angle `theta` radians, a point at distance `r`
from the rotation centre moves approximately `r * theta`. At 30 cm, a 10-degree
rotation gives roughly 5.2 cm displacement. This explains why an apparently small
opening orientation ambiguity can affect the relative layout of a second object.
It is a geometric example, not a measured project result.

A scale factor `1 + s` changes a displacement of length `d` by approximately
`abs(s) * d`. A 5% scale error over 40 cm contributes about 2 cm. A rigid alignment
cannot repair that scale. Camera and object motion errors can either cancel or
reinforce, so a low residual in one view does not establish correct world motion.

Temporal error matters most during fast motion. For approximately constant local
speed `v`, a timing displacement `dt` contributes about `v * dt` translation.
Preserving MP4 PTS while separately enforcing the reference/rollout clock avoids
silently treating distinct time bases as equivalent.

## Threshold ownership

Thresholds are request/configuration choices owned by the feature that applies
them. Their presence in source does not make them calibrated tolerances.

| Owner | Examples | Interpretation |
|---|---|---|
| Stereo/anchor | Disparity range, valid depth range, support share | Admission of observed depth and scale |
| Camera motion | Feature budget, inlier floor, mask dilation | Evidence needed for a camera observation |
| Object pose | Residual, surface support, motion and recovery limits | Evidence needed for a registered observation |
| Hand lift | Depth support, residual and shift limits | Whether the metric fit is admitted |
| Scene assembly | Gap length, resting criteria, contact distance, seating limit | Explicit reference-completion policy |
| Object reconstruction | Voxel size, support count, point/grid/face capacities | Resolution and bounded resource behavior |
| Submission selection | Challenger margin | Deliberate comparison policy with a human rationale |

For example, the current scene request contains a five-frame maximum gap and
5 cm seating limit. These are source defaults, not demonstrated safe values.
Do not increase them simply to hide an upstream failure. Similarly, a smaller
voxel is not automatically more accurate when the depth uncertainty is larger.

## Coverage belongs beside accuracy

Report what was observed before reporting a mean error. Invalid camera frames,
lost objects, rejected hands and unavailable reference meshes have different
meanings. They must not become zeros in an aggregate.

The implemented public mesh diagnostic supports same-object candidate/baseline
comparisons with matched references and sampling seeds. Deltas are candidate
minus baseline; a negative distance delta indicates improvement for that metric.
The known unavailable spoon scan contributes an explicit unscorable row, not a
zero distance. Mixed reports retain scored/unscorable counts and the documented
episode weighting. No comparison has been executed.

Selected hand previews cover only the requested frames and projected joints.
They can reveal a gross image-space mistake without establishing world depth,
full-video side identity or contact correctness. Retain rejected-fit outcomes
instead of showing only attractive successful frames.

## Keep three comparisons separate

1. **Perception against public references:** diagnostic-only, with development
   and frozen validation separation. It informs design without becoming a
   production artifact input.
2. **Achieved rollout against the reconstructed full scene:** the self-score
   used for local policy/fallback comparison. It can improve even if the scene
   itself is inaccurate.
3. **Submitted result against the external evaluator:** the eventual delivery
   outcome. A local self-score or successful packer exit cannot substitute for it.

A future authorized evidence report should name exact runs, source hashes,
episode/frame coverage, validity/fill proportions, metric definition and resource
measurements. State where observations end and derived geometry begins. Until
that evidence exists, the [delivery ledger](version1.md) remains incomplete.
