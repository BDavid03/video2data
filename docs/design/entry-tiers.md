# Entry tiers and selection

The full version 1 target includes tracked reconstruction and a trained policy.
A resting fallback is a delivery milestone and comparison baseline, not a reduced
definition of success. This page describes source behavior; no tier currently has
end-to-end runtime evidence.

## What each tier means

| State | Reference and policy | Required evidence | Limitation |
|---|---|---|---|
| `none` | No eligible explicitly promoted rollout | Missing, incomplete or blocking inspection | Cannot supply a complete pose submission |
| `resting` | Opening object poses, sourced parked hands, zero-update resting policy | Actual simulator rollout, nonblocking inspection, intact lineage and explicit resting promotion | Does not attempt the demonstrated object motion |
| `tracked` | Full reconstructed scene and trained policy | Actual trained-policy rollout, nonblocking inspection, explicit promotion and applicable comparison evidence | Training can still perform worse than fallback; perception errors remain |

Replay is a policy kind used by the policy/selection path, not a campaign
`tracked` promotion. A replay checkpoint or inspection cannot be relabelled as a
trained policy. The [campaign](../../track3/features/campaign/README.md) enforces
that the selected tier matches the inspection's actual policy kind.

Both resting and tracked scenes require a completed neutral parked-hand template,
metric meshes, supports, masses and exact camera/opening lineage. Neither can be
assembled from arbitrary hand coordinates. The resting entry still requires
retargeting, checkpoint creation and simulation; stationary reference arrays are
not achieved rollout evidence.

## Baseline and subsequent selections

The initial submission plan always selects resting rollouts across the complete
roster. This establishes a coherent baseline. Later plans name that baseline and
one stated change. Overrides are explicit recorded choices, separate from the
automatic comparison rule.

The rule called **D17** in source documentation applies to automatic comparison
when the campaign has a current tracked-scene role selection:

- Resting and challenger self-scores must use the exact same selected full scene.
- The challenger must be eligible, have intact required evidence and retain
  `left_working_area: false`.
- Select the best eligible challenger only if its score exceeds the resting score
  by the configured margin. An ineligible higher-scoring run must not hide an
  eligible candidate.
- Otherwise retain the resting fallback. Record comparison scene, fallback,
  candidate scores and margin in `selection_evidence`.

The margin needs a human rationale based on development work. It is not an
empirically established constant merely because it appears in a request. The
submission plan does not consume public diagnostic labels to make its choice.
Self-score measures agreement with the reconstructed scene; it cannot establish
agreement with a confidential evaluation trajectory.

Campaign promotion and submission selection are separate decisions. Ordinary
artifact pins do not erase the last explicit rollout promotion. A later resting
scene pin does not replace the tracked scene used for comparison. The append-only
pin history retains both roles and the reasons for changes.

## Mesh delivery is separate

A pose tier does not imply complete CD-O meshes. Every required episode-object
pair still needs eligible reconstructed geometry. Public scans can support
tracking without qualifying as submissions. Hull and TSDF choices are explicit;
TSDF surface coverage and hull concavity errors have different consequences.
The spoon collision frame remains a separate unresolved prerequisite.

## Promotion evidence and limits

Future evidence should identify the exact inspection, checkpoint, recipe, scene,
mesh and ancestor hashes; blocking findings; comparison clock; and reason for the
selection. A successful preview or pack operation cannot supply missing rollout
evidence. A complete resting roster cannot close the
[full version 1 ledger](version1.md), and a local tracked roster cannot substitute
for final packaging and accepted delivery receipts.
