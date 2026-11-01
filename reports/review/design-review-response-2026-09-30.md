# Design review response, 2026-09-30

Scope: all 39 findings in [the independent review](design-review-2026-09-30.md).
The original review and upstream references remain unchanged.

Status: corrections prepared in
[design-review-corrections-2026-09-30/](design-review-corrections-2026-09-30/README.md).
Writes to the original document paths failed with EPERM; the corrected copies
and patch are the deliverable pending file access. These rows describe the
prepared changes, not edits already applied to the original design set.

No tests, validation commands, model operations, notebook cells, dependency
installation, hash checks, uploads or messages were run. A user-requested
Playwright import failed because its installed entry point has an export error.
Node file access was used to read sources and write the correction package.
The first roadmap decision remains due 2 October. Episode 41 is excluded from
proposed execution; its frozen split membership and retained request are unchanged.

| ID | Prepared correction |
|---|---|
| E-01 | Retargeting states that empty support stages are rejected. Inspection records the support requirement; training applies scene.support=false when no Cylinder or Cube exists. The ground remains enabled and checkpoint configuration carries the setting into evaluation. |
| E-02 | Risks, overview and object_mesh distinguish slot 0's initially cancelled body offset from other slots' full offset from frame 0. Seven spoon episodes use slot 1. |
| E-03 | Submission lists the three supported Kaggle 2 authentication paths and explains the environment-variable choice through project storage constraints. |
| E-04 | D03, training, rollout and submission require an LF installation source at aee95f25, separate from the CRLF Windows reference. Future checks compare raw-module digests and any organiser allowlist. Added R14. |
| I-01 | Camera motion leaves NaN/invalid gaps; scene assembly owns bridging. Camera interpolation settings are removed. |
| I-02 | All retargeting operations move to Linux. Documents name the forward scene bundle and returning rollout Parquet, JSON and records. |
| I-03 | Overview and D11 name perception, Linux policy, isolated Windows MANO and Windows submission environments and their operations. |
| I-04 | Registry adds object_masks as a depth-grid reader and policy_rollout as a sequence reader; removes scene_assembly as a stereo reader and direct retargeting mesh consumption. |
| I-05 | Phase 0 includes depth split extension for evaluation and separately gated validation. Registry and depth README state the current restriction. |
| I-06 | Registry marks implemented depth records as nonconforming and lists kind, inputs, sources and provenance migration. Both depth READMEs repeat the required slice-owned work; untagged records are refused. |
| I-07 | Overview and D15 name depth's actual configs/perception/moge/local.json flag and distinguish proposed runtime files from implemented flags. |
| I-08 | Submission plan and check require completed video_only inputs and source=reconstructed meshes. Both diagnostic tags and missing tags are rejected. |
| I-09 | Mesh reuse defaults to disabled pending Q06; submission and object_mesh share the same source-episode rule. Rollouts remain episode-specific. |
| I-10 | D12 uses about ten scored submissions per competition from 18 October to the final window, with reset uncertainty. |
| I-11 | Rollout paths may repeat the same episode index; they contain no other episode_<digits> index. |
| I-12 | Registry documents policy and rollout as short directory names, plus submission and scratch_depth multi-episode exceptions and UUID suffix. |
| I-13 | G1 and Phase 1 propose development episode 39 and explicitly retain the episode 41 refusal. Public compute count is 19 proposed episodes; the frozen split is unchanged. |
| I-14 | Stereo baseline is the ruler independent of object identity; scan dimensions remain D07's cross-check/fallback. |
| I-15 | Contract report adds evaluator mandate/allowlists and mocap-missing reference rows to its open table, matching the existing JSON. |
| I-16 | Final wrist and finger rotations are both anatomical/un-mirrored. Intermediate outputs declare one form for both; kinematics converts them together exactly once. |
| I-17 | Hand README and G4 use models/hand/models/ for both MANO files, matching model readiness. |
| I-18 | Submission flags become pack_poses_authorized and pack_meshes_authorized. |
| I-19 | Artifact flow adds depth-to-mask/mesh/pose/hand, optional mask-to-stereo and sequence-to-rollout edges. |
| M-01 | Dockerfile/model-card statements cite source readings without Measured labels; placeholder masses cite the official dataset URDFs. |
| M-02 | Submission unit/scale claim cites Track 3 README line 36 and object_pose.py lines 141-142. |
| M-03 | Quaternion check targets near-zero norms and cites the scorer's normalization/rejection lines. |
| M-04 | Registry separates official rollout and CSV quaternion citations: rollout.py line 66 and object_pose.py lines 10,47. |
| M-05 | Tracking-termination disabling cites harness.py lines 228-251. |
| M-06 | Rollout states that the evaluator root already disables logging and keeps the explicit override. |
| M-07 | Silent frame mismatch is qualified for slower playback/higher control rate; faster playback/fewer steps can fail for missing rows. |
| M-08 | Submission distinguishes unreadable/unrecognised/duplicate Parquet failures from ignored valid non-scored episodes. |
| M-09 | D02 limits eval_e2e.py absence to the inspected trees and kit. |
| M-10 | Roadmap states the 100-hour reserve and approximately eight-hour policy threshold for Phase 3. Slower training spreads into Phase 4; a second pass is not assumed. |
| M-11 | Historical milestone wording becomes inference stage plus simulator smoke by 27 September, reconstruction by 4 October. |
| M-12 | D11 uses 62 GiB; R01 uses 2.4 GiB for Windows and other processes. |
| M-13 | Stereo transfer applies R1 transpose before A_T_C. |
| M-14 | Scene contacts use explicit closest-point-on-triangle calculations and solid-angle inside/outside classification, avoiding rtree-dependent calls and new packages. This design remains unmeasured. |
| M-15 | Object poses add world_valid = object valid AND camera valid, masking world arrays. Hand lifting propagates the same camera validity rule. |
| M-16 | Feature template moves request-file detail to Configuration and authorization, matching the operations tables. |

The organiser announcement is now included as user-supplied evidence for the
webinar, Discord channel, FAQ mention, quota and approximate L40S claim. It
does not resolve organiser questions or authorize any execution.

Future work remains explicit: implement the depth split/record migrations,
obtain the existing roadmap decisions, and later gather acceptance evidence
only for specifically authorized operations. Dependency proposals stay proposals.
