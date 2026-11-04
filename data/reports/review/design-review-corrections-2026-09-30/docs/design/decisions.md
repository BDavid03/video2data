# Decisions

Each entry records a choice, what it was chosen over, and what would reverse it.
Every decision is labelled **Design** in the sense of the
[conventions](README.md#conventions-used-in-every-document): none has been tested.
Entries marked *needs the user* cannot take effect until the user acts.

## Summary

| # | Decision | Status |
|---|---|---|
| D01 | Organise work as vertical feature slices | In force, user instruction |
| D02 | Treat the Submission Kit and the FlashCHORD commit it names as the contract of record | In force |
| D03 | Produce rollouts with the official evaluator, unmodified | In force |
| D04 | Train one policy per scored episode | In force |
| D05 | Use floating Sharpa hands | In force, reversible by an organiser answer |
| D06 | Keep rollout frames equal to video frames by training at unit motion speed | In force, reversible by an organiser answer |
| D07 | Take metric scale from the calibrated stereo pair | In force |
| D08 | Put the table at z = 0 and use the simulator's ground as the table | In force |
| D09 | Use public scans for tracking and simulation; submit reconstructed meshes to CD-O | In force, reversible by an organiser answer |
| D10 | Fill missing measurements in one place only | In force |
| D11 | Run perception on Windows and the policy stack in an isolated Linux environment | Needs the user |
| D12 | Submit early with a complete, weak entry, then improve | In force |
| D13 | Keep the project in a Git repository shared with the organisers | Needs the user |
| D14 | Run sub-agents on Opus 5.5 or Sonnet 5.5 | In force, user instruction |
| D15 | Keep every operation disabled until separately authorized | In force, user instruction |

---

## D01. Vertical feature slices

**Choice.** Each feature owns its request, handler, integrations, persistence,
configuration and documentation in `track3/features/<slice>/`. Slices share only
`track3/paths.py`, `track3/gpu.py` and `track3/provenance.py`, and exchange files.

**Over.** Horizontal service and repository layers.

**Why.** The user's instruction in `AGENTS.md`. It also fits the work: each slice
wraps a different external model with its own licence, environment and failure
modes, and little logic is shared between them.

**Reversed by.** The user.

## D02. Contract of record

**Choice.** The contract is the Submission Kit archive (SHA-256 `a2de2aee…91b3`,
retrieved 2026-09-30) and upstream commit `aee95f25`, both pinned in
`configs/sources.lock.json`.

**Over.** The challenge website's prose, which still names an `eval_e2e.py` script
that appears in none of the inspected trees or the kit.

**Why.** The kit is executable and specific. The website is neither.

**Reversed by.** A new kit. The URL is unversioned, so the archive hash is
compared before each packing run.

## D03. Official evaluator, unmodified

**Choice.** Every submitted trajectory is written by the kit's
`evaluate_policy.py`, run in the FlashCHORD runtime at the pinned commit.

The runtime must be installed from an LF checkout of `aee95f25`, with CRLF
conversion disabled, stored inside the project-owned WSL disk. The Windows
reference checkout under `upstream/` has CRLF in the four evaluation modules,
as recorded in review E-04, and must not be used as installation source.
`eval_code_sha256` hashes their raw bytes
(`FC/src/flash_chord/evaluation/harness.py:123-135`). Future rollout inspection
and submission checks compare this value with the LF source hash and any
published allowlist. No checkout or hash check is authorized now.

**Over.** Writing a compatible Parquet from another simulator, or post-processing
the evaluator's output. The packer would accept either: its checks cannot tell
producers apart, and the evaluator-hash allowlists are empty (**Official**,
`KIT/config/tracks.json:27-28`).

**Why.** Every instruction in the kit says to use the official evaluator. The
allowlists can be filled later without rescoring earlier submissions
(`KIT/config/tracks.json:17-22`), and an entry that depends on a gap in
enforcement can be invalidated at any time. Award eligibility also involves
organiser review.

**Reversed by.** An organiser statement that other producers are acceptable.

## D04. One policy per scored episode

**Choice.** Twenty policies, one per evaluation episode, each trained on that
episode's reconstructed demonstration.

**Over.** One policy shared across episodes or tasks.

**Why.** The kit states it: "This challenge trains one policy per sequence, so a
submission normally carries a different checkpoint for every episode"
(**Official**, `KIT/skills/v2d-submit/SKILL.md:52-54`). The provenance rules are
built around it.

**Reversed by.** Nothing foreseeable.

## D05. Floating Sharpa hands

**Choice.** Embodiment `sharpa_hands`, robot name `sharpa_wave`, recipe
`sharpa_flash_sac`.

**Over.** Dexmate arm with Sharpa hand, and the G1 humanoid with Dex3 hands.

**Why.** Sixteen scored episodes have two objects. The arm-mounted egocentric
route accepts exactly one (**Official**,
`RG/scripts/retarget/ego_recon_to_dexmate_sharpa.py:1068-1069`). The humanoid
route takes whole-body motion, which egocentric video does not show. Floating
hands also remove reachability as a failure cause.

**Reversed by.** An organiser statement that Track 3 requires another embodiment.

## D06. Rollout frames equal video frames

**Choice.** Train with the recipe's control rate of 20 steps per second, set
`task.motion_speed=1.0`, and declare the reference frame rate as exactly 20.0.
The reference then has one sample per video frame and the rollout has one step
per video frame.

**Over.** The released recipe's half speed, which gives two steps per video frame.

**Why.** The packer matches rows to the roster by integer frame and silently
discards extra frames. A half-speed rollout would be accepted and scored against
the wrong moments. Details in
[policy_rollout](../../track3/features/policy_rollout/README.md).

**Cost.** The released recipes were validated at half speed. Full speed is harder
to track.

**Reversed by.** An organiser statement on how their baselines align frames, or
confirmation that a rollout resampled to dataset frames is acceptable.

## D07. Metric scale from stereo

**Choice.** Scale comes from the calibrated mono stereo pair. Monocular depth
supplies dense shape and is rescaled to agree with stereo.

**Over.** Trusting the monocular model's own metric output; taking scale from
known object sizes.

**Why.** The scorer removes no scale error. The stereo baseline is a measured
length. A monocular model's scale is a learned guess. Known object sizes are
available for ten objects and would be a useful cross-check, and one object has
no scan.

**Reversed by.** Evidence that stereo depth is too noisy to anchor scale, in
which case scanned-object size becomes the anchor for episodes that have one.

## D08. Table at z = 0

**Choice.** The world's z axis is the table normal and the table top is z = 0.
The simulator's ground plane serves as the table.

**Over.** Keeping the table at its physical height with a separate support
surface.

**Why.** Upstream's support reconstruction assumes the floor at z = 0 and drops
supports at or below 5 cm (**Official**,
`RG/source/robotic_grounding/robotic_grounding/retarget/support_recon.py:39,741-757`).
Floating hands need no floor below the table. One surface is simpler than two.

**Assumption.** The table is level. No inertial data exists to check it.

**Reversed by.** A simulator run showing different contact behaviour on the
ground plane, or a recording where the table is visibly not level.

## D09. Meshes

**Choice.** For tracking and simulation, use the public scan where one exists,
and a reconstruction for `wooden_spoon`. For the CD-O leaderboard, submit
reconstructions for all 36 pairs.

**Over.** Reconstructing every object for every purpose; submitting public scans
to CD-O.

**Why.** Reference poses are expressed in the scan's frame, and the scorer does
not forgive a body-frame mismatch. Using the scan removes that error for ten of
eleven objects. Submitting the organisers' scans as reconstructions would not be
a reconstruction.

**Reversed by.** Organiser answers on whether scans may be used at evaluation,
whether they may be submitted, and what the `wooden_spoon` frame is.

## D10. One place for filling

**Choice.** Perception slices write NaN and a validity mask when they have no
measurement. Only [scene_assembly](../../track3/features/scene_assembly/README.md)
fills, and it flags every filled value and enforces a maximum gap.

**Over.** Letting each slice smooth and interpolate its own output.

**Why.** Filled values look like measurements. Filling in several places makes it
impossible to say later how much of a trajectory was observed.

**Reversed by.** Nothing foreseeable.

## D11. Runtime environments — needs the user

**Choice.** Perception runs in Windows environments inside this repository, like
the existing MoGe2 environment. Retargeting, training and rollout run in a new,
isolated Ubuntu 24.04 WSL distribution whose virtual disk is stored under
`.runtime/wsl/` on `Z:`.

All retargeting operations run in Linux, including export and asset writing.
Separate Windows environments under `.runtime/mano/` and
`.runtime/submission-python313/` host MANO kinematics and submission respectively.
The scene bundle goes to Linux; rollout Parquet, JSON and provenance records
return to Windows. Hand parameters and kinematics results pass between the
perception and MANO processes. See the environment table in
[system overview](system-overview.md).

**Over.** Installing into the existing shared Ubuntu distribution; running the
policy stack natively on Windows.

**Why.** FlashCHORD states Linux and a driver at or above 580 (**Official**,
`FC/README.md:11-14`). The existing Ubuntu is version 26.04, stores its disk on
`C:` with 62 GiB free, and is shared with other projects (**Measured**,
`reports/runtime/README.md`). `AGENTS.md` requires all artifacts inside this
repository and forbids modifying external environments.

**Needs.** The user's confirmation to create the distribution and to install the
locked FlashCHORD dependencies and the retargeting stack.

**Reversed by.** The policy stack failing to run under WSL 2 on this GPU.

## D12. Submit early, then improve

**Choice.** The first submission covers all 20 episodes and all 36 meshes with
whatever quality exists, as soon as the pipeline runs end to end.

**Over.** Waiting for a good result.

**Why.** A submission is rejected whole if any episode is missing, so
completeness is the first thing to prove. The organisers' message asks teams to
submit early. Track 3 has no local score, because its reference is confidential
(**Official**, `KIT/tools/pack_submission.py:447-451`); the leaderboard is the
only measurement on evaluation episodes. With the first submission planned by
18 October, there are about ten scored submissions per competition before the
unlimited final three days, subject to the unpublished weekly reset. The quota
and request to submit early are also in the
[user-supplied organiser announcement](../../reports/contract-evidence/2026-09-30/organiser-announcement.md).

**Reversed by.** Nothing foreseeable.

## D13. Git repository — needs the user

**Choice.** The project becomes a Git repository with a GitHub remote shared with
the organisers. A submission is packed only from a clean, pushed commit.

**Why.** Every submitted row carries a commit URL (**Official**,
`KIT/v2dlb/report_schema.py:55`). Winners must release their code under
Apache-2.0 (**Official**, `KIT/config/tracks.json:334-337`). The project root is
not a Git repository today.

**Needs.** The user to create the repository and decide its visibility. Before
the first commit, `coursework/` needs a decision: it holds course material and
two 347 MB embedding files, and it may not be the user's to publish.

**Reversed by.** Nothing; the commit URL is mandatory.

## D14. Sub-agent models

**Choice.** Sub-agents run on Opus 5.5 or Sonnet 5.5.

**Why.** The user's instruction on 2026-09-30. It replaces the model settings in
`TRACK3-PLAN.md`, which describe an earlier session.

## D15. Disabled by default

**Choice.** New slices will check an authorization flag in their slice's
`runtime.json` before importing a model, opening a video or writing an artifact.
scratch_depth already owns runtime flags. Implemented depth is the exception:
it checks `execution.model_execution_authorized` in
`configs/perception/moge/local.json`. Execution flags remain false; runtime files
for design-only slices remain proposed. No command-line switch bypasses a flag.

**Why.** The user's instruction in `AGENTS.md`: no testing, no model execution,
until the user changes the instruction.

**Consequence.** Nothing in this design has been observed to work. The
[roadmap](roadmap.md) lists the authorizations the plan depends on and the dates
by which each is needed.
