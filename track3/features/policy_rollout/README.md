# Policy rollout

Status: implemented, never run

The three commands use separate, immutable timestamped runs. `inspect.json` and
`self_score.json` therefore live in their own run directories and refer to the
original evaluator output by path and hash. Every operation and named episode
remains disabled in `runtime.json`; no CLI argument bypasses authorization.
`rollout` requires the dedicated Linux policy environment. Read-only inspection
and scoring also accept the dedicated Windows submission environment.

The fallback ladder is recorded, not launched automatically: a caller explicitly
requests an earlier checkpoint after a failure. A tracked rollout requires an
already inspected resting fallback. Invalid worlds fail the rollout with
`failure_reason=invalid_world`; the shared status remains `failed`.
An inspection with blocking findings keeps its report and has status `failed`,
so a completed inspection pin always denotes a passing result.
Self-scoring uses `campaign/selections.py` to require the current pin in the
tracked scene role. A later resting-scene pin does not replace it. The operation
records the pin and the hash of the exact log snapshot it read. Both resting and
trained rollouts are compared against this same full scene. It records filled
reference poses, and evaluates the final object origins against the table extent
after the kit's initial-pose alignment. It never reads public pose labels.

This slice runs the organisers' evaluator on one trained policy and keeps the
object trajectory it writes. That file is what the leaderboards score. `FC/`
means `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/`
and `KIT/` means `upstream/v2d_submission_kit/v2d_submission_kit/`.

## Role in scoring

The rollout Parquet is copied into the submission without any transform
(**Official**, `KIT/tools/pack_submission.py:297,345-346`). Whatever this slice
records is the score.

## How the evaluator decides which frames exist

This is the behaviour the rest of the design bends around. All **Official**,
read from `FC/src/flash_chord/`.

| Step | Source |
|---|---|
| The evaluator rebuilds the training configuration from the checkpoint, including the reference path, the control rate and the motion speed. | `evaluation/harness.py:376-420` |
| The reference is resampled to one sample per control step. At unit motion speed the sample count is `round((N-1) / source_fps * control_fps) + 1`. | `data/resampling.py:11-40` |
| The rollout takes one control step per resampled sample: `step_count = reference.num_frames - start_frame`. | `evaluation/harness.py:421-423` |
| Each row records `reference_frame = step + start_frame`, the index into the resampled reference. | `evaluation/rollout.py:186-190` |
| Tracking-failure terminations are disabled, so every world advances through the whole reference. | `evaluation/harness.py:228-251` |

The submission packer matches rows to the roster by exact integer frame, drops
extra frames without a message, and rejects only a rollout that is too short
(**Official**, `KIT/tools/pack_submission.py:256-297`). The roster's frames are
video frames.

Taken together: slower reference playback, or a control rate above the video's,
produces more rollout steps than video frames. The packer takes the first N steps
as if they were the N video frames and discards the rest. The file is accepted
with poses from the wrong moments. Faster playback or a lower control rate
produces fewer steps and can instead be rejected for missing rows
(`KIT/tools/pack_submission.py:277-293`).

The released floating-hand recipe runs at 20 control steps per second and half
motion speed (**Official**, `FC/src/flash_chord/configs/experiment/sharpa_flash_sac.yaml:20`,
`FC/src/flash_chord/configs/sim/rl.yaml`). For a 20 frame-per-second reference
that is two control steps per video frame. A policy trained with that recipe
unchanged would submit only the first half of each demonstration, stretched across
the roster.

**Design.** A rollout is accepted by this slice only when its step count equals
the episode's video frame count. [policy_training](../policy_training/README.md)
makes that true by keeping the recipe's control rate of 20 and setting
`task.motion_speed=1.0`. The reference frame rate is declared as exactly 20.0,
matching the MP4 frame rate, so the loader does not resample and returns the same
N frames (`FC/src/flash_chord/data/mano_sharpa.py:236-238`). The capture rate in
the public metadata is 20.25 Hz; the 1.3% difference in playback speed is
accepted. Tracking a demonstration at full speed is harder than at half speed,
and the released recipes were validated at half speed, so this choice has a cost
that only training runs can measure.

## Checkpoint admission

Before evaluation or trajectory inspection, `_checkpoint` uses the public
`policy_training/checkpoints.py` metadata contract. The selected file must be
owned by the named training run and match its artifact hash, complete resolved
recipe, reference clock, rigid-object metadata, budget counters and saved
admission inventory. The contract source is included in the rollout record.
This complements the official evaluator's tensor/schema checks; header admission
alone is not evidence that evaluation works. The entire path remains disabled
and unrun.

## Operations

Both disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `rollout` | The evaluator's trajectory Parquet and metrics JSON for one episode | Policy environment, Linux | Yes |
| `inspect` | `inspect.json`: findings from opening the Parquet | Policy environment or submission environment | No |
| `self-score` | `self_score.json`: the kit's metrics for this rollout against the episode's pinned reconstruction | Policy environment or submission environment | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `policy_<steps>.safetensors` | [policy_training](../policy_training/README.md) | Weights and embedded configuration |
| The training-ready sequence and its assets | [retargeting](../retargeting/README.md) | Read by the evaluator through the path stored in the checkpoint |
| `KIT/flash_chord/scripts/evaluate_policy.py` and its configuration files | Organisers | Run unchanged |
| `KIT/data/track_3_sample_submission.csv` | Organisers | Expected frame count per episode, for `inspect` |

## Method

1. **Run.** Call the kit's LF copy of `evaluate_policy.py` from the LF FlashCHORD
   checkout described in [policy_training](../policy_training/README.md), once per episode, with:

   | Setting | Value | Reason |
   |---|---|---|
   | `evaluation.world_count` | 1 | Only world 0 is scored, and one invalid world anywhere rejects the episode (`KIT/tools/pack_submission.py:266-273`) |
   | `evaluation.reset_mode` | `explicit` | Every attempt starts from frame 0 of the reference (`FC/docs/configuration.md`) |
   | `evaluation.start_frame` | 0 | The roster starts at frame 0 |
   | `logging.mode` | `disabled` | The evaluator's root config already disables logging; set it explicitly as well (`KIT/flash_chord/src/flash_chord/configs/evaluate_policy.yaml:8-9`). |
   | `evaluation.source_root` | Set when the checkpoint's stored path is not valid in this environment | Path remap, no data change |

   The kit's `examples/run_policy_evaluation.py` applies the same settings. Either
   entry point gives the same file.
2. **Inspect.** Open the Parquet and check:

   | Check | Expected |
   |---|---|
   | Metadata `flash_chord.step_count` | The episode's video frame count |
   | Metadata `flash_chord.control_fps` | 20.0 |
   | Metadata `flash_chord.episode_index` | The episode being rolled out |
   | `object_body_name` values in body order | The official slot order for the episode |
   | `environment_valid` | True on every row |
   | Metadata hashes `checkpoint_sha256`, `eval_code_sha256`, `eval_script_sha256` | Present; checkpoint hash equal to the file's |
   | Metadata `eval_code_sha256` | Equals the evaluator digest of the LF sources at `aee95f25`; matches any published organiser allowlist. The Windows CRLF reference is not the hash source (`FC/src/flash_chord/evaluation/harness.py:123-135`). |
   | Positions | Finite, and within a few metres of the origin |

   The evaluator takes the episode index from the first `episode_<digits>` in the
   reference path (`FC/src/flash_chord/evaluation/harness.py:154-156`). The
   retargeting slice's output path therefore contains no `episode_<digits>` text
   other than the episode's own index. The same index may appear more than once.
3. **Record.** Copy the evaluator's own metrics into the run record, labelled as
   tracking against the project's reconstructed reference. They are not
   leaderboard scores: the leaderboard compares with a confidential reference.
   One of them also differs in definition. The evaluator's `mppe_cm` takes the
   worst object in each frame, and the leaderboard's MPPE averages over objects
   (`FC/src/flash_chord/evaluation/metrics.py:261-278`,
   `KIT/v2dlb/object_pose.py:227-253`). The evaluator's
   `mean_per_point_error_cm` is the one that matches.

4. **Self-score.** There is no local score on evaluation episodes. To choose
   between an episode's candidate rollouts, compare each with the same
   yardstick: the object trajectory in the episode's pinned tracked scene bundle,
   which is the project's best estimate of what happened.

   | Item | Value |
   |---|---|
   | Achieved | The rollout's achieved pose columns, world 0 |
   | Reference | `world_T_object` from the pinned [scene_assembly](../scene_assembly/README.md) bundle built in the normal mode, converted to scalar-last quaternions here |
   | Code | The kit's `episode_metric` with `align_initial_pose=True` (`KIT/v2dlb/object_pose.py:296-315`) |
   | Point set | The kit's 500 sampled vertices of the public scan; for `wooden_spoon`, 500 vertices of the mesh the scene used, sampled the same way |
   | Extra | `left_working_area`: true when any object ends outside the table's inlier extent or below the table |

   A resting rollout is self-scored against the tracked bundle, not against its
   own resting reference. Its self-score is therefore how far the reconstruction
   says the objects moved. With no pinned tracked bundle there is no self-score,
   and the episode's resting rollout is the only candidate. The rule that uses
   these numbers is in [entry tiers](../../../docs/design/entry-tiers.md).

   The self-score is not a leaderboard estimate. It inherits every error in the
   reconstruction.

## Outputs

`runs/rollout/evaluation/episode_XXXXXX/<run-id>/`

| File | Content |
|---|---|
| `episode_XXXXXX.parquet` | One row per step, world and body. Achieved and reference pose, metres, scalar-last quaternions. Schema in `FC/src/flash_chord/evaluation/rollout.py:59-105`. |
| `episode_XXXXXX.json` | The evaluator's metrics report |
| `hydra/` | The evaluator's resolved configuration |
| `inspect.json` | One finding per check, with a blocking flag |
| `run.json` | Checkpoint path and hash, evaluator hashes, status, timing, device memory samples |

Pose rows are written after each control step, so row 0 is the state after the
first step from the reset and not the reset state itself
(`FC/src/flash_chord/evaluation/rollout.py:392-402`).

## Configuration and authorization

`runtime.json` holds `rollout_authorized`, `inspect_authorized` and
`self_score_authorized`, all false.
The request names one episode and one checkpoint.

## Dependencies

The policy environment described in [policy_training](../policy_training/README.md).
Nothing further.

## Resources

One world. Expected to be far lighter than training. Unmeasured.

## Failure handling

A world that records a non-finite pose is flagged by the evaluator and rejected by
the packer. With one world, that fails the episode. The run is marked
`invalid_world` and the episode falls back to the next checkpoint in the ladder:

| Order | Checkpoint |
|---|---|
| 1 | The selected trained checkpoint |
| 2 | An earlier saved checkpoint of the same training run |
| 3 | The episode's resting rollout: the policy trained on the resting scene ([entry tiers](../../../docs/design/entry-tiers.md)) |

The resting rollout is produced first, for every episode, so rung 3 exists
before any tracked policy is trained.

Every scored episode needs one valid rollout, because a submission with a missing
episode is rejected whole.

## Separation of public labels

Rollouts for submission use evaluation episodes and read no label. Rollouts on
public development episodes are allowed for diagnosis and carry
`public_label_diagnostic` when their reference came from public poses.

## Disabled achieved-behaviour comparison

`compare-achieved` consumes the immutable `policy_training.comparison` plan and
explicit completed self-score runs. It never launches composition, training,
inspection, simulation or scoring. `compare_achieved_authorized` is false;
named tuning-episode authorization and canonical Z: output storage are also
required. Existing command defaults remain unchanged.

The training owner freezes the baseline/feasibility/conditional-extension roster,
reference pins, recipes, seeds, budgets and final scheduled checkpoint rule in
`protocol.json` before any achieved rollout is produced. The comparison checks
that receipt, source hashes, scene and retargeting lineage, checkpoint counters,
zero-update baselines and rollout creation time. Episodes 39/42 only are admitted;
41 and consumed 43/44 cannot enter. No pre-existing-policy checkpoint has been
provided; its absence remains recorded by the plan rather than fabricated.

The request shape is `{"protocol_run": ".../run.json", "outcomes": [...]}`.
Supply exactly one row for every frozen `attempt_id`. Completed rows name
`self_score_run: {"path": ".../run.json", "sha256": "..."}`. Unscored rows
name `failure_reason` and optionally a pinned failed/interrupted `failure_run`.
An unassessed extension is retained in the planned denominator. An assessed
extension is refused unless all initial-stage attempts completed. Failure
evidence must belong to the specified recipe, episode and seed. The handler
does not retry, install, select checkpoints or promote a candidate.

`self-score` additionally produces per-object pinned-kit AUC/SP-SR/MP-SR/MPPE,
filled-reference counts and frame MPPE. RPE remains an episode/pair quantity;
the per-object field is explicitly unavailable. Initial alignment is applied
once to the complete attempt before frame attribution. A runtime worst-object
`mppe_cm` is never substituted for the pinned kit's mean-object definition.
All scores remain agreement with reconstructed reference poses and sampled
reconstruction meshes. They are neither public-label diagnostics nor independent
task-success measurements. Old self-score artifacts without these fields require
an explicitly authorized new self-score; the handler does not silently migrate
or recompute them.

`comparison.json` preserves per-attempt/object values, complete object-frame
denominators (including zero achieved coverage for failed/unassessed attempts),
conditional mean metric counts, matched objective pairs, failures and worst
completed attempts. These conditional means do not assign invented failure
scores. Whole-episode uncertainty remains weak with only two tuning episodes.

For future producer-attributed event support, the consumer accepts frozen
attempt `events` containing `kind`, `start_frame`, `stop_frame_exclusive` and
an immutable completed video-only `producer` run pin, plus optional `object_names`.
The current fixed plan supplies no event intervals;
contact loss, drop and recovery attribution is therefore unavailable. Event
MPPE uses the original full-attempt alignment, without realigning each event.
Extending the frozen producer to admit event source pins is required before
claiming measured event attribution. `left_working_area` remains an explicit
self-score finding and is not relabelled as a measured drop.

The report exports only immutable achieved-pose/checkpoint references for a
separately authorized `public_diagnostics` comparison. This owner reads no public
pose labels and does not mix diagnostic rankings into reconstruction scores.
Source implementation is unrun: no tests, imports, scoring or simulator commands
were executed for this buildout.

## Acceptance evidence

Once runs are authorized: for one episode, `inspect` passes every check, and the
kit's packer accepts the file.

## Open questions

| Question | Who can answer |
|---|---|
| How do the organisers' own baselines keep rollout frames equal to dataset frames, given that the released floating-hand recipe plays the reference at half speed? Is a half-speed rollout resampled to dataset frames acceptable? | Organisers. This is the first question for Discord and the 7 October webinar. |
| Is the state after the first control step the intended pose for roster frame 0? | Organisers |
| Does the confidential reference hold a pose for frames where motion capture lost the object? | Organisers |
