# Policy rollout

Status: design only. No code exists for this slice. Nothing was run while writing
this document. The evaluator behaviour described here was read from source, not
observed.

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

## Operations

Both disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `rollout` | The evaluator's trajectory Parquet and metrics JSON for one episode | Policy environment, Linux | Yes |
| `inspect` | `inspect.json`: findings from opening the Parquet | Policy environment or submission environment | No |

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

`runtime.json` holds `rollout_authorized` and `inspect_authorized`, both false.
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
| 3 | A checkpoint from a run trained on a simplified reference for the same episode |

Every scored episode needs one valid rollout, because a submission with a missing
episode is rejected whole.

## Separation of public labels

Rollouts for submission use evaluation episodes and read no label. Rollouts on
public development episodes are allowed for diagnosis and carry
`public_label_diagnostic` when their reference came from public poses.

## Acceptance evidence

Once runs are authorized: for one episode, `inspect` passes every check, and the
kit's packer accepts the file.

## Open questions

| Question | Who can answer |
|---|---|
| How do the organisers' own baselines keep rollout frames equal to dataset frames, given that the released floating-hand recipe plays the reference at half speed? Is a half-speed rollout resampled to dataset frames acceptable? | Organisers. This is the first question for Discord and the 7 October webinar. |
| Is the state after the first control step the intended pose for roster frame 0? | Organisers |
| Does the confidential reference hold a pose for frames where motion capture lost the object? | Organisers |
