# Policy training

Status: design only. No code exists for this slice, and the policy environment it
needs does not exist. Nothing was run while writing this document.

This slice trains one policy per episode to make the simulated robot hands move
the simulated objects the way the demonstration did. It uses the organisers'
FlashCHORD runtime unchanged and owns only the configuration, the bookkeeping and
the checks around it. `FC/` means
`upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/`. The
source reading is
[flashchord-runtime.md](../../../reports/contract-evidence/2026-09-30/flashchord-runtime.md).

## Role in scoring

The scored trajectory is what the objects do under this policy. If the policy
fails to grasp, the object stays where it started, and the episode is scored on a
static object.

## What the runtime is

All **Official**, from the pinned source.

| Property | Value | Source |
|---|---|---|
| Physics | Newton 1.2.0 with the MuJoCo Warp solver | `FC/pyproject.toml:6-26`, `src/flash_chord/runtime/sim.py:1-9` |
| Learning | FlashSAC or PPO, in JAX 0.10.2 with CUDA 13 | `pyproject.toml:31-39` |
| Python | 3.11 or 3.12 | `pyproject.toml:5` |
| System | Linux, NVIDIA driver 580 or newer | `README.md:11-13` |
| GPU JAX on Windows | Not available: the locked wheels are Linux only | `uv.lock:518-535` |
| Control and physics rate | 20 control steps and 100 physics steps per second | `src/flash_chord/configs/sim/rl.yaml:2-3` |
| Action, floating hands | 56 numbers: residuals added to the reference wrist pose and finger joints | `src/flash_chord/runtime/actions/hand_pose.py:159-186` |
| Observation | The hands' own state, each object's pose, the reference targets for the current frame, and contacts. No look-ahead. | `src/flash_chord/envs/observation.py:290-351,496-500` |
| Assistance | A virtual force holds the object on its reference early in training and is reduced to zero in stages | `src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml` |
| Checkpoint | Stores the full resolved configuration and the absolute path of the reference | `scripts/train_flash_sac.py:80-87` |
| Logging default | Online, to the organisers' Weights & Biases entity, with checkpoint upload | `src/flash_chord/configs/logging/wandb.yaml:1-8` |

The runtime's documentation names 48 GiB of GPU memory with 46,000 MiB free for
full training (`README.md:14`). No code checks it. Windows, WSL and the RTX 5080's
architecture are not mentioned anywhere in the tree.

## Operations

All disabled by default.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `compose` | The resolved training configuration for one episode, printed by the runtime without training | Policy environment | No |
| `train` | Checkpoints for one episode | Policy environment | Yes |
| `select` | The checkpoint chosen for rollout, with the reason | Policy environment | No |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| The processed sequence, object assets and support file | [retargeting](../retargeting/README.md) | By the runtime, through `task.parquet` |
| `inspect.json` | retargeting | Refuse when it holds a blocking finding |
| `budget.json` | This slice | World count, step budget and replay settings chosen from measurements |

## Method

1. **Start from the released recipe.** `experiment=sharpa_flash_sac`. The
   runtime's guidance is to override recipe values only deliberately
   (`FC/src/flash_chord/configs/README.md:15-17`). Every override below is
   recorded in the run record with its reason.

   | Override | Value | Reason |
   |---|---|---|
   | `task.parquet` | The episode's processed sequence | Required |
   | `task.motion_speed` | 1.0 | One control step per video frame. See [policy_rollout](../policy_rollout/README.md). The recipe uses 0.5. |
   | `logging.mode` | `disabled` | Nothing leaves the machine. The default uploads checkpoints. |
   | `output_dir`, `hydra.run.dir` | Inside the run directory | The defaults are relative to the working directory |
   | `scene.world_count` | From `budget.json` | Memory |
   | `scene.support` | `false` when retargeting reports no Cylinder or Cube; enable only with a usable stage | Empty stages fail scene setup. The ground plane stays on; the checkpoint stores this setting and the evaluator inherits it. |
   | `training.replay.capacity` | From `budget.json` | Memory. The buffer lives on the GPU. |
   | `training.replay.observation_storage_dtype` | `float16` | Memory. Two released recipes already use it. |
   | `training.total_environment_steps` | From `budget.json` | Time. Must be divisible by the world count. |
   | `reset.seed` | Fixed per run | Recorded |

   The environment variable `FLASH_CHORD_CACHE_DIR` is set to a folder under
   `.cache/`, because the runtime otherwise writes its convex-decomposition cache
   into the source checkout (`FC/src/flash_chord/assets/mesh.py:32-34`).
2. **Compose before training.** Print the resolved configuration with the
   runtime's own option and store it. Check the reference's frame count against
   the clip's, and apply the support requirement in retargeting `inspect.json`.
   A stage without Cylinder or Cube shapes requires `scene.support=false`.
3. **Train.** One episode at a time, holding the shared GPU lock.
4. **Select.** Roll out the latest checkpoint. If its rollout is valid, select
   it. The evaluator's own tracking metrics, against the project's reference, are
   recorded for comparison between checkpoints of the same episode.

### Rules the runtime enforces on a changed budget

| Rule | Source |
|---|---|
| The step budget must be divisible by the world count | `FC/src/flash_chord/training/flash_sac/config.py:326-330` |
| Replay capacity must be at least the world count | `config.py:339-342` |
| The curriculum's stages are placed by environment steps. With a smaller budget the stage boundaries must be rescaled, or training ends while assistance is still on. | `training/flash_sac/runner.py:437-446` |

The third rule matters most. The released curriculum lowers assistance in eight
steps and reaches zero at about 123 million of its 250 million steps
(`FC/src/flash_chord/configs/curriculum/tracking_250m_mixed_reset.yaml`). A run
stopped at 50 million steps with that curriculum would still have assistance at a
quarter of full strength. Its policy would never have moved an object without
help, and the evaluator turns help off
(`FC/src/flash_chord/evaluation/harness.py:228-237`). A reduced budget therefore
needs its own curriculum file with boundaries scaled to the budget.

### What the source does not say

It gives no guidance on training with fewer worlds, no recipe for less memory and
no training time. `budget.json` is filled from measurements on this machine and
is empty until then.

### Two properties of the task that need attention

**The idle hand is trained too.** There is no single-hand mode. A hand that does
nothing in the demonstration is still simulated and tracked, and during training
a wrist error above 15 cm ends the attempt
(`FC/src/flash_chord/configs/termination/tracking.yaml:2-5`). The parked pose from
scene assembly has to be easy to hold.

**The released objective trusts contacts.** The floating-hand objective gives
weight 10 to a contact-based term and leaves the alternative, force closure,
disabled (`FC/src/flash_chord/configs/objective/tracking.yaml:21-27,57-61`).
Upstream's own README recommends force closure for egocentric reconstructions,
because reconstructed contacts are noisy. Which to use is left to measurement;
the released setting is the starting point.

## Outputs

`runs/policy/<split>/episode_XXXXXX/<run-id>/`

| File | Content |
|---|---|
| `policy_<steps>.safetensors` | Actor weights with embedded configuration. The file the evaluator loads. |
| `state_<steps>.safetensors` | Learner state for resuming |
| `resolved-config.yaml` | From `compose` |
| `overrides.json` | Each override and its reason |
| `selection.json` | From `select` |
| `run.json` | Inputs with hashes, FlashCHORD commit, seed, status, timing, sampled device memory |

The checkpoint records the reference by absolute path and stores no hash of its
content. The run record stores the hash, so a later change to the reference is
detectable.

## Configuration and authorization

`runtime.json` holds `compose_authorized`, `train_authorized` and
`select_authorized`, all false. The request names one episode and one retargeting
run.

## Dependencies

The policy environment. Proposed, not created. It needs the user's confirmation.

| Item | Detail |
|---|---|
| System | A new Ubuntu 24.04 WSL distribution with its disk under `.runtime/wsl/` |
| Source | An LF checkout of `aee95f25` inside the project-owned WSL disk, with CRLF conversion disabled. The Windows checkout under `upstream/` is a read-only reference with CRLF evaluation modules, unsuitable for installation because evaluation hashes raw bytes. |
| Packages | The runtime's `uv.lock`, installed with `uv` into a virtual environment inside that distribution. No Docker: the runtime's container script writes under the home directory and changes display access (`FC/docker/run.sh:131,140`). |
| Robot assets | The Sharpa hand meshes are Git LFS objects and are not fetched yet. The local checkout holds 346 pointer files in their place (evidence report, observation 2). |
| Licence | Runtime source Apache-2.0. The Sharpa asset folders carry no licence file. |

Future installation records must include the source commit and byte hashes of
`evaluation/__init__.py`, `harness.py`, `metrics.py` and `rollout.py`, using the
evaluator's own digest scheme (`FC/src/flash_chord/evaluation/harness.py:123-135`).
No sources are converted, installed or hashed under the current authorization.

## Resources

Unmeasured. The measurements to take first, at 256, 512 and 1,024 worlds: sampled
device memory, steps per second, and time to a fixed step count.

## Failure handling

A run that stops early keeps its checkpoints and is marked `interrupted`. The
learner state file allows resuming. A run whose rollout has an invalid world is
marked and the next checkpoint down the ladder in
[policy_rollout](../policy_rollout/README.md) is tried.

## Separation of public labels

Policies for submission train on evaluation sequences. A policy trained on a
public episode's labelled poses tests control separately from perception. It
carries `public_label_diagnostic`, lives under `runs/policy/public/`, and cannot
be selected for a submission.

## Acceptance evidence

Once runs are authorized: on the upstream example sequence, a short run at a
reduced world count completes, saves a checkpoint, and the evaluator loads it and
writes a trajectory file. On one Track 3 episode, a trained policy's rollout
shows the object leaving the table and arriving near its reference end pose.

## Open questions

| Question | Who can answer |
|---|---|
| Do Warp, MuJoCo Warp and JAX at the pinned versions run on the RTX 5080 under WSL 2? | A run |
| How many worlds fit in 16 GB, and how does quality change with fewer? | A run; organisers may have advice |
| Does a policy track a full-speed reference at 20 control steps per second? | A run |
| Contact term or force closure for reconstructed demonstrations? | A run |
