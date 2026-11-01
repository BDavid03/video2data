# Policy training

Status: implemented, never run

The 2026-10-18 source build adds named `contact_aware`/`motion_only` recipes,
all-stage contact-weight suppression, the complete `contact_feasibility` budget
and an immutable matched-comparison planner. See
[research-protocol.md](research-protocol.md). New planner persistence and hybrid
training gates are false; existing execution gates stay false.

`hybrid.py` admits declared successful assistance-free achieved-rollout manifests
for the original 56D floating-Sharpa residual action contract and supplies an
optional tiny original PyTorch MLP in the already existing learning stack. It
does not manufacture actions from human video, truncate external action vectors,
fit a model, convert checkpoints into FlashSAC or establish evaluator compatibility.
Pinned action schemas, executed controls, before-action observations, timestamps,
achieved objects, success protocol/report and training-only normalization are
mandatory. Manifest admission checks declarations; actual payload hashing and
row-level admission remain prerequisites for a future fitting adapter. No such
corpus exists yet. No pretrained VLA/world-model stack or production dependency
is added. OpenVLA/OFT, openpi, GR00T and Cosmos remain provision-blocked.

Tiny-policy materialization additionally requires an explicit public train/tuning
`PolicyRequest`, owning runtime authorization, canonical project, named episode
authorization and admitted dedicated policy interpreter. It confines compiler,
model, temporary and bytecode writes before importing PyTorch. PyTorch must
already exist in that admitted environment; this source adds no installation
permission and no automatic environment modification. Fitting and FlashSAC export
adapters remain unimplemented until successful robot-action data exists.

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
no training time. No steps-per-second or per-world memory figure appears anywhere
in the tree. `budget.json` contains complete, unmeasured preparation entries;
it does not claim a measured capacity on this machine.

### Memory model (Derived)

Arithmetic on the source's own formulas, from a source reading on 2026-10-03.
Not a measurement.

The replay buffer is the one large allocation whose size the source states
exactly. It lives on the GPU. Bytes per transition are
`2·obs·s + 4·(56 + 11 + 2·ctx) + 2`, where `s` is 4 for float32 observations and
2 for float16 (**Official**, `FC/src/flash_chord/training/flash_sac/replay.py:184-198`).
With `B` object bodies, the observation has `286 + 218·B` numbers and the critic
context `4 + 6·B` (**Official**, `FC/src/flash_chord/envs/observation.py:290-370`,
`configs/observation/policy.yaml`, `envs/rl.py:66-74`).

| Episode | Observation storage | Bytes per transition | Capacity 4,194,304 (recipe) | Capacity 2,097,152 | Capacity 1,048,576 |
|---|---|---|---|---|---|
| One object | float32 (recipe) | 4,382 | 17.1 GiB | 8.6 GiB | 4.3 GiB |
| One object | float16 | 2,366 | 9.2 GiB | 4.6 GiB | 2.3 GiB |
| Two objects | float32 (recipe) | 6,174 | 24.1 GiB | 12.1 GiB | 6.0 GiB |
| Two objects | float16 | 3,286 | 12.8 GiB | 6.4 GiB | 3.2 GiB |

The recipe's buffer alone exceeds this GPU for every episode. With float16
storage and a quarter of the capacity, a two-object buffer is 3.2 GiB. Network
weights, optimiser state and target networks come to roughly 0.2 GiB.

What the source does not determine is the simulator's footprint. MuJoCo Warp is
sized per world by `sim.njmax` 512 and `sim.nconmax` 256
(`FC/src/flash_chord/configs/sim/rl.yaml:7-8`), and the objective keeps
`worlds × 2 × B × 512` wrench supports (`objectives/tracking.py:351-357`). Convex
decomposition has no cap on hull count; the only control is
`scene.decompose_objects=false`. The simulator's share is what the Phase 1 probe
measures.

### Scaling the recipe to a smaller budget (Design)

Several of the recipe's values are absolute step counts or are tied to the world
count only by arithmetic. Changing the world count or the step budget without
changing them alters the method, not just its size. `budget.json` therefore
holds a complete, consistent set, and `compose` refuses a partial one.

| Setting | Recipe at 4,096 worlds, 250 M steps | Rule when worlds `W` or total steps `N` change | Source |
|---|---|---|---|
| `curriculum.thresholds` | 11 absolute steps; assistance reaches zero at 122,880,000 | Keep each boundary at the same fraction of `N`: 0.0786, 0.1376, 0.1966, 0.2556, 0.3146, 0.3736, 0.4325, 0.4915, 0.5505, 0.8000, and `N` itself. Round each down to a multiple of `W`. They must stay strictly increasing. | `configs/curriculum/tracking_250m_mixed_reset.yaml`, `lifecycle/curriculum.py:95-111` |
| `training.updates_per_collection` | 8, with batch 2,048: four sampled transitions per new one | Updates per collection do not scale with `W` in the code. At 1,024 worlds the recipe's 8 gives a ratio of 16 and four times the optimiser work per step. Start from the recipe's ratio: `8·W/4096`. | `training/flash_sac/learner.py:1087-1118`, `config.py:358-366` |
| `training.replay.minimum_size` | 409,600, about 100 collections | Keep 100 collections: `100·W`. It must not exceed `N`. | `sharpa_flash_sac.yaml:47`, `config.py:322-356` |
| `training.replay.capacity` | 4,194,304, about 1,024 collections | From the memory table. Record how many collections it holds. | `config.py:339-342` |
| `training.checkpoint.save_interval_environment_steps` | 10,000,384 | About `N/25`, a multiple of `W` | `runner.py:530-533` |
| Learning-rate schedule | Cosine over all optimiser updates | Rescales by itself | `config.py:368-377` |
| `N` | 250,003,456 | A multiple of `W`, at least `minimum_size` | `config.py:326-330` |

A resumed run must use the same world count as the run that saved the state, and
its replay starts empty unless the replay was saved too
(`training/flash_sac/runner.py:465-492`, `FC/src/flash_chord/training/README.md:24-25`).
A long run split across nights therefore sets `save_replay`, and its disk cost is
the replay size in the table above per checkpoint kept.

Whether a policy trained at a quarter of the worlds and a fraction of the steps
grasps anything is the question Phase 1 and Phase 3 answer. The table only makes
sure a smaller run is the same method.

### Zero-update policies (Design)

Three facts from the source, all **Official** and none observed in a run:

| Fact | Source |
|---|---|
| The floating-hand action is a residual. A zero action makes the wrist and finger targets exactly the reference, tracked by the hands' PD controllers with gravity compensation. | `FC/src/flash_chord/runtime/actions/hand_pose.py:269-307`, `runtime/residual.py:61-100` |
| A newly initialised actor outputs a mean of exactly zero for every observation: its output head starts with zero gain and zero bias. | `training/flash_sac/network.py:72-88,331-363` |
| The evaluator acts deterministically on the mean, with assistance off, from the reference's first frame, with no settling steps. | `evaluation/harness.py:228-250,449-465` |

So a checkpoint saved before any optimiser update is a policy that replays its
reference through physics. The runtime permits such a run: a step budget of one
collection (`total_environment_steps = W`) with `minimum_size` at most `W` passes
the configuration checks, performs no update, and saves a policy file at the
final step (`config.py:322-356`, `runner.py:613-616`). This is a reading of the
source. It is not a documented feature, and Phase 1 tests it first.

The design uses zero-update policies in two places.

| Use | Reference | What the rollout is |
|---|---|---|
| Resting policy | The resting scene: constant objects, parked hands | Hands held at the parked pose by the controllers. Objects at rest. |
| Replay candidate | The tracked scene | The retargeted hand motion played open-loop against real physics. The objects move only as far as that motion carries them. |

Both take one collection at a small world count, so their cost is scene loading,
not training. Both are written by the official training script and rolled out by
the official evaluator. The submission record and the technical report name each
episode's policy kind: `resting`, `replay` or `trained`.

A zero-update run uses a small world count, a replay capacity of 2,048 (the
recipe's batch size, the smallest capacity upstream accepts), and
the unscaled curriculum, whose stage is irrelevant because the evaluator turns
assistance off. `budget.json` has a named entry for it.

The replay candidate matters for two reasons. It gives the tracked entry a first
rollout per episode without a training run, which removes R01 from the path to a
first tracked score. And it is the baseline a trained policy has to beat: the
learned residual is worth its GPU hours only where it improves on open-loop
replay.

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

The CLI is `python -m track3.features.policy_training <compose|train|select>
--request <file>`, inside the dedicated WSL policy Python 3.11 environment.
Compose requires a completed, nonblocking retargeting inspect run and hashes
the inspected asset closure again. It runs the unchanged training entry point
with Hydra's `--cfg job --resolve`, checks the printed configuration, and writes
every override with its reason. These Hydra framework switches are not spelled
out in the vendored FlashCHORD source; its `@hydra.main` entry point and locked
Hydra version are confirmed, but the framework's implementation is not present
in the read-only source tree. No command has been exercised.

`zero_update` is one 64-world collection with replay capacity 2,048,
minimum_size 64 and the unscaled curriculum. Its `updates_per_collection=1.0` gives the configuration a
positive optimizer schedule horizon; scaling that value down to 0.125 would
make the one-collection horizon zero and fail `OptimizationConfig.schedule_steps`.
The released three-step replay still has no sample after one collection. Train
also refuses a supposedly zero-update checkpoint whose saved
`global_update_step` is nonzero. `scaled_template` is a complete 1,024-world,
10,240,000-step arithmetic template, with the documented curriculum fractions,
100-collection warmup, 1,024-collection replay, two updates per collection and
400-collection checkpoint interval. Neither budget is a measured recommendation.

Train takes a completed compose run. It preserves all overrides except the
new run's output directories, captures a fresh resolved configuration, and
compares the complete resolved job with composition before launching upstream
while holding `GpuLease`. Both operations retain `resolved-config.yaml` and a
plain, fully resolved `resolved-config.json`. Only the top-level `output_dir` may
differ between the job configurations; Hydra's own run directory is omitted
from `--cfg job`. Checkpoints are retained on failure;
automatic resume is not implemented. Set `save_replay` in a complete budget if
a later separately planned resumed run needs replay persistence. Resting and
replay policies normally use zero_update; a resting policy can also use a
scaled budget for the documented holding-policy fallback. Trained policies use
tracked scenes; replay cannot use a scaled budget.

Select takes `training_run` and explicit `rollout_inspections`, and chooses the
latest checkpoint with a matching nonblocking rollout inspection. It does not
launch rollouts or assume that the latest checkpoint is valid. Inspection files
carry `training_run`, `checkpoint` path/hash entries, `blocking` and optional
`tracking_metrics`. Selection records those metrics as reference tracking only.

Checkpoint admission is owned by the public, standard-library
`checkpoints.py` contract. It reads a bounded safetensors header and requires the
pinned format/version, FlashSAC algorithm and actor-policy kind. The saved
configuration must match the owning run's complete resolved job. The only
upstream-added field is `reference`: its path, full frame count, 20 Hz playback
and empty rigid-object articulation contract must match the episode. Project
path aliases are normalised for the reference path, not used to excuse changes
to the recipe.

Filenames and environment-step counters must agree and fit whole collections
within the named budget. Zero-update runs must retain zero optimizer updates;
the final `trained` checkpoint must contain updates. Optimizer counters cannot
go backwards across saved checkpoints. The inventory retains configuration and
policy/critic-schema metadata hashes. These checks do not decode tensors or
prove that the evaluator can load a checkpoint; the official evaluator retains
its runtime schema and tensor checks.

Rollout and selection reapply the same admission contract against hashed training
artifacts and compare the result with the checkpoint inventory. Older records
without the resolved JSON snapshot and admission fields cannot silently satisfy
this boundary. No compose, training, selection, checkpoint inspection, validation
command or test was executed as part of this source-only change.

`runtime.json` holds `compose_authorized`, `train_authorized` and
`select_authorized`, all false. The request names one episode and one retargeting
run.

## Dependencies

Policy operations require the checkout at `/mnt/z/hugging-face/video2data`.
The shell selector and training subprocess route CUDA, extension, Inductor and
Triton caches into that project's `.cache/` tree; direct invocation cannot inherit
machine-wide compiler cache locations. These setup changes remain unexecuted.

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

Unmeasured. The measurements to take, in this order:

1. A zero-update run on the upstream example sequence at 64 worlds: does it
   save a policy file, and does the evaluator load it and write a trajectory?
2. At 256, 512 and 1,024 worlds, with float16 storage and the scaled settings
   above: sampled device memory before the replay fills, steps per second, and
   time to a fixed step count. Subtracting the replay size in the memory model
   gives the simulator's share per world.
3. With `scene.decompose_objects` on and off for one concave object: device
   memory and scene load time.

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

The new disabled `comparison` CLI route consumes [comparison.json](comparison.json)
and persists only the matched immutable attempt plan. References are explicit
resting/full scene inspections; `tracked` is the comparison role, not a new
scene mode. Placeholder paths require admitted completed producers. No plan,
composition, training or simulation was executed for this source buildout.

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
| Does a one-collection run save a loadable policy, as the source reading says? | A run |
| Does open-loop replay of the retargeted hands move the objects at all? | A run |
| Is a rollout from a policy with no optimiser updates an acceptable submission? | Organisers. The design discloses it either way. |
