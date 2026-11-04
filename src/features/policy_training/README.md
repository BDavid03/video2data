# Policy training

General tools to train one FlashCHORD policy per episode. Compose, train, select and comparison have never been run, and their flags in
`runtime.json` are false. They need the retargeting, scene_assembly and policy_rollout producers, which are in Git
history before the cleanup commit (parent `ef884fd`). Only `budget.json` is in current use, by the D0 loop in
[dynamic_feasibility](../dynamic_feasibility/README.md). The policies in the entry came from [pose_insurance](../pose_insurance/README.md). Episode 38's moving policy from [moving_policy](../moving_policy/README.md) is in the moving v5 file.

```bash
python -m src.features.policy_training <compose|train|select|comparison> --request <file>
```

Run these in the FlashCHORD environment (`scripts/linux/install-flashchord.sh`). `FC/` below means
`upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/`. Source reading:
[flashchord-runtime.md](../../../data/reports/contract-evidence/2026-09-30/flashchord-runtime.md). Protocol:
[research-protocol.md](research-protocol.md).

## Runtime facts, from the pinned source

| Property | Value | Source |
| --- | --- | --- |
| Physics | Newton 1.2.0 with MuJoCo Warp | `FC/pyproject.toml:6-26` |
| Learning | FlashSAC or PPO, JAX 0.10.2, CUDA 13 | `pyproject.toml:31-39` |
| Python, system | 3.11 or 3.12; Linux, NVIDIA driver 580 or newer | `pyproject.toml:5`, `README.md:11-13` |
| Rates | 20 control steps, 100 physics steps per second | `configs/sim/rl.yaml:2-3` |
| Action, floating hands | 56 residuals on reference wrist pose and finger joints | `runtime/actions/hand_pose.py:159-186` |
| Observation | Hand state, object poses, current reference, contacts; no look-ahead | `envs/observation.py:290-351,496-500` |
| Assistance | Virtual force reduced to zero in stages | `configs/curriculum/tracking_250m_mixed_reset.yaml` |
| Logging default | Online to the organisers' W&B, with checkpoint upload | `configs/logging/wandb.yaml:1-8` |

Upstream names 48 GiB of GPU memory for full training (`README.md:14`), though no code checks it. The runtime source is
Apache-2.0; the Sharpa asset folders carry no licence file.

## Operations

`compose` prints the resolved config with Hydra `--cfg job --resolve` and records every override with its reason. It needs a completed,
non-blocking retargeting inspect run. `train` takes a completed compose run, compares the resolved job, and launches upstream under `GpuLease`.
Checkpoints survive failure, and there is no automatic resume. `select` picks the latest checkpoint with a matching non-blocking rollout inspection.
`comparison` only persists a matched plan from [comparison.json](comparison.json).

Overrides from `experiment=sharpa_flash_sac`:

| Override | Value | Reason |
| --- | --- | --- |
| `task.parquet` | Episode's sequence | Required |
| `task.motion_speed` | 1.0 | One control step per video frame; the recipe uses 0.5 |
| `logging.mode` | `disabled` | The default uploads checkpoints |
| `output_dir`, `hydra.run.dir` | Run directory | Defaults are relative to the working directory |
| `scene.world_count`, `training.replay.capacity`, `training.total_environment_steps` | From `budget.json` | Memory and time |
| `scene.support` | `false` without a Cylinder or Cube | Empty stages fail setup |
| `training.replay.observation_storage_dtype` | `float16` | Memory |
| `reset.seed` | Fixed per run | Recorded |

Set `FLASH_CHORD_CACHE_DIR` under `data/.cache/`, because the runtime otherwise writes into its source tree (`assets/mesh.py:32-34`).

Output in `data/runs/policy/<split>/episode_XXXXXX/<run-id>/`: `policy_<steps>.safetensors`, `state_<steps>.safetensors`,
`resolved-config.yaml`, `resolved-config.json`, `overrides.json`, `selection.json`, `run.json`. The checkpoint stores the reference by path with
no hash, so the run record stores the hash. `checkpoints.py` admits checkpoints from their header and saved config; it does not
decode tensors or prove the evaluator can load them.

Policies trained on public labels carry `public_label_diagnostic` and cannot be selected for submission.

## Scaling the recipe (derived, not measured)

Rules the runtime enforces: the step budget must be divisible by world count, and replay capacity must be at least the world count. Curriculum stages are
absolute steps. The released curriculum reaches zero assistance at about 123M of 250M steps, and the evaluator turns help off, so a shorter run on that
curriculum would never have moved an object unaided. `compose` refuses a partial budget.

| Setting | Recipe at 4,096 worlds, 250M steps | Rule for W worlds, N steps |
| --- | --- | --- |
| `curriculum.thresholds` | 11 absolute steps | Fractions 0.0786, 0.1376, 0.1966, 0.2556, 0.3146, 0.3736, 0.4325, 0.4915, 0.5505, 0.8000 of N, and N; round down to multiples of W |
| `training.updates_per_collection` | 8 | `8*W/4096` |
| `training.replay.minimum_size` | 409,600 | `100*W`, at most N |
| `training.replay.capacity` | 4,194,304 | From the memory table |
| `training.checkpoint.save_interval_environment_steps` | 10,000,384 | About N/25, a multiple of W |
| `N` | 250,003,456 | A multiple of W, at least `minimum_size` |

Replay bytes per transition are `2*obs*s + 4*(56 + 11 + 2*ctx) + 2`, with s = 4 for float32 and 2 for float16, obs = `286 + 218*B`, and ctx = `4 + 6*B`
for B bodies. This is arithmetic from the source. The simulator's share is unknown.

| Episode | Storage | Bytes | 4,194,304 | 2,097,152 | 1,048,576 |
| --- | --- | ---: | ---: | ---: | ---: |
| One object | float32 | 4,382 | 17.1 GiB | 8.6 GiB | 4.3 GiB |
| One object | float16 | 2,366 | 9.2 GiB | 4.6 GiB | 2.3 GiB |
| Two objects | float32 | 6,174 | 24.1 GiB | 12.1 GiB | 6.0 GiB |
| Two objects | float16 | 3,286 | 12.8 GiB | 6.4 GiB | 3.2 GiB |

A resumed run must keep its world count, and its replay starts empty unless saved (`save_replay`).

`budget.json` holds `zero_update` (one 64-world collection, capacity 2,048, minimum 64, `updates_per_collection=1.0`) and `scaled_template` (1,024
worlds, 10,240,000 steps). Neither is a measured recommendation. Reading the source, a zero-update checkpoint replays its reference through physics,
because a new actor outputs exactly zero (`training/flash_sac/network.py:72-88,331-363`). The design uses this for resting and replay policies, and this
reading is untested here.

Two task properties to watch: the idle hand is trained too, and a wrist error above 15 cm ends the attempt
(`configs/termination/tracking.yaml:2-5`). The released objective trusts contacts, while upstream recommends force closure for egocentric data.

`hybrid.py` admits declared assistance-free rollout manifests for the 56D action contract, and defines a tiny MLP. It fits nothing, makes no FlashSAC
checkpoints, and no qualifying data exists.

## Open questions

- How does quality change with fewer worlds?
- Contact term or force closure for reconstructed demonstrations?
- Does open-loop replay of retargeted hands move the objects at all?
- Is a rollout from a policy with no optimiser updates an acceptable submission? The design discloses it either way; this is for the organisers.
