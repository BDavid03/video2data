# Policy training

Not used by the submission: general tools to train one FlashCHORD policy per episode. Only `budget.json` is
live (read by the D0 run in step 4's groundwork, [dynamic_feasibility](../dynamic_feasibility/README.md)).

The submitted policies come from elsewhere. Static poses (step 3): [pose_insurance](../pose_insurance/README.md).
Episode 38's moving policy (step 4): [moving_policy](../moving_policy/README.md).

## Status

The four operations, `compose`, `train`, `select` and `comparison`, have never run. Their flags in
`runtime.json` are off. They need the retargeting, scene_assembly and policy_rollout slices, which were deleted
on 4 Nov 2026 and are in Git history before the cleanup commit (parent `ef884fd`).

```bash
python -m src.features.policy_training <compose|train|select|comparison> --request <file>
```

Run in the FlashCHORD environment (`scripts/linux/install-flashchord.sh`). Protocol:
[research-protocol.md](research-protocol.md). Source notes:
[flashchord-runtime.md](../../../data/reports/contract-evidence/2026-09-30/flashchord-runtime.md).

- `compose` resolves the Hydra config and records each override with its reason.
- `train` launches upstream training under the GPU lease. Checkpoints survive failure. No automatic resume.
- `select` picks the latest checkpoint with a matching rollout check.
- `comparison` saves a matched plan from [comparison.json](comparison.json).
- `checkpoints.py` admits a checkpoint from its header and saved config. It does not load tensors.
- `hybrid.py` defines a tiny MLP for assistance-free rollouts. It fits nothing.

Outputs would go to `data/runs/policy/<split>/episode_XXXXXX/<run-id>/`. Policies trained on public labels
are tagged `public_label_diagnostic` and cannot be submitted.

## FlashCHORD facts (pinned source)

`FC/` = `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/`.

| Property | Value | Source |
| --- | --- | --- |
| Physics | Newton 1.2.0 with MuJoCo Warp | `FC/pyproject.toml:6-26` |
| Learning | FlashSAC or PPO, JAX 0.10.2, CUDA 13 | `pyproject.toml:31-39` |
| Python, system | 3.11 or 3.12; Linux, NVIDIA driver 580+ | `pyproject.toml:5`, `README.md:11-13` |
| Rates | 20 control steps, 100 physics steps per second | `configs/sim/rl.yaml:2-3` |
| Action | 56 residuals on the reference wrist pose and finger joints | `runtime/actions/hand_pose.py:159-186` |
| Observation | Hand state, object poses, current reference, contacts; no look-ahead | `envs/observation.py:290-351,496-500` |
| Assistance | Virtual force lowered to zero in stages | `configs/curriculum/tracking_250m_mixed_reset.yaml` |
| Logging default | Online to the organisers' W&B, with checkpoint upload | `configs/logging/wandb.yaml:1-8` |

- Upstream asks for 48 GiB of GPU memory for full training (`README.md:14`). No code checks it.
- Set `FLASH_CHORD_CACHE_DIR` under `data/.cache/`. Otherwise the runtime writes into its source tree (`assets/mesh.py:32-34`).
- A wrist error above 15 cm ends the attempt (`configs/termination/tracking.yaml:2-5`). The idle hand is trained too.
- A new actor outputs exactly zero, so a zero-update policy replays its reference through physics
  (`training/flash_sac/network.py:72-88,331-363`). Untested here.
- Runtime source is Apache-2.0. The Sharpa asset folders have no licence file.

Overrides we set on `experiment=sharpa_flash_sac`:

| Override | Value | Why |
| --- | --- | --- |
| `task.parquet` | The episode's sequence | Required |
| `task.motion_speed` | 1.0 | One control step per video frame (recipe uses 0.5) |
| `logging.mode` | `disabled` | The default uploads checkpoints |
| `output_dir`, `hydra.run.dir` | Run directory | Defaults are relative to the working directory |
| `scene.world_count`, `training.replay.capacity`, `training.total_environment_steps` | From `budget.json` | Memory and time |
| `scene.support` | `false` without a Cylinder or Cube | Empty stages fail setup |
| `training.replay.observation_storage_dtype` | `float16` | Memory |
| `reset.seed` | Fixed per run | Recorded |

## Scaling the recipe (arithmetic, not measured)

The step budget must divide by the world count. Replay capacity must be at least the world count. The released
curriculum reaches zero assistance at about 123M of 250M steps, so a shorter run on that curriculum never moves
an object unaided. `compose` refuses a partial budget.

| Setting | Recipe: 4,096 worlds, 250M steps | Rule for W worlds, N steps |
| --- | --- | --- |
| `curriculum.thresholds` | 11 absolute steps | Fractions 0.0786, 0.1376, 0.1966, 0.2556, 0.3146, 0.3736, 0.4325, 0.4915, 0.5505, 0.8000 of N, and N; round down to multiples of W |
| `training.updates_per_collection` | 8 | `8*W/4096` |
| `training.replay.minimum_size` | 409,600 | `100*W`, at most N |
| `training.replay.capacity` | 4,194,304 | From the memory table |
| `training.checkpoint.save_interval_environment_steps` | 10,000,384 | About N/25, a multiple of W |
| `N` | 250,003,456 | A multiple of W, at least `minimum_size` |

Replay bytes per transition = `2*obs*s + 4*(56 + 11 + 2*ctx) + 2`, with s = 4 (float32) or 2 (float16),
obs = `286 + 218*B`, ctx = `4 + 6*B`, B = number of bodies. The simulator's own memory is not included.

| Episode | Storage | Bytes | 4,194,304 | 2,097,152 | 1,048,576 |
| --- | --- | ---: | ---: | ---: | ---: |
| One object | float32 | 4,382 | 17.1 GiB | 8.6 GiB | 4.3 GiB |
| One object | float16 | 2,366 | 9.2 GiB | 4.6 GiB | 2.3 GiB |
| Two objects | float32 | 6,174 | 24.1 GiB | 12.1 GiB | 6.0 GiB |
| Two objects | float16 | 3,286 | 12.8 GiB | 6.4 GiB | 3.2 GiB |

A resumed run must keep its world count. Its replay buffer starts empty unless saved (`save_replay`).

`budget.json` holds `zero_update` (one 64-world collection, capacity 2,048, minimum 64,
`updates_per_collection=1.0`) and `scaled_template` (1,024 worlds, 10,240,000 steps). Neither is a measured
recommendation.
