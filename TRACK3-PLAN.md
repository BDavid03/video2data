# Track 3 Egocentric: local execution plan

Prepared 2026-09-24. Project root: `Z:\hugging-face\video2data`.
Compute constraint: existing local hardware only. Implementation of the first batch began on 2026-09-24; see [current execution evidence and blockers](reports/implementation-status.md). The original planning baseline below predates implementation. The initial foundation batch installed no production dependencies or weights; a synthetic CUDA runtime smoke ran before the no-testing direction. On 2026-09-25, the user approved the bounded MoGe2 installation recorded under `reports/installation/moge-20260925/`.

**Revision note, 2026-09-30.** The organisers published the Track 3 contract on
2026-09-29. It is recorded in [reports/contract.md](reports/contract.md). The
design that follows from it is in [docs/design](docs/design/README.md), and the
[roadmap](docs/design/roadmap.md) replaces the milestone dates below. This
document remains as the original plan and the record of the first batch. Where it
says the evaluator, embodiment or submission format is unknown, the contract
report now has the answer. Candidate components it names, such as Isaac Lab for
simulation, were superseded by the published runtime.

Current user override: **absolutely no testing**. No further test suites, smoke runs, validation commands or dependency-resolution experiments may run. Historical results remain recorded; implementation and documentation may continue. All generated artifacts must stay inside this repository.

## Objective and approach

Produce a reproducible Track 3 submission that converts the released egocentric videos into reconstructions and executable robot policies, with recorded evaluations packaged by the official evaluator. The challenge permits explicit reconstruction and retargeting, implicit learning, or hybrids. Award eligibility also requires a short technical report and agreement to release code under a research license. The leaderboard freezes November 4, 2026 at 17:00 EST, equivalent to November 5 at 09:00 AEDT in Sydney. Aim to have the final package ready November 2. [Challenge requirements](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#tracks)

Recommended approach: calibrated video -> camera motion and metric depth -> object and hand trajectories -> robot retargeting -> compact policy trained in simulation -> official evaluation. First establish a working path on one public episode, then expand coverage and improve the measured failure points.

Use pretrained perception components and a small policy network. The separate upstream GR00T workflow documents a 48 GB GPU and Sharpa embodiments; it is useful reference material, but its hardware and task configuration have not been validated for this machine or the Track 3 evaluator. [Upstream full workflow](https://github.com/nvidia-isaac/video_to_data/blob/main/docs/e2e_workflow.md)

## Verified starting point

| Item | Observation |
|---|---|
| GPU | NVIDIA RTX 5080, 16,303 MiB VRAM; Windows driver 596.21 |
| RAM | Approximately 64 GB installed |
| Storage | Approximately 1.48 TiB free on Z:, 62.5 GiB free on C: at inspection |
| Linux environment | Ubuntu under WSL 2; GPU visible from Ubuntu |
| Docker | 29.8.1 inside Ubuntu; daemon accessible as root; ordinary Ubuntu user receives socket permission denied |
| Docker storage | `/var/lib/docker` inside the Ubuntu virtual disk, currently backed by C: |
| Project contents | Existing Hugging Face dataset cache; no application/source checkout yet |
| Dataset revision | `5f68335f3acc802033d1e80728c1633197521de8` |
| Track 3 files | All 169 snapshot files resolve and match manifest byte sizes; checksums and decoding remain to be verified |
| Public set | 20 episodes, 10 tasks, 5,246 frames, 60 camera clips, object trajectories and assets |
| Evaluation set | 60 video clips across three streams; ground-truth trajectories are not distributed in this split |
| Starter release inspected | `release/0.2.0`, commit `43a3f5b4d479581f2e8bdf5e6ed7ca361d39ec5a` |
| Main branch inspected | Commit `33129dd0f2d2dcfd1164d43fd076542660756ed2` |

Inventory evidence came from local hardware commands, the cached tree manifest, resolved snapshot files, and public metadata. File-size checks do not establish video integrity.

The RGB camera is accompanied by a calibrated mono stereo pair. Object poses use metres and scalar-first quaternions. Invisible objects have invalid zero poses. Each episode has its own horizontal origin, and camera-to-mocap-world transforms are absent. [Track 3 data conventions](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_3/README.md)

## Milestone 0: settle the contract and runtime (September 24-27)

1. Pin the dataset and starter commits. Record environment, source, model, configuration, and artifact hashes in run manifests.
2. Obtain the official Track 3 evaluator and submission schema. Neither inspected GitHub tree contained `eval_e2e.py`, despite the website and dataset card referring to it. Track the exact source/version when it becomes available. [Dataset submission guidance](https://huggingface.co/datasets/nvidia/video_to_data_challenge)
3. Resolve these specific questions from official rules or organizer clarification before committing to policy architecture:
   - Track 3 embodiment, simulator version, allowed observations/actions, reset distribution, horizon, metrics, and aggregation.
   - Whether a policy may be optimized separately for each supplied demonstration or must be shared across tasks.
   - Whether all three ego streams, supplied calibration, known object meshes, and manual annotations are allowed at evaluation time.
   - Required reconstruction contents: hands, full body, object geometry/poses, world-frame alignment, and missing-track representation.
   - Exact package structure, evaluation episodes, registration and submission mechanism.
   These are open questions, not established restrictions. Do not inherit Track 2's G1 + Dex3 contract or the generic demo's Sharpa contract without verification.
4. Establish Docker access for the intended runtime user. Budget storage before building images: Ubuntu's apparent free virtual capacity is larger than its available C: backing space. Prefer a dedicated project runtime disk/location on Z:; any relocation of the existing shared Ubuntu/Docker setup needs a separate, explicit migration decision.
5. Smoke-test GPU inference, required CUDA extensions, headless simulator stepping, and a short rendered recording. WSL CUDA visibility alone does not prove simulator graphics work. If WSL graphics fail, evaluate native Windows Isaac Lab on the same machine, with perception stages in WSL and artifacts exchanged through Z:. Preserve the official simulator contract. [Isaac Lab installation platforms](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)
6. Build only required components. NVIDIA documents an `sm_120` incompatibility in the cuSFM-based HOI route. Use the ego reconstruction path with a separately validated camera-motion backend; confirm each extension actually runs on the RTX 5080. [Reconstruction compatibility](https://github.com/nvidia-isaac/video_to_data/blob/release/0.2.0/reconstruction/README.md)

Exit criteria: exact unresolved-rule list recorded, one compatible inference stage and simulator smoke completed within memory limits, and an agreed local storage layout. Missing evaluator details may leave packaging open while public-data reconstruction continues. If no compliant simulator runtime works locally, record that as a competition feasibility blocker.

## Milestone 1: data adapter and one reconstruction (September 28-October 4)

- Implement explicit episode discovery from filenames and metadata. The public metadata still says `train: 0:46`, although only 20 noncontiguous episode IDs are released; never generate episodes from that range.
- Decode all videos, compare frame counts and timestamps, validate object identities/assets, and verify hashes. Keep the input snapshot immutable and place derived files under the project output directory.
- Validate cross-camera pairing before stereo depth: timestamp origins, dropped/duplicate frames, and any observed offset. Record the synchronization tolerance and test stereo rectification plus RGB-to-stereo geometry on a short segment. Equal frame counts alone do not establish synchronization.
- Verify camera orientation before preprocessing: the ego rig calibration records a 180-degree rotation. Apply rotation, distortion correction, resize, and intrinsics transforms consistently; avoid double rotation.
- Start with one visually selected simple `Pan_Solo` episode. Use RGB for detection/hand appearance and evaluate stereo for metric depth if permitted by the contract. Keep a monocular route available.
- Candidate perception components: Grounding DINO and SAM2 for object masks, FoundationPose for known-mesh tracking, and WiLoR/HaMeR for hands. Prefer supplied meshes when allowed. Use the consolidated `run_ego_reconstruction.py` adapter and test DROID-SLAM for camera motion. Model assets such as MANO and gated weights must be available before their dependent stages run. [Ego setup and supported routes](https://github.com/nvidia-isaac/video_to_data/blob/release/0.2.0/reconstruction/docs/ego_e2e_setup.md)
- Estimate camera motion from background/static evidence, masking hands and manipulated objects. Recover scale from permitted calibrated stereo or known geometry; establish gravity and a support plane.
- Save explicit coordinate transforms: `world_T_object = world_T_camera * camera_T_object`. Treat the moving ego camera as time-varying. Public-label alignment may be used for clearly labelled diagnostics; evaluation inference must not depend on unavailable mocap poses or fitted hidden-label transforms.

Exit criteria: every frame represented with explicit track validity and confidence; finite poses for valid estimates; correct frame indexing and quaternion handling; an overlay video reviewed at start/contact/release/end; a metric scene with consistent camera, object, and hand transforms; stage runtime and peak total GPU memory measurements. Distinguish observed poses from interpolated estimates, set a maximum interpolation gap, and record reinitialization after occlusion. Invalid zero poses must never become training targets. Record numeric error baselines without inventing official thresholds.

Run a control feasibility pilot during Milestone 1 as soon as the official embodiment and policy interface are confirmed. Use a public trajectory to validate retargeting, a short training update, checkpoint reload, and one rendered evaluation independently of reconstruction. This checks simulator/contact/training memory before the larger October 12-18 training phase. Record peak total device VRAM, RAM, disk growth, and elapsed time.

## Milestone 2: public coverage and validation (October 5-11)

- Extend the adapter to multiple objects and both hands while preserving persistent identities and confidence/visibility masks.
- Freeze an episode-level development split: provisionally 13 training/development episodes and 7 validation episodes, choosing one held-out episode from each of the seven tasks with multiple examples. Keep all camera streams and frames from an episode together. Singleton tasks cannot support within-task validation; report this limitation explicitly.
- Process all 20 public episodes. Maintain an error table by task, object, visibility, and contact phase. Report both pooled and per-task results because `Pan_Solo` has more examples.
- Measure translation/rotation error and coverage on valid public poses, reprojection quality, temporal acceleration, drift, and hand-object penetration. State any alignment and symmetry handling alongside the metric. Treat these as engineering diagnostics until the official scoring contract is available.
- Freeze the metric protocol: pose direction, quaternion normalization, validity masks, and object symmetries. Use one fixed alignment per episode for labelled aligned diagnostics, and report relative-motion/drift diagnostics alongside them. Per-frame label-fitted alignment must not hide trajectory errors.
- Run the public-trajectory branch as a labelled diagnostic to isolate retargeting/control errors from reconstruction errors. Keep it out of the video-only evaluation path.

Exit criteria: every public episode has either a validated artifact or a recorded actionable failure; validation inputs are isolated from tuning; a ranked list identifies the dominant downstream errors.

## Milestone 3: retargeting and a working policy (October 12-18)

- Use the confirmed Track 3 embodiment and official task assets. Validate robot joint order, units, frame conversions, collision geometry, table/support alignment, and object inertia.
- Retarget hand/wrist trajectories with inverse kinematics and contact-aware constraints. Preserve object motion and interaction timing; check joint limits and penetration before training.
- Train a compact PPO-style reference-tracking policy using the permitted observations. Start with one task and a small number of parallel environments, increasing only after profiling. A state-based policy is contingent on the evaluator allowing those observations.
- Compare public ground-truth-reference training with reconstructed-reference training to locate bottlenecks. Evaluation policies receive only the inputs allowed by the official rules.
- Validate closed-loop manipulation, then add tasks. Log task success, object tracking, drop/contact failures, runtime, and seeds. Reaching a timeout or producing a plausible replay is insufficient evidence of success.

Exit criteria: one task succeeds reproducibly in the compliant simulator, with a saved checkpoint, recorded rollout, and quantified success rate over a fixed reset set. Benchmark training hours per task and recalculate full-coverage cost before further experiments.

## Milestone 4: targeted improvements (October 19-25)

Prioritize the failures observed in Milestones 2-3:

1. Better masks and reinitialization after occlusion; compare monocular and stereo depth where allowed.
2. Confidence-weighted temporal smoothing and contact-phase constraints that preserve abrupt legitimate motion.
3. Retargeting and policy robustness to the measured reconstruction errors, including pose noise, short dropouts, friction, and reset perturbations within the allowed training setup.

Use one-change-at-a-time ablations with fixed validation episodes and seeds. Promote changes only when they improve validation behavior without causing coverage or runtime regressions. Use a small set of repeat seeds on finalists, not an exhaustive hyperparameter sweep. Consider expensive mesh generation/refinement only if missing or inaccurate geometry is a demonstrated bottleneck.

Exit criteria: a reproducible comparison of the baseline and chosen method, including failures and compute cost; all required tasks covered or outstanding failures clearly enumerated.

## Milestone 5: submission rehearsal and freeze (October 26-November 4)

- Integrate the official evaluator as soon as available and rehearse packaging early. Replace provisional diagnostics with exact official scoring where applicable.
- Run the frozen pipeline over every required evaluation episode using only allowed inputs. Validate completeness, frame/episode IDs, shapes, finite values, units, and artifact hashes.
- Generate reconstructions, policy evaluation recordings, and all evaluator-required metadata. Reproduce a selected result from saved configuration and checkpoints.
- Prepare the technical report: method, training data, validation split, runtime/hardware, ablations, limitations, and reproducibility instructions. Preserve dataset and model attributions.
- Target the final package for November 2 and use the remaining time for submission validation and fixes. The site allows five submissions per week, with unlimited submissions during the final three days; follow the actual competition portal's rules when available. [Challenge timeline and submission cadence](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#timeline)

Exit criteria: official validator succeeds, all required episodes are included, a real policy evaluation accompanies the reconstructions, and the submitted artifact/report can be traced to fixed inputs, source, and checkpoints. A local package alone is not evidence of leaderboard acceptance.

## Project layout to create during implementation

```text
Z:\hugging-face\video2data\
  TRACK3-PLAN.md
  datasets--nvidia--video_to_data_challenge\  existing input cache
  upstream\video_to_data\                   pinned source checkout
  track3\                                   our Python adapters and pipeline
  configs\                                  paths, splits, experiments, runtime
  manifests\                                inventory and run provenance
  models\                                   permitted local model assets
  outputs\                                  per-episode intermediate artifacts
  runs\                                     policy checkpoints and metrics
  reports\                                  audits, overlays, ablation results
  submissions\                              versioned evaluator packages
```

The corresponding WSL data path is `/mnt/z/hugging-face/video2data`. Verify Windows cache symlinks resolve from Linux before container mounts; if needed, materialize a manifest-verified Track 3 input directory under this root. Benchmark mounted-drive I/O before introducing optional local Linux scratch storage.

## Local resource policy and first implementation batch

Use one GPU-heavy stage at a time. Set admission limits from measured total device VRAM, including the Windows display, renderer, CUDA contexts, and training buffers; PyTorch allocated memory alone is insufficient. Aim to retain approximately 2 GiB of device headroom during pilots, subject to measurement. Reduce resolution or batch size and cache outputs before adding more complex alternatives. Start rendered evaluation with one environment. Profile short segments before processing complete episodes.

Reserve a provisional 200-300 GB on Z: for selected images, models, outputs, and checkpoints, then revise from measured usage. Keep the first reconstruction pilot and training pilot small; estimate total runtime as measured per-episode/task time multiplied by required coverage, with a 30% rerun allowance. Avoid claiming a fixed training budget before those measurements exist.

The first implementation batch is: create the project layout and manifests; pin the upstream release; resolve evaluator/runtime questions; audit/decode the cached dataset; and complete one reconstruction plus one simulator smoke. New production dependencies require confirmation under the workspace agreement. If JavaScript files are changed later, run `npm test`; prefer `pnpm` for JavaScript dependency installation.

## Agent execution plan: 30 Astra-high assignments

**Superseded on 2026-09-30.** The user directed that sub-agents run on Opus 5.5 or
Sonnet 5.5. The Astra model settings and the Codex session limits below describe
the 2026-09-24 session and no longer apply. The assignment table remains as a
record of how the work was divided; ownership now follows the feature slices in
[docs/design/README.md](docs/design/README.md). Current practice: Opus for design,
source analysis and review; Sonnet for bounded lookups and mechanical edits; every
brief repeats the no-testing rule and names the only paths the agent may write.

The user requested 20-40 agents using Astra at high reasoning. Target 30 distinct, scoped agent assignments across implementation and independent review. Every newly spawned agent for this project uses `model: gpt-6-astra`, `reasoning_effort: high`, and a self-contained task brief with `fork_turns: none`. The two initial reviews predate this explicit model configuration; they are not counted toward the 30 assignments.

This session permits seven active agents including the root coordinator: at most six sub-agents run concurrently. Dispatch the 30 assignments in successive batches as dependencies become ready. Nested delegation cannot increase this session limit. The table groups assignments by purpose; these groups are not barriers that delay a ready critical-path task.

The user authorized the first implementation batch on 2026-09-24. Six foundation assignments (01-06) have produced initial artifacts and reviews; the runtime and reconstruction acceptance gates remain open. The remaining table entries are future assignments, not completed work. All generated artifacts must remain inside this repository, as the user subsequently specified.

| Group | ID | Assignment | Primary owned output |
|---|---|---|---|
| Foundation | 01 | Verify official evaluator, embodiment, policy observations/actions, reference access, and episode-specific policy rules | `reports/contract.md`, `configs/evaluator/` |
| Foundation | 02 | Prove local runtime/storage feasibility, including early simulation, training update, checkpoint reload, and recording | `configs/runtime/`, `reports/runtime/` |
| Foundation | 03 | Audit hashes, decoding, episode inventory, frame counts, and released metadata | `manifests/dataset/`, `reports/data-audit/` |
| Foundation | 04 | Verify synchronization, camera orientation, distortion, stereo rectification, and frame conventions | `reports/calibration/`, `configs/calibration/` |
| Foundation | 05 | Audit required perception models, weights, access, dependencies, Blackwell extensions, and memory assumptions | `manifests/models/`, `reports/model-readiness/` |
| Foundation | 06 | Define the held-out split and metric protocol, including gauge alignment, symmetries, visibility, and drift | `configs/splits/`, `reports/metric-protocol.md` |
| Perception | 07 | Implement the episode/video adapter and calibrated frame map | `track3/data/` |
| Perception | 08 | Implement object detection and mask tracking | `track3/reconstruction/masks/` |
| Perception | 09 | Implement and profile metric depth/scale estimation | `track3/reconstruction/depth/` |
| Perception | 10 | Implement camera motion and consistent coordinate transforms | `track3/reconstruction/camera/` |
| Perception | 11 | Implement object asset loading and pose tracking | `track3/reconstruction/objects/` |
| Perception | 12 | Implement hand tracking and hand/object alignment | `track3/reconstruction/hands/` |
| Integration | 13 | Implement persistent identities across objects, hands, views, and occlusion | `track3/reconstruction/association/` |
| Integration | 14 | Implement confidence, missing-track handling, bounded interpolation, and contact-aware smoothing | `track3/reconstruction/temporal/` |
| Integration | 15 | Assemble the metric scene, support surface, meshes, and portable reconstruction bundle | `track3/scene/` |
| Integration | 16 | Implement retargeting and diagnose reachability, joint limits, collisions, and contact | `track3/retargeting/`, `configs/retargeting/` |
| Integration | 17 | Implement the permitted policy interface and compact training/evaluation pilot | `track3/policy/`, `configs/policy/` |
| Integration | 18 | Implement metric aggregation, failure reports, overlays, and regression checks | `track3/metrics/`, `reports/validation/` |
| Experiments | 19 | Compare permitted depth/scale alternatives on frozen validation episodes | `configs/experiments/depth/`, `reports/experiments/depth/` |
| Experiments | 20 | Diagnose camera drift and coordinate errors with controlled comparisons | `configs/experiments/camera/`, `reports/experiments/camera/` |
| Experiments | 21 | Compare occlusion recovery, reinitialization, and multi-object tracking | `configs/experiments/occlusion/`, `reports/experiments/occlusion/` |
| Experiments | 22 | Compare contact constraints and retargeting choices using downstream behavior | `configs/experiments/contact/`, `reports/experiments/contact/` |
| Experiments | 23 | Evaluate policy robustness to measured reconstruction errors and permitted reset variation | `configs/experiments/policy/`, `reports/experiments/policy/` |
| Experiments | 24 | Measure throughput, VRAM/RAM, disk growth, caching, and full-coverage runtime | `reports/performance/` |
| Final review | 25 | Independently audit train/validation/evaluation separation and input provenance | `reports/review/data-provenance.md` |
| Final review | 26 | Independently review coordinate transforms, missing-track handling, and reconstruction regressions | `reports/review/reconstruction.md` |
| Final review | 27 | Independently verify policy loading, reset conditions, recorded behavior, and actual task success | `reports/review/policy.md` |
| Final review | 28 | Implement official submission packaging and episode-completeness validation | `track3/submission/`, `reports/submission-validation/` |
| Final review | 29 | Reproduce selected artifacts from pinned inputs/configuration and verify checksums | `reports/reproducibility/` |
| Final review | 30 | Prepare the technical report, ablation tables, limitations, and reproducibility instructions | `reports/technical-report/` |

### Scheduling and dependencies

Start with the six foundation assignments. The root coordinator consolidates their evidence into shared schemas, environment pins, and architecture decisions. Unresolved evaluator requirements remain explicit; agents can continue public-data work that does not depend on those answers.

Dispatch perception work when the input schema and model prerequisites are ready. Bring retargeting and the policy pilot forward as soon as the official contract, runtime, and public-trajectory adapter are available. Do not wait for the complete reconstruction pipeline or October 12 to test whether local control training is feasible.

Dispatch experiments only against a working baseline with frozen validation inputs. Each experiment assignment owns configuration and results. Changes to shared implementation return to its owner or receive an explicit ownership handoff, preventing concurrent edits to the same files. Choose the next experiment from demonstrated failure modes; refine its scope if its proposed comparison is no longer useful.

Independent reviewers receive the final artifacts and acceptance criteria. Reviewers report defects to the owner; the root coordinator verifies fixes and assembles the final submission. Assignment 28 should inspect the official packaging interface as soon as it exists, even though its full acceptance run occurs near completion.

### Coordination and resource ownership

The root coordinator owns shared artifact contracts, dependency manifests, environment changes, pipeline orchestration, experiment promotion, GPU scheduling, and final integration. Agent briefs specify the objective, input artifacts, writable paths, dependencies, acceptance criteria, and a stop condition. Each handoff includes changed paths, validation evidence, remaining issues, and measured resource use. Agents may inspect shared files but edit only their assigned paths unless ownership is explicitly transferred.

Only one GPU-heavy job runs at a time on the RTX 5080. Agents submit a command, expected inputs/outputs, and resource estimate to the coordinator before GPU work. Record peak total device VRAM, RAM, disk growth, elapsed time, and configuration; include display, renderer, CUDA contexts, and training buffers. CPU analysis, implementation, and review proceed concurrently while experiments queue.

The local-hardware-only constraint applies to dataset processing, model inference, simulation, and training. The Astra agents use the models provided by this Codex session; no additional cloud GPU jobs or hosted inference services are planned. Runtime feasibility and the missing official contract remain genuine dependencies regardless of agent count.

