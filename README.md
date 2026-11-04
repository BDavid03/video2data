# video2data: V2D Challenge Track 3 entry

Our entry in NVIDIA's V2D Challenge, Track 3.

**Start here.** The two-page technical report is
[docs/report/final/track3-technical-report.pdf](docs/report/final/track3-technical-report.pdf). The counted
submission is moving v5 (`track3_combined_v5.parquet`, SHA-256 `b476eb1c…`), listed with every other uploaded
file in [docs/submissions/counted.json](docs/submissions/counted.json). How to check and rebuild everything:
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Code lives in `src/features/`, one folder per job, each with its own README.

## What we did

1. **The task.** Track 3 gives human egocentric video of tabletop tasks. Each recording is an
   *episode*. For each scored episode we must hand in a mesh of every manipulated object and the
   object's pose in every frame. The poses must come from robot-hand policies trained by
   reinforcement learning (RL) in simulation. Six leaderboards score the entry: one shape board
   (CD-O) and five pose boards (AUC, SP-SR, MP-SR, RPE, MPPE). The terms are defined below.
2. **Shapes.** We run MoGe-2 depth on the video and measure each object's size. The sizes drive a
   hand-designed procedural shape for that object type (pot, pitcher, pan, rack, spoon and so on).
   The submitted file holds procedural shapes only.
3. **Static poses (19 of 20 episodes).** We place each object from calibrated stereo depth on the
   first frame. We train a short FlashSAC RL policy in a simulated box scene built from the video,
   with the hands parked, and the official evaluator exports its poses unchanged. The objects stay at rest, so
   these poses do not reproduce the manipulation.
4. **Episode 38 (moving).** We tracked the dust pan's path in the video. A FlashSAC policy, trained
   from scratch under FlashCHORD (66.76M transitions, seed 0), learned to carry the pan along that
   path. It went into the file only after passing admission gates written before the run.
5. **What we submitted.** One file, "moving v5": episode 38 moving, the other 19 static. Its
   scores are in the Results table below (AUC 0.32339, CD-O 0.56471 cm).

## Where things are

| Step | What it does | Code | Run | Result |
| --- | --- | --- | --- | --- |
| Shapes | Measure each object from video depth, build its mesh | `src/features/mesh_extents/` | `scripts/reproduce/d1_meshes.sh` | `data/outputs/mesh_extents/` |
| Static poses | Place each object, train a static policy | `src/features/pose_insurance/` | `scripts/reproduce/d2_poses_v5.sh` | `data/outputs/pose_insurance/` |
| Episode 38 | Track the pan in video, train the moving policy | `src/features/moving_policy/` | `scripts/reproduce/d2_poses_moving.sh` | `data/outputs/moving_policy/ep38-v1/` |
| Submitted file | Pack and check the file we uploaded | `src/features/pose_insurance/` | `scripts/reproduce/check_release.sh` | `submission/20261104T1422Z-d2-moving-v5/` |
| Report | Build the two-page report | `docs/report/track3/main.tex` | `scripts/reproduce/d3_report.sh` | `docs/report/final/track3-technical-report.pdf` |

- **Trained policies:** the 20 behind the submitted file, in `models/policies/track3-moving-v5/`.
- **Helpers:** depth `src/features/depth/`, stereo `src/features/stereo_depth/`, video tracking `src/features/moving_reference/`.

## Terms

- **Episode**: one recorded video of a person doing one task. Episodes are numbered.
- **Slice**: one folder under `src/features/`. It holds all the code, settings and README for one job.
- **Pack**: the single Parquet file the organisers' kit builds for upload, with its manifest (a list of
  every input file and its SHA-256 hash).
- **Admission gate**: a pass/fail rule written down before the result is known. A result that fails
  is not used. The rules are in `docs/chain/*-protocol.md`; the verdicts are in `*-outcome.md`.
- **CD-O**: chamfer distance between our object mesh and the organisers' scan of the object, in cm.
  Lower is better.
- **AUC**: area under the curve of pose success against the error threshold, from 0 to 1. Higher is better.
- **SP-SR / MP-SR**: share of episodes whose object poses stay within a loose limit (10 cm, 0.5 rad) or a
  tight limit (3 cm, 30 degrees). Higher is better.
- **RPE / MPPE**: position error relative to another object / mean error of six points on the object, in cm.
  Lower is better.

## Results

Leaderboard scores (public boards, 4 November 2026):

| Leaderboard record | Boards | Result |
| --- | --- | --- |
| Combined v5 (all static), 14:15 UTC | All six | AUC 0.30715, SP-SR 0.15, MP-SR 0.0, RPE 20.108 cm, MPPE 14.624 cm, CD-O 0.56604 cm |
| **Moving v5 (episode 38 moving), 14:28 UTC, the counted file** | All six | AUC 0.32339, SP-SR 0.20, MP-SR 0.0, RPE 20.108 cm, MPPE 14.296 cm, CD-O 0.56471 cm |
| D1 v5 meshes | CD-O | 0.53562 cm, recorded but not the counted file |
| D2 v5 poses | Five pose boards | Rejected: the kit replaced on 2 November needs poses and meshes in one file |

Our own measurements, none of them a leaderboard score:

| What | Value | Basis |
| --- | --- | --- |
| D1 v5 CD-O, object / pair mean | 0.564 / 0.621 cm | Public scans, 26 of 36 pairs |
| D1 v4 CD-O (procedural only), object / pair mean | 0.653 / 0.663 cm | Same |
| Static AUC | 0.29 to 0.30 | 8 train episodes, frozen kinematic projections of the static placement, new scorer, 3 registration seeds (not 3 trained policy sets) |
| Static SP-SR, MP-SR, RPE, MPPE | 0.25, 0, 19.5 to 21.1 cm, 12.7 to 15.7 cm | Same |
| Moving policy AUC on episode 39 | 0.775 old scorer, 0.669 new (0.591 with the tracking mesh) | Same hyperparameters as 38, but trained uninterrupted; labelled practice episode |
| Episode 38 error against its own video goal | 1.42 cm, 1.59 degrees | No ground-truth labels available to us |

What these numbers do not show:

- New-scorer figures use public geometry and assumed axes. The host's reference geometry is
  unpublished. They do not predict host scores. Pose numbers from before 4 November use the old scorer.
- We chose the mesh rules on the same public scans, so the mesh numbers are development estimates.
  Most of the v5 gain comes from one pitcher pair. Ten spoon pairs have no public scan.
- D1 v5 used two generated Hunyuan3D-2mini shapes (evaluation pot 5 and pitcher 20). The combined file
  does not, because they have no recorded mapping to the rollout frame. That is why its CD-O is worse.
- Episode 38 does not show that its goal is correct or that it beats the static rollout. Its training
  was stopped and resumed twice and lost its replay buffer each time. It failed its midpoint check, and
  we do not know why. It finished after the planned last-pack time, which the owner moved. It passed
  the final checks after disclosed tracker revisions, a retrospective alignment of the episode-39
  prerequisite and an exception allowing resumed training. The video tracker failed its first fixed
  rule and its amendment, then passed a conditional rule written after seeing both clips' gates and
  before any label ([moving-m1-outcome.md](docs/chain/moving-m1-outcome.md)). Discount it accordingly.

## Pictures

None of these is a challenge score. The full explanation is in the
[non-technical report](docs/report/nontechnical/report.md).

Episode 38 policy on real footage. Green is the pan mask from the video. Orange is the larger
simulated pan at the exported poses. The robot hands come from a CPU replay whose pan path differs
from the export by 0.63 cm on average (1.84 cm at most).
[Watch the clip](docs/report/nontechnical/figures/21-episode38-on-real-footage.mp4).

![Episode 38 policy drawn on real footage](docs/report/nontechnical/figures/21-episode38-on-real-footage.png)

Episode 38 tracking error against its video goal, at the failed midpoint check and at the end of training.

![Episode 38 tracking error at the midpoint and at the end of training](docs/report/nontechnical/figures/20-episode38-video-goal.png)

Practice episode 39 against motion capture at each stage (old scorer; the new scorer moves the video
policy from 0.775 to 0.669).

![Episode 39 at each stage against motion capture](docs/report/nontechnical/figures/18-policy-on-video-path.png)

The video tracker's pan outline on episodes 39 and 38.
[Watch the clips](docs/report/nontechnical/figures/16-tracked-pan.mp4).

![Tracked pan outline on real footage](docs/report/nontechnical/figures/16-tracked-pan.png)

## Reproduce

Packed files, run records, environments, the dataset and `upstream/` are not in git. The commands
need them in place. Container build files are in [scripts/container/](scripts/container/README.md).

```bash
bash scripts/linux/install-linux-cuda.sh   # project environment
bash scripts/linux/install-kit-env.sh      # submission kit environment
bash scripts/linux/install-flashchord.sh   # FlashCHORD runtime
# Check the counted file's recorded hashes, then run the full CPU release check
bash scripts/reproduce/verify.sh submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
bash scripts/reproduce/check_release.sh --combined submission/20261104T1422Z-d2-moving-v5/track3_combined_v5.manifest.json
# Print rebuild plans; add --run to execute
bash scripts/reproduce/d1_meshes_v5.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d
bash scripts/reproduce/d2_poses_v5.sh
bash scripts/reproduce/d3_report.sh
```

- The counted file's manifest pins the report source at its URL commit `c8ff8c1`. Run the two checks
  from that commit; at a later commit the report hash no longer matches.
- Use the full commit URL. It is packed into the file, so a short one changes the bytes.
- The scripts refuse changed sources. Older runs may need their recorded commit.
- Details: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [scripts/reproduce/README.md](scripts/reproduce/README.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/features/` | One slice per job, each with a README ([index](docs/design/README.md#feature-directory)) |
| `scripts/reproduce/` | Rebuild and check scripts |
| `docs/report/` | Technical report (`track3/`, PDF in `final/`), plain-language report and figures (`nontechnical/`) |
| `docs/chain/` | Dated working records, gates written before each run (`*-protocol.md`) and their verdicts (`*-outcome.md`) |
| `docs/design/` | Slice index, artifact contracts, code style |
| `data/configs/` | Pinned sources (`sources.lock.json`) and the frozen split (`splits/public-v1.json`) |
| `data/reports/` | [Contract](data/reports/contract.md), [metric protocol](data/reports/metric-protocol.md), [data audit](data/reports/data-audit/README.md), [model licences](data/reports/model-readiness/README.md) |

## Inputs and limits

- Dataset `nvidia/video_to_data_challenge` at `5f68335f3acc802033d1e80728c1633197521de8`.
  FlashCHORD at `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`. The organisers' submission kit.
- Episode 41 and holdout episodes 43 and 44 are excluded.
- Public labels were used to choose the static placement, for one motion-capture training run on
  practice episode 39, and for diagnostics. Episode 38's policy trained against a video-derived path.
- Depth scores compare to MoGe2 pseudo-labels, not measured depth. They are not part of this entry.
- Licence: [Apache-2.0](LICENSE).
