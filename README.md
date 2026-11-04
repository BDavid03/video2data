# video2data: V2D Challenge Track 3 entry

Our entry in NVIDIA's V2D Challenge, Track 3. Track 3 starts from human egocentric video of tabletop
tasks. It scores object shape (CD-O) and the object motion a simulated robot policy achieves (AUC,
SP-SR, MP-SR, RPE, MPPE).

The project was built with AI coding agents under the owner's direction. The dated working records,
including failed attempts, are in [docs/chain/](docs/chain/).

## Visualisations

More figures and the full explanation are in the
[non-technical report](docs/report/nontechnical/report.md). None of these is a
challenge score.

The trained policy on episode 38. Green is the pan mask from the video. Orange is
the larger simulated pan and the coloured shapes are the robot hands, drawn through
the fitted camera. Episode 38 has no labels, so this shows the policy following its
video-derived goal. It does not show that the goal is correct.
[Watch the clip](docs/report/nontechnical/figures/21-episode38-on-real-footage.mp4).

![Episode 38 policy drawn on real footage](docs/report/nontechnical/figures/21-episode38-on-real-footage.png)

Tracking error against that same goal, at the failed midpoint check and at the end
of training. Why the midpoint failed has not been diagnosed.

![Episode 38 tracking error at the midpoint and at the end of training](docs/report/nontechnical/figures/20-episode38-video-goal.png)

Practice episode 39, compared with motion capture at each stage. These numbers use
the scorer from the kit we had before the organisers replaced it on 2 November. A
rough re-check with the new scorer's code, using public shapes, moves the video
policy from 0.775 to 0.669.

![Episode 39 at each stage against motion capture](docs/report/nontechnical/figures/18-policy-on-video-path.png)

The video tracker's pan outline on real footage from episodes 39 and 38.
[Watch the clips](docs/report/nontechnical/figures/16-tracked-pan.mp4).

![Tracked pan outline on real footage](docs/report/nontechnical/figures/16-tracked-pan.png)

## Method

Meshes. We measure each object's proportions from video depth and build a closed procedural shape from
a rule for its object type. After a pilot on public scans failed on the rack, we generated Hunyuan3D-2mini shapes for pots and pitchers from a masked first frame. A shape needed a video check and a packed scan check. The video check has no validated link to CD-O: it also passed a pot 6 shape that the scan check vetoed. The check accepted evaluation pot 5 and
pitcher 20. The final combined file does not use those two shapes, because they have no recorded mapping
to the rollout frame, so it holds procedural shapes only.

Poses. Nineteen of the 20 evaluation episodes use static FlashCHORD rollouts: a policy with
parked hands in a box scene built from the video, exported by the unchanged evaluator. Placement comes
from calibrated stereo, with a heading correction on episode 19. The objects stay at rest, so these
rollouts do not reproduce the manipulation.

Episode 38 uses a moving RL policy (FlashSAC under FlashCHORD, 66.76M transitions, seed 0) that reached its scheduled end following an object path tracked from the video. It finished after the 12:00 UTC last-pack cut-off, the owner moved the cut-off, and it passed its frozen admission gates, so it is packed as the moving v5 file. Training was stopped and
resumed twice and lost its replay buffer each time. It failed its midpoint check, and we do not know why.
Two files exist: combined v5 (all 20 static, `ddfc98b3`) and moving v5 (38 moving, `1570037a`);
[the submission ledger](docs/chain/SUBMISSION.md) records which was uploaded and marked final.

## Results

Final scores: to be filled at release

| Leaderboard record | Boards | Result |
| --- | --- | --- |
| D1 v5 meshes | CD-O | 0.53562 cm, recorded but not the counted file |
| D2 v5 poses | Five pose boards | Rejected: the kit replaced on 2 November needs poses and meshes in one file |

Our own measurements, none of them a leaderboard score:

| What | Value | Basis |
| --- | --- | --- |
| D1 v5 CD-O, object / pair mean | 0.564 / 0.621 cm | Public scans, 26 of 36 pairs |
| D1 v4 CD-O (procedural only), object / pair mean | 0.653 / 0.663 cm | Same |
| Static AUC | 0.29 to 0.30 | 8 train episodes, new scorer, 3 seeds |
| Static SP-SR, MP-SR, RPE, MPPE | 0.25, 0, 19.5 to 21.1 cm, 12.7 to 15.7 cm | Same |
| Moving policy AUC on episode 39 | 0.775 old scorer, 0.669 new (0.591 with the tracking mesh) | Same recipe as 38, labelled practice episode |
| Episode 38 error against its own video goal | 1.42 cm, 1.59 degrees | No labels exist |

- New-scorer figures use public geometry and assumed axes, as the host's reference geometry is
  unpublished. They do not predict host scores.
- Every pose number in this repository from before 4 November uses the old scorer.
- We chose the mesh rules on the same public scans, so these are development estimates. Most of the v5
  gain comes from one pitcher pair, and ten spoon pairs have no public scan.
- We expect the combined file's CD-O to be somewhat worse than D1 v5's.
- Episode 38's numbers do not show that its goal is correct or that it beats the static rollout. The video tracker failed its first frozen rule and its amendment, and proceeded under a conditional rule written after seeing both clips' gates and before any label. It used a body convention read once from the public scan. Discount it accordingly ([moving-m1-outcome.md](docs/chain/moving-m1-outcome.md)).

## Reproduce

Packed files, run records, environments, the dataset and `upstream/` are not in git. These commands need
them in place. Container build files are in [scripts/container/](scripts/container/README.md).

```bash
bash scripts/linux/install-linux-cuda.sh   # project environment
bash scripts/linux/install-kit-env.sh      # submission kit environment
bash scripts/linux/install-flashchord.sh   # FlashCHORD runtime
# Check the combined file's recorded hashes, then run the full CPU release check
bash scripts/reproduce/verify.sh submission/20261104T1316Z-d2-combined-v5/track3_combined_v5.manifest.json
bash scripts/reproduce/check_release.sh --combined submission/20261104T1316Z-d2-combined-v5/track3_combined_v5.manifest.json
# Print rebuild plans; add --run to execute
bash scripts/reproduce/d1_meshes_v5.sh https://github.com/BDavid03/video2data/commit/b1617234a85e316baa4a67b7f10ff053f5f92c4d
bash scripts/reproduce/d2_poses_v5.sh
bash scripts/reproduce/d3_report.sh
```

Use the full commit URL: it is packed into the file, so a shortened one changes the bytes. Older runs may
need their recorded commit, because the scripts refuse changed sources. See
[scripts/reproduce/README.md](scripts/reproduce/README.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/features/` | One folder per feature, each with a README ([index](docs/design/README.md#feature-directory)) |
| `docs/report/` | Technical report source (`track3/`), plain-language report and figures (`nontechnical/`) |
| `docs/chain/` | Dated working records and decision protocols |
| `docs/design/` | Feature directory, artifact contracts, code style |
| `data/configs/` | Pinned sources (`sources.lock.json`) and the frozen split (`splits/public-v1.json`) |
| `data/reports/` | [Contract](data/reports/contract.md), [metric protocol](data/reports/metric-protocol.md), [data audit](data/reports/data-audit/README.md), [model licences](data/reports/model-readiness/README.md) |
| `AGENTS.md`, `CLAUDE.md` | Instructions for the coding agents |

Inputs are `nvidia/video_to_data_challenge` at `5f68335f3acc802033d1e80728c1633197521de8`, FlashCHORD at
`aee95f25ae0943f335d8bb42fa95b6a9acfa266f`, and the submission kit. Episode 41 and holdout episodes 43 and 44
are excluded from current work, and public labels are for diagnostics only. Scores in the archived depth study
compare to MoGe2 pseudo-labels, not measured depth, and are separate from this entry. Licence: [Apache-2.0](LICENSE).
