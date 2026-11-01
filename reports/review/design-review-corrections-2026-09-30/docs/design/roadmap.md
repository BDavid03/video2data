# Roadmap to the freeze

Written 2026-09-30. It replaces the milestone dates in `TRACK3-PLAN.md`, which
were set before the contract was published. Dates are Sydney time.

## Fixed dates

| Event | Organisers' time | Sydney time |
|---|---|---|
| Participant webinar | 7 October, 11:00 Pacific | 8 October, 05:00 |
| Unlimited submissions begin | Three days before the freeze | About 2 November, 09:00 |
| Leaderboard freeze | 4 November, 17:00 EST | 5 November, 09:00 |
| Winners announced | 12 November | 12 or 13 November |

The webinar date and time come from the
[user-supplied organiser announcement](../../reports/contract-evidence/2026-09-30/organiser-announcement.md);
access details are pending. Until the final three days, the limit is five
submissions per week (**Official**, `KIT/config/tracks.json:304,333`, also the announcement). The week's reset time is unpublished, so the
submission ledger counts a rolling seven days.

## Gates that need the user

The plan cannot start without these. Each has the latest date that keeps the
schedule below intact.

| Gate | Needed | By |
|---|---|---|
| G1 | Authorize execution of named operations. Proposed order: depth on public development episode 39, then perception on other permitted development episodes, then the policy runtime. The earlier refusal to run depth on episode 41 remains in force; it is excluded from proposed runs. | 2 October |
| G2 | Approve an isolated Ubuntu 24.04 WSL distribution under `.runtime/wsl/`, and installing the locked FlashCHORD dependencies in it | 2 October |
| G3 | Create the GitHub repository and share it with the organisers. Create a Kaggle API token. Accept the rules of the six Track 3 competitions. Confirm team registration. | 4 October |
| G4 | Register for MANO and place `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` under `models/hand/models/` | 4 October |
| G5 | Approve perception dependencies in batches: masks first, then the pose tracker and the hand model | 5 October |
| G6 | Post the organiser questions in [risks and open questions](risks-and-open-questions.md) | 2 October, and at the webinar |

## Phases

### Phase 0: design and gates, 30 September to 4 October

| Work | Output |
|---|---|
| Contract recorded from the Submission Kit and pinned source | `reports/contract.md`, `configs/evaluator/contract.json` |
| Design set | `docs/design/`, feature READMEs |
| Organiser questions drafted | Ready for the user to post |
| Implementation of slices that need no new dependency: stereo_depth, camera_motion, scene_assembly | Code, not run; assembly uses explicit triangle-distance calculations without `rtree` |
| Extend depth to named evaluation episodes and separately gated public validation; migrate depth and scratch_depth run records to the shared contract | Code, not run; evaluation pixel grids become possible only after this work and execution authorization |

Exit: gates G1 to G4 answered.

### Phase 1: two feasibility answers, 5 to 11 October

Two questions decide whether the plan is possible. Both are answered before
anything is built on top of them.

| Question | How |
|---|---|
| Does the policy stack run here, and at what cost? | Install the policy environment. Load the upstream example sequence in the scene viewer. Train briefly at 256, 512 and 1,024 worlds and record device memory and steps per second. Run the evaluator on the resulting checkpoint and pass its file through the packer's checks. |
| How good is perception? | Run depth, stereo depth, masks, camera motion and object pose on public development episode 39, then on other permitted development episodes, excluding episode 41. Score the reconstructed trajectories against public poses with the kit's metric code. |

Exit: a measured training cost per policy; a measured reconstruction score on
development episodes; a decision on world count and step budget. If training does
not fit, the user decides between a rented GPU and a reduced entry.

### Phase 2: one episode end to end, then all twenty, 12 to 18 October

| Work | Notes |
|---|---|
| One evaluation episode through every slice | Start with episode 38: one scanned object, 263 frames |
| Hands, scene assembly, retargeting | Needs G4 |
| Perception on all 20 evaluation episodes | The ten scanned-object episodes first |
| A provisional `wooden_spoon` mesh | Needed for the other ten |
| Twenty policies at a small step budget | Enough to produce valid rollouts |
| **Submission 1: poses and meshes** | Complete and weak, by 18 October |

Exit: scores on all six leaderboards.

### Phase 3: full training, 19 to 25 October

| Work | Notes |
|---|---|
| Twenty policies at the full step budget chosen in Phase 1 | One GPU job at a time; Phase 3 fits only if each averages about 8 hours or less, and other GPU work fits the remaining time |
| Validation episodes scored once, with settings frozen | Seven public episodes |
| Submissions 2 and 3 | After each completed training pass |

Exit: a trained policy for every episode, and a list of episodes ranked by the
evaluator's own tracking error.

### Phase 4: targeted improvement, 26 October to 1 November

Work on the largest measured error, one change at a time. Candidates, to be
ordered by evidence from Phases 1 to 3:

| Candidate | Helps |
|---|---|
| Better `wooden_spoon` mesh and frame | Ten episodes |
| Re-initialisation after occlusion | Episodes with long hand cover |
| Retargeting options: smoothing, penetration offsets | Policies that knock objects |
| Retraining the worst episodes with more steps | Those episodes |

Also in this phase: the technical report draft, and preparing the repository for
release.

Exit: the entry intended as final, submitted once before the unlimited window.

### Phase 5: final window, 2 to 5 November

| Work | Notes |
|---|---|
| Final package built from a clean, pushed commit | Target 3 November |
| Re-submissions only for measured improvements | Unlimited in this window |
| Choose the final submissions on each leaderboard | The kit's configuration allows two per competition (`KIT/config/tracks.json:326`) |
| Stop | 4 November, evening. The last 12 hours are reserve. |

## Compute budget

Unmeasured. Phase 1 fills this table.

| Item | Count | Cost each | Total |
|---|---|---|---|
| Perception, all slices, per episode | 20 evaluation, 19 proposed public (episode 41 excluded) | Unknown | Unknown |
| Policy training, small budget | 20 | Unknown | Unknown |
| Policy training, full budget | 20, preferably 40 | Unknown | Unknown |
| Rollout | 20 per submission | Unknown | Unknown |

From 12 October to 1 November there are 21 days, or 504 hours, on one GPU.
Reserve about 100 hours for perception, small-budget training, rollouts and
interruptions, leaving about 404 hours for full training. Forty policies then
require an average of about 10 hours each. Phase 3 alone has 168 hours: twenty
policies require at most 8.4 hours each before other GPU work, so the planning
threshold is about 8 hours. If that threshold is exceeded, spread one pass into
Phase 4 and reduce targeted improvement; a second full pass is not assumed.
Phase 1 measurements must replace these estimates before training is scheduled.

The frozen split remains 13 development and seven validation episodes. The
proposed execution roster omits episode 41, giving twelve development plus seven
validation episodes; split membership itself grants no execution permission.

## What is already behind

`TRACK3-PLAN.md` planned one compatible inference stage and one simulator smoke
by 27 September, and a data adapter with one reconstruction by 4 October. Neither run has happened. The depth slice is implemented and has never
executed. The schedule above assumes runs begin on 5 October; each day later
comes out of Phase 4.
