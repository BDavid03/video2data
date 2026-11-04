# Public diagnostics

Status: implemented, never run

All six operations are implemented and disabled. `oracle` accepts the same
trajectory producers as `score`. Its five independent substitutions report fit
parameters and metric gains; no corrected trajectory is persisted. Scale fits
translations about the aligned slot-0 start. Anchor fits a rigid transform to
all observed object keypoints and replaces the initial alignment. Body-frame
fits one object-side rigid offset; time searches the requested integer window
and counts out-of-range edges as misses; rest replaces label-stationary
object-frames within the request's position and rotation thresholds.

The pinned release's missing `wooden_spoon` scan now leaves pose ADD/AUC null
for the whole episode, with explicit reference-scan coverage and missing object
names. SP-SR, MP-SR, RPE and MPPE remain available on their declared trajectory
support. The report never substitutes vertices or averages ADD over only the
scanned objects. A missing scan for another object still fails. Frame-convention
reports list the spoon as unavailable instead of aborting the scan inventory.
Without ADD support, the oracle's AUC-selected time shift and agreement's AUC
ranking are explicitly unscorable; other oracle substitutions retain their
scan-independent metrics in declared order.

Pose summaries preserve the top-level valid-only metric keys and add separate
`valid_frames_only` and `missing_as_miss` means with per-metric episode counts,
requested and unscorable episode counts, and mean pose coverage. Null distances
and unavailable AUC values never become zero. Oracle gain summaries and agreement
summaries also expose their available episode support. Comparisons still require
matched support; these summaries do not make different supports comparable.
These are source-only reporting changes, with no executed diagnostic outcomes.

`agreement` uses `candidate_runs[episode]` lists of rollout run paths and
`self_score_runs[rollout_run]` paths. Each episode needs one resting candidate
and at least one replay/trained candidate, complete valid trajectories, and a
shared tracked scene in the self-score lineage. It reports pairwise AUC ranking
agreement and the smallest strict margin with supporting successful comparisons;
the margin is null if filtering leaves no evidence. Development and validation
remain separate. This is observed agreement, not a guarantee for unseen episodes.

`mesh` accepts same-episode reconstruction runs and uses the unchanged kit
`object_chamfer_cm` and `object_seed`, with the public scan as reference. It reads
object identity from the audit inventory without opening pose-label Parquets.
An optional `mesh_baseline_runs[episode]` mapping names a second reconstruction
for every requested episode, with the same object set. Both use the same reference
scan and metric sampling seed. Reports retain each method and artifact hash,
candidate and baseline distances, and candidate-minus-baseline deltas; negative
deltas mean lower distance on the objects actually scored. No method is selected
automatically and no geometry is transformed or exported by this diagnostic.

The pinned release has no `wooden_spoon` reference scan. Its row is explicitly
`unscorable` with a null distance, and the episode records reference-scan coverage.
A mixed episode is `partially_scored`; a spoon-only reconstruction remains
`unscorable`. An unexpectedly missing scan for another object is still an error.
Means exclude unavailable references and weight scored episodes equally, retaining
both scored and unscorable counts. A subset with no scored objects has null means.
These are original-mesh comparisons using the kit metric; they do not apply the
submission packer's decimation budget and are not predicted leaderboard scores.

All three operations retain the validation
gate, frozen-settings requirement and one-time claim used by the earlier ones.

The evaluation reference is confidential, so the leaderboard is the only score on
evaluation episodes (**Official**, `KIT/tools/pack_submission.py:447-451`). The 20
public episodes have pose labels. This slice scores the project's own outputs on
public episodes against those labels, with the organisers' metric code, so that a
change can be judged before it spends a submission. It is the owner of every
"score against public poses" step that other feature documents refer to.

Nothing this slice writes is ever an input to another slice. Its outputs are
reports for people.

## Role in scoring

None directly. It decides which errors are worth working on
([error budget](../../../docs/design/error-budget.md)) and whether the per-episode
choice between a resting and a trained rollout can be trusted
([entry tiers](../../../docs/design/entry-tiers.md)).

## Operations

All disabled by default. All CPU.

| Operation | Produces | Reads labels |
|---|---|---|
| `floor` | For each public episode, the five pose metrics of a trajectory that holds every object at its frame-0 label for the whole clip | Yes. Needs no project output at all. |
| `score` | The five pose metrics for one trajectory artifact of one public episode | Yes |
| `oracle` | The same score after replacing one error source with its best label-fitted value, one source at a time | Yes |
| `agreement` | For public episodes with several candidate rollouts: whether the label-free ranking used for evaluation episodes orders them the way the labels do | Yes |
| `mesh` | CD-O of a reconstructed mesh against the public scan of the same object | No. Reads a public scan as the reference shape. |
| `frame-convention` | A report on where the ten public scans put their origin and axes | Origin part: no. Resting-axis part: yes, frame-0 labels only. |

## Inputs

| Input | Producer | Fields read |
|---|---|---|
| `track_3/public/data/chunk-000/episode_XXXXXX.parquet` | Dataset | Object names, poses, `visible` |
| `track_3/public/mesh/<name>/<name>.glb` | Dataset | Vertices, for the ADD point set and for `mesh` |
| `configs/splits/public-v1.json` | Frozen split | Which episodes are development and which validation |
| `KIT/v2dlb/object_pose.py`, `vertices.py`, `object_mesh.py` | Organisers | Imported unchanged as the metric implementation |
| `poses.npz` | [object_pose](../object_pose/README.md) | `world_T_object`, `world_valid` |
| `trajectories.npz` | [scene_assembly](../scene_assembly/README.md) | `world_T_object`, `object_filled` |
| `episode_XXXXXX.parquet` | [policy_rollout](../policy_rollout/README.md), public split | Achieved pose columns |
| `<object>.glb`, `mesh.json` | [object_mesh](../object_mesh/README.md), public split | Reconstructed geometry |

A trajectory input from any split other than `public` is refused.

## Method

1. **Reference.** Read the episode's labels. Convert each pose from the dataset's
   `[x, y, z, qw, qx, qy, qz]` to the scorer's scalar-last order, once, here. All
   public poses are visible and valid (**Measured**, `reports/data-audit/README.md`);
   the reader still refuses an episode with an invisible or malformed label
   instead of filling it.
2. **Point set.** Sample 500 vertices per object from the public scan with seed 0,
   using the kit's `sample_vertices`. This is the kit's canonical set
   (**Official**, `KIT/v2dlb/vertices.py:36-37,50-60`).
3. **Score.** Call the kit's `episode_metric` with `align_initial_pose=True`, the
   Track 3 setting (**Official**, `KIT/v2dlb/object_pose.py:296-315`,
   `KIT/metric_code/track_3/AUC.py:221`). Report the five metrics per episode.
   The equal-weight mean over episodes is reported for development and for
   validation separately, never pooled.
4. **Incomplete trajectories.** The official metrics need a pose in every frame.
   A perception output with gaps is scored twice and both numbers are labelled:
   over valid frames only, and with every missing frame counted as a miss at
   every threshold. Coverage is always printed beside them. A trajectory whose
   slot-0 pose at frame 0 is missing cannot be aligned and is reported as
   unscorable.
5. **`floor`.** Build the resting trajectory from the labels themselves: every
   object keeps its frame-0 pose. Score it. The result says how much of each
   metric a rollout earns when nothing moves, per public episode and per task.
6. **`oracle`.** Re-score after each substitution, alone:

   | Substitution | Fitted from labels | Gain attributed to |
   |---|---|---|
   | Scale | One scale factor about slot 0's start | Metric scale |
   | Anchor | One rigid transform fitted over all frames and objects, replacing the frame-0 alignment | Slot 0's frame-0 pose |
   | Body frame | One constant transform per object, applied on the object side | Mesh frame and symmetry |
   | Time | One integer frame shift within a small window | Frame timing and lag |
   | Rest | Frames the labels show at rest replaced by the frame-0 pose | Drift and disturbance of resting objects |

   The gain from each substitution ranks the error sources for that episode. The
   substitutions are diagnostic fits. None is ever applied to an output.
7. **`agreement`.** For a public episode with a resting rollout and one or more
   trained rollouts, compute two rankings: by label score, and by the label-free
   self-score that [policy_rollout](../policy_rollout/README.md) writes. Report
   how often they agree, and the smallest self-score margin above which they
   always agreed. That margin is the evidence for the selection setting in
   [entry tiers](../../../docs/design/entry-tiers.md).
8. **`mesh`.** Call the kit's CD-O code on a reconstructed mesh and the public
   scan (**Official**, `KIT/v2dlb/object_mesh.py`). Optionally compare a named
   baseline reconstruction on the same objects; unavailable spoon references
   remain null and are reported in coverage counts.
9. **`frame-convention`.** For each public scan: the origin's offset from the
   vertex centroid, from the bounding-box centre and from the lowest point. For
   each public development episode: which body axis is closest to world up at
   frame 0, and the origin's height. The report states whether the ten objects
   share a convention. It informs the provisional `wooden_spoon` frame
   ([error budget](../../../docs/design/error-budget.md)).

## Relation to the earlier metric protocol

`reports/metric-protocol.md` and `track3/metrics/pose.py` were frozen on
2026-09-24, before the kit existed. They defined their own alignment and
summaries. The kit's code is now the headline metric, because it is what the
leaderboard runs. The earlier protocol's coverage counts and lagged relative
motion remain useful as supplementary diagnostics and keep their names. Where
the two disagree, the kit wins.

## Outputs

`outputs/public_diagnostics/<operation>/<run-id>/`. This slice reports across
episodes, so it does not use the per-episode directory layout.

| File | Content |
|---|---|
| `report.json` | One row per episode and metric; for `oracle`, one row per substitution; coverage; the input runs and their hashes |
| `report.md` | The same as tables, per task and per split |
| `run.json` | Request, inputs with hashes, status, provenance tag |

## Configuration and authorization

`runtime.json` holds one flag per operation, all false, and a separate
`validation_authorized: false`. A request names the episodes and the input runs.
Development episodes are accepted by default once an operation is authorized.
Validation episodes also need `validation_authorized` and a request field naming
the frozen settings record, so validation is scored once and on purpose.
Episode 41 stays excluded from every request until the user says otherwise.

## Dependencies

NumPy, `pyarrow` to read the public Parquet files, and `trimesh` for the vertex
sampler and the mesh metric. The locked MoGe2 environment has no `pyarrow`. Two
candidate hosts, neither chosen:

| Host | Status |
|---|---|
| The external Python the data audit used, reused read-only | Has `pyarrow`. `trimesh` unverified. |
| The proposed submission environment | Has all three by the kit's requirements. Not created; needs approval. |

`mesh` itself no longer imports `pyarrow`; its identity inventory is JSON and
its geometry is GLB. This source change does not establish environment readiness
or authorize running any diagnostic.

## Resources

CPU only. `floor` over 20 episodes is a few thousand pose rows. Unmeasured.

## Failure handling

An unscorable input produces a row that says why. A failed operation keeps its
partial report with status `failed`.

## Separation of public labels

This is the slice that reads them. Every run carries `public_label_diagnostic`,
except `mesh` and the origin part of `frame-convention`, which carry
`public_asset_diagnostic`. Both tags bar a run from feeding a submission
([artifact contracts](../../../docs/design/artifact-contracts.md)), and no slice
lists this slice's outputs as an input.

A person may read a report and change a setting in another slice's request. That
is parameter selection on development data, and the setting is recorded in that
slice's request with a note of which report motivated it. Settings are frozen
before validation episodes are scored.

## Acceptance evidence

Mesh baseline comparison and missing-reference coverage are implemented but have
not run. No tests, smoke checks, validation commands, project imports, metric
execution or installation accompanied these changes; all runtime flags remain
false.

Once runs are authorized: `floor` reproduces a perfect score when given the
labels themselves in place of the resting trajectory, and a known injected
offset changes each metric by the amount the
[error budget](../../../docs/design/error-budget.md) derives.

## Open questions

| Question | Who can answer |
|---|---|
| Does the kit's canonical vertex set equal the one the leaderboard uses? The kit says the evaluator's own sampler differs and small AUC discrepancies are expected (`KIT/v2dlb/vertices.py:10-15`). | Organisers |
| Which host environment? | The user, at authorization |
