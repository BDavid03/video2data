# Submission

Status: design only. No code exists for this slice. Nothing was packed, validated
or uploaded while writing this document.

This slice turns finished rollouts and meshes into the two files the leaderboards
accept, and sends them when the user says so. It wraps the organisers' Submission
Kit without changing it. Citations use `KIT/` for
`upstream/v2d_submission_kit/v2d_submission_kit/`; the full source reading is in
[submission-kit.md](../../../reports/contract-evidence/2026-09-30/submission-kit.md).

## Role in scoring

Every Track 3 score passes through this slice. It adds no accuracy, but it can
lose all of it: a single missing row fails the whole submission, and there is no
partial credit (**Official**, `KIT/metric_code/track_3/AUC.py:399-405`).

| File | Goes to | Scored on |
|---|---|---|
| Pose CSV, 9,120 rows | AUC, SP-SR, MP-SR, RPE and MPPE competitions | Object poses from the policy rollout of 20 evaluation episodes |
| Mesh Parquet | CD-O competition | One reconstructed mesh for each of 36 episode-object pairs |

## Operations

All five are disabled by default. `upload` is the only operation in the project
that sends data to an outside service, and it needs the user's confirmation every
time it runs.

| Operation | Produces | Runtime | GPU |
|---|---|---|---|
| `plan` | `submission.json`: the chosen rollout, checkpoint and mesh for every scored item | Submission environment | No |
| `check` | `check.json`: findings from opening every chosen input | Submission environment | No |
| `pack-poses` | `track3_poses.csv`, by running the kit's packer unchanged | Submission environment | No |
| `pack-meshes` | `track3_meshes.parquet`, by running the kit's mesh packer unchanged | Submission environment | No |
| `upload` | `upload-log.json` and a ledger entry, by running the kit's uploader unchanged | Submission environment, network | No |

## Inputs

| Input | Producer | Used for |
|---|---|---|
| `runs/rollout/evaluation/episode_XXXXXX/<run-id>/episode_XXXXXX.parquet` and `.json` | [policy_rollout](../policy_rollout/README.md) | Pose rows and provenance hashes |
| `outputs/object_mesh/evaluation/episode_XXXXXX/<run-id>/<object>.glb`, `mesh.json`, `run.json` | [object_mesh](../object_mesh/README.md) | CD-O geometry, reconstructed source, source episode and provenance |
| `KIT/data/track_3_sample_submission.csv` | Organisers | The row roster: 20 episodes, world `w000`, frames 0 to N-1 |
| `KIT/data/track_3_evaluation_objects.json` | Organisers | Object names in slot order |
| `KIT/data/track_3_mesh_sample_submission.parquet` | Organisers | Mesh row roster and vertex and face budget |
| Pushed Git commit URL | The user's GitHub repository | `code_commit_url` on every row |

`plan` refuses to continue when any scored episode or any episode-object pair has
no completed input. It requires `video_only` provenance on every selected input
and `mesh.json` `source = reconstructed` for CD-O. It never invents a pose or
substitutes another episode's rollout. A mesh from another evaluation episode of
the same object is permitted only after Q06 explicitly allows reuse; until then
`source_episode` must equal the pair's episode.

## Method

1. **Plan.** Choose one completed rollout per scored episode and one mesh per
   episode-object pair, and record their hashes. Require `video_only` provenance
   and reconstructed CD-O meshes, and enforce the source-episode reuse rule
   above. Require a clean worktree whose
   HEAD is pushed, and build the `/commit/<sha>` URL from it. The packer accepts
   any URL of the right shape and never checks that the commit exists
   (**Official**, `KIT/v2dlb/report_schema.py:55`), so this slice is the only
   place the link between code and submission is enforced (**Design**).
2. **Stage.** Copy exactly the 20 chosen Parquet files into a new, flat
   `results/` folder inside the submission directory. The packer reads every
   Parquet under its results folder recursively. It aborts on unreadable or
   unrecognised files and on duplicate episode files; valid trajectories for
   non-scored episodes are ignored with a note. The kit's preflight looks only
   at the top level (**Official**, `KIT/tools/pack_submission.py:57-64,221-224,231-233`,
   `KIT/tools/preflight.py:197`). Staging
   makes the two agree.
3. **Check.** Open each staged file and test the things the kit does not:

   | Check | Why the kit is not enough |
   |---|---|
   | Frames 0 to N-1 present for world 0 and every slot | The kit's preflight reads file names only (`KIT/tools/preflight.py:164-213`) |
   | `object_body_name` present and equal to the roster order | Without names the packer skips the order check and a swap passes silently (`KIT/tools/pack_submission.py:299-312`) |
   | One body per object | `object_body_id` is used as the object slot (`KIT/v2dlb/submission_parquet.py:149`) |
   | Every world valid, not only world 0 | One invalid world rejects the episode (`KIT/tools/pack_submission.py:266-273`) |
   | Positions plausible in metres | The unit key is not read by the packer; scale error is scored in full (`KIT/tracks/track_3/README.md:36`, `KIT/v2dlb/object_pose.py:141-142`) |
   | Quaternion norm not near zero | The scorer normalises quaternions and rejects norms below 1e-8; a rejected upload still costs quota (`KIT/metric_code/track_3/AUC.py:209-216,418-419`) |
   | All three provenance hashes present in Parquet metadata | `eval_code_sha256` has no fallback (`KIT/tools/pack_submission.py:380-388`) |
   | Input provenance is `video_only`; CD-O `mesh.json` has `source = reconstructed` | Diagnostic tags are refused; public scans used for tracking are not CD-O reconstructions. Recheck `source_episode` against Q06 reuse authorization. |
   | Recorded `eval_code_sha256` equals the evaluator digest of the installed LF sources at `aee95f25`, and matches any published allowlist | The digest covers raw bytes of four evaluation modules. Installing from the Windows CRLF reference produces a different digest (`FC/src/flash_chord/evaluation/harness.py:123-135`). |
   | Kit scripts byte-identical to `KIT_MANIFEST.json` | The packer and uploader hashes are written into every row |

4. **Pack.** Run the kit's packers unchanged, from the kit root. The SKILL file in
   the kit warns against building the CSV by hand, because frame alignment and
   object order are easy to get wrong in ways that score badly without failing.
5. **Upload.** Only on the user's explicit instruction for that specific file.
   Run the kit's uploader, then write the outcome of every competition leg to the
   ledger. On a partial failure, retry only the failed legs with `--only`.

The kit's LF scripts are kept byte-identical to their manifest. Separately,
`eval_code_sha256` covers the installed FlashCHORD evaluation modules, not just
the evaluator script. The Windows reference checkout under `upstream/` has CRLF
in those modules (review E-04). The policy environment must be installed from an
LF checkout of the pinned commit with CRLF conversion disabled; preserving the
kit alone does not fix a runtime-source hash mismatch. These hash checks remain
future, disabled operations.

## Outputs

`submissions/<submission-id>/`, where the id is a UTC timestamp and a short label.
A submission directory is never overwritten.

| File | Content |
|---|---|
| `submission.json` | Chosen inputs with hashes, commit URL, kit archive hash, FlashCHORD commit, operation status |
| `run.json` | Shared run record, inherited provenance and operation statuses |
| `results/episode_XXXXXX.parquet` | The 20 staged rollouts, unchanged copies |
| `meshes/episode_XXXXXX/<object>.glb` | The 36 staged meshes, unchanged copies |
| `check.json` | One finding per check and per episode, with a blocking flag |
| `track3_poses.csv` | Packer output: 14 columns, positions and quaternions to 6 decimals, scalar-last quaternions |
| `track3_meshes.parquet` | Mesh packer output: `row_id, x, y, z, code_commit_url` |
| `upload-log.json` | Per-competition result, message, time and remaining quota as reported |

`submissions/ledger.json` lists every upload attempt across all submissions.

## Configuration and authorization

`runtime.json` holds five flags, all false: `plan_authorized`, `check_authorized`,
`pack_poses_authorized`, `pack_meshes_authorized` and `upload_authorized`. `upload` also
needs a request file naming the exact submission directory, so an approval for one
file cannot send another.

The Kaggle token is read from the `KAGGLE_API_TOKEN` environment variable of the
user's shell. It is never written to a file in this repository or to a log.
The uploader also supports `kaggle auth login` and `~/.kaggle/access_token`;
legacy `kaggle.json` is unsupported in Kaggle 2 or newer
(**Official**, `KIT/submit/v2d_submit.py:29-30,76-77`). This design chooses the
environment variable because the other supported paths store credentials in
the user profile outside the project root.

## Dependencies

Proposed, not installed. Each needs the user's confirmation.

| Package | Needed by | Source |
|---|---|---|
| `numpy>=1.24`, `pandas>=2.0`, `pyarrow>=14.0` | Packers, `check` | `KIT/requirements.txt` |
| `trimesh>=4.0`, `fast_simplification>=0.1` | Mesh packer | `KIT/requirements.txt` |
| `kaggle>=2.0`, `kagglesdk` | Uploader only | `KIT/requirements.txt` |

Proposed location: a separate environment at `.runtime/submission-python313`, so
the Kaggle client is never importable from a perception or training process.
Importing `kaggle` may authenticate as a side effect
(`KIT/submit/v2d_submit.py:65-66`).

## Resources

CPU only. The pose CSV in the kit is 4.6 MB and the mesh sample is 1.5 MB.
Estimated disk per submission directory is under 100 MB, unmeasured.

## Failure handling

A failed operation leaves its partial directory in place with status `failed` and
the error text. `upload` records each competition leg separately, because five
legs can end in different states.

## Separation of public labels

This slice reads evaluation rollouts and meshes only. It never opens public pose
labels. `plan` and `check` accept only completed `video_only` inputs. Both
`public_label_diagnostic` and `public_asset_diagnostic`, and missing tags, are
rejected. CD-O meshes must additionally record `source = reconstructed`; a
`public_scan` mesh remains ineligible even when its run is `video_only`.

## Quota

| Fact | Source |
|---|---|
| Five submissions per week, unlimited in the final three days | **Official**, challenge site and `KIT/config/tracks.json:304,333` |
| One pose upload uses one submission in each of the five pose competitions | **Official**, `KIT/submit/v2d_submit.py:426-524` |
| A rejected submission still costs quota | **Official**, `KIT/skills/v2d-submit/SKILL.md:16` |
| Kaggle is configured for five per day, which is looser than the published rule | **Official**, `KIT/config/tracks.json:325`; the site's policy prevails (`:313-316`) |
| Two final submissions per competition | **Official**, `KIT/config/tracks.json:326` |

**Design.** The ledger enforces the stricter rule: at most five pose uploads and
five mesh uploads in any rolling seven days, until the organisers say when the
week resets.

## Acceptance evidence

Once runs are authorized: the packer accepts all 20 staged rollouts, the uploader's
dry run reports every competition as entered, and a first upload returns a score
on each of the six Kaggle leaderboards.

## Actions only the user can take

| Action | Why |
|---|---|
| Create a GitHub repository for this project and share it with the organisers | Every row carries a commit URL; the rules reference a repository shared with the organisers |
| Register the team through the form on the challenge site | Entry |
| Create a Kaggle API token and set `KAGGLE_API_TOKEN` | Uses the environment variable so credentials stay out of user-profile files; login and access-token-file authentication are also supported |
| Open all six Track 3 competitions and accept their rules | There is no API for accepting rules |
| Approve each upload | It is outward-facing and spends quota |

## Open questions

| Question | Who can answer |
|---|---|
| When does the submission week reset? | Organisers, Discord `#v2d-challenge` |
| Must the rollout come from the official FlashCHORD evaluator? The READMEs say to use it; the code does not check. | Organisers |
| Will the evaluator hash allowlists be filled later? They are empty today. | Organisers |
| Does Kaggle reject extra rows before scoring? | Organisers, or the first upload |
| Is the mesh budget 4,096 vertices and 4,096 faces, as the config says? The sample Parquet was not opened. | Reading the sample once `check` is authorized |
