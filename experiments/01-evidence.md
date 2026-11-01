# 01 — Evidence reconciliation

Status: complete as a metadata/source reconciliation, 2026-10-18. No campaign
was relaunched. No new model, label-array, metric or verification execution.
Owners: learning_experiments, public_diagnostics, submission and producers.

Subsequent source build: `campaign/evidence.py` adds disabled explicit metadata
inventory and conditional freeze handlers using these historical identities.
See [the build record](16-freeze-build.md) and [current build map](BUILDOUT.md).
No inventory/export operation was executed and no historical artifact changed.

## Evidence inventory

All artifact paths below are relative to the **canonical**
`/mnt/z/hugging-face/video2data/outputs/` (Windows `Z:/hugging-face/video2data/outputs/`).
They are existing artifacts, not outputs created in this implementation checkout.
Digests are identities recorded by the completed producers/report, not a new
integrity-check claim. Existing source/config hashes remain historical even when
source changes here. No weight or prediction arrays were opened for this review.

| Record | State and supported claim | Important limit |
|---|---|---|
| `learning_experiments/execute/20261018T105355.351623Z-37d0aa68/result.json` | Completed bounded campaign, 12 trials, zero failed, 200 teacher frames, 40 retained pairs | Four early frames/episode, not full clips; no holdout |
| `learning_experiments/tune/20261018T105658.305032Z-7407936a/selection.json` | Transformer `transformer-c4d255682d988b57440a753f`, tuning log-depth MAE 0.15815437453249034 | MoGe2 agreement, not independent measured depth |
| `learning_experiments/investigate/20261018T113050.188730Z-d4ce9207/result.json` | Completed 35 stage outcomes, two failed compiler attempts, nine new fits and reused incumbent, 1470.21 s | Counts are stages, not independent models; one seed |
| Same directory, `frozen-selection.json` | Equal log-depth ensemble `scratch_eager`/`scratch_bf16`, weights 0.5/0.5; tuning MAE 0.12395560418526001 | Research incumbent only; no production promotion |
| Same campaign, historical held-out results | Episodes 43/44 already assessed; frozen winner MAE 0.1521616316967859 | Consumed holdout; no later recipe or mixture selection from these outcomes |
| `learning_experiments/report/20261018T120125.303833Z-cb2e5071/` | Completed report, candidate CSV, coverage tables, comparison figure | Teacher agreement remains separate from geometry and robot behaviour |
| `opening_experiments/20261018T115633.072019Z-b1450593/result.json` | Rectification failed OpenCV `gemm` dimensional assertion | Later stages blocked |
| `opening_experiments/20261018T120039.470932Z-e0b15dc0/result.json` | Rectify/stereo/anchor/camera completed after contiguous calibration-array correction | Clusters failed `opening_not_quiet: an above-table cluster touches the image border`; hull and TSDF blocked |
| `scratch_depth/compile_experiment/20261018T115052.455224Z-8142557a/run.json` | TorchInductor failed `TritonMissing` | Backend did not execute |
| `scratch_depth/compile_experiment/20261018T115112.404037Z-099fd706/run.json` | CUDA 13.3 failed: CCCL rejected MSVC traditional preprocessor | Backend did not execute |
| `scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json` | External read-only toolchain completed one real batch, zero optimizer steps | Single-batch numerical/timing evidence only; see item 15 |
| `synthetic_depth/20261018T115837.781490Z-06a78723/` | Separate completed 128-sample procedural training-only pilot | Created after campaign; no synthetic mixing or learning benefit assessed |

No currently running stage is established by these finished records. Later
production geometry, hand/contact refinement, policy training and achieved robot
behaviour have no completed evidence in this inventory. Source presence is not
runtime success. The synthetic run's generic `video_only` tag is insufficient:
its explicit `data_origin: procedural_synthetic` and `training_only: true` govern
interpretation; do not present these samples as captured video evidence.

## Dataset, source and freeze identities

Dataset revision: `5f68335f3acc802033d1e80728c1633197521de8`.
Research grouping: train **0/11/12/18/21/23/26/31**, tuning **39/42**,
consumed final holdout **43/44**, excluded **41**. All cameras, frames, objects
and derivatives remain grouped by episode. The older outer public development /
validation split is a different partition; its seven validation episodes were
not used by this campaign, but their global untouched status is **not certified**
by this inventory. Do not automatically designate them a new holdout.

Expanded preparation retained 128 training plus 32 tuning pairs at 128×96;
scratch preparation retained only 128 training pairs. Holdout preparation lists
192 rows including prior train/tuning, with the scorer selecting 32 held-out
rows. Sixteen sparse frames/episode are spread across the clips, not contiguous
sequences. Baseline early-frame scores and expanded scores are not matched.

| Historical identity | SHA-256 recorded in evidence |
|---|---|
| Baseline selection | `9f75f0625755fb16af762f0c4499717659d6ce5b4f91ee52207204d40106d16a` |
| Expanded owning run | `198b8f8292a0aa1959e5a339e7e1cd4e2030ea3f0e77e4e43176db7369505357` |
| Expanded result | `5c7ba878341c2c1cc07d9502b3da86024a7da0cafa3d4f46cf853f57f3374886` |
| Frozen selection | `c25178a18f6ea8a295c4556483a4e493af9c404dfcecf23a26f00dcfe1f3d857` |
| Expanded dataset `scratch_depth/prepare/20261018T113353.167080Z-383fc5ae/dataset.json` | `35d66341ac1e625d872e807bafa5ddcf8478545d99ddfafc3542bde324ff71e1` |
| Train-only scratch dataset `.../20261018T113402.984050Z-d446c33e/dataset.json` | `0b52a8b3182e8d5a36ed4bd242c7c9f1a1d025b661d9ac48ea2315877b421131` |
| Holdout dataset `.../20261018T115239.341106Z-7d30e324/dataset.json` | `6993ea8a90b1a7efdb490aa46cf58e3a212b2f4f1b3f9a3f662cf80802832c1f` |
| Historical `investigate.py` | `d8176947012c48e5197a01fec94e48e758450c698eb542eb8f34bdf4741f65f7` |
| Historical `investigate.json` | `395e40522844e766b564c07566c800a0066b51c4e481c8d8f550ff16af2acf4f` |
| Historical source lock | `9cb4be3c5aa2275e7defb1436f1ee4e2d2595a12e605d38c20daea64e3984a20` |
| Synthetic dataset | `b221825f424030ed31dff9089ad0bcd66f85f11fcbee10a3035a32b029cee042` |

`investigate.py` writes frozen choices before held-out teacher preparation and
checks the frozen identity afterwards. Last tuning ensemble score begins
11:52:12 UTC; first held-out teacher begins 11:52:13 UTC. There is no separate
freeze timestamp field: source control flow and recorded ordering support the
freeze claim, not an independently signed timestamp. Preserve the historical
winner even when another candidate's historical holdout number looks preferable.

## Pinned evaluation and producer contract map

Reuse [the kit reading](../reports/contract-evidence/2026-09-30/submission-kit.md),
[FlashCHORD reading](../reports/contract-evidence/2026-09-30/flashchord-runtime.md)
and [artifact contracts](../docs/design/artifact-contracts.md).
Kit archive SHA is `a2de2aeed5821c28efedf36ddbe08a7e595f9b7d7298e4be2c07fe4665cf91b3`;
FlashCHORD commit is `aee95f25ae0943f335d8bb42fa95b6a9acfa266f`.
Online research never replaces their metric code.

| Boundary | Contract and source |
|---|---|
| Optical camera → world | x right, y down, z forward; world +z table-up. `world_T_object = world_T_camA @ camA_T_object`; camera_motion and object_pose own composition |
| Calibrated depth | Optical-axis z in metres; one saved pinhole grid/K after all 14 distortion coefficients. Grid identity must match masks/hands/stereo |
| Stereo → colour | C-left/B-right; one rectification call; `R1.T` to C optical then `inverse(C_T_A)` to A; nearest-depth projection, invalid holes remain invalid |
| Object body frame | Mesh/URDF root equals pose frame; no universal centroid convention established; spoon remains provisional |
| Quaternion boundary | Dataset and FlashCHORD demonstration use wxyz; rollout/scorer use xyzw; conversion is explicit in producer/diagnostic adapters |
| Time | Source frame index + integer PTS, timebase 1/10240; current clips nominally 20 Hz. FlashCHORD exports uniform FPS, so irregular PTS must be refused rather than silently resampled |
| Missing observations | NaN plus valid/observed masks; scene fills are separately recorded, never relabelled observations |
| Pose score | Pinned `v2dlb/object_pose.py`: one rigid alignment from slot 0/frame 0, no scale fit; metres input, RPE/MPPE centimetres |
| Submission | Achieved simulated object poses, exact episode/object roster and ordering; reconstruction targets alone are ineligible |
| Mesh score | CD-O registration removes Sim(3) mismatch; low CD-O does not establish metric collision dimensions or correct body origin |

Identified implementation gaps: spoon's absent public scan aborts pose diagnostics;
aggregate diagnostic reports omit missing-as-miss summaries; scene assembly does
not check PTS spacing before uniform export. Source corrections are tracked in
item 02. Unresolved pinned runtime ambiguity: simulation advances before recording
step 0 but exports `reference_frame=step+start_frame`; do not shift trajectories
without a fixed external contract. Public canonical scan vertices also differ
from evaluator simulator-shape samples according to pinned `v2dlb/vertices.py`.

## Authorization and disposition

Reconciliation is complete; measured geometry and achieved behaviour remain
unestablished. No consumed holdout was used to change a recipe. Current sandbox
allows writes only here and `/tmp`; canonical Z: is outside those roots and
approval is unavailable. Thus **new artifact-producing research is blocked**,
even where the user's research execution scope otherwise permits it. No local
runtime artifacts may substitute. Tests, validation commands, dependency
experiments, simulation, notebooks and uploads remain prohibited. New production
dependencies still need confirmation. Existing environments remain read-only.

Next: item 02 specifications and source fixes; later independent research can
continue. Required outcome evidence remains blocked by canonical write access,
admitted geometry support and, downstream, simulation permission/assets.
