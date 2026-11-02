# Track 3 extended abstract

Anonymous, two-page CoRL-workshop abstract about video-derived primitive meshes
and static object poses. The packaged mesh comparison uses matching kit seeds:
**0.9257 cm per object versus 1.0091 cm for boxes**. The **0.926 cm** cross-fitted
score is selection evidence on reused public scans, not an independent test.
Stereo shrinkage is an exploratory, post-result pose choice; both trajectory
success rates remain zero, and the controls do not demonstrate calibration benefit.
MoGe-2 supplies predicted depth, not measured depth. No metric, submission file,
or model changed in this editorial revision.

## Build and figures

From the checkout root, the normal build is:

```bash
bash scripts/report/build_pdf.sh docs/report/track3/main.tex
```

This writes `main.pdf`, `main.log` and `main.blg` beside the source and uses the
existing Tectonic cache. The anonymous template and 10 pt body font are unchanged.
For the restricted editorial job, the unchanged script and report inputs were
copied to `.runtime/tmp/d3-plain/build/`; its binary is a symlink to the existing
Tectonic executable and its cache is a separate copy. The exact command above ran
inside that scratch mirror. The resulting PDF and logs were copied to
`.runtime/tmp/d3-plain/main.{pdf,log,blg}`; the checkout's previous build products
were not overwritten. Rebuild that mirror after an edit with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 .runtime/tmp/d3-plain/build.py
```

Regenerate the three included PDFs on CPU from saved data:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.runtime/tmp/d3-plain/matplotlib"
.runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py
.runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py --pipeline
.runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py --pose-offsets
```

Each command accepts `--preview <png>`. Outputs are respectively
`figures/d1-v2-qualitative.pdf`, `figures/pipeline.pdf`, and
`figures/d2-slot1-offset.pdf`. The existing `--bars` option still draws the unused
legacy score chart; that PDF is not included or regenerated in this revision.
Plotting functions take arrays and plain values; their readers load saved records.
There is no scoring, video decoding, training or simulation in figure generation.

## Claim-to-source map

Paths below are relative to the checkout root. Select array entries by name,
not by position. Short source names:

- **Rules:** `outputs/mesh_eval/rules/summary.json`; **Handle** is its
  `ranking[rule="R2 vessel-handle"]` entry.
- **Pairs:** `outputs/mesh_eval/pairs/summary.json`.
- **Scale:** `outputs/pose_insurance/scale/summary.json`.
- **Shrink:** `outputs/pose_insurance/shrink/summary.json`.
- **Build:** `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/`.
- **Fitted scores:** the unique `outputs/mesh_eval/*/v2-cdo.json` file;
  **Box scores:** the unique `outputs/mesh_eval/*/v1-box.json` file. These
  selectors match exactly one file each. The figure code requires uniqueness.

Displayed values retain their recorded precision and rounding. Four-decimal
packaged results and three-decimal cross-fitted results describe different
sampling procedures, not alternate rounding of the same statistic.

| Current claim | Recorded source and interpretation |
| --- | --- |
| D1 meshes and D2 policy-rollout poses are scored separately; CD-O removes placement and scale under the mesh budget | Read-only kit `upstream/v2d_submission_kit/v2d_submission_kit/tracks/track_3/README.md` and mesh metric implementation; paper citation `challenge`. The scan-use restriction is also in the organisers' FAQ (`faq` in `references.bib`). |
| MoGe-2 predicted depth; above-table clusters; manual object naming; omit merged/clipped observations; normalisation preserves distortion; medians over frames then episodes | `track3/features/mesh_extents/README.md`, `_geometry.py:extents`, `measure.py`, and `build.py`. **Build** `run.json:request.measure_runs` pins the consumed measurements; each measurement records manual assignments and accepted frames. `assignments.json` holds names and cluster IDs. No measured-depth accuracy claim is made. |
| Cylinder-plus-handle definition: horizontal long/short ratio > 1.30; circular body diameter = short extent, full height; closed bar carries the long-axis excess, height 60%, width 15% of short extent, overlap 5% of short extent; otherwise elliptical cylinder; non-vessels remain boxes; internal surfaces retained | `track3/features/mesh_extents/_shapes.py:vessel_with_handle,cylinder`; `_rule_scores.py:shape_for`; `rules.json:rules[name="R2 vessel-handle"]`; `build.py` implements the strict `pooled[0]/pooled[1] > request.aspect_threshold` comparison. **Build** `run.json:request.aspect_threshold=1.3,handle_height_fraction=0.6,shape="vessel_rule",shape_overrides={}`. The bar length is `long-short + 0.05*short`, with the overlap placed inside the body, so total extent is preserved. Concatenation retains internal faces. |
| Three categories, two folds of five; select parameters on one fold and score on the other; acceptance >0.03 cm mean gain, <=0.15 cm worst regression, no pair-mean regression | `docs/chain/mesh-selection-protocol.md`, “Rules, not objects”, “Two-fold leave-objects-out”, and “Adoption”; `_rule_scores.py:CATEGORIES,FOLDS,select`. **Rules** records the fixed protocol hash and decisions. Cross-fitting concerns parameters within a rule; rule-family selection still uses these previously inspected scans. |
| Cylinder and cylinder-plus-handle rules pass; handle rule selected; tool-head rule fails gain criterion | **Rules** `ranking[rule="R1 vessel-cylinder"/"R2 vessel-handle"/"R3 thin-tool-head"].{gates,adopt}`. Cylinder-plus-handle ranks first; tool-head `gain_gt_0.03_cm=false`. |
| Packaged boxes: 1.0091 object / 1.2547 pair cm; packaged handle meshes: 0.9257 / 1.0850 cm | **Pairs** `means.v1={per_object_cm:1.0091025260092832,pair_weighted_cm:1.254731056017533}`; `means.v2={per_object_cm:0.9257250385709256,pair_weighted_cm:1.085023637656605}`. Both use each entry's own exact kit seed, recorded in `pairs[].seed`. Group scores by object, average within object, then equally over ten objects for object means. Pair means average all 26 covered `(episode,slot,object)` entries equally. |
| Cross-fitted selection: 0.926 object / 1.087 pair cm | **Handle** `cross_fitted_mean_cm=0.9257219019644454,pair_weighted_mean_cm=1.0874159144755273`. Each object has a three-seed score; pair weighting reuses it according to **Rules** `pair_counts`, rather than independently resampling every entry. Seeds derive from episodes 1, 3 and 5, not training replications. |
| Cube baseline: 2.523 object / 2.772 pair cm | `outputs/mesh_eval/baseline-cube.json:mean_cd_o_cm=2.523`; the unique `outputs/mesh_eval/*/d1-v1-cdo.json:["Cube (floor)"].pair_weighted_mean_cm=2.772`. Pair weighting reuses object scores, as in the selection block of Table 1. |
| Worst cross-fitted regression 0.106 cm (pitcher); mean gain 0.082 cm; sample SD 0.180, SE 0.057 | **Handle** `worst_held_out_regression_cm=0.10599337236949369`. Object gains = **Rules** `baseline_per_object_cm[name]` minus **Handle** `per_object_cm[name]`. Mean **0.08191885568138821**, sample SD (denominator 9) **0.18012167185925354**, SE = SD/sqrt(10) **0.056959473903269685**. These are descriptive statistics on adaptively selected, reused objects. |
| Scans reused in earlier box/ellipsoid choice and episode 39 measurement thresholds; rule design followed the pot box losing to a cube | `docs/chain/SUBMISSION.md`, “D1 v0”; `docs/chain/IDEAS.md`, idea 15; `docs/chain/mesh-selection-protocol.md` opening disclosure. The later fixed folds do not undo this prior development use. |
| Ten scanned objects; spoon unscanned, box retained; ten of 36 pairs excluded | **Pairs** `covered_pairs=26,total_pairs=36,unscored` (ten spoon entries); **Build** `objects.json:objects.wooden_spoon.shape="box"`. Both reported means exclude the spoon. |
| FlashCHORD simulates stationary video-sized boxes with parked hands, rather than the submitted handle meshes | `track3/features/pose_insurance/README.md`, `outputs/pose_insurance/v1/report.md`, and task/scene producer records. The D2 lineage admits the earlier video-box build, while **Build** provides the D1 handle-rule meshes. Paper citation `flashchord` identifies the pinned source. |
| First achieved pose of first object, including rotation, aligns to its first reference mocap pose; second object's offset scored in first object's body frame; camera-to-mocap heading unknown | `docs/chain/pose-placement-protocol.md`, “Why placement is scored in slot 0's body frame”; kit `metric_code/track_3/AUC.py:_warp_to_reference_start`. The reference pose is a scoring input, not a scene-construction input. The protocol's frame audit records that no camera-to-mocap transform is released. |
| Earlier rule: video centroids/defaults with shared-cluster stacking; one measurable confirmation, post-result scene-wise acceptance, reused training analogue | `docs/chain/pose-placement-protocol.md`, candidate definition and its two final post-result decision sections; `outputs/pose_insurance/placement/confirmation.md`; `outputs/pose_insurance/v2/physical_checks.json`. The measurable confirmation was public episode 2; public episode 27 had separate clusters. Training episode 26 informed selection for submission episode 28. Earlier rule in Table 2/**Scale** means stacking on training episodes 0 and 26, earlier placement on the other six, including 31. |
| Table 2 earlier rule: AUC 0.2688, SP-SR 0, RPE 39.36 cm, MPPE 22.82 cm; stereo shrinkage: 0.2690, 0, 35.34, 20.84 | **Scale** `means.incumbent` and `means.S1`. Raw earlier RPE/MPPE **39.36384551739198 / 22.818618462986215**; stereo **35.33558264341916 / 20.843564644963294**. Equal means over training episodes 0, 11, 12, 18, 21, 23, 26 and 31. **Shrink** `means8.I/S1` reproduces these records. |
| Table 2 uniform shrinkage: 0.2690, 0, 35.28, 20.80; zero offset: 0.2748, 0, 19.54, 14.13 | **Shrink** `means8.G`: raw `(AUC,SP_SR,RPE_cm,MPPE_cm)=(0.2690123323497656,0,35.27602887377869,20.801243810949728)`; `means8.C0=(0.27477472005330295,0,19.542964153582133,14.133047790824333)`. |
| Reference static diagnostic: AUC 0.5319, SP-SR 0.625, RPE 15.11 cm, MPPE 7.40 cm | `outputs/pose_insurance/placement/summary.json:oracle_static` / `summary.md`, equal means over the same eight episodes. Holds reference first poses for diagnosis only; this is not a submitted policy. |
| Both success rates zero for video-derived rows; MP-SR always zero, including reference static; metric definitions and thresholds | **Scale** and **Shrink** per-episode and aggregate `SP_SR,MP_SR`; reference-static row in placement summary. Kit metric functions and Track 3 README define AUC, SP-SR (mean <=10 cm/0.5 rad), MP-SR (all-object <3 cm/30 degrees), RPE and six-attached-point MPPE. Thresholds are unchanged from the previous paper. These diagnostics use scan surface samples in place of official evaluation vertices. |
| Stereo cameras and estimator; training factors 0.819–0.923; second-object table-plane offset only; heights, rotations and sizes unchanged | `track3/features/stereo_depth/README.md:Inputs` names `ego_cam_c` and `ego_cam_b`. `docs/chain/pose-scale-protocol.md:Scale estimate,Candidates` fixes camera C left, B right, median stereo/monocular ratios over jointly valid pixels, then median over observed frames. **Scale** `stereo[episode].s` ranges **0.8185996684784886–0.9233432798833405** on the eight training episodes. The scale is not independently validated. |
| Training mocap distances motivated shrinkage; tuning episodes 39 and 42 have one object; adoption after results and candidate-wise single-object preservation check | `docs/chain/pose-scale-protocol.md:Disclosure,Decision after the train result`. **Scale** `invariance.comparisons["39:S1"/"42:S1"]` preserves every metric exactly. The original job's global rejection reading retained the earlier rule; the later lead decision adopted stereo shrinkage under a candidate-wise reading. The paper calls this exploratory rather than defending it as confirmatory evidence. |
| Uniform factor 0.857; stereo fails calibration-benefit criterion; 0.04 cm MPPE gap below 0.25 cm margin does not establish equivalence | **Shrink** `assignments.s_bar=0.8569522079777245`; `means8.S1.MPPE_cm-means8.G.MPPE_cm=0.04232083401356945` → **0.04**. `verdicts["1"]` and `docs/chain/pose-shrink-protocol.md:Verdicts` record the unmet calibration-benefit criterion and practical margin. The controls establish neither causal calibration benefit nor equivalence. |
| Zero offset overlaps boxes; shorter wrongly directed offsets can lower error | `docs/chain/pose-shrink-protocol.md:Candidates` and **Shrink** `verdicts["2"]`. This is a kinematic diagnostic, not an adopted scene. Possible shrinkage benefit from wrong direction is a disclosed interpretation, not a causal finding. |
| Clearance-limited follow-up: 1 cm footprint separation; 17.02 cm MPPE; AUC -0.00018; episode-31 MPPE +0.69 cm >0.5 cm limit | `docs/chain/pose-shrink-protocol.md:Pre-registered follow-up H`; **Shrink** `means8.H.MPPE_cm=17.020548795924775`, `H.gates.guarded_changes.AUC=-0.00018028846153844702`, `episodes.H["31"].metrics.MPPE_cm-episodes.S1["31"].metrics.MPPE_cm=0.6933918802549996`. `H.gates.passed=false`; the follow-up was not packed. |
| Direction errors about 2–130 degrees; evaluation calibration absent, public calibration assumed | **Scale** `episodes.S1[episode].frame0.direction_error_deg` spans **2.464691495435456–129.39456399854512**. `docs/chain/pose-scale-protocol.md` and `outputs/pose_insurance/v3/report.md:Caveats and provenance` disclose the calibration assumption. The undefined residual-ratio sentence was removed from the paper. |
| Physical checks on 13 scenes: intended-pose error <=0.049 cm and drift <=0.048 cm | `outputs/pose_insurance/v3/audit.json`: `accepted` has 13 episodes; `maximum_first_pose_cm=0.048607143738936404`, `maximum_drift_cm=0.04769638180732727`. These are errors against intended initialisation / first exported pose, not motion-capture accuracy. No physical evaluator was rerun for this revision. |
| Earlier policy retained on stacked episodes 1 and 28, single-object episodes 35–38, and one scene after unresolved numerical failure; submission factors 0.768–0.948 | `outputs/pose_insurance/v3/audit.json:unchanged=[1,5,28,35,36,37,38]`; `report.md` explains the no-ops and episode-5 fallback. `outputs/pose_insurance/shrink/episode5_diagnosis.{md,json}` records non-finite state without finding its origin. `v3/stereo_*.json:s` spans **0.7678335235402176–0.947996387351935**. The paper compresses episode 5 to one sentence. |
| No demonstrated manipulation or official-score gain; missing concavities, manual identities, coordinate uncertainty and unscanned spoon limit generality | Static trajectory construction and diagnostic sampling described above, **Build** primitive definitions, manual `assignments.json`, and the missing heading/calibration disclosures. This is a limitation statement, not an unmeasured performance claim. |

## Figure provenance

Figures follow reading order after moving the pipeline onto page 1: **Figure 1**
is the qualitative panel, **Figure 2** the pipeline, and **Figure 3** the offset
plot. These last two numbers are interchanged relative to the reviewed version.

Figure 1 reads the fitted-score files above, not cross-fitted selection scores.
It displays **ours 1.328 / box 1.818 cm** for the pot,
**ours 1.683 / box 1.577 cm** for the pitcher, and
**ours 2.115 / box 2.115 cm** for the rack. Raw saved scores are
**1.3277 / 1.8182**, **1.6829 / 1.5769**, and **2.1148 / 2.1148**.
No rescoring is involved; the existing three-seed fitted scores remain unchanged.

RGB comes from the left half of each measurement's `overlay_000000.png`, the
remapped RGB before the overlay. The assigned mask uses
`labels_000000.npz:label in [cluster_id+1]`, with a 10% crop margin.
The figure checks these assignments against `assignments.json`:

| Object | Frame-0 producer under `outputs/mesh_extents/public/` | Assignment key and cluster IDs |
| --- | --- | --- |
| Pot | `episode_000042/20261101T103715.090424Z-d410dfac/run.json` | `public:42`, `white_pot=[0]` |
| Pitcher | `episode_000021/20261101T103707.151959Z-05bf6bac/run.json` | `public:21`, `water_pitcher=[0]` |
| Dish rack | `episode_000012/20261101T103703.949004Z-546c56c2/run.json` | `public:12`, `plastic_dish_rack=[0,2,3,4,5,6]` |

Submitted mesh arrays are the fitted-score `objects[object=name].candidate`
files in **Build** `objects/`. Public scans come from the read-only snapshot
`datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh/<name>/<name>.glb`.
Only these three scans are displayed. All scan triangles are shown, whereas
recorded CD-O uses the kit mesh budget. No display operation changes a candidate.

Display-only scan rotations map `(x,y,z)` to `(z,-y,x)` for pot/rack and
`(y,-x,z)` for pitcher. Candidates retain their axes. Each mesh is independently
centred and uniformly scaled by its longest extent, with the same orthographic
camera (elevation 25 degrees, azimuth -65 degrees), light and limits. This is
not metric-size or frame-camera alignment. The canvas remains 3.18 by 3.05 inches,
with vector text and mesh surfaces rasterised at 300 dpi.

The pipeline is a schematic of the two described branches, with manual naming
visible and stereo cameras identified; it runs no reconstruction. The offset
plot reads **Scale** `episodes.incumbent/S1[episode].frame0.offset_error_cm` for
episodes 0, 11, 12, 18, 21, 23, 26 and 31. All sixteen bar heights are unchanged.
Its legend says “earlier rule” and “stereo shrinkage”; its y label is
“Frame-0 offset error of second object (cm)”, wrapped across two lines to fit.
The 9 cm surface-point threshold and its caption explanation were removed,
because they concern a different quantity. Stars mark the unchanged stacked
training episodes 0 and 26.

## Cold-review response and inspection

- **2a:** The abstract states zero for both trajectory success rates and keeps
  the uniform-shrinkage comparison and absence of demonstrated calibration benefit.
- **2b:** Adoption is explicitly exploratory and post-result. Two plain sentences
  replace the procedural defence. The failed clearance-limited follow-up remains.
- **2c:** The abstract compares packaged meshes and boxes with matching per-pair
  seeds. Table 1 separates packaged scoring from selection estimates and defines
  both averaging schemes. The reused-scan disclosure and descriptive uncertainty
  remain; none of the reported numbers changed.
- **2d:** The paper identifies the achieved/reference alignment poses and the
  first object's body frame, names both stereo cameras, and retains the unknown
  heading, unvalidated scale and absent evaluation calibration. Residual ratios
  were cut rather than leaving an undefined numerator/denominator.
- **2e:** The horizontal aspect ratio and resulting body/bar geometry are
  defined from `_shapes.py`, `_rule_scores.py`, `rules.json` and the recorded build.
- **3:** Descriptive names replace internal labels. Episode lists use words or
  ranges; no amendment numbers or quarter-offset diagnostic remain in the paper.
- **4:** Figure headings, explicit fitted score labels and descriptive legends
  replace codes; the offset threshold line is gone; manual naming is visible in
  the pipeline on page 1. Figure numbering follows the new reading order.
- **5:** Protocol history and numerical-state detail are compressed; episode 5
  occupies one sentence. The threshold-sensitivity sentence was cut. Both packaged
  rows remain to show the requested matched comparison directly.

No requested review point was declined. The optional suggestion to drop the
packaged mesh row was not used: retaining it makes the matched comparison visible.
The final PDF has **three pages: two main pages and references only on page 3**.
All three final 110 dpi pages were visually inspected; figures, text, labels and
references fit without clipping. The build has no overfull boxes or undefined
references. Its sole TeX error is the known microtype tracking warning; existing
font-encoding and underfull-box warnings remain.

Build/page evidence is in `.runtime/tmp/d3-plain/main.log` and `build.log`.
The figure equivalence and numeric evidence record is
`.runtime/tmp/d3-plain/editorial-evidence.json`. Syntax compilation, a CPU import,
old-versus-new bar equivalence and read-only arithmetic on recorded scores were
performed. No test suite, new experiment, dependency installation, GPU work,
submission change or excluded-episode access was performed.

Render the scratch PDF with the existing Poppler installation:

```bash
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png \
  .runtime/tmp/d3-plain/main.pdf .runtime/tmp/d3-plain/p
```

Rendered pages: `.runtime/tmp/d3-plain/p-1.png`, `p-2.png`, `p-3.png`.
The scratch directory also contains the before-edit sources, build mirror,
preview PNGs, and command/evidence scripts. Historical experimental results stay
in their original recorded sources; removed paper claims are not new findings.
