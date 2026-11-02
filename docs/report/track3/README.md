# Track 3 extended abstract: D1 v3

Anonymous two-page CoRL-workshop abstract describing the packaged **peg-rack
rule (D1 v3)** and the unchanged D2 v3 static-pose entry. D1 now scores
**0.7711 cm per object / 0.8115 cm per pair**, versus the replaced
cylinder-plus-handle meshes at **0.9257 / 1.0850 cm** (boxes:
**1.0091 / 1.2547 cm**), with matching per-pair kit seeds. Round-2 cross-fitted
selection gives **0.773 cm** on reused public scans. The round was pre-registered,
but selecting among three nested rules is optimistic; video attributes were
chosen after the authors had already seen public scan renders.
Stereo shrinkage remains exploratory and post-result, both video-derived
trajectory success rates remain zero, and calibration benefit is unestablished.
MoGe-2 supplies predicted depth, not measured depth. This edit reports completed
runs; it does not rescore, repack or modify a submission.

## Build and figures

The normal build from the checkout root is:

```bash
bash scripts/report/build_pdf.sh docs/report/track3/main.tex
```

It writes PDF/logs beside the source and uses the existing Tectonic cache. To
respect this job's write scope, the unchanged script, report inputs and a copy
of that cache were placed in `.runtime/tmp/d3-fix/build/`; its Tectonic binary is
a symlink to the existing executable. The exact command above ran inside that
scratch mirror. Rebuild the mirror after an edit with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 .runtime/tmp/d3-fix/build.py
```

The resulting PDF/logs are copied to `.runtime/tmp/d3-fix/main.{pdf,log,blg}`.
The previous checkout build products were not overwritten. The anonymous
option, 10 pt body font and template margins are unchanged. Body paragraph
spacing is 0.25 baseline (3 pt), reduced from the template's 0.5 baseline (6 pt).
Figures and tables use inline minipages at paragraph boundaries; short method
blocks are kept together. This removes the critic's sentence continuations
below displaced floats without shrinking the body text.

Regenerate the included figures from saved data on CPU:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.runtime/tmp/d3-fix/matplotlib"
.runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py
.runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py --pipeline
.runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py --pose-offsets
```

Each accepts `--preview <png>`. Outputs are `figures/d1-v3-qualitative.pdf`,
`figures/pipeline.pdf` and `figures/d2-slot1-offset.pdf`. The previous
`d1-v2-qualitative.pdf` remains as an unused historical artifact. `--bars` draws
the unused round-1 score chart; it is not included or regenerated here.
Plotting functions take arrays and plain values; readers load saved records.
There is no scoring, video decoding, training or simulation in figure generation.

## Claim-to-source map

Paths are relative to the checkout root; select array entries by name rather
than position. Short source names:

- **Rules:** `outputs/mesh_eval/rules/summary.json`; **Handle** is its
  `ranking[rule="R2 vessel-handle"]` entry (round 1).
- **Round 2:** `outputs/mesh_eval/rules-round2/summary.json` and `summary.md`.
- **Pairs:** `outputs/mesh_eval/pairs/summary.json` (boxes versus v2).
- **Packed:** `outputs/mesh_eval/pairs/mesh-round2-v2-v3/summary.json` and `summary.md`.
- **Build:** `outputs/mesh_extents/build/20261102T031152.089350Z-ea88a295/`.
- **Veto:** `outputs/mesh_eval/rules-round2/veto.json`.
- **Scale:** `outputs/pose_insurance/scale/summary.json`.
- **Shrink:** `outputs/pose_insurance/shrink/summary.json`.

Table 1 separates exact packaged means (four decimals) from three-seed selection
estimates (three decimals). Pair means average the 26 covered episode--object
entries equally. Object means first average each object's entries, then average
the ten objects equally. Selection pair means reuse each object's three-seed
score with its roster count. The spoon is excluded from both means.

| Current or supporting claim | Recorded source and interpretation |
| --- | --- |
| Shipped D1 v3: exact object mean 0.7711 cm / pair mean 0.8115 cm; v2 0.9257 / 1.0850 cm | **Packed** `means.v3.per_object_cm=0.7710770332786878,pair_weighted_cm=0.8114951342417491`; `means.v2.per_object_cm=0.9257250385709256,pair_weighted_cm=1.085023637656605`. Every `pairs[].seed` is the kit's own per-entry seed; no seed or scorer constant changed. Values differ from cross-fitted development estimates. |
| Exact packed change: 19 improved pairs, seven unchanged, none worse | **Packed** `improved_pairs=19,unchanged_pairs=7,worsened_pairs=0`; independently count the signed `pairs[].difference_cm` entries. **Veto** repeats these counts and `veto_pass=true`. |
| Second round pre-registered before scores; two five-object folds retained | `docs/chain/mesh-selection-protocol.md`, “Amendment 1 (round 2)”, committed as `c7e2b566ab67dd49bbdb6a86b506bf00e6e6c049` at **2026-11-02 02:55:35 UTC** (`git show -s --format='%H %cI' c7e2b56` reports the equivalent +10:00 time). **Round 2** pins protocol SHA-256 `518976a9a1a765f3d6fdb18b80d38bff90285d93f973f244f632974fc14974b4`, attributes and queue hashes. The same protocol defines both five-object folds and parameter selection on the opposite fold. |
| Round-2 incumbent is the fixed-threshold cylinder-plus-handle rule, not its round-1 cross-fitted result | Protocol incumbent: threshold **1.3**, handle height **0.6**. **Round 2** `baseline_mean_cm=0.9250628448040409` (**0.92506**), `baseline_pair_mean_cm=1.087678420439796` (**1.08768**), and `round2_inputs.incumbent_scores`. Three kit seeds derive from episodes 1, 3 and 5. These fixed-threshold means differ from the round-1 cross-fitted **0.9257219019644454 / 1.0874159144755273**. |
| Round-2 acceptance: gain >0.03 cm, no object regression >0.15 cm, no pair-mean regression; lowest mean among passing nested rules wins; exact veto against packaged v2; no retuning or rule switching | Protocol amendment “Incumbent and adoption”; **Round 2** `ranking[].gates`, `winner_pending_exact_veto`; **Veto** `selected_rule="R6 peg-rack",gates={pair_weighted_cm:true,per_object_cm:true},veto_pass=true`. Both exact packed means had to be no worse than v2. The development summary remains labelled veto-pending; the separate veto record establishes acceptance. |
| Three nested rules: rim handle fails gain; hollow vessels and peg rack pass; peg rack has lowest cross-fitted mean, no object regression | **Round 2** `ranking[rule=...]`: rim handle **0.9206074175594295** (0.92061), gain gate false; hollow vessels **0.8455545563468723** (0.84555), pair **0.9559956660132265** (0.95600), all gates true; peg rack **0.7731489706385066** (Table 1 **0.773**), pair **0.8167541550356001** (**0.817**), all gates true. Both passing rules have `worst_held_out_regression_cm=0.0`. The rules add refinements cumulatively. |
| Shell wall/radius grid 0.02, 0.05, 0.10; thickness 0.10 selected on both folds; thicker walls not tried | Protocol “Video attributes and geometry”; `track3/features/mesh_extents/rules-round2.json`; **Round 2** hollow/peg `chosen_on_fold={A:0.1,B:0.1},in_sample_parameter=0.1`. Their `trials[].parameter` contain only the three listed fractions. **Build** `run.json:request.rule_parameter=0.1`. This is a grid endpoint, not an established thickness optimum. |
| Reproducible inherited cylinder-plus-handle body/bar geometry, rim handle, open shells and peg rack | `track3/features/mesh_extents/_shapes.py:cylinder,vessel_with_handle,_attach_vessel_handle,rim_handle,vessel_shell,peg_rack`; protocol geometry definitions and `rules-round2.json`. For horizontal long/short extents L/W >1.30, body diameter W and height H, bar dimensions `(L-W+0.05W,0.15W,0.60H)` with 0.05W overlap; otherwise elliptical cylinders, and boxes outside vessels. Rim-flat pot bar height **0.15H**, top flush; pitcher bar stays vertically centred. Shell radius W/2, wall=floor=fW/2, open mouth with annular rim, no mouth cap; unhandled shells stretch along L/W. Rack: **0.25H** slab, **two rows of six** box pegs, cell spacing L/6, peg length half-cell, width **0.3W**, rows span W with half-width margins, base overlap **0.01H**, top at full H. Concatenation retains internal faces; the shell body is connected and watertight. |
| Video attributes: pot rim-flat handle; cups, pot and pitcher open-topped; two rows of six rack tines; missing rack compartment/tray rim and pitcher spout | `track3/features/mesh_extents/attributes.json:objects.*` provides observations, camera/frame paths and hashes. Open mouths: cups in public episode 11, pot in public 42 and evaluation 5, pitcher in public 21. Rack evidence includes public 12 and unobstructed evaluation episodes **15 and 16, frame 0, ego_cam_a**. Its note explicitly records the omitted compartment and tray rim. Pitcher observation records the pouring lip; the constant-profile shell/bar construction omits a spout, as visible in Figure 1. |
| Attributes are not blind to public scans; selecting among three nested rules on ten reused scans is optimistic, not independent assessment | Attributes pin video-frame evidence, but do not erase prior scan viewing. Previous report commit `5594d55` already displayed public scan renders in Figure 1, before round-2 attribute selection; this follow-up explicitly requires that disclosure. The protocol amendment itself says family selection among three nested rules is optimistic and thickness cross-fitting does not cross-fit family selection. No claim of independent testing is made. |
| Earlier shape choices and episode-39 measurement thresholds also used these scans | `docs/chain/SUBMISSION.md`, “D1 v0”; `docs/chain/mesh-selection-protocol.md` opening disclosure; `docs/chain/IDEAS.md`, idea 15. Earlier box/ellipsoid choices and the pot-box/cube comparison preceded the new rules. |
| Exact shipped artifact, source build and veto-build equivalence | `submissions/20261102T0313Z-d1-v3/manifest.json` names `track3_meshes_v3.parquet`, SHA-256 **a61f02b12078a05c940037034e37ec4201e3e919a55e4bea9e0058dfa2f45f40**, `rule="R6 peg-rack, wall/radius 0.10..."`, final source **Build**, and veto build `outputs/mesh_extents/build/20261102T030138.153550Z-118eeb04/`. Its reproduction manifest and current read-only hash comparison establish identical mesh bytes. The packed code URL is still a placeholder (`final=false`); “packaged/shipped v3” here names the prepared artifact, not an upload or a final public release. |
| D1 meshes and D2 policy-rollout poses are scored separately; CD-O removes placement and scale under the mesh budget | Read-only kit `upstream/v2d_submission_kit/v2d_submission_kit/tracks/track_3/README.md` and mesh metric implementation; paper citation `challenge`. The scan-use restriction is also in the organisers' FAQ (`faq` in `references.bib`). |
| MoGe-2 predicted depth; above-table clusters; manual object naming; omit merged/clipped observations; normalisation preserves distortion; medians over frames then episodes | `track3/features/mesh_extents/README.md`, `_geometry.py:extents`, `measure.py`, and `build.py`. **Build** `run.json:request.measure_runs` pins the consumed measurements; each measurement records manual assignments and accepted frames. `assignments.json` holds names and cluster IDs. No measured-depth accuracy claim is made. |
| Packaged boxes: 1.0091 object / 1.2547 pair cm; packaged handle meshes: 0.9257 / 1.0850 cm | **Pairs** `means.v1={per_object_cm:1.0091025260092832,pair_weighted_cm:1.254731056017533}`; `means.v2={per_object_cm:0.9257250385709256,pair_weighted_cm:1.085023637656605}`. Both use each entry's own exact kit seed, recorded in `pairs[].seed`. Group scores by object, average within object, then equally over ten objects for object means. Pair means average all 26 covered `(episode,slot,object)` entries equally. |
| Round-1 cross-fitted selection: 0.926 object / 1.087 pair cm | **Handle** `cross_fitted_mean_cm=0.9257219019644454,pair_weighted_mean_cm=1.0874159144755273`. Each object has a three-seed score; pair weighting reuses it according to **Rules** `pair_counts`, rather than independently resampling every entry. Seeds derive from episodes 1, 3 and 5, not training replications. |
| Cube baseline: 2.523 object / 2.772 pair cm | `outputs/mesh_eval/baseline-cube.json:mean_cd_o_cm=2.523`; the unique `outputs/mesh_eval/*/d1-v1-cdo.json:["Cube (floor)"].pair_weighted_mean_cm=2.772`. Pair weighting reuses object scores, as in the selection block of Table 1. |
| Ten scanned objects; spoon unscanned, box retained; ten of 36 pairs excluded | **Pairs** `covered_pairs=26,total_pairs=36,unscored` (ten spoon entries); **Build** `objects.json:objects.wooden_spoon.shape="box"`. Both reported means exclude the spoon. |
| FlashCHORD simulates stationary video-sized boxes with parked hands, rather than the submitted shell-and-peg meshes | `track3/features/pose_insurance/README.md`, `outputs/pose_insurance/v1/report.md`, and task/scene producer records. The D2 lineage admits the earlier video-box build, while **Build** provides the D1 shell-and-peg meshes. Paper citation `flashchord` identifies the pinned source. |
| First achieved pose of first object, including rotation, aligns to its first reference mocap pose; second object's offset scored in first object's body frame; camera-to-mocap heading unknown | `docs/chain/pose-placement-protocol.md`, “Why placement is scored in slot 0's body frame”; kit `metric_code/track_3/AUC.py:_warp_to_reference_start`. The reference pose is a scoring input, not a scene-construction input. The protocol's frame audit records that no camera-to-mocap transform is released. |
| Earlier rule: video centroids/defaults with shared-cluster stacking; one measurable confirmation, post-result scene-wise acceptance, reused training analogue | `docs/chain/pose-placement-protocol.md`, candidate definition and its two final post-result decision sections; `outputs/pose_insurance/placement/confirmation.md`; `outputs/pose_insurance/v2/physical_checks.json`. The measurable confirmation was public episode 2; public episode 27 had separate clusters. Training episode 26 informed selection for submission episode 28. Earlier rule in Table 2/**Scale** means stacking on training episodes 0 and 26, earlier placement on the other six, including 31. |
| Table 2 earlier rule: AUC 0.2688, SP-SR 0, RPE 39.36 cm, MPPE 22.82 cm; stereo shrinkage: 0.2690, 0, 35.34, 20.84 | **Scale** `means.incumbent` and `means.S1`. Raw earlier RPE/MPPE **39.36384551739198 / 22.818618462986215**; stereo **35.33558264341916 / 20.843564644963294**. Equal means over training episodes 0, 11, 12, 18, 21, 23, 26 and 31. **Shrink** `means8.I/S1` reproduces these records. |
| Uniform shrinkage in Table 2; supporting zero-offset diagnostic: 0.2690, 0, 35.28, 20.80; zero offset: 0.2748, 0, 19.54, 14.13 | **Shrink** `means8.G`: raw `(AUC,SP_SR,RPE_cm,MPPE_cm)=(0.2690123323497656,0,35.27602887377869,20.801243810949728)`; `means8.C0=(0.27477472005330295,0,19.542964153582133,14.133047790824333)`. |
| Supporting reference static diagnostic (Table 2 caption: AUC 0.532, MPPE 7.40): AUC 0.5319, SP-SR 0.625, RPE 15.11 cm, MPPE 7.40 cm | `outputs/pose_insurance/placement/summary.json:oracle_static` / `summary.md`, equal means over the same eight episodes. Holds reference first poses for diagnosis only; this is not a submitted policy. |
| Both success rates zero for video-derived rows; supporting reference static has zero MP-SR; metric definitions and thresholds | **Scale** and **Shrink** per-episode and aggregate `SP_SR,MP_SR`; reference-static row in placement summary. Kit metric functions and Track 3 README define AUC, SP-SR (mean <=10 cm/0.5 rad), MP-SR (all-object <3 cm/30 degrees), RPE and six-attached-point MPPE. Thresholds are unchanged from the previous paper. These diagnostics use scan surface samples in place of official evaluation vertices. |
| Stereo cameras and estimator; training factors 0.819–0.923; second-object table-plane offset only; heights, rotations and sizes unchanged | `track3/features/stereo_depth/README.md:Inputs` names `ego_cam_c` and `ego_cam_b`. `docs/chain/pose-scale-protocol.md:Scale estimate,Candidates` fixes camera C left, B right, median stereo/monocular ratios over jointly valid pixels, then median over observed frames. **Scale** `stereo[episode].s` ranges **0.8185996684784886–0.9233432798833405** on the eight training episodes. The scale is not independently validated. |
| Training mocap distances motivated shrinkage; tuning episodes 39 and 42 have one object; adoption after results and candidate-wise single-object preservation check | `docs/chain/pose-scale-protocol.md:Disclosure,Decision after the train result`. **Scale** `invariance.comparisons["39:S1"/"42:S1"]` preserves every metric exactly. The original job's global rejection reading retained the earlier rule; the later lead decision adopted stereo shrinkage under a candidate-wise reading. The paper calls this exploratory rather than defending it as confirmatory evidence. |
| Uniform factor 0.857; stereo fails calibration-benefit criterion; 0.04 cm MPPE gap below 0.25 cm margin does not establish equivalence | **Shrink** `assignments.s_bar=0.8569522079777245`; `means8.S1.MPPE_cm-means8.G.MPPE_cm=0.04232083401356945` → **0.04**. `verdicts["1"]` and `docs/chain/pose-shrink-protocol.md:Verdicts` record the unmet calibration-benefit criterion and practical margin. The controls establish neither causal calibration benefit nor equivalence. |
| Zero offset overlaps boxes; shorter wrongly directed offsets can lower error | `docs/chain/pose-shrink-protocol.md:Candidates` and **Shrink** `verdicts["2"]`. This is a kinematic diagnostic, not an adopted scene. Possible shrinkage benefit from wrong direction is a disclosed interpretation, not a causal finding. |
| Clearance-limited follow-up: 1 cm footprint separation; 17.02 cm MPPE; AUC -0.00018; episode-31 MPPE +0.69 cm >0.5 cm limit | `docs/chain/pose-shrink-protocol.md:Pre-registered follow-up H`; **Shrink** `means8.H.MPPE_cm=17.020548795924775`, `H.gates.guarded_changes.AUC=-0.00018028846153844702`, `episodes.H["31"].metrics.MPPE_cm-episodes.S1["31"].metrics.MPPE_cm=0.6933918802549996`. `H.gates.passed=false`; the follow-up was not packed. |
| Evaluation calibration is absent; episode 5 keeps earlier poses because policy training produced invalid numbers, with cause unknown | `docs/chain/pose-scale-protocol.md` and `outputs/pose_insurance/v3/report.md:Caveats and provenance` disclose the public-calibration assumption. `outputs/pose_insurance/v3/audit.json` retains episode 5; `outputs/pose_insurance/shrink/episode5_diagnosis.{md,json}` records non-finite state without locating its origin. No physical evaluator or checkpoint inspection was rerun here. |
| No demonstrated manipulation or official pose-score gain; manual names, uncertain coordinate frames and the unscanned spoon limit generality | Static trajectory construction, the unknown heading/calibration disclosures and diagnostic scan sampling above. These are limitations, not claims of measured official performance. |

| New critic-fix claim | Recorded source and recomputation |
| --- | --- |
| Headline compares v3 against the replaced v2 rule before boxes | **Packed** `means.v3={per_object_cm:0.7710770332786878,pair_weighted_cm:0.8114951342417491}`; `means.v2={per_object_cm:0.9257250385709256,pair_weighted_cm:1.085023637656605}`. Parenthetical boxes are **Pairs** `means.v1={per_object_cm:1.0091025260092832,pair_weighted_cm:1.254731056017533}`. Four-decimal displayed values are unchanged; the comparator order changed. |
| Sampling variability: mean per-object seed SD 0.0069 cm, maximum 0.0214 cm | Deduplicate `outputs/mesh_eval/rules-round2/ledger.jsonl` by `identity` (70 rows, 70 unique). Select `rule="R6 peg-rack",parameter=0.1` (ten objects; both held-out folds chose this parameter), then take the arithmetic mean and maximum of `seed_std_cm`: **0.006887335425561346 / 0.02136714096735155**, rounded to four decimals. `scripts/track3/local_cdo.py:score` defines each saved SD with `np.std(values)` (population denominator 3) over kit-derived sampling seeds for episodes 1, 3 and 5. No new meshes were sampled here. These are sampling fluctuations for these scans, not new-object uncertainty or training-seed variability. |
| Shared wall-fraction trade-off: cups best at 0.10, pot and pitcher best at 0.02; pot 0.834 vs 0.863 cm, pitcher 1.461 vs 1.473 cm at 0.02 vs 0.10 | Same deduplicated ledger, `rule="R6 peg-rack"`, group `cd_o_cm` by object/parameter. Pot raw **0.83411640467235 / 0.8631161086673155**; pitcher **1.4605896239060332 / 1.472552596081469** at **0.02 / 0.10**. Beige cup grid scores at 0.02/0.05/0.10: **0.3805679252331655 / 0.3485410734730012 / 0.3207342096946351**; blue cup: **0.48278002033991035 / 0.43389635905454166 / 0.380004128502687**. Thus each cup's minimum is 0.10, while pot/pitcher minima are 0.02. A shared fraction compromises per-object optima on this limited grid; thicker walls remain untried. |
| Rim-handle refinement affects only the pot; added pegs only the rack, approaching the per-object-choice prohibition | `attributes.json` gives `rim_flat` only to the pot and confirmed pegs only to the rack; `_rule_scores.py:shape_for` and `rules-round2.json` apply these attributes as frozen rules. Ledger comparison of rim-handle scores with the fixed incumbent changes only `white_pot`. At identical wall fraction 0.10, comparing peg-rack versus hollow-vessel `inputs.candidate_sha256` changes only `plastic_dish_rack`. “Only rack” means the added peg refinement relative to hollow vessels; the full peg-rack rule still includes the vessel refinements. Protocol “Rules, not objects” prohibits choosing each object's shape from its scan score. The paper identifies the proximity to that prohibition, not an independently demonstrated protocol violation or a defence based on registration. |
| Largest gains came from rack pegs and the pot shell | At the same 0.10 fraction, ledger hollow-vessel minus peg-rack CD-O gives rack-only gain **0.7240558570836573 cm**. Rim-handle minus hollow-vessel gives pot-shell gain **0.4200176461186095 cm**, versus pitcher **0.2102997639533848**, blue cup **0.061865978547312905**, beige cup **0.058345223506265476**; other shell gains are zero. These are the largest two feature/object increments. **Packed** also ranks rack and pot as the two largest total per-object v2→v3 gains. Remaining compartments/spout are documented in attributes and Figure 1; both pose success rates remain zero in **Scale/Shrink**. |
| Plain explanation of the stacking check and episode 5 | Public episode **2** was the measurable stacking confirmation: `outputs/pose_insurance/placement/confirmation.md` and the final decision in `docs/chain/pose-placement-protocol.md`. Acceptance became per-scene only after results. `outputs/pose_insurance/shrink/episode5_diagnosis.{md,json}` finds non-finite policy/training tensors and does not identify where invalid values began. The new wording says training produced invalid numbers and the cause is unknown; it does not assert a located numerical root cause. |

## Figure provenance

Figure 1 now reads **Build** `objects.json:objects[name].file` and displays the
actual submitted v3 PLY files. Its labels read **Packed** `objects[object=name]`,
which averages all covered kit-seeded entries within each object; they are
**exact packaged means, not three-seed fitted or cross-fitted estimates**.

| Object | v3 mean (cm) | v2 mean (cm) | Figure label |
| --- | ---: | ---: | --- |
| Pot | 0.8583628989575244 | 1.3262550320406155 | v3 0.858 / v2 1.326 cm |
| Pitcher | 1.4871715535982997 | 1.7231591423732386 | v3 1.487 / v2 1.723 cm |
| Rack | 1.3678618752518905 | 2.1027812706787152 | v3 1.368 / v2 2.103 cm |

Other changed objects, also from **Packed** `objects`: blue cup v2
**0.44086543542223333** to v3 **0.3813848177210324**; beige cup
**0.37377830838198145** to **0.3255779904456613**. These round to
0.441 → 0.381 and 0.374 → 0.326 cm. They are supporting records, not extra
paper or panel claims.

RGB still comes from the left half of each measurement's `overlay_000000.png`,
the remapped RGB before the overlay. The mask uses
`labels_000000.npz:label in [cluster_id+1]`, with a 10% crop margin.
The figure checks assignments against `assignments.json`:

| Object | Frame-0 producer under `outputs/mesh_extents/public/` | Assignment key and cluster IDs |
| --- | --- | --- |
| Pot | `episode_000042/20261101T103715.090424Z-d410dfac/run.json` | `public:42`, `white_pot=[0]` |
| Pitcher | `episode_000021/20261101T103707.151959Z-05bf6bac/run.json` | `public:21`, `water_pitcher=[0]` |
| Rack | `episode_000012/20261101T103703.949004Z-546c56c2/run.json` | `public:12`, `plastic_dish_rack=[0,2,3,4,5,6]` |

Only these three public scans are displayed, from the read-only snapshot
`datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh/<name>/<name>.glb`.
All triangles are shown; recorded CD-O instead uses the kit mesh budget.
Display-only scan rotations map `(x,y,z)` to `(z,-y,x)` for pot/rack and
`(y,-x,z)` for pitcher. Each mesh is centred and uniformly divided by its longest
extent. Camera, light, axis limits and 300 dpi surface rasterisation are unchanged. The qualitative canvas is
3.18 by 2.65 inches and the pipeline 3.18 by 1.65 inches; the meshes, masks,
labels, view, light and numerical scores are unchanged. These sizes keep both
main pages readable. The display implies neither metric size nor camera-frame
alignment. No candidate file is modified.

Figure 2's pipeline now says “Manual names + attributes” and “Shells + peg rack
(D1)”; its D2 branch is unchanged. Figure 3 still reads **Scale**
`episodes.incumbent/S1[episode].frame0.offset_error_cm` for training episodes
0, 11, 12, 18, 21, 23, 26 and 31. All sixteen bar heights, descriptive legends,
wrapped y label and stars on stacked episodes 0 and 26 are unchanged. There is
no surface-point threshold line on this offset-error plot.

## Critic fixes and inspection

1. The abstract now compares v3 first with the replaced cylinder-plus-handle v2
   meshes, retaining boxes as a parenthetical baseline. All exact values use the
   same sources and averaging rules above.
2. Pose prose names the public stacking check (episode 2), says that new placements
   were retained where objects stayed put after seeing results, explains unchanged
   single-object scores directly, and names episode 5's invalid training values
   without claiming that their origin is known.
3. Two final sentences name the largest measured shape gains, remaining rack
   compartments/pitcher spout, and zero pose success rates.
4. “Closed solids with cup-shaped cavities” replaces the ambiguous phrase about
   watertight open mouths; the constructor and the geometry have not changed.
5. Disclosure now says the rim-handle refinement affects only the pot and the
   additional peg refinement only the rack, close to the protocol's ban on
   per-object shape choice. It retains prior scan-render exposure and optimistic
   selection on ten reused scans.
6. Per-object three-seed standard deviations have mean **0.0069 cm**, maximum
   **0.0214 cm**. They describe sampling variability, not generalisation uncertainty.
7. The shared wall fraction is a compromise: cups prefer 0.10, pot/pitcher 0.02
   on this grid, with the four reported values checked from the ledger.
8. Figures/tables stay at paragraph boundaries; the first-round geometry,
   round-2 rules, selection results and pose alignment remain complete within
   their columns. The body font and margins are unchanged; figure height and
   paragraph gaps were reduced. The pose table focuses on the earlier rule,
   stereo shrinkage and uniform shrinkage. The zero-offset overlap caveat/value
   remain in prose; the reference-static baseline is retained as supporting
   provenance above rather than a paper row.
9. The optional all-ten-object panel was omitted: items 1–8 fill the two main pages.

No caveat about scan reuse, non-blind attributes, the unscanned spoon, manual
identities/measurement distortion, unknown heading, unvalidated stereo scale,
training-episode selection, post-result/exploratory adoption, missing evaluation
calibration, absent demonstrated calibration benefit or equivalence, physically
overlapping zero offsets, the failed clearance-limited follow-up, episode 5's
unknown failure cause, approximate scan sampling, absent official pose-score gain,
internal mesh faces, untried thicker walls, or zero success rates was removed.

The final PDF has **three pages: two main pages and references only on page 3**.
The pipeline remains on page 1; both tables and the offset plot are on page 2.
Every final 110 dpi page was inspected for clipping, complete sentence flow and
readable labels. No overfull boxes or undefined references remain. The sole TeX
error is the known microtype tracking warning; existing font-encoding and
underfull warnings remain. No requested number lacked a recorded source.

Syntax compilation, CPU import, unchanged-pose/numeric-helper equivalence and
read-only arithmetic checks are recorded in
`.runtime/tmp/d3-fix/editorial-evidence.json`. The deduplicated 70-row ledger
aggregation, per-object spreads, thickness scores and changed-object sets are in
`ledger-evidence.json` there; build evidence is in `main.log` and `build.log`.
No tests, new scoring, model runs, repacking, installs, uploads, GPU work or
excluded-episode access occurred. The lead reviews and commits the edits.

Render with the existing Poppler installation:

```bash
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png \
  .runtime/tmp/d3-fix/main.pdf .runtime/tmp/d3-fix/p
```

Pages: `.runtime/tmp/d3-fix/p-1.png`, `p-2.png`, `p-3.png`.
The scratch directory also contains before-edit sources, previews, the build
mirror, command logs and the full file inventory (`files-changed.txt`).
