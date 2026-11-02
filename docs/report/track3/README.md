Anonymous Track 3 abstract with two pages of main text, covering the fixed mesh-rule protocol and the D2 v3 static-object pose entry with stereo-anchored offset scale. Build from the worktree root with `bash scripts/report/build_pdf.sh docs/report/track3/main.tex`; the pinned Tectonic writes `main.pdf`, `main.log`, and `main.blg` beside the source. The unchanged template classes use the anonymous option to suppress names and the logo. Three column-width figures (qualitative meshes, shipped-rule/S1 frame-zero pose offsets, and the D1/D2 pipeline) complement the mesh and pose tables; the legacy bar chart remains available with `make_figures.py --bars` but is not included. Panel scores describe the final fitted meshes, not the cross-fitted headline. Below, `S` = `outputs/mesh_eval/rules/summary.json`, `R2` = `S:ranking[rule="R2 vessel-handle"]`, `V` = `outputs/mesh_eval/d1-ctx03/v2-cdo.json`, `B` = `outputs/mesh_extents/build/20261101T160402.316899Z-0c4f8b6f/`, `V1` = `outputs/mesh_eval/d1-ctx02/v1-box.json`, `P` = `outputs/pose_insurance/`, and `M` = `outputs/mesh_extents/public/episode_000011/20261101T103701.928174Z-f63cc238/run.json`. Name selectors mean lookup by value, not array position; displayed values use decimal half-up rounding. The current mesh table shows aggregate scores; historical per-object mesh-table provenance is retained below. The table uses full-precision rule trials, avoiding a second rounding of the fresh v2 score file; the text rounds that fresh fitted score to the same three decimals: **0.925 cm** throughout. Raw source precision is retained in this README, not mixed into the prose. Cross-fitting applies to rule parameters on the same public references, not an untouched test, and the D2 placement diagnostic uses eight public train two-object episodes plus one fresh measurable confirmation. These kinematic results do not establish a host-score gain or manipulation performance.

| Claim in text or table | Recorded source and key / derivation |
| --- | --- |
| R2 headline 0.926 cm per object; 1.087 cm pair-weighted | `R2.cross_fitted_mean_cm=0.9257219019644454`; `R2.pair_weighted_mean_cm=1.0874159144755273`. Recomputed as the unweighted and `S.pair_counts`-weighted means of `R2.per_object_cm`. |
| Box v1 1.008 cm per object; 1.257 cm pair-weighted | `S.baseline_mean_cm=1.0076407576458337`; `S.baseline_pair_mean_cm=1.2566198957481125`. The table uses `S.baseline_per_object_cm`. These are full-precision rescored v1 boxes, consistent with the older four-decimal file at its rounding precision. |
| Cube 2.523 cm per object; 2.772 cm pair-weighted; all ten cube table entries | `outputs/mesh_eval/baseline-cube.json:mean_cd_o_cm=2.523` and `objects[status="scored"].{object,cd_o_cm}`; weighted mean recorded in `outputs/mesh_eval/d1-ctx02/d1-v1-cdo.json:['Cube (floor)'].pair_weighted_mean_cm=2.772`. |
| Every R2 CF table value | `R2.per_object_cm[object_name]`: beige cup 0.3656637065258667, blue cup 0.4418701070499999, dust pan 1.0612569261637064, sweeper 0.723585214852021, dish rack 2.1147920862043694, pitcher 1.6828523600348537, pot 1.327688027232037, lid 0.6955886791970488, wood 1 0.44443347270991, wood 2 0.39948843967464204. |
| R2 fit table values; fitted object mean 0.925 cm and pair mean 1.088 cm | `R2.trials[parameter=1.3].objects[].{object,cd_o_cm}`; object mean `R2.in_sample_mean_cm=0.9250628448040409`; pair mean `sum(score*S.pair_counts[name])/26=1.087678420439796`. Fit differs from CF for beige cup (0.3790794332009006) and lid (0.6755823809179679); all other values agree. |
| Fresh final-mesh fitted score: 0.925 cm in text (raw saved value 0.9251 cm) | `V.mean_cd_o_cm=0.9251`. `V.source` points to `B/objects`. This fresh score records object scores to four decimal places; it is not the cross-fitted adoption score. `outputs/mesh_eval/d1-ctx03/v2-mesh-equivalence.json` records equivalence with the loop's R2/1.3 candidates. |
| Qualitative panel: final R2 / v1 box CD-O (cm) | `V.objects[object=name].cd_o_cm` / `V1.objects[object=name].cd_o_cm`: `white_pot` = 1.3277 / 1.8182 → **1.328 / 1.818**; `water_pitcher` = 1.6829 / 1.5769 → **1.683 / 1.577**; `plastic_dish_rack` = 2.1148 / 2.1148 → **2.115 / 2.115**. These are the saved final-build scores, averaged over sampling seeds derived from episodes 1/3/5, with no new scoring. The pot improves, pitcher regresses, and rack has the largest final score among all ten `V.objects` entries. The reference scan has no separate candidate score; both numbers label the submitted-mesh row. |
| R1 0.956 cm; R3 0.990 cm, R3 fails gain gate | `S.ranking[rule="R1 vessel-cylinder"].cross_fitted_mean_cm=0.9564430821692316`; `S.ranking[rule="R3 thin-tool-head"].cross_fitted_mean_cm=0.9902723134347186`, `.gates.gain_gt_0.03_cm=false`, `.adopt=false`. Both R1 and R2 have all gate booleans true; R2 ranks first. |
| Per-object paired gains and object-level uncertainty | Recomputed directly from the 80 unique rows in `outputs/mesh_eval/rules/ledger.jsonl`, using `rule`, `parameter`, `object`, `cd_o_cm` and the protocol folds. Select the lowest training-fold mean in fixed grid order, then use the opposite fold: A selects 1.3, B selects 1.05. For each object, gain = box minus held-out R2 score. Gains (cm): beige cup `0.23928082287039348`, blue cup `0.22211950083414095`, pot `0.49047122601327375`, pitcher `-0.10599337236949369`, lid `-0.026689620534432468`; the other five are zero. Arithmetic mean `0.08191885568138821`; sample SD (denominator 9) `0.18012167185925354`; SE = SD/sqrt(10) = `0.056959473903269685`; mean/SE = `1.4381954408586235`. These are descriptive object-level summaries of adaptively selected results, not independent assessment or a correction for selection. |
| Pot supplies 60% of net object-mean gain and 67% of net pair-weighted gain | From the ledger-recomputed gain vector above: `gain[white_pot]/sum(gain)=0.5987281217902899`; `6*gain[white_pot]/sum(count[name]*gain[name])=0.6689302857430074`. Counts come from object-name occurrences in `outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff/objects.json:pairs_written` (six pot pairs, 26 scanned pairs); equivalently `S.pair_counts`. Shares use net gain, including negative pitcher/lid gains. |
| R2 threshold scores span about 0.002 cm; effective selection is rule-family choice over R1 | From ledger rows with `rule="R2 vessel-handle"`, average `cd_o_cm` over all ten objects separately for each `parameter`: 1.05 → `0.9257219019644455`, 1.15 → `0.9270634746319489`, 1.30 → `0.9250628448040408` cm. Max minus min = `0.0020006298279081625` cm, slightly greater than a strict 0.002 bound. The paper therefore says **about** 0.002. R1/R2 family selection uses these same previously inspected scans; cross-fitting only chooses the threshold within R2. |
| Largest held-out R2 regression 0.106 cm, on pitcher | `R2.worst_held_out_regression_cm=0.10599337236949369`; `R2.per_object_cm.water_pitcher - S.baseline_per_object_cm.water_pitcher`. |
| Two folds, five objects each; three rules; fixed categories and adoption gates >0.03 cm gain, <=0.15 cm regression, no pair regression | `docs/chain/mesh-selection-protocol.md`, “Rules, not objects”, “Two-fold leave-objects-out”, and “Adoption”; its current SHA-256 matches `S.protocol.sha256=ce5e6c3a11139a81588aaaab92f3d2481ec14386ac529b47dc0a154b00548097`. `track3/features/mesh_extents/_rule_scores.py:FOLDS,CATEGORIES,select` implements these constants; `R2.gates` records the three decisions. `S.ranking` contains all three tried rules, including R3. |
| R2 fold A threshold 1.30; fold B 1.05; final threshold 1.30 | `R2.chosen_on_fold={A:1.3,B:1.05}` means chosen on that fold, scored on the opposite one. `R2.in_sample_parameter=1.3`; `B/run.json:request.aspect_threshold=1.3`, `request.shape="vessel_rule"`, `request.shape_overrides={}`. |
| Threshold grid 1.05, 1.15, 1.30; handle height 60%, width 15%; shape mappings | `R2.trials[].parameter`; `track3/features/mesh_extents/rules.json:rules[name="R2 vessel-handle"].{grid,shapes,handle_height_fraction}`. `B/run.json:request.handle_height_fraction=0.6`. Width 0.15 is an implementation constant in `track3/features/mesh_extents/_shapes.py:vessel_with_handle`, not a scored outcome; this source is pinned in `B/run.json:sources[path="track3/features/mesh_extents/_shapes.py"].sha256`. Internal faces are disclosed in the slice README “Shape rules” and `outputs/mesh_eval/rules/summary.md`. |
| Three kit-derived sampling seeds and fixed mesh budget | `V.episodes_for_seeds=[1,3,5]`; trial entries `R2.trials[].objects[].inputs.scoring.{seed_episodes,budget}`. Seed episodes map through the kit's `_object_seed(episode,0)`; they are sampling seeds, not training replicates. Both references and candidates pass the kit budget in `scripts/track3/local_cdo.py`. |
| Ten scanned objects, 26 scanned pairs, unmeasured spoon in 10 of 36 pairs | `len(S.baseline_per_object_cm)=10`; `sum(S.pair_counts.values())=26`; `B/objects.json:pairs_written` has 36 entries, ten ending `:wooden_spoon`. The spoon has no entry in `V.objects`; its `B/objects.json:objects.wooden_spoon.shape="box"`. Pair means weight object scores by these counts, rather than rerunning every pair's exact seed. |
| Opening frames 0–19; evaluation episode 1 exception 110–230; 1–9 episodes per object, one for lid | Each measurement record in `B/run.json:request.measure_runs` has `request.frames`; evaluation 1's `outputs/mesh_extents/evaluation/episode_000001/20261101T103716.335071Z-0993ba31/run.json:request.frames=[110,130,150,170,190,210,230]`. Other measurement frames fall within 0–19. `B/objects.json:objects.*.episodes` lengths span 1–9; lid has one. |
| PCA 2nd–98th percentile range, 98th percentile height; mask acceptance 0.8–1.25×; scale-normalized medians | `M:request.pixel_ratio_range=[0.8,1.25]`; `frames[].objects.*.{in_plane_m,height_m}` and `objects.*.frames_used` store measurements. Percentiles are implementation constants in `_geometry.py:extents`, pinned by `M:sources[path="track3/features/mesh_extents/_geometry.py"].sha256`. Pooling is pinned in the recorded measure/build source hashes; the measure version is retrievable at commit `93038d2`. Manual names and notes are in `M:request.{objects,assignment_note}` and `track3/features/mesh_extents/assignments.json`. |
| v1 selection used the public scans; measurement thresholds used episode 39 | Research-history disclosure in `docs/chain/SUBMISSION.md`, “D1 v0”; this is procedural history, not an inferred numeric score. The cylinder/handle idea was motivated by the pot box losing to a cube (`docs/chain/IDEAS.md`, idea 15; `docs/chain/handoffs/HANDOFF-20261101T1508Z-02.md`, next steps). Rule design therefore followed inspection of these references. The later fixed-protocol claim is bounded by `S.protocol` and the logged rule trial provenance in `outputs/mesh_eval/rules/ledger.jsonl`. |
| Pose training: 512 environment steps, one world, 479 critic and 240 actor Adam updates per episode | `P/v1/budget_selection.json:selected.budget.{environment_steps,worlds}={512,1}`. Match the 20 paths in `P/v1/checkpoints.json` to `P/v1/ledger.jsonl:checkpoint`; every selected record has `optimizer_updates=479` and `actor_adam_steps=240`. Actual saved-state Adam counters are independently recorded in `P/v1/optimizer_audit.json`. |
| Larger budgets 4,096 and 16,384 steps failed on episodes 5 and 3, respectively | `P/v1/medium_budget_failure.json` and `P/v1/large_budget_failure.json`: `error="ValueError: Training state contains nonfinite tensors"`; their `run` fields select producer records with `episode_index=5,3` and `request.budget.environment_steps=4096,16384`. They passed the episode-10 pilots but failed during the broader campaign; these budgets are not the packed policies. |
| D2 v2 exports: 20 episodes, 9,120 rows; kit preflight Ready to submit | `submissions/20261101T2315Z-d2-v2/manifest.json:rows=9120`, `len(exports)=20`, `preflight_verdict="Ready to submit"`, `results_dir="outputs/pose_insurance/results-v2"`. `composition.K1a=[1,28]`, `composition.v1` contains the other 18 episodes including 30; `exports[].unchanged_v1` and hashes retain source identity. `commands` records the official packer (`--skip-crosscheck`) and kit preflight (`--skip-kaggle`). Manifest and `P/v2/report.md` read here; no repacking or preflight rerun. |
| Historical D2 v1 drift below 0.05 cm (superseded for the v2 entry) | Maximum `checks.drift_cm` over checkpoint-selected records in `P/v1/ledger.jsonl` is `0.04769628867506981`. Drift is relative to the first exported pose, not a measure of initial placement accuracy. |
| First-version spurious 33–93 cm clearance shifts in six episodes corrected | `P/v1/displacements.json`: six records have `before_max_cm>30`, for episodes 3,6,14,20,30,32; exact range `33.11440626257501` to `93.13235084718853` cm. Their `after_max_cm` values are all zero. This does not assert zero clearance for every episode: episode 28's explicit defaults require 3.776 cm of clearance. |
| Episode-14 cup/rack identity swap corrected | `P/v1/displacements.json[episode=14].slots`: cup `v0_clusters=[0]` becomes `v1_clusters=[1]`; rack `[1]` becomes `[0]`. `P/v1/tasks.json["14"].scene[]` records the reviewed identities, notes and overlay paths. |
| Explicit placements in nine episodes, six uncertain | `P/v1/tasks.json`: episodes 1,3,6,14,20,25,28,30,32 have a scene slot with `placement_source="placements.json"`. Of these, episodes 1,20,25,28,30,32 have at least one slot with `identity_confidence="uncertain"`. The count is episodes, not slots. `track3/features/pose_insurance/placements.json` and `P/v1/report.md` retain the decisions and reasons. |
| Historical D2 v1: five of 36 scored pairs used defaults; v2 retains the episode-30 sweeper default only | `track3/features/pose_insurance/placements.json:placements[cluster_ids=null]` contains exactly `(1,white_pot)`, `(1,white_pot_lid)`, `(28,plastic_dish_rack)`, `(28,white_pot)`, `(30,mini_sweeper)`, with merged/unnamed-cluster notes. The 36-pair denominator is the scored mesh roster `B/objects.json:pairs_written`, also matching the pose task slots. `P/v1/tasks.json[episode].scene[]` records `clusters=null`, horizontal `position_before_clearance[:2]=[slot*0.5,0]` m and `quaternion_wxyz=[1,0,0,0]` (identity yaw) for these five slots. Defaults seat at half box height; these are initialization choices. In v2, `pose-placement-protocol.md` final Decision and the manifest replace the episode-1/28 scenes with K1a; episode 30 retains v1, including its sweeper default. The old four default slot entries in 1/28 no longer describe the submitted scenes. |
| Simulated video-box assets differ from submitted vessel-rule mesh assets; v1 lineage | Every producer selected by `P/v1/tasks.json:*.run` records `inputs` admitting `outputs/mesh_extents/build/20261101T103838.400965Z-117fbcff/{run.json,objects.json,objects/*.ply}`: video-box build used by pose simulation, with per-episode MoGe size restoration/default pooled size. The submitted vessel-rule assets instead come from `B` (`V.source`; `B/run.json:request.shape="vessel_rule"`). Both producer lineages are video-derived reconstructions. Example task producer: `outputs/pose_insurance/build-v1/20261101T165137.956282Z-b83602fa/episode_000001/run.json`. |
| Historical single-object tuning 39/42 unchanged: AUC 0.175/0.194, SP-SR 0/0, matching oracle-static AUC/SP-SR | `outputs/pose_insurance/diagnostic-v1/20261101T170255.684375Z-4b00de97/run.json` and `outputs/pose_insurance/diagnostic-v1/20261101T170326.563872Z-4028f8b6/run.json`: `metrics.learned.AUC=0.17496081504702193,0.19406614785992216`; `metrics.learned.SP_SR=0,0`; these equal `metrics.oracle_static.AUC,SP_SR` and the previous recorded diagnostics. `sampling` and `metrics.note` disclose own surface samples approximating host vertices. With a constant single-object predicted trajectory and initial-pose alignment, matching the oracle static AUC/SP-SR is by construction, not evidence of placement accuracy; see kit metric alignment and `track3/features/pose_insurance/diagnostic.py`. Those single-object episodes do not measure relative placement; the new two-object diagnostic below does. Other metrics are not asserted identical: episode 42 learned/oracle MPPE is 7.38046781421122/7.371285267790487 cm. |
| Metric size, static boxes, parked hands, remaining pose limits | `track3/features/pose_insurance/README.md`, `P/v1/report.md`, recorded task scenes and policy producers describe the implementation. MoGe size remains unanchored and uncalibrated; parked hands leave objects stationary. The older tuning episodes each contain one object. The new placement diagnostic below measures relative-placement and body-frame errors on public development data. The manifest pins actual working-source hashes and a placeholder commit URL; its preflight verdict does not establish a completed upload. |
| Pose alignment: slot 0 anchors the full initial pose, slot 1 is evaluated relative to its body frame | `docs/chain/pose-placement-protocol.md`, “Why placement is scored in slot 0’s body frame” and Amendment 1; kit `upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/AUC.py:_warp_to_reference_start` uses `reference[0,0,0]` and `achieved[0,:,0]`, including quaternion alignment, then applies the same world transform to both slots. A slot-0 orientation convention error rotates the entire inter-object offset. This is a metric property, not a newly measured outcome. |
| Eight public train two-object episodes; v1 AUC 0.2688 versus oracle-static 0.5319 | `P/placement/summary.json:means.v1.AUC=0.26884406311899633`, `oracle_static.AUC=0.5319185200141404`; the latter is also printed to six decimals as 0.531919 in `summary.md`. Equal-episode averaging over protocol episodes `[0,11,12,18,21,23,26,31]`, with evaluation-analogue slot ordering. Kinematic static poses, one world, nearest-visible reference fill, official kit functions; AUC uses 1,000 surface samples per public scan, seed 0, to approximate host vertices. Oracle-static is a comparator, not an optimum over static poses. |
| Historical unscaled v1/K1a diagnostic: offsets approximately 10–74 cm; AUC/SP-SR tied | `P/placement/summary.md`, “Frame-zero relative-transform decomposition”, 40 rows (five candidates × eight episodes): rounded extrema 10.105 cm (`K1a`, episode 26) and 74.117 cm (e.g. v1, episode 21). The context-05 figure plotted v1/K1a; context 06 replaces it with v2/S1 (see Figure 2 row below). The former prose extrema were approximate, not strict bounds. Kit `AUC.py:_ADD_THRESHOLDS_M=np.arange(0.01,0.10,0.01)` ends at 0.09 m = 9 cm. `summary.json:means` gives identical AUC 0.26884406311899633 and SP_SR 0 for all candidates; the protocol Amendment 2 records zero slot-1 AUC credit. This is observed saturation for these candidates, not a general inference that a frame-zero offset alone fixes an entire trajectory’s score. |
| Clean episodes 11/18/21/23: video/mocap inter-object distance ratio 1.37–1.80 | `P/placement/summary.md`, decomposition rows for v1 on those episodes: length ratios 1.370, 1.800, 1.484, 1.522 respectively. The ratio is the magnitude of achieved relative translation divided by reference relative translation; it includes video-centroid versus scan-origin offsets, so it is not a pure depth-scale calibration error. |
| K1a lowers mean RPE 47.51 → 36.80 cm and MPPE 26.66 → 21.63 cm; all fixed gates fail | `P/placement/summary.json:means.v1.{RPE_cm,MPPE_cm}={47.507593808999744,26.66119273895194}`; `means.K1a` gives `{36.79569731467336,21.632295386379496}`. `gates.*.passed=false`; all fail `checks.auc_gain` under Amendment 1’s required mean AUC gain ≥0.02. K1a stacks objects sharing a recorded frame-0 merged cluster at its table-plane centroid. These are selection results on the eight train episodes, not physical rollouts or an unbiased assessment. |
| K2 body-up calibration gives no aggregate gain and is not used by D2 v2; heading is unobservable | Protocol Amendment 1 fits the signed body-up vector from public train mocap leave-one-episode-out (singleton z-up fallback); `summary.json:means.K2` has AUC 0.26884406311899633, RPE 48.6984541398602 cm and MPPE 27.62321616761419 cm, versus the v1 values above. `gates.K2.passed=false`; final manifest composition uses K1a/v1 only. K3 is omitted because no camera-to-mocap transform is released; gravity alone does not establish world heading. Source: protocol Amendment 1 and `summary.md`, “Limits and follow-up”. No new frame convention is fitted here. |
| Post-result amendment; one fresh confirmation, public 2: RPE 54.39 → 21.79 cm, MPPE 28.29 → 12.43 cm; AUC/SP-SR/MP-SR unchanged | Protocol Amendment 2 was fixed after the eight-episode results, before preparing/scoring fresh public 2/27; recorded overlay assignments preceded candidate scoring. `P/placement/confirmation.json:episodes["2"].v1` gives RPE 54.389153734707975 cm, MPPE 28.286727421462004 cm; `.K1a` gives RPE 21.788548057666155 cm, MPPE 12.432519320531478 cm. Both have AUC 0.3342747813411079, SP_SR 0 and MP_SR 0. `confirmation_episodes=1`, `verdict="adopt"`. `confirmation.md` records the same results. This is a kinematic confirmation on one episode, not a new multi-episode independent test. |
| Public 27 unmeasurable; train 26 is scored 28’s selection analogue | `P/placement/confirmation.json:episodes["27"].measurable=false`: rack is cluster 1, pot body 0 and handle 2, so no merged cluster exists; no score is assigned. Protocol Amendment 2 explicitly allows exclusion for this condition. Its final Decision and `P/v2/report.md` disclose that confirmation rests only on public 2, while train 26 (rack/pot analogue of scored 28) was selection data. |
| Physical validity filter: K1a accepted on scored 1/28; first-pose agreement 0.062 cm and drift 0.058 cm | `P/v2/physical_checks.json:episodes[episode=1,28]` have `accepted=true`; `first_pose_cm=0.062205004537963786,0.06220133621018252`; `stats.drift_cm=0.05764961242675781,0.05764365196228027`. Reported values are the rounded maxima over those two accepted scenes. First-pose error is deviation from the **intended initial scene**, not mocap accuracy. The protocol Amendment 1 requires first pose within 1 cm / 2° and the slice’s drift check. The final Decision applies this filter per scene. |
| Episode 30 falls about 17 cm and keeps v1’s default; eighteen v1 exports retained | `P/v2/physical_checks.json:episodes[episode=30]` has `accepted=false`, `stats.drift_cm=17.002592235803604`, `first_pose_cm=1.4715029858594941`. Its `support_geometry` and `P/v2/report.md` place the sweeper outside the rack footprint. Protocol final Decision and v2 manifest `composition` adopt K1a only on `[1,28]`, retaining eighteen byte-identical v1 exports, including 30. The raw physical-check top-level `physical_adoption=false,packed=false` and the confirmation Markdown’s final rejection paragraph are historical pre-Decision verdicts; the final protocol Decision, v2 report and manifest supersede them without rewriting the historical records. |
| Pose Table 2, eight-episode rows: v1 / K1a / oracle static | `P/placement/summary.json:means.v1`, `means.K1a`, `oracle_static`. Displayed `(AUC, SP-SR, RPE cm, MPPE cm)` = **(0.2688, 0, 47.51, 26.66)** / **(0.2688, 0, 36.80, 21.63)** / **(0.5319, 0.625, 15.11, 7.40)**. Oracle full precision: `RPE_cm=15.114438767442412`, `MPPE_cm=7.403291369009795`, `SP_SR=0.625`. These are recorded equal-episode means, not newly scored results. |
| Pose Table 2, fresh episode-2 rows: v1 / K1a | `P/placement/confirmation.json:episodes["2"].v1` / `.K1a`, also `confirmation.md`. Displayed vectors = **(0.3343, 0, 54.39, 28.29)** / **(0.3343, 0, 21.79, 12.43)**. AUC rounds `0.3342747813411079` to four places. `MP_SR=0` in all seven displayed rows, including the added v2/S1 rows; episode 27 has no measurable merged cluster and no invented score. |
| Figure 2: v2 rule versus S1 frame-0 slot-1 offset errors (cm), train 0/11/12/18/21/23/26/31 | `P/scale/summary.json:episodes[candidate][str(episode)].frame0.offset_error_cm`, keyed by candidate (`incumbent`, `S1`) and episode value. `incumbent` is the shipped v2 rule: K1a on 0/26 only, v1 on the other six, including 31. In the stated episode order, v2: **12.982646, 18.529993, 61.094409, 48.052814, 74.116841, 14.646197, 10.104506, 39.110273**; S1: **12.982646, 16.438327, 55.094711, 42.403529, 70.757237, 8.281592, 10.104506, 30.051638**. These match `P/scale/summary.md` at its six-decimal precision; the figure plots full JSON precision. Stacked 0/26 are starred; S1 is an exact no-op there. Episode 23 falls below the 9 cm translation reference, which alone does not establish surface-point or trajectory success. The prior all-episode K1a bars are replaced, not relabelled. |
| Figure 3: D1/D2 pipeline, no new measurements | `make_figures.py:pipeline_panel` draws the documented routes: video → MoGe-2 depth plus calibrated stereo scale → object extents → R2 vessel/handle rule (else box) → D1 meshes; frame-0 placement → static scene with S1 offset scale → kit `evaluate_policy.py` → D2 trajectories. Shape implementation/provenance is in the R2 rows above; pose scale and failed-check fallbacks are in the S1/v3 rows below. Stereo scale affects pose offsets; CD-O removes mesh scale. The diagram summarizes the existing entry and does not claim dynamic manipulation, a shared mesh/policy geometry, or new evaluator results. |
| Figure 2 threshold and Table 2 metric definitions (9 cm, 10 cm/0.5 rad, 3 cm/30 degrees, six attached points) | Kit `upstream/v2d_submission_kit/v2d_submission_kit/metric_code/track_3/AUC.py:19–23`, `_add_auc`, `_spider_success`, `_maniptrans_success`, `_relative_position_error_cm`, `_mean_per_point_error_cm`; plain-language definitions in `tracks/track_3/README.md:66–70`. The largest of `np.arange(0.01,0.10,0.01)` metres is 9 cm. This surface-point threshold is a scale reference for initial translation error, **not** a sufficient condition for trajectory success or a new proof of AUC saturation. |
| Slot 0/1, v1, K1a, K2, oracle static, FlashCHORD and FlashSAC definitions | Slot order: kit `data/track_3_evaluation_objects.json`; pose alignment: `AUC.py:113`, `_warp_to_reference_start`. Placements/calibration/oracle: `P/placement/summary.md` and `docs/chain/pose-placement-protocol.md`. Kit `flash_chord/scripts/evaluate_policy.py:24` and `flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml:1` identify `flash_chord.training.flash_sac`. The pinned full source `upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/training/flash_sac/__init__.py:3–10` explicitly calls it a native JAX reinforcement-learning algorithm derived from Holiday Robotics FlashSAC. Its `scripts/train_flash_sac.py:4,41` describes training the reference-tracking policy. The report defines this role without inventing an acronym expansion. |
| Post-result Amendments 2 and 3, not an unchanged original protocol | `docs/chain/pose-placement-protocol.md`, named Amendments 2/3 (Amendment 3 starts at line 164). Amendment 2 follows the failed eight-episode gates; Amendment 3 follows physical checks and changes the all-scenes requirement to a scene-wise filter. The main text explicitly retains both amendments, the single measurable confirmation, train 26 as selection data, and episode 30 retaining v1. |
| Table 2: v2 rule / + stereo scale (S1), eight train means | `outputs/pose_insurance/scale/summary.json:means.incumbent` / `.S1`. Displayed `(AUC, SP-SR, RPE cm, MPPE cm)` = **(0.2688, 0, 39.36, 22.82)** / **(0.2690, 0, 35.34, 20.84)**. Full precision: incumbent `AUC=0.26884406311899633`, `RPE_cm=39.36384551739198`, `MPPE_cm=22.818618462986215`; S1 `AUC=0.2690243515805348`, `RPE_cm=35.33558264341916`, `MPPE_cm=20.843564644963294`. Both `SP_SR` and `MP_SR` are zero. The incumbent is **K1a on train 0/26 only, v1 on the other six including 31**, as fixed in `docs/chain/pose-scale-protocol.md:Candidates`; it is not the all-episode K1a row. |
| Stereo method, calibrated 75.47 mm C/B baseline; train s 0.819–0.923; MoGe-2 overstates size by 8–22% against calibrated stereo | `docs/chain/pose-scale-protocol.md:Scale estimate` fixes C-left/B-right, 75.47 mm, semi-global block matching and median valid stereo/monocular pixel ratios, followed by the median over observed frames. `outputs/pose_insurance/scale/summary.json:stereo[episode].s` for train `[0,11,12,18,21,23,26,31]` spans **0.8185996684784886–0.9233432798833405**. Relative to stereo the MoGe-2 overstatement is **100*(1/s-1) = 8.302082420131395–22.159834471796902%**, rounded to 8–22%. This is a calibrated-stereo comparison on these eight episodes, not measured depth ground truth; depth-training scores remain against MoGe-2 pseudo-labels. |
| S1 definition and train selection because tuning 39/42 contain one object | `docs/chain/pose-scale-protocol.md:Candidates,Scoring,Disclosure`. S1 multiplies only slot 1's initial table-plane offset from slot 0 by s, preserving heights, rotations and sizes; stacked offsets are zero. Single-object tuning cannot measure relative placement. The scale idea arose from train mocap distance ratios. S2, mentioned only to explain the control dispute, scales the whole scene. |
| Post-result scale decision: S1 passed controls exactly; the job's global fail-closed reading kept v2, the lead's candidate-specific reading adopted S1 | The unchanged `outputs/pose_insurance/scale/summary.json:decision` / `summary.md` retain the job's **blocked** verdict. `invariance.comparisons["39:S1"]` and `["42:S1"]` pass exactly; S2 differs on 39 in MPPE by `7.105427357601002e-15` cm. `docs/chain/pose-scale-protocol.md:Decision after the train result` is explicitly post-result, fixes the lead's candidate-specific interpretation, and adopts S1 without changing estimator/gates/physical filters. The main text discloses this departure; it does not present adoption as an unambiguous preregistered outcome. |
| Residual S1 distance ratios 1.18–2.23 **excluding default-offset episode 31 (3.58)**; direction errors about 2–130 degrees | `outputs/pose_insurance/scale/summary.json:episodes.S1[episode].frame0.{distance_ratio,direction_error_deg}`. Excluding 31, ratios span **1.1804346050295738–2.228294376715972**; episode 31 is **3.5793787338584875**. The source report/protocol's 1.18–2.23 shorthand is therefore not an all-eight-episode range. Directions span **2.464691495435456–129.39456399854512 degrees**; 2–130 is approximate. Shorter offsets may hedge incorrect directions rather than establish correct metric scale. |
| Length-only oracle AUC 0.2767 versus 0.2688; no evaluation calibration | `outputs/pose_insurance/placement/summary.md:Diagnostic ceilings`, v1 `Length oracle AUC=0.276705`, versus baseline `0.268844`; this is reference-informed diagnostic evidence, not an adopted method. `docs/chain/pose-scale-protocol.md:Decision after the train result` and `outputs/pose_insurance/v3/report.md:Caveats and provenance` disclose that evaluation ships no calibration and public stereo calibration is assumed. |
| Current v3 pose file: 20 episodes, 9,120 rows, Ready to submit; 13 S1 scenes, intended-pose error <=0.049 cm, drift <=0.048 cm | Read-only recorded sources `outputs/pose_insurance/v3/report.json` / `report.md` and `audit.json`. `rows=9120`, `len(exports)=20`, `preflight_verdict="Ready to submit"`. Audit `accepted=[3,6,8,10,14,15,16,19,20,25,30,32,33]`; maxima `maximum_first_pose_cm=0.048607143738936404`, `maximum_drift_cm=0.04769638180732727`. These errors are against intended initialization / first exported pose, not mocap. Packed file: `submissions/20261102T0032Z-d2-v3/track3_poses_v3.csv`, SHA-256 `1dcbca75223873e233f187c781457cd5cbeab8b0fcce0bf955933e83fe6ca416`. No file was repacked or preflight rerun for the abstract. |
| v3 composition: v2 kept on stacked 1/28, failed-training 5 and single-object 35–38; scored s 0.768–0.948 | `outputs/pose_insurance/v3/audit.json:unchanged=[1,5,28,35,36,37,38]`; `report.md` explains S1 no-op on stacked 1/28, nonfinite optimizer state on 5 before the evaluator (no retry), and unchanged single-object 35–38. The remaining 13 episodes are the accepted list above. `v3/stereo_*.json:s` spans **0.7678335235402176–0.947996387351935**, rounded to 0.768–0.948. Episode 30 now uses S1 applied to its v2 default-offset scene; its earlier rejected K1a stack was not restored. |

Qualitative panel reproduction and image provenance:

```bash
mkdir -p .runtime/tmp/d3-qualitative
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/d3-qualitative/matplotlib" \
  .runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py \
  --preview .runtime/tmp/d3-qualitative/panel.png
```

The script writes `figures/d1-v2-qualitative.pdf` and, with `--preview`, the same Matplotlib
figure as a 300 dpi PNG. The one-column canvas is 3.18 × 3.05 inches; text remains vector,
and the mesh surfaces are rasterised at 300 dpi. It uses the existing Matplotlib 3.11.2,
trimesh 5.2.0 and NumPy 2.2.6 on CPU; nothing was installed. The PNG was visually inspected
for crop/assignment correspondence, mesh orientation, label clipping and overlap.

The RGB comes from the **left half** of each producer's `overlay_000000.png`: this is the
unmodified RGB after the measurement handler's remapping, not a fresh video decode.
`labels_000000.npz:label` holds one-based cluster labels; the displayed union uses
`label in [cluster_id + 1]`. A blue translucent fill and boundary highlight only the assigned
object. The crop adds a 10% margin around that union and does not rotate the RGB.
All three producers are in `B/run.json:request.measure_runs`; their recorded request IDs
are checked against `track3/features/mesh_extents/assignments.json`:

| Panel row | Frame-0 measurement producer under `outputs/mesh_extents/public/` | Assignment key and zero-based cluster IDs |
| --- | --- | --- |
| Pot (`white_pot`) | `episode_000042/20261101T103715.090424Z-d410dfac/run.json` | `episodes["public:42"].objects.white_pot = [0]` |
| Pitcher (`water_pitcher`) | `episode_000021/20261101T103707.151959Z-05bf6bac/run.json` | `episodes["public:21"].objects.water_pitcher = [0]` |
| Dish rack (`plastic_dish_rack`) | `episode_000012/20261101T103703.949004Z-546c56c2/run.json` | `episodes["public:12"].objects.plastic_dish_rack = [0,2,3,4,5,6]` |

Submitted surfaces are the exact `V.objects[object=name].candidate` files in
`B/objects/<name>.ply`. Reference surfaces come from
`datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public/mesh/<name>/<name>.glb`.
Only those three scans are read. Full scan triangles are displayed; recorded CD-O instead
uses the kit mesh budget, as described above. Scans are shown solely as scoring references,
and no display transform changes any candidate file or recorded score.

For display, each mesh is independently centred on its bounding box and divided **uniformly**
by its longest extent. Candidate axes are unchanged. Display-only right-handed rotations
map scan coordinates `(x,y,z)` to `(z,-y,x)` for pot/rack and `(y,-x,z)` for pitcher,
so vessels stand upright and the rack lies flat. All six meshes use the same orthographic
camera (elevation 25°, azimuth −65°), light and axis limits. This is a canonical illustration,
not the scorer's optimized registration or surface-RMS normalization; hence the caption says
**shown at normalised scale**. Neither metric size nor frame-camera alignment is implied.

Build and page-budget evidence (2026-11-02): the pinned Tectonic 0.17.0 was invoked with the
same `-X compile -Z continue-on-errors --keep-logs` arguments as `scripts/report/build_pdf.sh`.
For this job's restricted write scope, its existing `.cache/tectonic` contents were copied to
`.runtime/tmp/d3-qualitative/tectonic` and `TECTONIC_CACHE_DIR` redirected there. Reproduce the
ordinary build with the command in the opening paragraph, or use the scoped-cache invocation:

```bash
TECTONIC_CACHE_DIR="$PWD/.runtime/tmp/d3-qualitative/tectonic" \
  .runtime/tools/tectonic-0.17.0/tectonic -X compile -Z continue-on-errors --keep-logs \
  docs/report/track3/main.tex
rg -n '^! |TRACK3-|Output written|Overfull' docs/report/track3/main.log
```

`main.log` records `TRACK3-MAIN-TEXT-END-PAGE: 2`, `TRACK3-TABLE-PAGE: 2`,
`TRACK3-FIGURE-PAGE: 1` and `Output written on main.xdv (3 pages, ...)`.
After adding the D2 v2 findings, the PDF has **3 pages**: main text ends on page 2,
and page 3 contains references only. The original font sizes and template geometry are retained. The sole `!` error is
`Package microtype Error: The tracking feature only works with pdftex 1.40`; there are no
overfull boxes or undefined references. Existing font-encoding and underfull-box warnings
remain. No test suite, GPU work, holdout access or metric rerun was performed.

D2 v2 report revision and PDF inspection (2026-11-02): the pose section replaces the older
single-object tuning/default-placement account with the eight-episode alignment diagnostic,
failed fixed gates, explicitly post-result amendment, one-episode confirmation and scene-wise
physical filter. The original qualitative mesh panel and numeric table are retained. The v2
CSV is package-ready according to its manifest; it has not been uploaded and has no reported
host-score gain. Only source/report files were edited here; historical metrics, protocol,
physical-check records, exported trajectories and submissions were read without modification.

Pages were rendered at 110 dpi using the already installed Ubuntu Poppler tool and all three
PNGs were visually inspected. Page 3 contains only references; the two main-text pages retain
readable text, the mesh panel and table without clipping or overlap. Commands:

```bash
mkdir -p .runtime/tmp/d3-pose
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png \
  docs/report/track3/main.pdf .runtime/tmp/d3-pose/page
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdfinfo docs/report/track3/main.pdf
```

Page images: `.runtime/tmp/d3-pose/page-1.png`, `page-2.png`, `page-3.png`.
The build uses the scoped-cache Tectonic command above; no new dependency, GPU job,
metric run, preflight run or test suite was needed for this prose revision.


Pose table, offset figure and final page inspection (context 05, 2026-11-02):

`main.tex:123` adds the compact booktabs pose table; `main.tex:149` includes
`figures/d2-slot1-offset.pdf`. The figure generator adds `--pose-offsets`, preserving the
existing default mesh panel and `--bars` behavior. The drawing function accepts arrays;
the orchestration function reads the recorded ledger and decompositions. No kit metric,
model, evaluator, preflight or GPU operation is executed.

```bash
mkdir -p .runtime/tmp/ctx05-pages .runtime/tmp/ctx05-matplotlib .runtime/tmp/ctx05-tectonic
PYTHONPYCACHEPREFIX="$PWD/.runtime/tmp/ctx05-pycache" \
  python3 -m py_compile docs/report/track3/figures/make_figures.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/ctx05-matplotlib" \
  .runtime/linux-cuda/bin/python -c 'import docs.report.track3.figures.make_figures'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/ctx05-matplotlib" \
  .runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py \
  --pose-offsets --preview .runtime/tmp/ctx05-pages/offset-preview.png
cp -a .cache/tectonic/. .runtime/tmp/ctx05-tectonic/
TECTONIC_CACHE_DIR="$PWD/.runtime/tmp/ctx05-tectonic" \
  .runtime/tools/tectonic-0.17.0/tectonic -X compile -Z continue-on-errors --keep-logs \
  docs/report/track3/main.tex
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png \
  docs/report/track3/main.pdf .runtime/tmp/ctx05-pages/p
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdfinfo docs/report/track3/main.pdf
```

The direct Tectonic invocation uses the same engine and arguments as
`scripts/report/build_pdf.sh`; its cache is redirected into the allowed `ctx05-*` scratch
scope rather than the wrapper's fixed `.cache/tectonic` destination. The final build has
**3 pages: 2 main-text pages plus 1 references-only page**. `main.log` records
`TRACK3-MAIN-TEXT-END-PAGE: 2`, mesh/pose tables on page 2, mesh panel on page 1 and
offset figure on page 2. Every final image was inspected:
`.runtime/tmp/ctx05-pages/p-1.png`, `p-2.png`, `p-3.png`. No clipped labels, overlapping
content, orphaned headings or overfull boxes were found; both tables and both figures
are readable at their printed sizes. Template fonts and margins are unchanged; float-to-text
spacing is 10 pt. The existing recoverable XeTeX/microtype tracking error remains, along with
font-encoding and underfull-box warnings; there are no other TeX errors or undefined references.

The figure is 3.18 × 1.65 inches, uses blue/orange plus hatching, and removes CreationDate
and ModDate metadata. Regeneration produced the same PDF SHA-256:
`71e9e9a64c09ca43f0953dd02d564cd79527744078f9b60759ccf3d63ddf71df`.
The CPU audit `.runtime/tmp/ctx05-audit.py` confirms every new pose-table literal against
saved JSON with decimal half-up rounding, all five MP-SR zeros, and identical old/new outputs
for the unchanged `cluster_crop` and `normalised_vertices` functions. Its result is
`.runtime/tmp/ctx05-pages/audit.json`; figure source values are also printed to
`.runtime/tmp/ctx05-pages/offset-data.json` (JSON followed by the artifact path).

To fit the new material, detailed handle proportions, per-object episode counts, clean-episode
distance ratios and individual paired mesh gains were removed from main text; their recorded
values and sources remain in this provenance table. All caveats and post-result disclosures
remain. No requested table row or figure was dropped. No dependencies, references, scientific
metrics, protocol files or submission files were changed. Complete changed-file inventory,
including generated scratch/cache files: `.runtime/tmp/ctx05-files-changed.txt`.


D2 v3 abstract update and final page inspection (2026-11-02):

The abstract and pose section now report the calibrated-stereo size finding, S1's recorded
train result, v3 composition and fallbacks, train-only selection, the length-oracle ceiling,
residual direction/ratio errors, missing evaluation calibration, and the post-result
invariance interpretation. Amendments 2/3 and the one-episode confirmation remain explicit.
The mesh table is condensed to aggregate means; its earlier per-object values remain in the
provenance table and qualitative figure. Mesh result prose and method detail were shortened.
All scientific caveats/disclosures remain. The offset figure is unchanged and its caption now
identifies it as the unscaled v1/K1a diagnostic, rather than implying S1 also earns zero slot-1
AUC. No figure source or figure PDF was changed in this follow-up.

`\raggedbottom` removes stretched column glue and the large gap above section 2 on page 1;
heading/font sizes and template margins are unchanged. The final PDF is **3 pages: exactly
2 main-text pages and 1 references-only page**. Every final 110 dpi page was rendered and
visually inspected: `.runtime/tmp/ctx05-pages-v3/p-1.png`, `p-2.png`, `p-3.png`. Both figures
and tables remain legible; no clipped content, margin overflow, overlapping material or orphaned
heading was found. `main.log` records main-text end, both tables and the offset figure on page 2,
and the qualitative panel on page 1. There are no overfull boxes or undefined references.
The sole recoverable TeX error remains the existing XeTeX/microtype tracking incompatibility;
font-encoding and underfull-box warnings remain.

Commands used (the scoped Tectonic invocation matches the build wrapper's engine/arguments):

```bash
PYTHONDONTWRITEBYTECODE=1 .runtime/linux-cuda/bin/python .runtime/tmp/ctx05-v3-audit.py
TECTONIC_CACHE_DIR="$PWD/.runtime/tmp/ctx05-tectonic" \
  .runtime/tools/tectonic-0.17.0/tectonic -X compile -Z continue-on-errors --keep-logs \
  docs/report/track3/main.tex
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png \
  docs/report/track3/main.pdf .runtime/tmp/ctx05-pages-v3/p
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdfinfo docs/report/track3/main.pdf
git diff --check -- docs/report/track3/main.tex docs/report/track3/README.md
```

The CPU saved-record audit writes `.runtime/tmp/ctx05-pages-v3/audit.json`: both new table rows,
train/scored scales, percent conversion, the residual-ratio exception, physical maxima,
composition, row count and package identity match the recorded sources. It computes no pose
or depth metric. No tests, metric reruns, model/evaluator jobs, installs, uploads, submission
edits or mutating git commands were performed. Files written by this follow-up are enumerated
in `.runtime/tmp/ctx05-v3-files-changed.txt`; backups are in `.runtime/tmp/ctx05-v3-backup/`.


Context 06 presentation revision (2026-11-02; current build):

The title is **Video-Measured Meshes and Static Poses**. Figure 2 now compares the
shipped v2 rule (`incumbent`) with S1; the unused old v1/K1a figure reader is replaced.
The default qualitative panel and legacy `--bars` behavior are preserved. Figure 3 is a
Matplotlib PDF of the mesh and static-pose branches; it fits on page 2. Both new/revised
canvases are 3.18 × 1.95 inches. The offset chart uses 9 pt labels (template table size),
blue/orange plus hatching, units in cm, a 9 cm dashed reference, and starred stacked episodes.
The diagram uses 10 pt text. Body, caption and table fonts and template geometry are unchanged.
No TikZ or new LaTeX package is needed. `\clearpage` before the bibliography ensures that
references start on page 3; the post-result placement paragraph is kept together.

The fitted D1 v2 score is **0.925 cm** everywhere in the abstract, prose and table.
The raw saved `V.mean_cd_o_cm=0.9251` and the all-ten trial mean
`R2.in_sample_mean_cm=0.9250628448040409` both round to that value. Other table values
and scientific outcomes are unchanged. All public-scan reuse, pseudo-label,
train-selection, post-result amendments/adoption, hedge, episode-5 fallback,
calibration, sampling and static-manipulation caveats remain. Repeated method detail
was shortened; numeric provenance above retains the threshold grid and other omitted detail.

Page-2 text extraction gives **650 → 576 whitespace-delimited words**, a net removal
of **74 (about 75)** including chart/diagram labels and captions. Some material moved
between pages/columns: the mesh table now fits on page 1. Pose-section prose alone
fell from about **433 to 340 words** (about 90 removed), excluding table/figure blocks.
The page count is **3: two main-text pages, then one references-only page**. Log markers:
main text ends on 2, mesh table/panel on 1, pose table/offset chart/pipeline on 2.
All final 110 dpi PNGs were visually inspected: no clipping, overlaps or overfull boxes;
page 2 ends with the pipeline caption, and page 3 begins with References.

Reproduction and checks (all CPU; from this worktree root):

```bash
PYTHONPYCACHEPREFIX="$PWD/.runtime/tmp/ctx06/d3/pycache" \
  python3 -m py_compile docs/report/track3/figures/make_figures.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/ctx06/d3/matplotlib" \
  .runtime/linux-cuda/bin/python -c 'import docs.report.track3.figures.make_figures'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/ctx06/d3/matplotlib" \
  .runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py \
  --pose-offsets --preview .runtime/tmp/ctx06/d3/offset-preview.png
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/ctx06/d3/matplotlib" \
  .runtime/linux-cuda/bin/python docs/report/track3/figures/make_figures.py \
  --pipeline --preview .runtime/tmp/ctx06/d3/pipeline-preview.png
bash scripts/report/build_pdf.sh docs/report/track3/main.tex
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftoppm -r 110 -png \
  docs/report/track3/main.pdf .runtime/tmp/ctx06/d3/page
LD_LIBRARY_PATH=.runtime/tools/poppler/root/usr/lib/x86_64-linux-gnu \
  .runtime/tools/poppler/root/usr/bin/pdftotext -f 2 -l 2 \
  docs/report/track3/main.pdf .runtime/tmp/ctx06/d3/after-page2.txt
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.runtime/tmp/ctx06/d3/matplotlib" \
  .runtime/linux-cuda/bin/python .runtime/tmp/ctx06/d3/audit.py
```

Page PNGs: `.runtime/tmp/ctx06/d3/page-1.png`,
`.runtime/tmp/ctx06/d3/page-2.png`, `.runtime/tmp/ctx06/d3/page-3.png`.
The saved-record audit is `.runtime/tmp/ctx06/d3/audit.json`: all 16 bar heights match
JSON exactly and the Markdown table at six decimals; the threshold matches the kit;
0/26 are exact no-ops. The unchanged `cluster_crop` and `normalised_vertices` helpers
match the backed-up implementation exactly on fixed arrays. Source lookup is by named
candidate/episode; no positional assumption or new metric evaluation is involved.
The build log is `.runtime/tmp/ctx06/d3/build.txt`. The sole TeX error remains the
known recoverable XeTeX/microtype tracking incompatibility; font-encoding and underfull
warnings remain. No undefined references or overfull boxes occur.

The reviewer concern that remains strongest is **train-selected, post-result S1 adoption**:
the interpretation of an integrity gate changed after seeing results. Better figures
make that limitation clearer; they do not repair the selection evidence. No result from
the separate pose-diagnostic job is included. No metric/model/evaluator runs, test suites,
installs, uploads, dependency changes, references edits or mutating git commands occurred.
Every written file (including scoped scratch artifacts and backups) is listed in
`.runtime/tmp/ctx06/d3/files-changed.txt`.

Current `d2-slot1-offset.pdf` SHA-256: `707c689a1bc1fc47677c4db8efc33d9514b75e245da9381a659e61911fc6bbdf`.

Current `pipeline.pdf` SHA-256: `e01940d105b6af91e502c6469bd26f760c96ba175041e995a0d722084102f055`.
