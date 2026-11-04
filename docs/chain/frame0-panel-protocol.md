# Frame-0 panel protocol: W1.1 masks, then H3 past episode 19 on a train panel (written before any number below exists)

## Why

Two findings set the shape of this work. T1's stereo offset reaches more scored pairs only if merged episodes get per-object masks. And
position alone cannot fix direction: the kit's one alignment rotates any offset by slot 0's heading error (`position-oracle-protocol.md`). H3
fixes that heading for non-cup objects. It shipped on episode 19 on the strength of one train pair, and the owner has now allowed it past
that episode (`PROGRAM.md` section 0, 2026-11-04 01:53 UTC). So masks on train 0, 26 and 31 do two jobs: they test W1, and they make a panel on
which H3's rotation can be judged on pairs other than 18.

**Departures from the plan's W1.1 gate, and why.** The plan asks for vector body-origin error within 3 cm, at least two of three. A vector error needs slot
0's rotation in a common frame, and there is no camera-to-mocap extrinsic. With T1's rotation the recorded aligned errors on clean episodes were 15 to 59 cm whatever
the offset's quality, so that gate would fail for a reason it does not test. The vector gate therefore moves to Stage C, where the registered anchor supplies the
rotation. Its limit is 5 cm, with 3 cm reported beside it: H3's shipped result on train 18 has `b` 9.3 degrees on a 22.56 cm offset, about 3.66 cm, and raised that episode's AUC by
0.172, so a 3 cm limit would reject a result already known to be useful. This limit is set before any registration of 0, 26 or 31 exists.

**Train labels decide here.** `PROGRAM.md` section 0 and the plan's section 3 put go/no-go on train/tuning labels under a protocol written first; tuning 39/42 are single-object and cannot test a pair. These are
disclosed train-panel results, not independent validation.

## What was already known when this was written (disclosure)

- Recorded: T1 and the oracles per episode. O3's rows on 0/26/31 are the source of the plan's 0.387. W1.0's arms were computed while this was written; the lead did not read them until this file was committed.
- HR2's frozen method on 18, 23, 24, 27, selected rotations ignoring declines: `b` 4.2 / 9.5 / 172.6 / 8.5 degrees, `C` 5.6 / 13.7 / 11.7 / 3.0 degrees; the wooden pieces' absolute up was inverted by about
  179 degrees (`pose-heading-hr2-outcome.md`). H3's TW on 18: `b` 9.3, `C` 5.6, AUC 0.3006 to 0.4729; its flip left AUC at 0.3006 while RPE rose from 29.58 to 41.71 cm and MPPE from 19.48 to 25.16 cm, because AUC has a floor
  (`pose-heading-h3-outcome.md`). So pair 18 is not fresh and pair 23 has been inspected. Pairs 0, 26 and 31 have no registration on record.
- HR1 found cup heading unrecoverable from stereo shape, so cup-anchored pairs are out of scope here.
- T1's saved 0 and 26 have zero horizontal offset but heights of about 11.25 and 18.95 cm: they are not zero in 3D.

## Order

**All label-free work is saved before any new label-dependent number is computed:** masks, offsets, registrations and candidate constructions. Only a label-free input failure may stop registration early. No arm or threshold is revised from these outcomes,
and a failed gate is not repaired on the same pairs.

## Stage A: masks and offsets (label-free; a few GPU-minutes; train 0, 26, 31)

Input: the frame-0 camera-A remapped RGB and the stereo run T1's code uses. Rule, with no hand selection and no re-prompts:

1. **Detector.** Grounding DINO tiny, revision `a2bb814`, the object's text from `track3/features/object_masks/prompts.json` (committed 2026-11-01), box threshold 0.4, text threshold 0.3. Highest-scoring box, ties to the lowest index. No
   box: the pair fails Stage A.
2. **Segmenter.** SAM2.1 Hiera-Large image predictor, the box as the only prompt, `multimask_output=True`, the highest predicted IoU, largest connected component only, as G3 did.
3. **Overlap.** Pixels in both objects' masks go to the smaller mask; equal areas go to slot 0. Box IoU above 0.8 between the two objects means the detector did not separate them, and the pair fails Stage A. Overlap ownership does not establish object
   separation or body origins, and the most likely failure is a pot mask that includes a seated lid, so that subtraction leaves a partial shell.
4. **Admission, per object, after overlap, recorded before any label is read:** at least 300 valid stereo pixels. Mask area is reported in pixels and as an image fraction, with no area window: the pot's range does not fit a rack.
5. **Offset.** `d = B (x1 - x0)`, the mean stereo points of each mask in the table basis `B` of the T1 build, keeping all three coordinates. This extends T1's horizontal-only measurement. Masks must match the recorded camera-A remap, frame index zero, stereo grid
   hash and intrinsics; no raw-camera mask is admitted. Visible-surface centroids are not body origins; that bias is what the panel measures.

**Control, descriptive only.** The rule also runs on clean episodes 11, 12, 18 and 21, and IoU against their recorded clusters is reported. A low IoU is not a veto: those clusters are depth clusters, not segmentation truth, and a better silhouette can score lower. Pairs 23 and 25
are not re-detected, because both wooden pieces share the prompt "wooden block" and independent selection would pick one instance twice; they keep their recorded masks and instance identities.

## Stage B: registration (label-free; CPU; saved before any label is read)

The frozen HR2 method (`docs/chain/pose-heading-method-hr2.json`, sha256 `c1fd8677...`, `_registration.py` unchanged), on both slots of 0, 26 and 31, with Stage A's masks. Pairs 18 and 23 reuse their recorded registrations. A slot is **admitted** if it has a mask, a loadable scan and subsampling stability
<= 10 degrees. A rival basin within 5 % does not decline a slot, because that rule rejected correct rotations in HR2; the flip stress below prices it instead. This relaxation is deliberate and is disclosed.

## Placement rule (fixed now, applied the same way on train and scored; label-free)

The pair's T1 boxes, at offset `d`, are checked for footprint clearance of at least 1 cm by T1's `footprint_clearance`. **Clear:** `p1 = p0 + d`. **Not clear:** a contact pair, which keeps its incumbent positions, so a stack stays the stack that already passed the physical check. No expansion, because the incumbent offset is
zero for the stacked pairs. No geometry is invented after seeing a score. Every construction keeps slot 0's position, and physical boxes and inertia are counter-rotated as D2 v5 did, so rotations change the body frame and not the physical placement.

## Stage C: arms and gates (train labels; kinematic; eight-episode means; episodes not named keep T1)

| Arm | Episodes changed | Construction |
| --- | --- | --- |
| P | 0, 26, 31 | placement rule; rotations T1 |
| P-free | 0, 26, 31 | `p1 = p0 + d` with no clearance rule; descriptive, not gated |
| TW4 | 0, 26, 31, 23 | P's positions on the first three, T1's on 23; `R_i = B R_i^cam` on both slots of a pair with both admitted |
| AN4 | 0, 26, 31, 23 | same positions; slot 0 as TW if admitted, slot 1 keeps T1's rotation |
| TW5, AN5 | also 18 | descriptive, so the recorded H3 row stays comparable |

Pair 18 is not in the gated arms: it is not fresh and its gain is already shipped on episode 19. H3's scoring wrapper refuses changed positions, so these arms take P as their construction baseline. Frame-0 errors come from `_heading_h3.errors`: `offset_error_m`, the full 3D `b` angle, and the `C` geodesic. A pair with an unadmitted slot keeps T1 for that pair
and fails its frame-0 limits. A truly zero 3D offset has no direction and cannot pass an angular limit. Missing or non-finite values fail; denominators are never reduced. Gated arms score at surface seed 0 as every earlier diagnostic; seeds 1 and 2 are reported beside them, and HR2's subsampling stability is reported for every slot.

**Frame-0 limits for a pair.** TW: `|b_pred - b_ref| <= 5 cm`, `b` angle under 30 degrees and `C` under 30 degrees. AN: the same `b` limits, no `C` limit. 3 cm passes are reported beside them.

**Gate P (diagnostic, with one route below):** mean RPE falls by at least 1 cm and MPPE by at least 0.5 cm; AUC, SP-SR and MP-SR do not fall (1e-12); no episode loses more than 0.02 AUC or gains more than 0.5 cm MPPE.

**Gate TW:**
1. the frame-0 limits hold on at least two of the three newly registered pairs 0, 26, 31; pair 23 is reported separately as previously inspected evidence;
2. mean AUC at least +0.02 against T1, and SP-SR above 0;
3. mean RPE and MPPE not worse;
4. every changed episode loses at most 0.02 AUC and gains at most 2 cm MPPE;
5. no train episode's `|b_pred - b_ref|` exceeds T1's by more than 0.5 cm;
6. flip stress: means are `(1/8) sum_episode mean_over_distinct_applied_combinations(metric)`, keeping T1 on unchanged episodes. AUC gain at least +0.01 and expected RPE and MPPE not worse than T1; every combination respects item 4's per-episode limits. This is a stipulated stress scenario and not a probability, and it covers only the one recorded rival per changed slot.

**Gate AN:** AN's frame-0 limits on at least two of 0, 26, 31; mean AUC at least +0.01; items 3, 4 and 6 with AN's arm, varying slot 0 only.

## Scored stage (only for gates passed; per-episode rules, fixed now; no scored label exists)

| Scored episode | Objects | Train analogue | Arm |
| --- | --- | --- | --- |
| 25 | wooden pieces | 23 | TW |
| 28 | rack, pot | 26 | TW |
| 30 | sweeper, rack | 31 | TW |
| 1 | pot, lid | 0 | TW |
| 3, 5, 6 | pot, spoon | 0, pot in slot 0; anchor object only, because slot 1 is a spoon | AN |
| 15, 16 | rack, spoon | 26, rack in slot 0; anchor object only | AN |

An episode changes only if all of the following hold:
1. its arm's gate passed;
2. its train analogue individually met its frame-0 limits and did not lose AUC to T1;
3. the inputs exist: on 1, 28 and 30, Stage A's rule recovers both masks and the offset, then the placement rule applies. On 3, 5, 6, 15, 16 and 25, the shipped v5 positions and the recorded independent masks are kept; episode 5 keeps its inherited fallback offset, with no T1-placement claim made for it. Missing inputs keep v5. Evaluation reuses the public-rig calibration assumption T1 made;
4. Stage B's admission holds for every slot the arm changes;
5. the physical check holds: first pose within 1 cm and 2 degrees, drift under 1 cm and 2 degrees, through the real evaluator on the slice's existing budget.

If Gate TW fails and Gate P passes, episode 30 alone may take P's construction, under rules 3 and 5. Any failure keeps that episode's v5 export, reading no score. Registrations of scored episodes are saved before any construction. Scans enter only measurement and registration; no scan geometry enters a mesh or a simulated asset. **A pack needs at least one changed episode, `check_release.sh` READY and a manifest that does not pin `AGENTS.md`,** and the abstract discloses that scans determined the rotations. The lead independently recomputes each gate's headline number before a pack.

## Integrity

Episodes 41, 43, 44 are never read. Validation 2, 13, 22, 24, 27 are not used here. The outcome goes in `docs/chain/frame0-panel-outcome.md`, whichever way each gate falls.
