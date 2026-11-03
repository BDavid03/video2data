# Mesh G3: automatic frame-0 pot masks for the pairs G2 could not reach

Written before any G3 mask, generation or score exists; the commit time is the record. Route:
`outputs/chain/codex/20261103T223809Z-route-g3-21.answer.md`. G2 and its results are in
`docs/chain/mesh-generative-protocol.md` and stay as they are. Incumbent: D1 v5 (`bcb93d57...`).

## Why

G2 replaced v4 meshes only where a frame-0 mask assignment had been recorded. Four evaluation pot pairs (1, 3, 6, 28)
and two development pot pairs (public 0, 26) kept v4 only for lack of a mask: `mesh_extents` clusters merged the pot with
the lid or the rack, or the recording rule excluded a pot with its handle end out of frame. SAM2.1 Hiera-Large and
`IDEA-Research/grounding-dino-tiny` are owner-approved (`AGENTS.md`, 2026-11-03 19:40 UTC), each in its own environment.

## Disclosure

- G2's selection stands: the pot was kept after its G1 result on the same public scan. Every pot pair scores against
  that one scan, so development and packed scores here are evidence on reused scans, not an untouched validation.
- Before this protocol the lead saw the frame-0 overlays of evaluation 1, 3, 28 and public 26, the recorded
  exclusion reasons, and the G2 results for pots 42, 5 and 8. No scan score exists for any G3 crop.
- Evaluation pot 8 stays rejected under G2 and is not regenerated. The post-hoc score of its rejected mesh happens only
  after this commit and changes nothing here.

## Pairs (fixed now)

White pot only. The pitcher has no unreached pair. Candidates: public 0, 26; evaluation 1, 3, 6, 28. Pots public 42 and
evaluation 5, 8 keep their G2 decisions and bytes. Everything else keeps its v5 bytes. Episodes 41, 43, 44 are never read.

## Mask rule (fixed now, no hand selection, no re-prompts)

Input: the frame-0 camera-A remapped RGB, the left half of the recorded `mesh_extents` overlay, as in G2.

1. **Lid rule, before inference.** If the episode's object list (kit roster for evaluation; the historical object-name
   inventory used by `object_masks` for public) contains `white_pot_lid`, the pair keeps v4: the lid is a separately
   scored object, and a shape generated from a pot with its lid on would merge the two.
2. **Detector.** Grounding DINO tiny, revision `a2bb814`, text `"white cooking pot."` (the prompt committed on 2026-11-01
   in `track3/features/object_masks/prompts.json`), box threshold 0.4, text threshold 0.3. Take the highest-scoring box;
   ties go to the lowest index. No box: keep v4.
3. **Segmenter.** SAM2.1 Hiera-Large image predictor, the box as the only prompt, `multimask_output=True`, the mask with
   the highest predicted IoU; ties go to the lowest index. No post-processing beyond the largest connected component.
4. **Mask rejection (each keeps v4, recorded in the ledger before any scan score):**
   - area outside 10,000-45,000 pixels at 576 x 768 (recorded pots: about 22,000), for contamination by rack or arm;
   - more than 10 % of the mask's boundary pixels within 2 pixels of the image border or the remap-invalid region, for a
     heavily cut object (a handle end out of frame stays allowed);
   - fewer than 2,000 valid observed depth points inside the mask.

## Control, before any generation

On the pairs with recorded masks (public 42, evaluation 5, 8), run steps 2-3 and compare with the recorded assigned
clusters. Every control must reach IoU >= 0.70, a test that the prompt finds the right object. A failure ends G3: v5
stands, with no change to the prompt, thresholds or rule. Controls produce no meshes.

## Generation and admission (G2's rules, unchanged)

Same code, weights and settings as G2: crop from the mask with 10 % margin, white outside, seed 12345, 30 steps,
octree 380, ordinary marching cubes, kit `budget_mesh(path, 4096, 4096)`, one generation per pair, no re-rolls.
Per-pair admission: after one Sim(3) fit to the pair's frame-0 observed points in the new mask, the generated median
point-to-surface residual must be <= 1.5 x that of the v4 mesh fitted the same way; otherwise v4. Decisions are sealed
in a video-only `admissions.json` before any scan score.

## Gates (fixed now; all required, no repair, no runner-up, no per-pair selection after scoring)

Incumbent numbers from `outputs/mesh_generative/g2/summary.md` and the packed v5.

**Development** (public pot 0/26/42 and pitcher 21, kit seeds 1/3/5, matched folds, the G2 aggregation):
- at least one newly admitted public pair, otherwise G3 has no out-of-pilot evidence and stops;
- equal-object mean < 0.56244822 cm and pair-weighted mean < 0.61972984 cm;
- no object worse than v4 by > 0.15 cm; seed spread reported.

**Packed veto** (production tree = v5 bytes plus admitted evaluation pairs, placeholder URL, `mesh_eval` pairs):
- at least one evaluation pair changed;
- object mean < 0.56443301 cm and pair mean < 0.62080727 cm;
- no object worse than packed v5 by > 0.15 cm; unchanged rows byte-identical to v5; kit preflight Ready (Kaggle skipped).

Pass: D1 v6 at a new real-URL commit with NEEDS-YOU commands, done by 12:00 UTC 4 Nov. Fail, or implementation
unfinished by 08:00 UTC 4 Nov: v5 stands.

## Reporting

Every candidate is reported with its mask, box score, mask checks, residual ratio and final decision, including the
rejected ones. A pass is described as pot shapes on reused-scan evidence, with G1's rack failure and G2's post-pilot
selection stated alongside it.
