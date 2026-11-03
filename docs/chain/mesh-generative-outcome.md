# What the pretrained mesh experiments showed

Track 3 reconstructs objects from video; the submission contains triangle meshes.
D1 v4 used measured primitives; v5 replaced two evaluation meshes with generated shapes.
CD-O is the kit’s public-scan shape error in centimetres; lower is better.
The kit budgets meshes to 4,096 vertices and faces before scoring.

G1 tried Hunyuan3D-2mini on one video crop each of a pitcher, pot and dish rack.
Generation used seed 12345, 30 diffusion steps and octree resolution 380.
Pitcher error improved from 1.48717 to 0.60673 cm; pot from 0.85836 to 0.72912 cm.
The rack worsened from 0.86484 to 1.09643 cm, a 0.23158 cm regression.
That exceeded the 0.15 cm object limit: G1 failed and its full panel was stopped.

G2 then selected only pot and pitcher using those same pilot outcomes.
Each new mesh was fitted to its own frame-zero masked depth cloud.
Admission required its median surface residual to be at most 1.5 times v4’s.
This residual uses video depth, not scans; its units are unanchored depth units.
G2 passed development and the separate exact-packed numerical check.
Packed pair-weighted error improved from 0.66292148 to 0.62080727 cm.
Packed equal-object error improved from 0.65338396 to 0.56443301 cm.
Only evaluation pot 5 and pitcher 20 changed; this became D1 v5.

G3 used automatic detection and segmentation to recover missing pot masks.
Its controls and development passed, but its packed numerical check failed.
Pot 6 passed admission at a residual ratio of 0.96201 yet worsened the scan score.
Its error rose from 0.85055790 to 1.00590635 cm; the handle is a stub and body distorted.
Packed pair error rose to 0.62678221 cm and equal-object error to 0.56702215 cm.
The G3 candidate was rejected; shipped v5 stands.

Post-hoc diagnostic A scores all nine generated pot/pitcher shapes, including rejections.
Admission ratio versus CD-O change has Spearman rho 0.38333 (n=9).
The ratio does not reliably predict CD-O change in these nine shapes.
The rejected evaluation shapes are episodes 3, 8, 28.
Admitting every generated evaluation shape would give pair error 0.63251184 cm
and equal-object error 0.56950499 cm, versus v5’s 0.62080727 / 0.56443301.
The combined G2/G3 admissions give 0.62678221 / 0.56702215 cm.
The video admission reduced mean error relative to admitting every generated shape.
The packed numerical veto protected v5 from G3; video admission alone did not reject pot 6.
All other objects and pairs without a generated shape retain their recorded v5 scores.
The nine input masks total 0 border-contact pixels and 0 remap-invalid pixels.
The assignment notes still flag cut handles in evaluation 3 and 6, and merging in 26/28.
A segmented visible body can stop before a missing handle; this mask check cannot prove completeness.
No isolated handle annotation exists, so handle pixel counts were not invented.

Post-hoc diagnostic B repeats three unchanged crops with generation seeds 12345–12349.
public-000021-water_pitcher: CD-O mean 0.53430, sample SD 0.09122, range 0.41766–0.62061 cm;
video ratio mean 1.15711, sample SD 0.16915, range 0.96608–1.42089.
public-000042-white_pot: CD-O mean 0.96608, sample SD 0.56984, range 0.48970–1.82112 cm;
video ratio mean 0.52773, sample SD 0.25045, range 0.20478–0.71763.
evaluation-000005-white_pot: CD-O mean 0.81332, sample SD 0.33492, range 0.58617–1.40215 cm;
video ratio mean 2.30089, sample SD 1.37019, range 1.24110–3.99463.
Within-crop pot seed variance / variance across seven original pot views is 749.06%.
This uses two replicated pot crops; it is descriptive, not a causal decomposition.
Each crop keeps its scoring seeds fixed; different views also differ in scoring seeds.
Geometry variation across seeds shows that one generated mesh does not define model quality.

The experiment does not separate cropping, occlusion, depth error and prior mismatch causally.
The rack failure shows that semantic familiarity alone does not ensure usable geometry.
The G3 false acceptance shows that fitting visible depth can miss errors in unseen shape.
There is only one public scan per object; views and seeds are not independent objects.
Objects were pilot-selected on these reused scans; this is not untouched validation.
No new seed was selected, no threshold changed, and no diagnostic mesh entered a pack.
All 255 protected historical hashes, including all 36 shipped PLYs, match before and after.
Full tables and the two labelled figures are in outputs/mesh_generative/insights/summary.md.

## What this means for D1 v5 (lead, checked against the tables above)

The pitcher gain is robust: all five seeds on the public crop score 0.418-0.621 cm against v4's 1.473 cm.
The pot gain is not: on the same crop the seeds span 0.490-1.821 cm (3 of 5 beat v4's 0.863 cm), and on evaluation 5
they span 0.586-1.402 cm (4 of 5 beat v4's 0.873 cm), with a five-seed mean of 0.813 cm.
The shipped pot 5 (0.627 cm) is therefore a favourable draw. Seed 12345 was fixed before any generation and not selected,
so the shipped mesh's score stands, but "a pretrained prior beats primitives on the pot" is only weakly supported.
The video ratio cannot pick seeds: its within-crop rank correlation with CD-O is -0.90 on pot 42 (the two worst
shapes fit the visible depth best) and +0.70 on pot 5; over the seven pot shapes in table A it is 0.14. Choosing a seed by
scan score would use the scan as a selector, which the project does not do, so no re-roll follows from these numbers.
G3's stop stands. Post-contact fusion (F2) was routed NO-GO for this deadline (5-15 % subjective chance, 8-13 h;
`outputs/chain/codex/*route-f2-21.answer.md`): the rack stays on the table and the pot is a textureless, near-symmetric
registration target. Before the deadline no further reconstruction method is built.
