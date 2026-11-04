# What the pretrained mesh experiments showed

Track 3 reconstructs triangle meshes from video. CD-O is scan error in cm; lower is better.
G1 tried one Hunyuan3D-2mini crop each for pitcher, pot and dish rack.
The fixed generation used seed 12345, 30 steps and octree 380, without scan input.
Pitcher improved 1.48717 → 0.60673 cm; pot improved 0.85836 → 0.72912 cm.
Rack worsened 0.86484 → 1.09643 cm: +0.23158 exceeded the 0.15 cm limit.
G1 failed and its full panel stopped.

G2 then selected pot and pitcher using those pilot results on the same scans.
Admission fitted each mesh to its own masked video-depth cloud with one Sim(3).
The generated median surface residual had to be at most 1.5 times v4's.
G2 passed development and the exact-packed check; evaluation pot 5 and pitcher 20 changed.
V4 → v5 pair mean: 0.66292148 → 0.62080727 cm; object mean: 0.65338396 → 0.56443301 cm.

G3 added automatic pot masks: detection followed by SAM segmentation.
Controls and development passed; the exact-packed check failed, so v5 stands.
Pot 6 passed admission at ratio 0.96201 but worsened 0.85055790 → 1.00590635 cm.
The generated handle is a stub and the body distorted.
G3 pair/object means were 0.62678221 / 0.56702215 cm, both worse than v5.
Public 26 was G3's only new development evidence; its mask and mesh include rack fragments.
It nevertheless passed the area check (17,094 mask pixels).
Boundary and depth-count checks never discriminated among the four inferred candidates:
all boundary fractions were 0%, and valid depth-point counts equalled mask pixels.
Pots 3 and 6 passed those mask checks despite handles recorded as cut by the frame.

Diagnostic A scores all nine original generated shapes, including rejected ones.
Spearman ratio versus CD-O change: 0.38333 (n=9); within pots: 0.14286 (n=7).
There is no validated predictive signal: nine dependent observations.
All-generated evaluation pair/object means would be 0.63251184 / 0.56950499 cm.
The +0.00572963 cm pair-mean gap from G3 comes from rejected pot 28's +0.27698888 cm,
offset by rejected pots 3 and 8 improving -0.02980174 and -0.09821666 cm (all divided by 26).
Admission helped pot 28 and hurt pots 3/8; an overall protective claim is unwarranted.
The packed numerical veto, rather than video admission alone, protected v5 from pot 6.
The three handle-in-view pot crops used cluster masks; the four others used SAM.
Handle visibility is confounded with mask source, and seed spread exceeds the apparent gap.
No handle explanation is established. Border contacts and invalid-remap overlaps were zero.
No isolated handle annotation exists; recorded assignment notes supply visibility context.

Diagnostic B uses five seeds (12345–12349) for each of three unchanged crops.
Public pitcher 21: CD-O mean 0.53430, sample SD 0.09122, range 0.41766–0.62061 cm.
Public pot 42: mean 0.96608, SD 0.56984, range 0.48970–1.82112 cm.
Evaluation pot 5: mean 0.81332, SD 0.33492, range 0.58617–1.40215 cm.
The pitcher replication covers public 21 only, not shipped evaluation pitcher 20.
Pot 42's five-seed mean 0.96608 cm is worse than v4's 0.86312 cm.
Shipped pot 5 is a favourable fixed-seed draw; no scan-based seed selection occurred.
Within-crop seed / between-pair pot variance ratio is 7.49056, not a fraction explained.
Only two pot crops were replicated; this is descriptive, not a causal decomposition.
Scoring seeds stay fixed within crops, but differ between development and evaluation views.

One public scan per object, pilot selection and reused scans bound this evidence.
The experiment does not causally separate occlusion, mask source, depth error and shape prior.
No new generation, score, threshold or pack was selected by the diagnostic rerender.
Full tables, v4 reference markers and the audit sidecar are in outputs/mesh_generative/insights/.
