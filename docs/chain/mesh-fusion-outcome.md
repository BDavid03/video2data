# Mesh fusion F1 outcome (context 20, 3 Nov 2026): Stage A KILL for both objects; D1 v4 stands

Protocol `docs/chain/mesh-fusion-protocol.md` (committed `277c7df` before any fused geometry). Evidence:
`outputs/mesh_fusion/summary.md`, run records `outputs/mesh_fusion/stage_a/public12/20261103T161154.712555Z-662a7738/`
(rack, train 12) and `.../public42/20261103T161158.370986Z-8e0ad798/` (pot, tuning 42). Slice
`track3/features/mesh_fusion/`. No scan was opened and no CD-O was computed.

| Object | Pre-contact frames | Candidates | Accepted | Frame 5 / 10 / 15 reprojection (px) | Background disagreement (mm) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Dish rack, train 12 | 0-17 | 0/5/10/15 | 0 only | 2.18 / 4.53 / 4.09 | 31 / 60 / 60 | KILL |
| White pot, tuning 42 | 0-18 | 0/5/10/15 | 0 only | 1.65 / 3.38 / 5.13 | 34 / 63 / 87 | KILL |

Frozen acceptance: >= 30 PnP inliers, median reprojection <= 1 px, independent background disagreement <= 3 mm. Every
later frame had enough inliers (121-455) but failed both error limits, so each object has fewer than 3 accepted frames.

**What the kill does and does not show.** The verdict is the frame-count rule: this registration pipeline accepted no
second frame in either window. A lead diagnostic outside the protocol (`outputs/mesh_fusion/lead-span-check.py`, output
beside it) takes the *rejected* poses at face value: the camera centre moves 2.5 cm (rack) and 5.3 cm (pot) at about
0.59 m from the object, an angular span of 2.4-2.5 and 4.9-5.2 degrees. That is an uncertain estimate from poses that
failed their own checks, and it is not a kill by itself: the rule needs low angle *and* less than 10 % added support,
and the misregistered clouds would add 68-75 % voxels (noise, not shape; Codex critic-20). Bounded claim: in these two
pre-contact windows (frames 0-17 / 0-18, about 0.6 s before the hand arrives) our background-PnP registration could not
add a view, and the rough motion estimate suggests little viewpoint change to gain. Other episodes, longer static
intervals or a better registration are untested.

**Execution note.** The first Stage A run mis-counted within-frame region growth as additional support (41.5 % / 18.4 %
with only the identity frame accepted). Codex fixed the baseline and re-ran with no threshold, frame or parameter
change; both verdicts were KILL either way by frame acceptance. `outputs/mesh_fusion/corrections.md` names the
superseded runs.

**Consequence.** More views of these objects would come from frames after contact, where the object moves with the
hand; that needs per-frame object masks and object-motion compensation, which this project has not built (our masks are
opening-frame clusters; SAM2 through the clip is one option, an unapproved install, `NEEDS-YOU.md`). The deformation
alternative (IDEAS 56) would face the same shortage of registered views. D1 v4 stays the mesh file. A learned shape prior (G1, Hunyuan3D-2mini, awaiting approval) is the remaining structural mesh
route that does not need more viewpoints.
