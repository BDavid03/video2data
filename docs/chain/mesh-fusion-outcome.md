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

**Why it is a robust kill, not only a strict threshold.** A lead diagnostic outside the protocol
(`outputs/mesh_fusion/lead-span-check.py`, output beside it) takes the *rejected* poses at face value. The camera
centre moves 2.5 cm (rack) and 5.3 cm (pot) over the pre-contact window at about 0.59 m from the object: an angular
span of 2.4-2.5 degrees and 4.9-5.2 degrees, below the 10-degree gate even if every frame had been accepted. The hand
reaches the scene within about 0.6 s (first possible contact at frame 18 / 19, clear by 20), so the static part of these
clips holds almost no new viewpoint.

**Execution note.** The first Stage A run mis-counted within-frame region growth as additional support (41.5 % / 18.4 %
with only the identity frame accepted). Codex fixed the baseline and re-ran with no threshold, frame or parameter
change; both verdicts were KILL either way by frame acceptance. `outputs/mesh_fusion/corrections.md` names the
superseded runs.

**Consequence.** Multi-frame reconstruction of these objects must use frames after contact, where the object moves with
the hand: that needs per-frame object masks and object-motion compensation (masks through the clip need SAM2, an
unapproved install; `NEEDS-YOU.md`). The deformation alternative (IDEAS 56) inherits the same viewpoint shortage. D1 v4
stays the mesh file. A learned shape prior (G1, Hunyuan3D-2mini, awaiting approval) is the remaining structural mesh
route that does not need more viewpoints.
