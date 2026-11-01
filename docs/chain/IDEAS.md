# Ideas ledger

The road-less-travelled record (see `PROGRAM.md` section 3). One row per idea, kept whether it
won or lost. A lost idea is a finding. Source: Claude (C), Codex (X), scout sub-agent (S).

| # | Stage | Idea | Why it might win | Cost | Cheapest test | Result | Context |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mesh | Submit a box/ellipsoid with each object's three principal extents measured from video (C) | CD-O removes scale and pose; proportions are most of the shape signal | low (CPU, existing depth/stereo) | measure extents on public episodes, score with `local_cdo.py` | analysis: oracle-extents box 1.17 cm, ellipsoid 1.18, cylinder 1.31 vs plain cube 2.52, sphere 2.82 (`outputs/mesh_eval/`); video-measured extents not yet tried | 01 |
| 2 | mesh | One reconstruction per physical object, pooled across every scored episode it appears in (S, X) | 10 episodes show the spoon, 6 the pot: more views, less occlusion; each pair is scored independently | low-medium (cross-episode registration) | single-episode vs pooled extents on public cups/pot | not tried; Codex: pool selectively, clean views only | 01 |
| 3 | mesh | Low-dimensional procedural shapes fitted to video: revolved profiles (cups, pot, lid, pitcher), bowl+handle spoon, bar rack (X, S) | the hull fills cavities; area-weighted samples reward hollow walls and thin parts | medium | dust pan on public ep 39: hull vs tray/rim/handle model from three clean views, 3 seeds | not tried | 01 |
| 4 | mesh | Stereo ego_cam_b/c depth instead of monocular MoGe2 (S) | metric, no teacher bias | low (`stereo_depth` slice exists) | extents from stereo vs MoGe on one public episode | not tried | 01 |
| 5 | mesh | Visual hull from segmentation masks over the clip (S) | captures silhouette concavities from many views | high: SAM2/Grounding DINO not installed (owner approval) | - | parked: needs a new dependency | 01 |
| 6 | metrics | Score candidates against budget-decimated references, through the host's own `budget_mesh` (X) | exact parity with the host, which decimates both sides | done | - | built into `scripts/track3/local_cdo.py`; self-check 0.000 cm | 01 |
| 7 | mesh | Spend effort by score weight: spoon, pot, blue cup, rack are 26/36 = 72 % of CD-O (X) | a 1 cm spoon gain moves the mean by 0.28 cm | none | - | adopted as priority order | 01 |
| 8 | mesh | Submit the public scans decimated to budget (S) | near-zero CD-O on 10 objects | - | - | rejected: the FAQ says submit your own reconstructions | 01 |
| 9 | pose | Objects-stay-at-first-pose baseline: score it on public mocap with the kit pose metrics (S) | slot 0's first pose is aligned for free; SP-SR allows 10 cm / 0.5 rad; static objects need no grasp | very low (CPU) | per public episode, AUC/SP-SR/MPPE of a constant pose vs mocap | not tried | 01 |
| 10 | pose | Optimise the hand *reference* so a lightly trained policy drags the object along the video track (sim in the loop) (S) | the policy acts as residuals on the reference, so the reference is the real control | high | one public episode, zero-update checkpoint, 1 world | not tried | 01 |
| 11 | pose | Track object 6-DoF from stereo with our own meshes, smooth over the clip, use it as the task reference (S) | scale error is scored; stereo gives metric scale | medium-high | track one public episode, compare with mocap offline | not tried | 01 |
| 12 | pose | Anchor slot 1's start relative to slot 0 from stereo geometry (S) | only slot 0's first pose is aligned for free; RPE and MP-SR need the relative placement | low once tracks exist | - | not tried | 01 |
| 13 | depth | Use the project's 256 px depth student as a fast dense depth for fusion (C) | ONNX, runs on every frame cheaply | low | extents from student vs MoGe on one public episode | not tried | 01 |
