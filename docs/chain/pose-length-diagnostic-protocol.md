# Pose length diagnostic protocol (fixed 2026-11-03, context 11, before any value is computed)

## Why

D2 v3 places slot 1 at its stereo-scaled (S1) table-plane offset from slot 0. On the eight train
two-object episodes (0, 11, 12, 18, 21, 23, 26, 31) the mocap/incumbent inter-object distance ratio is
0.23-0.73, so the video offsets are 1.4-4.4 times too long, and S1's mean absolute frame-0 length error
is 14.71 cm over the six changed episodes (`outputs/pose_insurance/shrink/summary.md`). The kit's
largest AUC threshold is 9 cm, so slot 1 earns no AUC while this error stands. A semantic-heading
route was declined (route call `route-heading-11`): even an exact direction leaves AUC at 0.271
because of the length error. This diagnostic asks where the length error comes from. It is
**descriptive only: it chooses nothing, changes no scene and repacks nothing.**

## Data and exclusions

Train episodes 0, 11, 12, 18, 21, 23, 26, 31 only (frame 0 of each, the static opening). Tuning 39/42
are single-object and give no offset. Episodes 41, 43, 44 are never read. Public mocap and public scans
enter as measurement only (as in the earlier pose protocols); nothing from them enters a scene.

## Quantities (per episode, horizontal table-plane distance between slot 0 and slot 1, cm)

- **M_o**: mocap distance between the two scan-frame origins (the scored quantity's reference).
- **M_c**: mocap distance between the two scans' axis-aligned bounding-box centres, each transformed by
  its mocap pose (where the scored origin sits relative to the object's geometric middle).
- **V_inc**: the incumbent video distance exactly as built for S1 (reproduced from the recorded scenes;
  must match the recorded S1 offsets within 1e-6 m).
- **V_moge**: MoGe-2 cluster centroids, unscaled; **V_stereo**: the same clusters' centroids from the
  calibrated ego_cam_b/c stereo depth points directly (metric, no MoGe), where stereo covers the
  cluster with at least 300 valid pixels; otherwise reported missing, never substituted.
- **V_hull**: the incumbent clusters' footprint-bounds centres (instead of point means) from the same
  points as V_inc, to show visible-surface centroid bias.
- Per episode also: cluster pixel counts, whether the two objects share a merged cluster or used the
  0.5 m default, and the camera-ray component of the offset error (error projected on the camera's
  horizontal viewing direction vs perpendicular to it).

## Report

A table of all quantities and their ratios to M_o; mean absolute length error vs M_o for each video
construction over the episodes where it exists (with the count); the same against M_c; the
decomposition origin offset (M_o vs M_c) vs video error (V vs M_c); along-ray vs cross-ray error.
Merged-cluster and default-offset episodes (0, 26, 31) are reported separately and also pooled.

## What a result can authorize (fixed now)

Nothing is adopted by this diagnostic. A **later, separately pre-registered candidate** is warranted
only if one video-only construction (V_stereo or V_hull, or a fixed combination named here: V_stereo
centroids with V_hull footprint centres) has mean absolute length error against M_o at least **5 cm
lower than V_inc** over the five clean episodes (11, 12, 18, 21, 23) and is lower on at least four of
them. Otherwise the length error is recorded as not explained by depth source or centroid choice, and
the pose line stays at v3.

## Amendment 1 (2026-11-03, after the train result, before any data on validation episodes exist)

Train result (`outputs/pose_insurance/length/summary.md`): V_stereo clean-five mean absolute length error
0.581 cm vs M_o (V_inc 11.754 cm), lower on 5/5, so the fixed test warrants a later candidate. Because
the result is surprisingly good, it gets one independent check on fresh public episodes before it is
believed: **validation 13, 22, 24** (two objects each, never used in any pose work). 2 and 27 are run if
their frame-0 clusters separate the two objects; otherwise reported as merged and not counted.

- Same code and definitions as above; new MoGe-2 frame-0 depth, mesh_extents clusters with identities
  assigned from the RGB overlay before any mocap is read, and calibrated stereo for each episode.
- V_inc is not defined there (no S1 scene); its stand-in is **V_s = V_moge x s**, s the episode stereo/MoGe
  scale computed exactly as `scale_stereo.py` does.
- **Confirmed** only if V_stereo mean absolute length error vs M_o over the counted episodes is below 3 cm
  and at least 5 cm below V_s, and V_stereo is lower than V_s on every counted episode. Otherwise the
  train result is recorded as not confirmed and no candidate follows.
- A confirmed result still adopts nothing. The candidate (slot-1 offset from direct stereo centroids)
  needs its own protocol with kinematic gates on the eight train episodes before any policy run or repack.
