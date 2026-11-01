# Scoreboard

Each lens is scored 0-5 by a fresh-context critic, never by the context that did the work (see
`PROGRAM.md` section 2). A score rises only with new evidence. Two critics score each pass, a
Claude sub-agent and Codex; where they differ the lower score is kept.

| Lens | Score | Evidence | Next improvement | Last scored |
| --- | --- | --- | --- | --- |
| Submission readiness | 2 (Claude 2, Codex 2) | D1 v1 packed, rows identical to the kit sample, vertex extents equal to the build's, preflight "Ready to submit" (`submissions/20261101T1500Z-d1-v1/`, `SUBMISSION.md`); placeholder URL; D2, D3, D4 open | owner push, then repack with the real URL; start the static-objects D2 insurance file | 02, 2026-11-01 |
| Models | 2 (Claude 2, Codex 2) | mesh stage: video-measured boxes 1.008 cm per-object vs cube 2.52 and scan-PCA oracle 1.17 (`outputs/mesh_eval/d1-ctx02/v1-box.json`); proportions consistent across 2-9 episodes per object; shape chosen on the scans (disclosed); spoon (10 pairs) unchecked; depth line frozen; no pose/policy model | a written tuning protocol (leave-objects-out on the scans) before the next mesh choice; pot and spoon shapes (IDEAS 15, 17) | 02 |
| Metrics | 3 (Claude 3, Codex 3) | pair-weighted CD-O beside the per-object mean, recomputed by both critics (video box 1.257 vs cube 2.77 vs oracle 1.34, 26 scanned pairs); kit pose functions; depth gates | per-pair seeds (`_object_seed(episode, slot)`) and provenance (hashes, command) in the `local_cdo.py` output | 02 |
| Visualisations | 2 (Claude 2, Codex 2) | `outputs/mesh_eval/d1-ctx02/d1-v1-cdo.png` (per-object bars: cube, oracle, video; pair counts); per-episode measure overlays | one panel per object: frame-0 overlay, box, scan; show the pot and rack failures | 02 |
| Presentation | 1 (Claude 1, Codex 2) | depth-only draft `main.tex`, unbuilt, describes MAE-only selection, count mismatch, no anonymous option | install Tectonic, build an anonymous two-page Track 3 skeleton PDF whose numbers match the records | 01 |
| Insights | 2 (Claude 3, Codex 2) | box vs ellipsoid ablation; input-split ablation (9 common objects: eval-only 1.024 vs public-only 1.063 cm); floor-not-table finding (IDEAS 18); scale-free proportions; white_pot is the one object where the box loses to a cube (1.82 vs 1.79) | isolate why video boxes beat the scan-PCA oracle (a table-aligned bounding-box oracle, analysis only); explain white_pot | 02 |
| Engineering | 2 (Claude 2, Codex 2) | `mesh_extents` slice with gates, run.json, video_only admission, revision and duplicate checks, assignment guard (reference pixel counts); Codex review fixes; requests in `scripts/track3/requests/d1/` | one manifest-driven `d1_reproduce.sh` that finds depth runs by (split, episode); provenance in the `local_cdo.py` output | 02 |
