# Recycle 20261103T1614Z (engineering debt from critic-19)

Moved here instead of deleted (owner, 2026-11-03: "can move everything thats gonna be deleted into a recycle folder").
Each file keeps its original path below this folder; restore with `git mv recycle/20261103T1614Z/<path> <path>`.
Evidence: the D1 campaign now runs as `track3/features/dynamic_feasibility/scaling_queue.json` through the slice's one
loop, and the slice READMEs were merged; the D0, D1 and S1 summaries and figures re-render byte-identically from the
existing ledgers with the new code (18 of 18 SHA-256 matches, re-hashed by the lead). No file here is pinned by a pack
manifest; historical `outputs/` records still name these paths as their sources and stay valid.

| Original path | Why |
| --- | --- |
| `track3/features/dynamic_feasibility/scaling/__init__.py` | retired package marker; D1 is a queue of the parent slice |
| `track3/features/dynamic_feasibility/scaling/__main__.py` | replaced by the parent CLI with `--queue scaling_queue.json` |
| `track3/features/dynamic_feasibility/scaling/_sizing.py` | folded into the parent `_sizing.py`, which sizes on measured training throughput x 1.1 |
| `track3/features/dynamic_feasibility/scaling/handler.py` | training, leases and checkpoint retention folded into the parent `handler.py` |
| `track3/features/dynamic_feasibility/scaling/loop.py` | campaign routing and ledger persistence folded into the parent `loop.py` |
| `track3/features/dynamic_feasibility/scaling/measure.py` | folded into the parent `measure.py --campaign D1` |
| `track3/features/dynamic_feasibility/scaling/probe.py` | environment-only throughput probe; obsolete for sizing (the D1 failure cause) |
| `track3/features/dynamic_feasibility/scaling/queue.json` | its decision is preserved in the parent `scaling_queue.json` |
| `track3/features/dynamic_feasibility/scaling/runtime.json` | same flags in the parent `scaling_runtime.json` |
| `track3/features/dynamic_feasibility/scaling/README.md` | merged into the parent README, section "D1 scaling" |
| `track3/features/pose_insurance/README_shrink.md` | merged into the slice README, "Pose shrink control" |
| `track3/features/pose_insurance/H3_README.md` | merged into the slice README, "H3 table-world heading" |
| `track3/features/pose_insurance/SILHOUETTE_README.md` | merged into the slice README, "S1 silhouette diagnostic" |
| `track3/features/pose_insurance/SCALE_V3.md` | merged into the slice README, "Scored stereo-anchored S1 follow-up" |
