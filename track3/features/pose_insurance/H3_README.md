# H3 table-world heading

Implements the precommitted `docs/chain/pose-heading-h3-protocol.md` using additive CPU handlers.
The episode-19 adapter verifies the frozen HR2 method and source hashes, admits the exact D2 v4
frame-zero producers, and calls the original recorded registration operation. The heading loop
places saved scan rotations through the T1 measurement's signed camera-to-table basis, keeps
positions, scores only train 18, and aggregates with the other seven historical T1 rows.

Run in this order from the repository root:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.registration_h3_loop
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.heading_h3_loop
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B -m track3.features.pose_insurance.heading_h3_check
```

Queues live beside the handlers. Each loop has a trial/time budget, immutable per-trial
`run.json`, input/source identities, a resumable ledger, and invocation reports. Repeating
unchanged queues reuses completed trials. Predictions are recorded before the label-free
scored-scene comparison, which is recorded before train scoring reads mocap. Frozen HR2
acceptance is retained; H3's G3 separately admits stable fits, and episode-19 ambiguity makes
G2 required. All qualifying recorded-rival combinations enter G2 with the selected pair at
equal weight. Missing registration fails closed.

Rosters and gate thresholds live in the queues. Contract hashes bind those decisions to
the fixed protocol, and registration thresholds must also match the frozen HR2 method.
Handlers admit the runtime before importing numerical code. `_registration_h3.py` owns
the pure mask, validity and point-cloud extraction.

Outputs: `outputs/pose_insurance/registration-h3/latest.json`,
`outputs/pose_insurance/heading-h3/summary.md` and `summary.json`; immutable reports and
per-trial artifacts remain in their corresponding output directories. The summary includes
all five kinematic metrics, full 3D frame-zero b/C errors, A reproduction, signed-basis and
quaternion checks, G1/G2/G3, and the episode-19 relative-pose change. Historical H2S protocol
hash differences are disclosed without modifying old files. Existing runtime flags are reused.

The queue pins the durable pre-job inventory at
`outputs/pose_insurance/heading-h3/integrity-before.json`; the report compares the old
heading and HR2 inventories and bytes against it. The maintained `heading_h3_check.py`
entry point independently recomputes geometry from raw NPZ/mocap, imports no H3 module,
and reads the same durable inventory. Its command above writes
`outputs/pose_insurance/heading-h3/independent-check.json`. Scratch compilation and logs
stay under `.runtime/tmp/h3/`.

This operation stops at evidence. A D2 v5 build additionally requires the protocol's owner
approval, all required gates and the specified physical checks.

Latest result (3 November 2026): all required gates pass. H3-sel episode-18 AUC is
0.472910765; its eight-episode mean is 0.298238811 versus T1 0.276704640. The sweeper
flip returns episode-18 AUC to T1's 0.300637394; the equally weighted mean gain is
0.010767086, only 0.000767086 above G2's threshold. G1/G2 expected position metrics
improve, and G3 has both masks, loaded scans and stable registrations. Episode 19's
sweeper retains the original ambiguity decline (3.442% rival margin, 179.978 degrees);
its dust pan is accepted. This makes G2 required. Descriptive A reproduces every
historical metric exactly; A/B are not selected. Resume executes zero trials in both
loops (two registrations and four heading rows reused). The 72 old heading results,
57 HR2 trial/report files, and all 345 protected files remain byte-identical.
