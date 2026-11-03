# D1 outcome (context 19, 15:48 UTC 3 Nov 2026): NO-GO by the completion rule; tracking unmeasured

Protocol `docs/chain/dynamic-scaling-protocol.md` (committed `5941707`, 14:33:09 UTC, before any D1 run). Evidence:
[D1 slice documentation](../../track3/features/dynamic_feasibility/README.md#d1-scaling), `outputs/dynamic_feasibility/d1/sizing-fixed/` (ledger,
`audit.json`, `commands.json`).

- Sizing probe (reads no error): 256 / 512 / 1,024 worlds at 28.7k / 39.6k / 49.5k environment steps/s, peak 2.6 /
  3.3 / 4.1 GiB. The rule chose 1,024 worlds and 66,763,776 transitions per seed, frozen 14:48:28.27 UTC before seed 0
  started (14:48:28.28).
- Training ran at about 12.9-13.6k steps/s: the probe timed environment stepping without learner updates and
  checkpoints, so the budget was about 3.7 x too large. Both seeds hit their 30-minute allowance before the final
  checkpoint (seed 0: 21.6M transitions, 2,632 learner updates; seed 1: 22.6M, 2,757), with training VOC still 0.05.
  No final export exists, so the GO rule's completion condition fails: **NO-GO**. GPU 60.0 of 65 minutes.
- Foreseeable (critic-19): D0's own loop already sized L2 on measured training seconds per step with a 1.1 safety
  factor (`dynamic_feasibility/loop.py:189-191`, `queue.json:41`); D1 did not reuse it.
- Deviation from the protocol text, disclosed: the protocol said D1 would reuse D0's slice code with a new queue entry;
  it ran on a new sub-slice `dynamic_feasibility/scaling/` (its own loop, handler, probe), a copy of D0's loop
  machinery. A first D1 run stopped at 1.39 GPU-minutes on the setup error and printed a NO-GO
  (`outputs/dynamic_feasibility/d1/setup-failure.log`); it was superseded by the `d1/sizing-fixed/` run reported here,
  and both summaries are kept. The unscored partial checkpoints sit under `.runtime/tmp/dynamic-scaling-seed*-*/`.
- Deviation, disclosed: three probe attempts failed on an observation-interface error, were fixed before training,
  kept on record and charged to the caps. Nothing changed after selection; no seed retried.
- Reading: a protocol-design failure (the sizing rule measured the wrong throughput), not evidence about whether the
  policy stage can learn transport, and no minimum training time was measured. Stopping the dynamic route for this
  deadline is a planning decision, not a measured impossibility: transport is still unmeasured, video hand references
  need installs not yet approved, and about 20 hours remain for references, training and packing across the scored
  episodes. The structural effort moves to meshes (G1, `docs/chain/mesh-generative-protocol.md`, awaiting the owner's
  install approval). A future attempt should size on measured training throughput (about 13k transitions/s at 1,024
  worlds, 4.7 GiB) and run 3-4 h per seed, as the public recipe's 3.2 GPU-hours suggests.
