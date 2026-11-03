# D1 outcome (context 19, 15:48 UTC 3 Nov 2026): NO-GO by the completion rule; tracking unmeasured

Protocol `docs/chain/dynamic-scaling-protocol.md` (committed `5941707`, 14:33:09 UTC, before any D1 run). Evidence:
`track3/features/dynamic_feasibility/scaling/README.md`, `outputs/dynamic_feasibility/d1/sizing-fixed/` (ledger,
`audit.json`, `commands.json`).

- Sizing probe (reads no error): 256 / 512 / 1,024 worlds at 28.7k / 39.6k / 49.5k environment steps/s, peak 2.6 /
  3.3 / 4.1 GiB. The rule chose 1,024 worlds and 66,763,776 transitions per seed, frozen 14:48:28.27 UTC before seed 0
  started (14:48:28.28).
- Training ran at about 12.9-13.6k steps/s: the probe timed environment stepping without learner updates and
  checkpoints, so the budget was about 3.7 x too large. Both seeds hit their 30-minute allowance before the final
  checkpoint (seed 0: 21.6M transitions, 2,632 learner updates; seed 1: 22.6M, 2,757), with training VOC still 0.05.
  No final export exists, so the GO rule's completion condition fails: **NO-GO**. GPU 60.0 of 65 minutes.
- Deviation, disclosed: three probe attempts failed on an observation-interface error, were fixed before training,
  kept on record and charged to the caps. Nothing changed after selection; no seed retried.
- Reading: a protocol-design failure (the sizing rule measured the wrong throughput), not evidence about whether the
  policy stage can learn transport. With moving-object policies also needing video hand references (installs not
  approved) and per-episode training of at least an hour on this GPU, the dynamic route cannot land a pack by 12:00 UTC
  4 Nov. The structural effort moves to meshes (G1, `docs/chain/mesh-generative-protocol.md`, awaiting the owner's
  install approval). A future attempt should size on measured training throughput (about 13k transitions/s at 1,024
  worlds, 4.7 GiB) and run 3-4 h per seed, as the public recipe's 3.2 GPU-hours suggests.
