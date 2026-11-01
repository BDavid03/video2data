# 15 — Runtime and optimization

Status: completed optimization evidence reconciled; no new backend justified.
Owners: scratch_depth and affected producers. No profiling, compiler, numerical
comparison, test, validation, gradcheck or sanitizer was run in this pass.

Reuse `scratch_depth/{_optimization.py,compiler_experiments.py,_cuda_loss.py,cuda/}`,
the completed expanded campaign and the external-toolchain record in item 01.
Do not rerun successful variants or overwrite failed compiler outcomes.

| Existing matched scratch fit, seed 31 | Owned training wall s | Allocated peak MiB | Tuning teacher MAE |
|---|---:|---:|---:|
| Eager FP32 | 101.794 | 120.794 | 0.132154 |
| BF16 | 110.417 | 233.348 | 0.138548 |
| Channels-last | 88.518 | 111.462 | 0.148336 |
| Fused Adam | 84.634 | 120.809 | 0.146228 |
| BF16/channels-last/fused | 92.812 | 92.371 | 0.202355 |

Each is one owned fit including bookkeeping/I/O. Different numerical paths also
change training trajectories: these are not proof of equal numerical workload,
isolated acceleration or statistically replicated speedups. Retain the eager /
BF16 frozen research ensemble for its **tuning selection**, not a runtime or
holdout-based choice. Reserved peaks in the exact owning records are different
from allocated peaks. Sampled whole-device memory includes other processes and
may miss transients; unavailable memory is null, not zero.

Primary TorchInductor attempt
`outputs/scratch_depth/compile_experiment/20261018T115052.455224Z-8142557a/run.json`
failed `TritonMissing`, backend unexecuted. Primary custom CUDA attempt
`.../20261018T115112.404037Z-099fd706/run.json` failed because CUDA13.3 CCCL rejected
MSVC traditional preprocessor and requested `/Zc:preprocessor`. Do not label
eager fallback a compiled success.

The later external read-only Python/Torch2.12.1+cu132 attempt
`outputs/scratch_depth/external_cuda_experiment/20261018T115832.389940Z-02e77ccb/run.json`
completed compilation and one real training batch, zero optimizer steps:
59.7557 s compile, 83.1539 s total, warm eager/CUDA forward+backward
0.0048373/0.0043066 s for one call each. Historical loss difference
1.1920928955078125e-7 and maximum parameter-gradient difference
1.862645149230957e-9 were within recorded rtol1e-4/atol1e-6 for that sample.
This prior evidence establishes neither primary-environment compatibility,
general gradient correctness nor steady-state/end-to-end speedup. No new
correctness checks are authorized by the backlog.

Correctness approaches were investigated by documentation only:
[PyTorch gradcheck](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html)
compares finite-difference and analytic derivatives; its default tolerances target
double precision, with caveats at nondifferentiable points and overlapping tensor
storage. These current docs do not change the pinned runtime. NVIDIA's
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
provides memory, shared-memory race, uninitialized-memory and synchronization
checks. Both are deferred explicitly by the no-testing instruction; no successful
compiler call substitutes for those checks, and neither tool was invoked.

## Proposed measurements, not commands

Hypothesis: optimize a measured dominant component of an admitted stable
candidate; no component is yet demonstrated to dominate full-chain runtime.
First compare identical training-only source clips, grids, batch sizes, numerical
mode, parameters, output policy and thread settings. Frozen inventory: two
32-frame windows each from episodes0/11, side128, batch4, no new teacher target
generation. A completed representative workload must exist before backend timing.

Record decode, remap/preprocessing, CPU→GPU transfer, model, loss/backward/update,
GPU→CPU transfer, checkpoint and artifact I/O separately, including overlap
semantics. Cold first-call and compilation are distinct from warm costs; GPU
boundaries require explicit synchronization/events within an authorized research
measurement. Record allocated/reserved peaks and sampled device usage separately.
Retain process/environment/compiler version, source hashes and exact failure.

Proposed ceiling: one baseline and one demonstrated-bottleneck challenger,
three cold calls and twenty warm calls per arm (after five declared warmups),
≤15 minutes total, 2 GiB artifacts and unchanged memory/headroom limits. No
unbounded autotuning, installations, repeated search or silent fallback. Stop
on budget, admission or backend failure. Correctness/gradient/sanitizer evidence
is a separate **prohibited** prerequisite for production promotion, even if a
timing experiment is otherwise authorized. Existing-toolchain compilation may
be research-authorized but cannot create artifacts outside canonical Z:.

Investigated approaches: eager baseline retained; BF16, channels-last and fused
optimizer already assessed; TorchInductor failed in primary environment; custom
CUDA failed there and has limited external success; decode/transfer/I/O tuning
deferred pending decomposition. No new kernel or compiler dependency justified.
Exact blockers: full-chain stable workload/evidence absent, canonical storage
unwritable, correctness checks prohibited, no demonstrated material bottleneck.
Next item16 report distinguishes research evidence from production promotion.

## Source buildout: producer receipt accounting

Implemented `track3/features/runtime_measurement/` as a disabled feature-owned
request/handler/configuration/CLI slice. It consumes explicitly hashed,
producer-bound phase receipts; records cold/warmup/warm durations, partial
failures, missing phases, decode/preprocess/transfers/model/loss/update/compile/I/O
and end-to-end wall time; preserves allocated/reserved/sampled-device memory
separately. Workload declarations must match except backend before the report
marks them declared-comparable. No speedup, bottleneck or correctness claim is
derived. Historical producer source hashes and receipt identities are retained.

Added the dependency-free opt-in `track3/timing.py` span recorder for producer
integration. GPU host-wall spans require explicit caller synchronization; nested
and concurrent span times are not additive. Disabled recording invokes no timers.
New scratch `research_train` source now explicitly opts into producer-owned
timing with `trace_runtime`, false by default; no measurement receipt was
generated in this pass. Existing campaign artifacts were not altered or rerun.

Bounded protocol: one baseline/one challenger, 64 frames each, side128, batch4,
three cold/five warmup/twenty warm iterations, 900 seconds combined wall time;
receipt and span sizes are bounded. Failed producer receipts retain timings
without being treated as successful numerical workloads. All runtime flags are
false; public training/tuning only, excluding 41/43/44 before receipt access.

A separate whole-arm `aggregate_grouped` adapter admits the new scratch research
trace without pretending it is a per-episode matched benchmark. It binds the
terminal producer and prepared dataset, enforces frozen train/tuning episode
groups, and preserves sampled phases, whole-arm failures and distinct memory
fields. Its output is explicitly accounting-only and comparison-ineligible;
changing training parameters and first32 sampled examples cannot demonstrate
steady-state backend speedup. The 1800-second arm budget remains separate from
the 900-second matched benchmark protocol; exceeded arm time is exposed. Missing
RGB decode/full-pipeline phases and unavailable memory remain explicit.

Disposition: runtime accounting source implemented, unrun. Eager/BF16/channels-last/
fused optimizer/Inductor/custom CUDA retain their recorded historical outcomes
above. New backend and kernel work remains unjustified. Exact remaining blockers:
canonical storage unwritable; producers lack admitted phase receipts; stable
representative workload evidence absent; correctness checks prohibited. No tests,
checks, imports, profiling, compilation, training or dependency installation ran.
