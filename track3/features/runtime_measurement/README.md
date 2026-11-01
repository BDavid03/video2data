# Producer-owned runtime receipts

Source implementation only. `aggregate_authorized`, `benchmark_authorized` and
`validation_authorized` remain false. No timer, model, profiler, compiler, import
check, numerical comparison or test was executed for this feature. No package
was added. Execution requires canonical Z: storage and explicit episode admission.
Only public training/tuning episodes are accepted; 41/43/44 are excluded before
reading receipt contents. Historical compiler evidence remains in item 15.

This slice accounts for existing producer receipts. It does not run benchmarks,
launch producers, synthesize missing timings or interpret eager fallback as
compiled execution. Legacy records without receipts remain historical evidence;
their total wall time cannot reconstruct phase decomposition.

`track3.timing.PhaseRecorder` is a small opt-in in-memory helper. A producer can
use `span(phase, regime, iteration, device="cpu")`, and `snapshot()` returns spans.
GPU spans require a producer-supplied `synchronize` callback, called before/after
the span; the helper imports no GPU package. Exceptions retain failed spans and
propagate. The default disabled recorder invokes no clocks or synchronizers.
The producer remains responsible for its runtime gate, measurement authorization,
budget, receipt persistence, memory sampling and environment identity. This
pass does not instrument or execute existing producers automatically.

Receipt protocol version 1 is `runtime_trace.json` beside the producer's
`run.json`, under `outputs/<feature>/public/episode_NNNNNN/<run>/`. The producer
binds its SHA256 in `artifacts`; an aggregation request predeclares that same
hash. Terminal completed and failed video-only producers are accepted so partial
failure timings stay visible. The receipt contains:

- `schema_version: 1`, public `split`, `episode_index`, `producer_status` and
  `sources` identical to the producer's historical source path/hash entries.
- `workload`: exact `input_sha256`, `parameters_sha256`, `frames`, `max_side`,
  `batch_size`, `numerical_mode`, `output_policy`, `threads`, `seed`,
  `environment_identity`, `backend`. Input/parameter hashes identify the entire
  frozen workload, including selected clip/frame identities and training state.
  An environment identity must identify package/compiler/driver/thread settings
  and device. Backend is the actually executed backend, never a requested one.
- `wall_seconds` covering the producer's whole owned operation, not the sum of
  span durations. The owning producer must stop on budget; aggregation only
  refuses out-of-budget receipts after the fact.
- `memory` with distinct nullable integer-byte `allocated_peak_bytes`,
  `reserved_peak_bytes`, `sampled_device_peak_bytes`. Unavailable is null. Reset
  allocator peaks at the declared producer boundary; device sampling includes
  unrelated processes and can miss transients. These declarations cannot prove
  sampling completeness or memory headroom.
- `spans` from the recorder, phases `decode`, `preprocess`, `transfer_to_device`,
  `model`, `loss_backward_update`, `transfer_to_host`, `compile`, `artifact_io`,
  `end_to_end`; regimes `cold`, `warmup`, `warm`. Iteration indices start at zero
  separately per regime. Failed spans name the exception type. No stage is
  silently assigned zero when absent.

The request bounds one baseline plus at most one challenger, 64 frames per
receipt, side128, batch4, ≤3 cold iterations, ≤5 warmups and ≤20 warm iterations,
≤4096 spans/receipt and ≤900 seconds combined producer wall time. Receipt files
are limited to 4 MiB. This accounting operation creates only a small JSON report;
it does not authorize new measurement execution or grant a separate run budget.

The handler reports raw successful span durations, failures, missing phases,
whole-operation wall time and all three memory fields. Concurrent/nested spans
are never summed. Cold calls, warmups, warm calls and compilation stay separate.
Declared workload equality excludes only backend; differing numerical modes,
inputs, parameter state or output policy refuse the matched declaration. Even
equality does not demonstrate equal arithmetic or numerical correctness, and no
speedup or dominant-bottleneck conclusion is computed. Historical source hashes
are preserved rather than compared with current modified source. Receipt hashes
establish identity, not authenticity of a producer's declarations.

The new scratch `research_train` source can optionally emit a different,
whole-training-arm trace via `trace_runtime` (false by default). Its dedicated
`--grouped-accounting` adapter uses `grouped-accounting.json`, a separate false
`aggregate_grouped_authorized` gate, and explicit actual training-episode
admission. It binds the trace to terminal `scratch_depth.research_train`, its
exact prepared dataset, train/tuning partition inventory and historical sources.
It accepts completed/failed/interrupted arms and preserves their failures.

This adapter reports **accounting only**, with `comparison_eligible: false`.
The first 32 sampled clip examples change parameters throughout training; they
are not repeated identical warm benchmark calls. NPZ reads are identified as
decode, while upstream RGB video decode is unmeasured. The arm's training budget
is 1800 seconds, separate from the 900-second matched benchmark proposal. Recorded
over-budget arms are flagged rather than reclassified as compliant benchmarks.
The report names actual executed training episodes, preserves the declared
groups, never allocates whole-arm runtime to individual episodes, and exposes
missing phases and null unavailable memory. Full end-to-end spans include
initialization/checkpoint after admission; other spans cover a sampled subset.

Unrun limitations: neither trace producer nor aggregation was executed in this
pass; historical records lack this phase breakdown. Admitted measurement is
needed before actual decomposition exists. Device events, independent profiling,
gradient comparisons and sanitizers remain separately constrained. No new kernel
or compiler dependency is justified by this source implementation.
