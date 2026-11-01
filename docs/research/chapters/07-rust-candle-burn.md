# Rust, Candle and Burn for depth estimation

## Scope and conclusion

This chapter evaluates Rust execution paths for the existing randomly initialized depth CNN and, prospectively, the surrounding video perception pipeline. Research cutoff: **2026-10-18**. This is a source-based design study. No dependency installation, model execution, compilation, test, validation command or benchmark was performed. Existing environments remain read-only, episode 41 remains excluded, and a future implementation introducing production dependencies still requires confirmation.

The practical recommendation is to retain PyTorch as the reference and consider two separately scoped Rust branches: a faithful FP32 inference transplant in Candle, and a broader training/compiler investigation in Burn. There is a concrete obstacle to training the current decoder on either reviewed stock GPU path: **Candle 0.11.0 rejects bilinear upsample backward; Burn 0.22.0 delegates interpolation backward to CubeK 0.3.0, which implements nearest-neighbor mode only.** Replacing bilinear with nearest-neighbor resize changes the architecture. It does not resolve a parity exercise. [Candle release backward source](https://raw.githubusercontent.com/huggingface/candle/0.11.0/candle-core/src/backprop.rs), [CubeK release interpolation source](https://raw.githubusercontent.com/tracel-ai/cubek/v0.3.0/crates/cubek-interpolate/src/lib.rs).

Rust is an implementation language, not a model architecture or a new source of geometric supervision. Changing framework does not remove teacher bias, increase episode diversity, fix temporal leakage, or establish physical depth accuracy. The deployment and compiler questions must remain separate from the scientific question of whether the learned representation transfers.

## Version and evidence ledger

| Component | Reviewed baseline | Evidence and interpretation |
|---|---|---|
| Candle | 0.11.0, published 2026-06-26 | The official package history identifies this release before the cutoff; implementation links below use tag `0.11.0` |
| Burn | 0.22.0, released 2026-10-06, commit `b8d75ad` | Major public API change; implementation links use `v0.22.0` |
| CubeCL | 0.11.0 | Version declared by the Burn 0.22 release workspace |
| CubeK | 0.3.0 | Version declared by the same workspace; interpolation derivative inspected at `v0.3.0` |
| Rust compiler for Burn | `rust-version = "1.95"` | Declared by the Burn 0.22 package; no local compiler compatibility assessment was executed |

Sources: [Candle package history](https://docs.rs/crate/candle-core/latest), [Burn 0.22 release](https://github.com/tracel-ai/burn/releases/tag/v0.22.0), [Burn release dependency manifest](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/Cargo.toml), [Burn package manifest](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn/Cargo.toml).

The package-history link is a rolling page: the observed history supplies the publication date; it is not a reproducible source snapshot. Release tags improve specificity but a later implementation should also record resolved commit hashes, crate checksums and the complete Cargo lockfile. A dependency range such as `0.11.0` ordinarily permits compatible patch updates; it is not an immutable dependency closure.

Official hosted documentation can mix generations. An inspected Burn interpolation page identified itself as 0.19.1, while release-tagged 0.22 source includes alignment configuration and different output-size constraints. Release source takes precedence here. Rolling documentation is marked as such and is used to explain architecture, not to claim an uninspected release feature.

## Framework architecture and current API

### Candle

Candle separates tensor/device functionality (`candle-core`), layers and optimizers (`candle-nn`), and existing model implementations (`candle-transformers`). The official repository explicitly includes training. It is inaccurate to characterize Candle as an inference-only framework. Its direct tensor programming model is attractive for a small, manually transcribed network, but a forward API and an autodiff framework do not imply that every forward operation has a derivative. [Official Candle repository](https://github.com/huggingface/candle).

The release feature manifest exposes optional CUDA, cuDNN, NCCL, Metal, MKL and Accelerate integration. `cudnn` implies CUDA. The inspected `candle-core` manifest has no WGPU feature. Those feature flags select different native-library and target requirements; they do not make all backends interchangeable. [Candle 0.11 feature manifest](https://raw.githubusercontent.com/huggingface/candle/0.11.0/candle-core/Cargo.toml).

The distinction between `Tensor` and trainable `Var` is operationally important. A port must create trainable parameters as variables and preserve their identities through the forward graph. Loading inference tensors and then calling backward is not automatically a training transplant. The model should retain an explicit parameter registry with stable names, shapes and dtypes.

### Burn

Burn 0.22 removes backend type parameters from public models and tensors. Current application code uses `Tensor<D>` and a runtime-selected `Device`; training enables gradients through `device.autodiff()` before model and input initialization. The release removed the Candle backend and deprecated NdArray and LibTorch. Older examples using `Tensor<B, D>` describe a previous API generation and should not be copied into a new design. [Burn 0.22 release](https://github.com/tracel-ai/burn/releases/tag/v0.22.0).

The release manifest separately exposes CUDA, ROCm, WGPU, Vulkan, WebGPU, Metal, CPU and Flex backends; training, storage, extensions, fusion, capture and autotuning have their own features. This makes a Rust training branch plausible, but increases configuration surface. A future artifact needs both the selected backend and the enabled feature set in its provenance. [Burn 0.22 package configuration](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn/Cargo.toml).

Burn's backend abstraction and public tensor rank serve different purposes. `Tensor<4>` constrains rank in the Rust type system; dimensions such as batch, channels, height and width remain values. Neither static rank nor memory safety establishes that two tensors have compatible channel counts for concatenation or correct spatial axes. Explicit shape contracts remain necessary.

### Comparative judgment

| Question | Candle route | Burn route |
|---|---|---|
| Best initial project use | Focused inference transplant | Training/backend/compiler research branch |
| Model representation | Direct Rust composition of tensor/layer calls | Rust modules and current runtime device dispatch |
| GPU choices examined | CUDA and Metal features | CubeCL-backed CUDA/WGPU family |
| Existing decoder forward | Bilinear API accepts explicit corner alignment | Interpolation supports explicit output size and alignment |
| Existing decoder GPU training | Missing bilinear derivative | Missing bilinear derivative in CubeK dependency |
| Training control | Explicit parameter/gradient/optimizer loop | Optimizers, autodiff and training infrastructure |
| Main scientific risk | Accidentally changing operators while porting | Confusing abstraction-level support with backend support |
| Main engineering risk | Manual parity work and derivative implementation | API churn, compiler dependencies, caches and backend constraints |
| Performance conclusion | Unmeasured | Unmeasured |

The final two rows are judgments for this project, not framework benchmark claims. Rust can reduce some host-language overhead; GPU kernel selection and memory behavior still determine much of the execution cost.

## Exact local model contract

The reference is [the existing network](../../../track3/features/scratch_depth/_network.py), with execution options in [the optimization adapter](../../../track3/features/scratch_depth/_optimization.py). The network consumes RGB `[0,1]` in NCHW layout and predicts positive optical-axis depth in N1HW layout. It rejects non-four-dimensional input, channels other than three and spatial dimensions below eight.

Each convolution block is

\[
\mathcal B_{a\to b}(x)=\operatorname{ReLU}\bigl(W_2*\operatorname{ReLU}(W_1*x+b_1)+b_2\bigr),
\]

where both kernels are 3 by 3, padding one, stride one, with biases. For base width \(c=24\), encoder widths are \(c,2c,4c\); the bottleneck has width \(8c\) and channel dropout probability 0.1 by default. Decoder input/output widths are \(12c\to4c\), \(6c\to2c\), and \(3c\to c\). The concatenation order is **resized decoder features first, skip features second**.

\[
e_1=\mathcal B_{3\to c}(x),\quad
e_2=\mathcal B_{c\to2c}(P(e_1)),\quad
e_3=\mathcal B_{2c\to4c}(P(e_2)),
\]
\[
b=\operatorname{Dropout2d}(\mathcal B_{4c\to8c}(P(e_3))),
\]
\[
d_3=\mathcal B_{12c\to4c}([U(b,e_3),e_3]),\quad
d_2=\mathcal B_{6c\to2c}([U(d_3,e_2),e_2]),
\]
\[
d_1=\mathcal B_{3c\to c}([U(d_2,e_1),e_1]),\qquad
\widehat d=\operatorname{softplus}(W_h*d_1+b_h)+10^{-3}.
\]

Here \(P\) is max pool with kernel and stride two, and \(U(a,s)\) resizes to the actual spatial shape of \(s\) using bilinear interpolation with `align_corners=false`. It must not assume that all inputs are divisible by eight: the reference supports odd spatial dimensions and uses skip sizes to reconcile them.

For the contemplated batch-four, width-128, height-96 case:

| Activation | NCHW shape | FP32 activation bytes, excluding allocator overhead |
|---|---|---:|
| Input | `[4,3,96,128]` | 589,824 |
| First skip | `[4,24,96,128]` | 4,718,592 |
| Second skip | `[4,48,48,64]` | 2,359,296 |
| Third skip | `[4,96,24,32]` | 1,179,648 |
| Bottleneck | `[4,192,12,16]` | 589,824 |
| Final concatenation | `[4,72,96,128]` | 14,155,776 |
| Output | `[4,1,96,128]` | 196,608 |

These are arithmetic storage estimates, not measured peak memory. Training retains additional intermediates and gradients, convolution algorithms may require workspaces, and aliases or fusion may avoid some materializations. The large final concatenation illustrates why copying skip tensors and layout conversion can matter even in a relatively small CNN.

## Operator support and parity

| Required operation | Candle 0.11 | Burn 0.22 / CubeCL | Required treatment |
|---|---|---|---|
| Conv2d, including head | Forward and backward source paths | Forward and input/weight backward source paths | Match weight layout, bias, padding and stride |
| ReLU | Tensor primitive | Activation primitive | Preserve operation and zero convention |
| MaxPool2d(2) | Backward requires kernel equal to stride | Indexed forward/backward integration | Inspect tie-gradient behavior; do not infer exact parity |
| Bottleneck Dropout2d | General dropout is elementwise | General dropout is elementwise | Build `[N,C,1,1]` channel mask |
| Bilinear resize forward | Explicit target height/width and alignment | Explicit output size and alignment | Pass actual skip dimensions, alignment false |
| Bilinear resize backward | Explicitly unsupported | CubeK nearest-only derivative | Custom derivative or differentiable primitive composition |
| Channel concat | Tensor concat | Tensor concat | Axis one, decoder before skip |
| Softplus | Stable primitive composition can express it | Softplus operation available | Match reference numerical branch/threshold |
| Valid masked log loss | Primitive composition | Primitive composition | Supplied mask AND finite positive teacher depth |
| Adam with coupled weight decay | Built-in AdamW is a different decay convention | Adam implementation available | Preserve effective-gradient rule and state |
| Save/load | Named VarMap and SafeTensors | Burnpack, SafeTensors and PyTorch interoperability | Explicit names, dimensions, adapters and metadata |

Release evidence: [Candle differentiation source](https://raw.githubusercontent.com/huggingface/candle/0.11.0/candle-core/src/backprop.rs), [Burn CubeCL module integration](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-cubecl/src/ops/module.rs). The derivative sources establish implementation paths, not measured equality of floating-point results. In particular, comparing gradients at max-pool ties needs the reference's exact index-selection convention.

## Bilinear interpolation and its missing derivative

### Forward coordinates

For output row \(j\), area-based pixel-center mapping is

\[
s_j=(j+\tfrac12)\frac{H_{\rm in}}{H_{\rm out}}-\tfrac12.
\]

After edge handling, define lower and upper input indices and interpolation fraction \(\alpha\); define horizontal fraction \(\beta\) analogously. Each output is

\[
y_{j,k}=(1-\alpha)(1-\beta)x_{i_0,\ell_0}
 +(1-\alpha)\beta x_{i_0,\ell_1}
 +\alpha(1-\beta)x_{i_1,\ell_0}
 +\alpha\beta x_{i_1,\ell_1}.
\]

For corner alignment, the corresponding nondegenerate coordinate is instead \(s_j=j(H_{\rm in}-1)/(H_{\rm out}-1)\). These expressions produce different edge and interior values. A one-pixel output dimension requires an explicit special case, so a generic implementation must not divide blindly by \(H_{\rm out}-1\).

Candle exposes `upsample_bilinear2d(H,W,align_corners)`. Its `interpolate2d` method is nearest-neighbor interpolation, despite the generic name. [Candle tensor API, observed version 0.11](https://docs.rs/candle-core/latest/candle_core/struct.Tensor.html). Burn's release module defaults `align_corners` to **true** and requires exactly one of output size or scale factor; both defaults must be handled deliberately. [Burn 0.22 interpolation module](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-nn/src/modules/interpolate/interpolate2d.rs).

### Reverse-mode derivative

With fixed shapes the resize is linear: \(y=Ax\). Therefore

\[
\nabla_xL=A^\mathsf T\nabla_yL.
\]

A forward output contributes its upstream gradient, multiplied by each interpolation weight, to four input pixels. Contributions from every overlapping output must accumulate. When clamped boundary indices coincide, their weights contribute to the same input element. A scatter implementation may require atomic additions; an alternative gather implementation assigns one input-gradient element per worker and sums all contributing outputs. The latter can avoid concurrent writes but needs a correct contributing-output range.

Candle's pinned implementation explicitly errors for `UpsampleBilinear2D` backward. Burn's autodiff wrapper records state and delegates to the inner backend; it does not generate the missing derivative. Its CubeCL integration reaches CubeK, whose pinned mode match accepts only nearest-neighbor backward. [Burn autodiff wrapper](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-autodiff/src/ops/module.rs), [Burn interpolation integration](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-cubecl/src/kernel/interpolate/base.rs), [CubeK derivative mode match](https://raw.githubusercontent.com/tracel-ai/cubek/v0.3.0/crates/cubek-interpolate/src/lib.rs).

### Prospective implementation choices

One faithful approach composes differentiable gathers, fixed interpolation weights, multiplications and sums. Its advantage is an inspectable chain rule using existing primitives; its cost may be temporary tensors and extra launches. Another implements a custom operator and explicit backward. Candle's custom-operation interface includes backend forward hooks and a `bwd` hook. [Candle custom-operation API](https://docs.rs/candle-core/latest/candle_core/trait.CustomOp1.html).

For a custom implementation, retain source dimensions, destination dimensions and exact coordinate conventions in immutable operator state. Do not retain an entire input tensor merely to compute a derivative whose weights depend only on shape. Boundary arithmetic, accumulation dtype and deterministic ordering are independent choices. They should be documented before any later comparison is run.

Inference remains a useful first target because it does not require this backward kernel. A training investigation should not begin by quietly substituting a different resize operation or changing decoder padding to avoid the problem.

## Dropout and train/evaluation behavior

The reference uses channel dropout at the bottleneck:

\[
m_{n,c}\sim\operatorname{Bernoulli}(1-p),\qquad
Y_{n,c,h,w}=X_{n,c,h,w}\frac{m_{n,c}}{1-p}.
\]

The mask shape is `[N,C,1,1]`, broadcast over space. Its expectation preserves each activation, but its spatial covariance differs from elementwise dropout. Elementwise masks can leave fragments of every feature map; channel masks remove a whole feature map for a sample.

Candle's reviewed dropout samples the full input shape and takes an explicit training flag. Burn's reviewed dropout also creates an input-shaped random mask; application depends on both a training flag and an autodiff device. Using either ordinary module directly would change `Dropout2d` semantics. [Candle release dropout source](https://raw.githubusercontent.com/huggingface/candle/0.11.0/candle-nn/src/ops.rs), [Burn release dropout source](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-nn/src/modules/dropout.rs).

An exact inference transplant disables dropout. A faithful training transplant must preserve mask granularity and scaling, while recognizing that sharing an integer random seed across frameworks does not ensure identical random streams or masks. Equivalent stochastic algorithms need not have identical training trajectories.

## Loss mathematics and numerical behavior

The supplied validity map is part of supervision. The exact usable-pixel mask is

\[
M_i=\operatorname{bool}(v_i)\land\operatorname{isfinite}(t_i)\land(t_i>0),\qquad K=\sum_iM_i.
\]

The local reference rejects \(K=0\). For usable labels it retains teacher scale:

\[
L=\frac1K\sum_{i:M_i=1}|\log\widehat d_i-\log t_i|.
\]

This is neither scale-aligned depth error nor a confidence-weighted loss. Reducing per-image means and then averaging would weight images differently when valid-pixel counts differ. The Rust route must preserve the global selected-pixel mean used by the reference.

Invalid teacher values must be replaced **before** logarithms. The optimization adapter also protects masked-out predictions before calling the eager loss and checks that supervised predictions are finite and positive. A robust transcription uses safe value one wherever \(M_i=0\), making log contributions zero while preserving zero gradient at invalid locations. Multiplying a NaN by a zero mask after arithmetic is insufficient.

For valid pixels away from the absolute-value kink:

\[
\frac{\partial L}{\partial\widehat d_i}=
\frac{\operatorname{sign}(\log\widehat d_i-\log t_i)}{K\widehat d_i},
\]
\[
\frac{\partial L}{\partial z_i}=
\frac{\operatorname{sign}(\log\widehat d_i-\log t_i)}{K\widehat d_i}\sigma(z_i).
\]

The positive offset bounds prediction away from zero, but it does not make arbitrary pre-head arithmetic finite. Stable softplus can be written

\[
\operatorname{softplus}(z)=\max(z,0)+\log(1+\exp(-|z|)).
\]

This avoids large positive exponentials. Exact reference matching also includes the selected threshold/branch convention, dtype casts and derivative at branch boundaries. Those are numerical details, not changes to the intended depth function.

Candle's dtype enum includes BF16, F16, F32 and F64, but enum membership is not evidence that every operator/backend supports each dtype. [Candle dtype API, observed version 0.11](https://docs.rs/candle-core/latest/candle_core/enum.DType.html). The project's BF16 option retains FP32 master parameters and performs logs/reduction outside autocast in FP32. A blanket conversion of all Rust parameters and losses to BF16 would be a different recipe.

## Gradients and optimizer equivalence

For any node \(y=f(x_1,\ldots,x_m)\), reverse mode propagates

\[
\overline x_k\mathrel{+}=\left(\frac{\partial f}{\partial x_k}\right)^\mathsf T\overline y.
\]

Skip paths require additive accumulation into shared ancestors. Concatenation partitions an upstream gradient using the original channel offsets; broadcast operations sum over replicated dimensions. The optimizer must receive the gradients for the same parameter identities that produced the forward pass.

The local reference constructs **`torch.optim.Adam` with weight decay**, whose default coupled-L2 convention requires

\[
\widetilde g_t=g_t+\lambda\theta_{t-1},\qquad
m_t=\beta_1m_{t-1}+(1-\beta_1)\widetilde g_t,
\]
\[
v_t=\beta_2v_{t-1}+(1-\beta_2)\widetilde g_t^2,\qquad
\theta_t=\theta_{t-1}-\eta\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
\]

AdamW instead applies decay separately from the moment calculation. Supplying the same nonzero number as Candle AdamW's weight decay does **not** reproduce the reference Adam. With zero decay the mathematical decay difference disappears; floating-point update order and state conventions still need attention.

Candle's release optimizer module provides SGD and AdamW, with default AdamW decay 0.01 and first/second moment variables using the parameter dtype. Thus BF16 parameters also imply BF16 moments in this implementation, another mismatch with the project's FP32-master setup. [Candle optimizer source](https://raw.githubusercontent.com/huggingface/candle/0.11.0/candle-nn/src/optim.rs). Burn has an Adam implementation and recordable state, but its selected configuration must still match the coupled-decay reference. [Burn Adam source](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-optim/src/optim/adam.rs).

A transplant manifest should record optimizer type, decay convention, epsilon position, betas, step count, clipping, gradient accumulation, master-parameter dtype and moment dtype. A model-weights file alone cannot resume the original optimization state. Identical initialization values are also necessary to study trajectory differences; matching initializer names does not guarantee matching random parameters.

## CubeCL, fusion and memory movement

Fusion can turn a chain such as

\[
u=ax+b,\qquad y=\max(u,0)
\]

into a kernel that keeps \(u\) in registers instead of writing it to global memory and reading it back. Approximate saved traffic scales with the number and size of eliminated intermediate arrays. That benefit must be weighed against register pressure, occupancy, larger kernels and scheduling constraints.

Burn's rolling fusion documentation describes deferred operations grouped into optimizations. Its rolling extension guide makes clear that custom forward registration does not derive custom backward logic. These pages explain concepts but should be rebound to a release snapshot during implementation. [Burn fusion documentation](https://burn.dev/docs/burn/backend/fusion/index.html), [Burn backend extension guide](https://burn.dev/books/burn/advanced/backend-extension/index.html).

The rolling CubeCL repository describes Rust macro-based kernel construction, an IR and compilation targets such as PTX, WGSL, SPIR-V and MSL. Its unit/plane/cube concepts correspond to a CUDA thread/warp/block; feature support differs across targets. It describes JIT kernels and cached compilation rather than universal ahead-of-time device binaries. [CubeCL official repository, rolling architecture description](https://github.com/tracel-ai/cubecl).

For this CNN, pointwise activations and the positive head are plausible fusion opportunities. Concatenation may become an explicit memory copy, and layout changes can erase some savings. Burn's release interpolation integration explicitly permutes NCHW to NHWC and makes its forward input contiguous before invoking CubeK. That is a reason to account for layout traffic, not a measured bottleneck. [Burn release interpolation integration](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-cubecl/src/kernel/interpolate/base.rs).

A useful unmeasured cost model is

\[
T_{\rm step}=T_{\rm host}+T_{\rm transfer}+T_{\rm launch}+T_{\rm convolution}+T_{\rm reduction}+T_{\rm optimizer}.
\]

Rust might reduce host overhead and fusion might reduce launches; neither proves those terms dominate. The batch-four, small-image workload could be sensitive to launch overhead, whereas optimized convolution implementations may dominate another configuration. Cold JIT, warm-cache inference and steady-state training are separate regimes. No speedup or memory-saving percentage is claimed here.

Autotuning itself executes candidate kernels or benchmarks. It is not merely passive configuration. A future run must account for cold tuning cost and its cache artifacts under the applicable execution authorization; no autotuning was invoked for this document.

## Checkpoint and architecture interchange

Candle's `VarMap` saves named variables as SafeTensors and loads values into variables already present in the map. Extra file variables absent from the map are not retained. Therefore the Rust architecture and complete expected name map must exist before loading. [Candle release variable-map source](https://raw.githubusercontent.com/huggingface/candle/0.11.0/candle-nn/src/var_map.rs).

Burn's release storage includes Burnpack, SafeTensors, PyTorch readers, tensor remapping and framework adapters. Burnpack retains parameter identifiers useful for stateful training. [Burn 0.22 storage source](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/crates/burn-store/src/lib.rs). A PyTorch reader transfers tensors; it does not execute the original Python `forward` method. SafeTensors similarly contains values and metadata, not the graph semantics required to infer skip order or interpolation configuration.

The proposed conversion manifest should enumerate every convolution weight and bias, expected shape, source key, destination key, dtype and checksum. For this architecture, weights follow the explicit cross-correlation tensor convention `[out_channels,in_channels,3,3]`, with the head using `[1,c,1,1]`; any adapter must be inspected before assuming no transpose is needed. Missing or unexpectedly initialized parameters should be treated as an import failure, not silently accepted partial loading.

Burn's current official ONNX workflow generates ordinary Rust model code and Burnpack weights during a build. Operator coverage and custom-operation hooks are explicit; generated `Model::new` alone initializes structure without loading exported weights. [Burn ONNX guide, rolling 0.22 examples](https://burn.dev/books/burn/onnx-import.html). This is a prospective route for supported exports, not evidence that MoGe2 or this project's full teacher pipeline imports unchanged. A forward export also does not preserve optimizer state or prove trainability of every generated operation.

## RTX 5080 and toolchain requirements

NVIDIA identifies the GeForce RTX 5080 as compute capability **12.0**. This is different from CUDA toolkit version 12.0 and from datacenter Blackwell capability 10.0. Compatibility requires appropriate native code or usable PTX, a compatible driver, supported libraries and the selected compiler path. An accelerator feature in Cargo is insufficient evidence. [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus), [NVIDIA CUDA 12.9 archived Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html).

The future build manifest should record host OS/architecture, Rust version, crate lockfile, feature flags, CUDA driver/runtime/toolkit, cuDNN if selected, compiler backend and target architecture. For WGPU it should additionally identify the selected graphics API, adapter and negotiated feature limits. Hardware BF16 capability does not guarantee BF16 support through every WGPU shader/compiler path.

Do not transplant datacenter-only assumptions such as a particular tensor-core instruction or architecture-specific target to the consumer GPU without checking the exact path. Similarly, a CUDA installation already working for PyTorch does not imply that a Rust crate compiling or JIT-generating its own kernels has the same compatibility closure.

All future Cargo build products, kernel caches, conversions and artifacts must be redirected to `Z:/hugging-face/video2data` under the repository's storage rule. Defaults such as a local `target/`, home-directory Cargo cache or current-directory runtime logs must be considered in a prospective implementation. No environment was modified to investigate those defaults.

## Prospective vertical-slice integration

A future slice could live at `track3/features/rust_depth/`, alongside the existing [scratch-depth slice](../../../track3/features/scratch_depth/). Keep its request schema, handler, integration adapters, persistence format, configuration and README together. Do not create horizontal framework-service or model-repository layers solely for this experiment.

The request should identify a frozen reference checkpoint, reference preprocessing manifest, framework/version/backend, precision and output destination. Its response should report artifact provenance and completion/failure explicitly. A missing CUDA kernel should fail with its documented limitation rather than silently selecting CPU, nearest interpolation or a different dtype.

The intended sequence is:

1. Freeze model structure, preprocessing, image orientation, depth convention and weight manifest from the existing reference.
2. Transcribe FP32 inference with dropout disabled. Choose Candle first if the question is a small Rust deployment; choose Burn if the question is backend/compiler integration.
3. Document exact operator and checkpoint contracts, including resize coordinates, names and concat order.
4. If training is later selected, implement a faithful differentiable resize and channel dropout, and preserve coupled-decay Adam semantics.
5. Define a separately fixed comparison protocol before any later numerical or timing assessment. Distinguish inference parity, gradients, training behavior, cold compilation and steady-state performance.

These are proposed future stages, not checks performed during this research. New production dependencies require confirmation before implementation. The existing PyTorch implementation remains authoritative until evidence supports a clearly defined replacement.

Evaluation must use appropriate training/tuning material. Consumed holdout outcomes from episodes 43/44 must not determine framework choice, model changes, precision mode or mixing ratios. Episode 41 remains excluded. Any fresh unbiased assessment requires an independently fixed protocol; an engineering migration does not reset prior exposure to holdout results.

## Cost, benefit and unresolved questions

The Candle inference branch has a bounded purpose: express this small model in Rust, load frozen weights and integrate its outputs with a Rust consumer. Its main costs are manual operator transcription, explicit checkpoint mapping and incomplete derivative support if scope expands to training.

The Burn branch offers a wider research surface: runtime backend selection, training modules, custom operations, fusion and portable-kernel investigation. Its costs include the 0.22 API transition, larger compiler/backend dependency closure and the identified bilinear derivative gap. Portable source does not mean identical supported instructions, accumulation precision or performance across devices.

Unresolved empirical questions include exact FP32 forward differences, gradient differences at ties and boundaries, convolution algorithm behavior, BF16 precision through a chosen backend, cold-start compilation cost and warm-cache latency on the actual RTX 5080 environment. They remain unresolved deliberately because this task authorizes research and documentation, not a new execution campaign.

The official repositories identify MIT/Apache-2.0 licensing for these framework projects. Dependency licenses and downloaded model-weight terms remain separate provenance facts; this observation is not legal advice. [Candle repository](https://github.com/huggingface/candle), [CubeCL repository](https://github.com/tracel-ai/cubecl), [Burn release workspace metadata](https://raw.githubusercontent.com/tracel-ai/burn/v0.22.0/Cargo.toml).
