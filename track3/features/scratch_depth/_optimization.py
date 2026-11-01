"""Explicit training opt-ins; no imports of PyTorch or execution at import time.

The eager FP32, contiguous-layout recipe and ordinary Adam kwargs remain default.
Compilation and CUDA loss require separate runtime authorization before even
creating a run. GPU capability checks and lazy backend loading happen only in
the caller's GPU lease. No gradient accumulation or automatic fallback is used.
"""

from contextlib import nullcontext


OPTION_FIELDS = ("bf16_autocast", "channels_last", "torch_compile", "fused_adam")


def optimization_options(request):
    return {**{name: getattr(request, name) for name in OPTION_FIELDS},
            "loss_backend": request.loss_backend}


def require_compile_authorization(request, runtime):
    execution = runtime["execution"]
    if request.torch_compile and execution.get("pytorch_compile_authorized") is not True:
        raise PermissionError("Scratch-depth torch.compile requires pytorch_compile_authorized=true")
    if request.loss_backend == "cuda" and execution.get("cuda_compile_authorized") is not True:
        raise PermissionError("Scratch-depth CUDA loss requires cuda_compile_authorized=true")


def configure_training(model, request, torch):
    """Return a forward callable and Adam while retaining the original model.

    Called only inside the admitted GPU lease. BF16 uses FP32 master parameters
    and no scaler; unsupported hardware is refused rather than silently changing
    the requested recipe. Compile defaults are pinned explicitly here.
    """
    if request.bf16_autocast and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("BF16 autocast requires native CUDA BF16 support")
    if request.channels_last:
        model.to(memory_format=torch.channels_last)
    optimizer_options = {"fused": True, "foreach": False} if request.fused_adam else {}
    optimizer = torch.optim.Adam(model.parameters(), lr=request.learning_rate,
                                 weight_decay=request.weight_decay, **optimizer_options)
    forward = (torch.compile(model, backend="inductor", mode="default",
                             fullgraph=False, dynamic=False)
               if request.torch_compile else model)
    return forward, optimizer


def forward_context(request, torch):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if request.bf16_autocast else nullcontext())


def training_loss(prediction, target, valid, request, torch, eager_loss):
    """FP32 logs/reduction outside forward autocast, retaining autograd casts.

    CUDA adapter API: _cuda_loss.masked_log_depth_loss(prediction, target, valid),
    lazily loaded by the adapter itself. It receives contiguous CUDA FP32 inputs
    and a contiguous boolean mask. This function runs only within the GPU lease.
    """
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid must have identical shapes")
    with torch.autocast(device_type="cuda", enabled=False):
        prediction, target = prediction.float(), target.float()
        if request.loss_backend == "cuda":
            from ._cuda_loss import masked_log_depth_loss

            return masked_log_depth_loss(prediction.contiguous(), target.contiguous(),
                                         valid.bool().contiguous())
        mask = valid.bool() & torch.isfinite(target) & (target > 0)
        supervised = prediction.masked_select(mask)
        if not bool((torch.isfinite(supervised) & (supervised > 0)).all()):
            raise FloatingPointError("Supervised depth predictions must be finite and positive")
        # Do not evaluate log on masked-out predictions: invalid positions must
        # contribute neither NaNs nor gradients to the existing network loss.
        safe_prediction = torch.where(mask, prediction, torch.ones_like(prediction))
        return eager_loss(safe_prediction, target, valid)
