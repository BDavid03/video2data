"""Explicitly selected, lazy CUDA log-depth MAE; no PyTorch fallback here.

Importing this module does not import torch, compile, or create artifacts. Each
call independently requires runtime.execution.cuda_compile_authorized == true,
the physical Z: project checkout and the configured existing interpreter. The
caller must also authorize its operation before creating run artifacts.

Compilation requires explicit authorization and existing local tools. The
18 October 2026 external-interpreter experiment compiled these kernel sources
and recorded one real batch's loss/gradient agreement; it does not establish this
production interpreter's compatibility. This adapter never installs dependencies
or downloads sources. Source hashes
are included by the slice's recursive operation_run provenance inventory.

Only contiguous float32 CUDA prediction/target and an equal-shaped contiguous
bool CUDA mask are accepted. The scalar and reduction arithmetic are float32;
integer counts are int64. Invalid labels/pixels have zero prediction gradients,
target gradients are deliberately absent, and equality has gradient zero.
Forward synchronizes its small count/error scalars to reject invalid batches.
Only first-order autograd is supported; CUDA graph capture is unsupported.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
from threading import RLock

from track3.paths import interpreter, permitted_root, output_path, project_path
from track3.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parent
_LOCK = RLock()
_EXTENSION = None
_FUNCTION = None


def _authorize_compile():
    """Strict standalone gate, including when a module is already loaded."""
    runtime_path = project_path(ROOT, FEATURE / "runtime.json")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8-sig"))
    execution = runtime.get("execution")
    if not isinstance(execution, dict) or execution.get("cuda_compile_authorized") is not True:
        raise PermissionError("Scratch-depth CUDA compilation remains disabled")
    permitted_root()
    if Path(sys.executable).resolve() != interpreter(ROOT, runtime["python"]):
        raise RuntimeError("Select the approved project interpreter (runtime.json python for this platform)")


def _confine_caches():
    # Called only after authorization, before torch or extension-loader imports.
    folders = {
        "TEMP": ".runtime/tmp", "TMP": ".runtime/tmp", "TMPDIR": ".runtime/tmp",
        "TORCH_HOME": ".cache/torch", "TORCH_EXTENSIONS_DIR": ".cache/torch-extensions",
        "CUDA_CACHE_PATH": ".cache/cuda", "XDG_CACHE_HOME": ".cache/xdg",
        "PYTHONPYCACHEPREFIX": ".cache/pycache",
    }
    for name, folder in folders.items():
        path = output_path(ROOT, folder)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # tempfile may already have cached an external directory before this call.
    tempfile.tempdir = os.environ["TMPDIR"]
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]


def _require_inputs(torch, prediction, target, valid):
    for name, tensor, dtype in (("prediction", prediction, torch.float32),
                                ("target", target, torch.float32),
                                ("valid", valid, torch.bool)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (tensor.layout != torch.strided or tensor.dtype != dtype
                or not tensor.is_cuda or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous strided {dtype} on CUDA")
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid must have identical shapes")
    if prediction.device != target.device or prediction.device != valid.device:
        raise ValueError("prediction, target and valid must use the same CUDA device")
    if prediction.numel() == 0:
        raise ValueError("Training batch contains no usable depth labels")


def _load_extension():
    global _EXTENSION
    _authorize_compile()
    if _EXTENSION is None:
        sources = [project_path(ROOT, FEATURE / "cuda" / name)
                   for name in ("masked_log_depth.cpp", "masked_log_depth.cu")]
        # Local source identity separates rebuilds; load also tracks tool flags.
        digest = "".join(sha256_file(path) for path in [Path(__file__), *sources])
        from hashlib import sha256
        name = "scratch_depth_masked_log_mae_" + sha256(digest.encode("ascii")).hexdigest()[:16]
        directory = output_path(ROOT, Path(".cache/torch-extensions/scratch_depth") / name)
        from torch.utils.cpp_extension import load
        # Recheck immediately before any build directory or compiler invocation.
        _authorize_compile()
        directory.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load(
            name=name, sources=[str(path) for path in sources],
            build_directory=str(directory), with_cuda=True,
            extra_cflags=["/Zc:preprocessor"] if os.name == "nt" else [],
            extra_cuda_cflags=["--ftz=false", *(["-Xcompiler=/Zc:preprocessor"]
                                              if os.name == "nt" else [])], verbose=False,
        )
    return _EXTENSION


def _autograd_function(torch):
    global _FUNCTION
    if _FUNCTION is None:
        from torch.autograd.function import once_differentiable

        class MaskedLogDepth(torch.autograd.Function):
            @staticmethod
            def forward(ctx, prediction, target, valid, extension):
                loss, mask, count = extension.forward(prediction, target, valid)
                ctx.save_for_backward(prediction, target, mask, count)
                ctx.extension = extension
                ctx.set_materialize_grads(False)
                return loss

            @staticmethod
            @once_differentiable
            def backward(ctx, gradient):
                if gradient is None or not ctx.needs_input_grad[0]:
                    return None, None, None, None
                prediction, target, mask, count = ctx.saved_tensors
                result = ctx.extension.backward(
                    gradient.contiguous(), prediction, target, mask, count,
                )
                return result, None, None, None

        _FUNCTION = MaskedLogDepth
    return _FUNCTION


def masked_log_depth_loss(prediction, target, valid):
    """Return fp32 mean |log(prediction)-log(target)| over usable labels.

    Usable means valid & isfinite(target) & (target > 0). Selected prediction
    values must be finite and positive. Errors propagate without a downgrade;
    the caller explicitly selects the standard PyTorch backend when desired.
    """
    with _LOCK:
        _authorize_compile()
        _confine_caches()
        import torch
        _require_inputs(torch, prediction, target, valid)
        with torch.cuda.device(prediction.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Scratch-depth CUDA loss does not support CUDA graph capture")
        extension = _load_extension()
        function = _autograd_function(torch)
    return function.apply(prediction, target, valid, extension)
