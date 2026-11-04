# Dot-source before running tools: . ./scripts/use-local-artifacts.ps1
# This creates directories and configures only the current process environment.
$track3Root = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd([char[]]@('\', '/'))
$track3CanonicalRoot = 'Z:\hugging-face\video2data'
# Refuse other checkouts before creating directories or changing the environment.
if (-not [string]::Equals($track3Root, $track3CanonicalRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Local artifacts require the checkout root '$track3CanonicalRoot'; found '$track3Root'."
}
$track3Paths = @{
    TEMP = 'data/.runtime/tmp'
    TMP = 'data/.runtime/tmp'
    TMPDIR = 'data/.runtime/tmp'
    HF_HOME = 'data/.cache/huggingface'
    HF_HUB_CACHE = 'data/.cache/huggingface/hub'
    HF_DATASETS_CACHE = 'data/.cache/huggingface/datasets'
    TORCH_HOME = 'data/.cache/torch'
    TORCH_EXTENSIONS_DIR = 'data/.cache/torch-extensions'
    TORCHINDUCTOR_CACHE_DIR = 'data/.cache/torch-inductor'
    TRITON_CACHE_DIR = 'data/.cache/triton'
    XDG_CACHE_HOME = 'data/.cache/xdg'
    PIP_CACHE_DIR = 'data/.cache/pip'
    UV_CACHE_DIR = 'data/.cache/uv'
    UV_PYTHON_INSTALL_DIR = 'data/.runtime/python'
    CUDA_CACHE_PATH = 'data/.cache/cuda'
    MPLCONFIGDIR = 'data/.cache/matplotlib'
    PYTHONPYCACHEPREFIX = 'data/.cache/pycache'
}
foreach ($track3Entry in $track3Paths.GetEnumerator()) {
    $track3Directory = Join-Path $track3Root $track3Entry.Value
    New-Item -ItemType Directory -Force -Path $track3Directory | Out-Null
    [Environment]::SetEnvironmentVariable($track3Entry.Key, $track3Directory, 'Process')
}
