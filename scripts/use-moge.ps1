# Dot-source to select the installed environment and pinned source trees.
# This does not execute Python, import a model, or run any checks.
. "$PSScriptRoot/use-local-artifacts.ps1"
$track3Root = Split-Path -Parent $PSScriptRoot
$env:VIRTUAL_ENV = Join-Path $track3Root 'data/.runtime/moge-python313'
$env:PATH = (Join-Path $env:VIRTUAL_ENV 'Scripts') + [IO.Path]::PathSeparator + $env:PATH
$env:PYTHONPATH = @(
    (Join-Path $track3Root 'upstream/moge'),
    (Join-Path $track3Root 'upstream/moge-utils3d'),
    (Join-Path $track3Root 'upstream/moge-pipeline')
) -join [IO.Path]::PathSeparator
$env:PYTHONNOUSERSITE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
