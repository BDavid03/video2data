# Dot-source to reuse installed packages without loading MoGe source trees.
# This selects the environment only; it does not enable or execute any operation.
. "$PSScriptRoot/use-local-artifacts.ps1"
$scratchDepthRoot = Split-Path -Parent $PSScriptRoot
$env:VIRTUAL_ENV = Join-Path $scratchDepthRoot '.runtime/moge-python313'
$env:PATH = (Join-Path $env:VIRTUAL_ENV 'Scripts') + [IO.Path]::PathSeparator + $env:PATH
$env:PYTHONPATH = $scratchDepthRoot
$env:PYTHONNOUSERSITE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
