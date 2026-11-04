param(
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$track3Root = Split-Path -Parent $PSScriptRoot
$track3Policy = Get-Content -LiteralPath (Join-Path $track3Root 'data/configs/project.json') -Raw | ConvertFrom-Json
if ($track3Policy.execution_policy.allow_validation_runs -ne $true) {
    throw 'Foundation checks are disabled by the current user instruction: absolutely no testing.'
}
. "$PSScriptRoot/use-local-artifacts.ps1"
$track3ReportDir = Join-Path $track3Root 'data/reports/foundation'
New-Item -ItemType Directory -Force -Path $track3ReportDir | Out-Null
$track3Stages = [System.Collections.Generic.List[object]]::new()
$track3AnyFailure = $false

function Invoke-Track3Stage {
    param([string]$Name, [string[]]$StageArguments)
    $stageStart = [DateTime]::UtcNow
    $stageLog = Join-Path $track3ReportDir "$Name.log"
    & $Python @StageArguments *> $stageLog
    $stageExit = $LASTEXITCODE
    $track3Stages.Add([ordered]@{
        name = $Name
        executable = $Python
        arguments = $StageArguments
        started_utc = $stageStart.ToString('o')
        finished_utc = [DateTime]::UtcNow.ToString('o')
        exit_code = $stageExit
        status = $(if ($stageExit -eq 0) { 'passed' } else { 'failed' })
        log = "data/reports/foundation/$Name.log"
    })
    if ($stageExit -ne 0) { $script:track3AnyFailure = $true }
    Write-Output "$Name exit=$stageExit; data/reports/foundation/$Name.log"
}

Push-Location $track3Root
try {
    Invoke-Track3Stage -Name 'tests' -StageArguments @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    Invoke-Track3Stage -Name 'data-audit' -StageArguments @('-m', 'src.data.audit', '--decode')
    Invoke-Track3Stage -Name 'calibration' -StageArguments @(
        '-m', 'src.calibration.audit', '--snapshot',
        'data/datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8',
        '--output', 'data/reports/calibration', '--image-check'
    )
    $track3Run = [ordered]@{
        schema_version = 1
        scope = 'CPU foundation verification; no installs, model inference, or simulator runs'
        status = $(if ($track3AnyFailure) { 'failed' } else { 'passed_scoped_checks' })
        stages = $track3Stages.ToArray()
        first_batch_complete = $false
        outstanding = @('full_video_reconstruction', 'simulator_step_and_render', 'official_evaluator_contract')
    }
    $track3Run | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $track3ReportDir 'run.json') -Encoding utf8
    & $Python -m src provenance --output data/manifests/runs/foundation.json --label foundation `
        --artifact-dir reports --artifact-dir data/manifests/dataset --artifact-dir data/manifests/models
    if ($LASTEXITCODE -ne 0) { $track3AnyFailure = $true }
} finally {
    Pop-Location
}
if ($track3AnyFailure) { exit 1 }
