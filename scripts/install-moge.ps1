param()

# Authorized installation only. No tests, imports, inference, or validation runs.
$ErrorActionPreference = 'Stop'
$track3Root = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot/use-local-artifacts.ps1"
$track3InstallDirectory = Join-Path $track3Root 'reports/installation/moge-20260925'
New-Item -ItemType Directory -Force -Path $track3InstallDirectory | Out-Null
$track3InstallSteps = [System.Collections.Generic.List[object]]::new()
$track3InstallRecord = [ordered]@{
    schema_version = 1
    authorized_date = '2026-09-25'
    authorization = 'User approved the MoGe2 installation proposal; no testing or model execution.'
    started_utc = [DateTime]::UtcNow.ToString('o')
    status = 'installing'
    environment = '.runtime/moge-python313'
    package_lock = 'configs/perception/moge/pylock.python313.toml'
    source_pins = 'manifests/models/moge-source-pins.json'
    model = [ordered]@{
        repo_id = 'Ruicheng/moge-2-vitl-normal'
        revision = 'cb0e8bbd6b1e243589717c78e750b1ba4c093acf'
        file = 'model.pt'
        destination = 'models/moge/model.pt'
    }
    tests_run = $false
    validation_commands_run = $false
    model_executed = $false
    source_build_hooks_run = $false
    steps = @()
}

function Save-Track3Installation {
    $track3InstallRecord.steps = $track3InstallSteps.ToArray()
    $track3InstallRecord | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $track3InstallDirectory 'installation.json') -Encoding utf8
}

function Invoke-Track3InstallStep {
    param([string]$Name, [string]$Executable, [string[]]$Arguments)
    $track3Step = [ordered]@{
        name = $Name
        executable = $Executable
        arguments = $Arguments
        started_utc = [DateTime]::UtcNow.ToString('o')
        status = 'running'
        log = "reports/installation/moge-20260925/$Name.log"
    }
    $track3InstallSteps.Add($track3Step)
    Save-Track3Installation
    Write-Output "Installing: $Name"
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $track3InstallDirectory "$Name.log")
    $track3Exit = $LASTEXITCODE
    $track3Step.exit_code = $track3Exit
    $track3Step.finished_utc = [DateTime]::UtcNow.ToString('o')
    $track3Step.status = $(if ($track3Exit -eq 0) { 'completed' } else { 'failed' })
    Save-Track3Installation
    if ($track3Exit -ne 0) { throw "Installation step '$Name' exited with $track3Exit. See $($track3Step.log)." }
}

Push-Location $track3Root
try {
    Save-Track3Installation
    # Existing destinations are not removed or cleared. uv reuses this dedicated
    # project environment on a resume; only it is synchronized to the lock.
    Invoke-Track3InstallStep -Name 'environment' -Executable 'uv' -Arguments @(
        'venv', '.runtime/moge-python313', '--allow-existing', '--python',
        'C:/Users/db009/AppData/Roaming/uv/python/cpython-3.13.14-windows-x86_64-none/python.exe',
        '--no-python-downloads'
    )
    Invoke-Track3InstallStep -Name 'packages' -Executable 'uv' -Arguments @(
        'pip', 'sync', 'configs/perception/moge/pylock.python313.toml',
        '--python', '.runtime/moge-python313/Scripts/python.exe',
        '--only-binary', ':all:', '--no-python-downloads'
    )
    $track3Sources = @(
        @('moge', 'https://github.com/microsoft/MoGe.git', '07444410f1e33f402353b99d6ccd26bd31e469e8'),
        @('moge-utils3d', 'https://github.com/EasternJournalist/utils3d.git', '3fab839f0be9931dac7c8488eb0e1600c236e183'),
        @('moge-pipeline', 'https://github.com/EasternJournalist/pipeline.git', '866f059d2a05cde05e4a52211ec5051fd5f276d6')
    )
    foreach ($track3Source in $track3Sources) {
        $track3Destination = "upstream/$($track3Source[0])"
        if (-not (Test-Path -LiteralPath $track3Destination)) {
            Invoke-Track3InstallStep -Name "$($track3Source[0])-clone" -Executable 'git' -Arguments @(
                'clone', '--filter=blob:none', '--no-checkout', $track3Source[1], $track3Destination
            )
        }
        Invoke-Track3InstallStep -Name "$($track3Source[0])-checkout" -Executable 'git' -Arguments @(
            '-C', $track3Destination, '-c', 'core.autocrlf=false', '-c', 'core.longpaths=true',
            'checkout', '--detach', $track3Source[2]
        )
    }
    Invoke-Track3InstallStep -Name 'model-download' -Executable 'hf' -Arguments @(
        'download', 'Ruicheng/moge-2-vitl-normal', 'model.pt',
        '--revision', 'cb0e8bbd6b1e243589717c78e750b1ba4c093acf',
        '--local-dir', 'models/moge'
    )
    $track3InstallRecord.status = 'installed_unexecuted'
} catch {
    $track3InstallRecord.status = 'installation_failed'
    $track3InstallRecord.error = $_.Exception.Message
    throw
} finally {
    $track3InstallRecord.finished_utc = [DateTime]::UtcNow.ToString('o')
    Save-Track3Installation
    Pop-Location
}
