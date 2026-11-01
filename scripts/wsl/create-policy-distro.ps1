param(
    [Parameter(Mandatory = $true)][string]$RootfsTarball,
    [switch]$Confirm
)
$ErrorActionPreference = 'Stop'
$track3Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$policyConfig = Get-Content -LiteralPath (Join-Path $track3Root 'configs/runtime/policy-env.json') -Raw | ConvertFrom-Json
if (-not $Confirm -or $policyConfig.create_authorized -cne $true) {
    throw 'Distribution creation requires -Confirm and create_authorized: true.'
}
if ($policyConfig.distro -cne 'video2data-policy' -or $policyConfig.ubuntu_version -cne '24.04') {
    throw 'Only the isolated video2data-policy Ubuntu 24.04 distribution is permitted.'
}
if ($track3Root -ine [IO.Path]::GetFullPath($policyConfig.windows_root)) {
    throw 'This script must live in the configured project.'
}
$policyDisk = [IO.Path]::GetFullPath((Join-Path $track3Root $policyConfig.vhdx_directory))
$expectedDisk = [IO.Path]::GetFullPath((Join-Path $track3Root '.runtime/wsl/video2data-policy'))
if ($policyDisk -ine $expectedDisk) { throw 'Unexpected policy VHDX directory.' }
$policyRootfs = (Resolve-Path -LiteralPath $RootfsTarball).Path
if (-not $policyRootfs.StartsWith($track3Root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Place the user-supplied Ubuntu 24.04 rootfs tarball inside the project.'
}
$policyDistros = & wsl.exe --list --quiet
if ($LASTEXITCODE -ne 0) { throw 'Could not list WSL distributions.' }
$policyNames = @($policyDistros | ForEach-Object { ($_ -replace "`0", '').Trim() })
if ($policyNames -icontains $policyConfig.distro) { throw 'Refusing to touch an existing distribution.' }
if (Test-Path -LiteralPath $policyDisk) { throw 'Refusing to reuse an existing VHDX directory.' }
New-Item -ItemType Directory -Path $policyDisk | Out-Null
& wsl.exe --import $policyConfig.distro $policyDisk $policyRootfs --version 2
if ($LASTEXITCODE -ne 0) { throw "WSL import failed with exit code $LASTEXITCODE; retained $policyDisk." }
# No default-distribution change, launch, installation, or automatic cleanup.
