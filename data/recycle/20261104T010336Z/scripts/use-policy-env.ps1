# Dot-source for a named WSL shell launcher. This does not start WSL or run code.
$env:VIDEO2DATA_POLICY_DISTRO = 'video2data-policy'
$env:VIDEO2DATA_POLICY_ROOT = '/mnt/z/hugging-face/video2data'
function Enter-Video2DataPolicy {
    param([ValidateSet('policy', 'retarget')][string]$Environment = 'policy')
    & wsl.exe --distribution video2data-policy --cd /mnt/z/hugging-face/video2data --exec bash --rcfile "/mnt/z/hugging-face/video2data/scripts/wsl/shell-$Environment.sh" -i
}
