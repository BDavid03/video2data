#!/usr/bin/env bash
set -euo pipefail
root=/mnt/z/hugging-face/video2data
[[ "${WSL_DISTRO_NAME:-}" == video2data-policy ]] || { echo 'Wrong WSL distribution.' >&2; exit 1; }
[[ $EUID == 0 ]] || { echo 'Run as root in the dedicated distribution.' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 24.04 ]] || { echo 'Ubuntu 24.04 is required.' >&2; exit 1; }
configuration=$(python3 - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
config = json.loads((root / 'configs/runtime/policy-env.json').read_text())
lock = json.loads((root / 'configs/sources.lock.json').read_text())['policy_runtime']
if config.get('install_authorized') is not True:
    raise SystemExit('Policy installation remains disabled.')
expected = {'distro': 'video2data-policy', 'ubuntu_version': '24.04', 'linux_root': str(root),
            'installation_root': '/opt/video2data', 'checkout': '/opt/video2data/video_to_data',
            'policy_environment': '/opt/video2data/policy-python311',
            'retarget_environment': '/opt/video2data/retarget-python311',
            'kit': '/opt/video2data/submission-kit', 'installation_record': 'reports/runtime/policy-env.json'}
if any(config.get(key) != value for key, value in expected.items()):
    raise SystemExit('Unexpected installation paths or distribution.')
if config['source_commit'] != lock['commit'] or config['source_url'] != lock['url']:
    raise SystemExit('Source pin differs from configs/sources.lock.json.')
if Path(config['installation_root']).exists() or (root / config['installation_record']).exists():
    raise SystemExit('Refusing to replace an existing or partial installation.')
print(config['source_url'])
print(config['source_commit'])
PY
)
mapfile -t source_pin <<< "$configuration"
source "$root/scripts/wsl/use-policy-env.sh" policy
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$FLASH_CHORD_CACHE_DIR" /opt/video2data/bin
export UV_PYTHON_INSTALL_DIR=/opt/video2data/python
export UV_TOOL_DIR=/opt/video2data/uv-tools
export UV_TOOL_BIN_DIR=/opt/video2data/bin
record_installation() {
    python3 - "$root" "$1" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root, status = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
from track3.provenance import run_fields, write_json
path = root / 'reports/runtime/policy-env.json'
config = json.loads((root / 'configs/runtime/policy-env.json').read_text())
if path.exists():
    record = json.loads(path.read_text())
else:
    record = run_fields(root, 'policy_environment', 'install', config, inputs=[
        'configs/runtime/policy-env.json', 'configs/sources.lock.json',
        'scripts/wsl/install-policy-env.sh', 'scripts/wsl/retarget-requirements.txt'])
    record.update(started_at=datetime.now(timezone.utc).isoformat(), **{
        key: config[key] for key in ('distro', 'source_commit', 'checkout', 'policy_environment', 'retarget_environment', 'kit')})
record['status'] = status
if status == 'completed':
    directory = Path(config['checkout']) / 'robotic_grounding/flash_chord/src/flash_chord/evaluation'
    digest, modules = hashlib.sha256(), {}
    for name in sorted(('__init__.py', 'harness.py', 'metrics.py', 'rollout.py')):
        content = (directory / name).read_bytes()
        if b'\r' in content:
            raise SystemExit('Evaluation source is not LF.')
        digest.update(name.encode()); digest.update(b'\0'); digest.update(content)
        modules[name] = hashlib.sha256(content).hexdigest()
    record.update(eval_code_sha256=digest.hexdigest(), evaluation_modules=modules)
    record['uv_lock_sha256'] = hashlib.sha256((directory.parents[2] / 'uv.lock').read_bytes()).hexdigest()
    record['kit_manifest_sha256'] = hashlib.sha256((Path(config['kit']) / 'KIT_MANIFEST.json').read_bytes()).hexdigest()
    record['completed_at'] = datetime.now(timezone.utc).isoformat()
write_json(path, record)
PY
}
record_installation running
trap 'status=$?; if [[ $status != 0 ]]; then record_installation failed; fi' EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git git-lfs build-essential ffmpeg libgl1 libglib2.0-0
export GIT_LFS_SKIP_SMUDGE=1
git -c core.autocrlf=false clone --no-checkout "${source_pin[0]}" /opt/video2data/video_to_data
git -C /opt/video2data/video_to_data config core.autocrlf false
git -C /opt/video2data/video_to_data checkout --detach "${source_pin[1]}"
[[ "$(git -C /opt/video2data/video_to_data rev-parse HEAD)" == "${source_pin[1]}" ]] || exit 1
git -C /opt/video2data/video_to_data lfs install --local
git -C /opt/video2data/video_to_data lfs pull --include='robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**,robotic_grounding/flash_chord/src/flash_chord/assets/**,robotic_grounding/assets/**'
curl --fail --location --silent --show-error https://astral.sh/uv/install.sh --output "$TMPDIR/uv-install.sh"
UV_INSTALL_DIR=/opt/video2data/bin UV_NO_MODIFY_PATH=1 bash "$TMPDIR/uv-install.sh"
export PATH="/opt/video2data/bin:$PATH"
uv python install 3.11
cd /opt/video2data/video_to_data/robotic_grounding/flash_chord
UV_PROJECT_ENVIRONMENT=/opt/video2data/policy-python311 uv sync --locked --python 3.11 --extra train
uv venv --python 3.11 /opt/video2data/retarget-python311
uv pip install --python /opt/video2data/retarget-python311/bin/python -r "$root/scripts/wsl/retarget-requirements.txt"
uv pip install --python /opt/video2data/retarget-python311/bin/python --no-build-isolation -e /opt/video2data/video_to_data/robotic_grounding/source/robotic_grounding
python3 - "$root" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
source = root / 'upstream/v2d_submission_kit/v2d_submission_kit'
manifest = json.loads((source / 'KIT_MANIFEST.json').read_text())
for name, digest in manifest['files'].items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
        raise SystemExit('Submission Kit differs from its manifest: ' + name)
shutil.copytree(source, '/opt/video2data/submission-kit')
PY
uv pip freeze --python /opt/video2data/retarget-python311/bin/python > "$root/reports/runtime/retarget-installed.txt"
record_installation completed
trap - EXIT
