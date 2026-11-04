#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. V2D_HISTORICAL=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
data/.runtime/linux-cuda/bin/python -B - "$@" <<'PY'
import hashlib, json, re, subprocess, sys, tempfile
from pathlib import Path
release = json.loads(Path('scripts/reproduce/release.json').read_text())
from src.paths import current_name  # recorded names may be in the old layout
for _entry in release.values():
    for _key in ('path', 'manifest'):
        if _key in _entry:
            _entry[_key] = current_name(_entry[_key])
Path('data/outputs/report').mkdir(exist_ok=True)
work = Path(tempfile.mkdtemp(prefix='release-', dir='data/outputs/report'))
failed = []
def check(label, action):
    try: ok, detail = action()
    except Exception as error: ok, detail = False, str(error)
    print(f"{'OK' if ok else 'FAIL'} {label}: {detail}", flush=True)
    if not ok: failed.append(label)
    return ok
def match(entry):
    actual = hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()
    return actual == entry['sha256'], f"{entry['path']} expected={entry['sha256']} actual={actual}"
def command(label, argv, marker=''):
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = work / (label + '.log'); log.write_text(result.stdout)
    return result.returncode == 0 and marker in result.stdout, f"exit {result.returncode}; {log}"
def report_file():
    pdf = Path(re.search(r'^Outputs/logs: (.+)$', (work/'report.log').read_text(), re.M)[1])/'src/docs/report/track3/main.pdf'
    data = pdf.read_bytes(); actual = hashlib.sha256(data).hexdigest(); entry = release['report']
    if actual != entry['sha256']: return False, f"expected={entry['sha256']} actual={actual}"
    destination = Path(entry['path'])
    if destination.exists(): return match(entry)
    with destination.open('xb') as stream: stream.write(data)
    return True, f"upload {destination} ({actual})"
combined_manifest = sys.argv[2] if len(sys.argv) == 3 and sys.argv[1] == '--combined' else None
check('arguments', lambda: (len(sys.argv) == 1 or combined_manifest is not None,
                           'optional --combined <release manifest>'))
for name in ('mesh', 'pose'): check(name+' packed file', lambda: match(release[name]))
for name in ('mesh', 'pose'): check(name+' manifest', lambda: command(name+'-manifest', ['bash', 'scripts/reproduce/verify.sh', release[name]['manifest']]))
moving_manifest = release['pose'].get('moving_manifest')
if not moving_manifest and json.loads(Path(release['pose']['manifest']).read_text()).get('kind') == 'pose_insurance.moving_pack':
    check('moving manifest declared', lambda: (False, 'a moving pose requires the optional moving_manifest entry'))
if moving_manifest:
    moving_manifest = current_name(moving_manifest)
    check('moving manifest identity', lambda: (
        moving_manifest == release['pose']['manifest']
        and current_name(json.loads(Path(moving_manifest).read_text())['file']['path']) == release['pose']['path'],
        'optional moving_manifest must name the selected pose manifest and packed file'))
    check('pose provenance', lambda: command('pose-provenance', [
        'data/.runtime/linux-cuda/bin/python', '-B', '-m',
        'src.features.pose_insurance.pack_moving', '--verify-manifest', moving_manifest]))
else:
    check('pose provenance', lambda: command('pose-provenance', ['bash', 'scripts/reproduce/d2_poses_v5.sh']))
for track in ('track_3_mesh', 'track_3'):
    check(track+' preflight', lambda: command(track, ['data/.runtime/kit-env/bin/python', '-B', 'upstream/v2d_submission_kit/v2d_submission_kit/tools/preflight.py', '--track', track, '--skip-kaggle'] + ([] if track == 'track_3_mesh' else ['--results', str(Path(current_name(json.loads(Path(release['pose']['manifest']).read_text())['exports'][0]['path'])).parent)]), 'Ready to submit'))
if combined_manifest:
    combined_spec = release['combined_pose']
    combined_kind = json.loads(Path(combined_manifest).read_text()).get('kind')
    combined_module = ('src.features.pose_insurance.pack_combined_moving'
                       if combined_kind == 'pose_insurance.combined_moving_pack' else combined_spec['module'])
    if check('combined pose provenance', lambda: command('combined-provenance', [
            'data/.runtime/linux-cuda/bin/python', '-B', '-m', combined_module,
            '--verify-manifest', combined_manifest, '--release'])):
        combined = json.loads(Path(combined_manifest).read_text())
        check('combined pose file', lambda: match(combined['file']))
        check('combined current-kit preflight', lambda: command('combined-preflight', [
            'data/.runtime/kit-env/bin/python', '-B', combined_spec['kit']+'/tools/preflight.py',
            '--track', 'track_3', '--skip-kaggle', '--results',
            str(Path(combined['exports'][0]['path']).parent)], 'Ready to submit'))
if check('report build', lambda: command('report', ['bash', 'scripts/reproduce/d3_report.sh', '--run'])): check('report upload file', report_file)
print('NOT READY' if failed else 'READY', flush=True)
sys.exit(bool(failed))
PY
