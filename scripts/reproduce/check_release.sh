#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
.runtime/linux-cuda/bin/python -B - "$@" <<'PY'
import hashlib, json, re, subprocess, sys, tempfile
from pathlib import Path
release = json.loads(Path('scripts/reproduce/release.json').read_text())
Path('outputs/report').mkdir(exist_ok=True)
work = Path(tempfile.mkdtemp(prefix='release-', dir='outputs/report'))
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
check('arguments', lambda: (len(sys.argv) == 1, 'no arguments accepted'))
for name in ('mesh', 'pose'): check(name+' packed file', lambda: match(release[name]))
for name in ('mesh', 'pose'): check(name+' manifest', lambda: command(name+'-manifest', ['bash', 'scripts/reproduce/verify.sh', release[name]['manifest']]))
check('pose provenance', lambda: command('pose-provenance', ['bash', 'scripts/reproduce/d2_poses_v5.sh']))
for track in ('track_3_mesh', 'track_3'):
    check(track+' preflight', lambda: command(track, ['.runtime/kit-env/bin/python', '-B', 'upstream/v2d_submission_kit/v2d_submission_kit/tools/preflight.py', '--track', track, '--skip-kaggle'] + ([] if track == 'track_3_mesh' else ['--results', str(Path(json.loads(Path(release['pose']['manifest']).read_text())['exports'][0]['path']).parent)]), 'Ready to submit'))
if check('report build', lambda: command('report', ['bash', 'scripts/reproduce/d3_report.sh', '--run'])): check('report upload file', report_file)
print('NOT READY' if failed else 'READY', flush=True)
sys.exit(bool(failed))
PY
