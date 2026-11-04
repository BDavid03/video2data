"""Resume the ordered spoon mask screen; a passing screen yields to the camera phase."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .spoon import ROOT, event
from src.runs import now_utc


def main():
    queue_path = Path(__file__).with_suffix('.json')
    queue = json.loads(queue_path.read_text())
    if queue['episodes'] != [35, 36, 37] or queue['object_name'] != 'spoon':
        raise ValueError('The protocol fixes the order and object')
    ledger = ROOT / 'data/outputs/moving_policy/spoon-ledger.jsonl'
    for episode in queue['episodes']:
        directory = ROOT / f'data/outputs/moving_reference/ep{episode}'
        final = directory / 'gates-final.json'
        if final.exists():
            result = json.loads(final.read_text())
            if not result['passed']:
                print(f'ep{episode}: already stopped; no retry', flush=True)
                continue
        command = [sys.executable, '-B', '-m', 'src.features.moving_reference.spoon']
        if not (directory / 'masks/once.json').exists():
            subprocess.run([*command, 'masks', '--episode', str(episode)], cwd=ROOT, check=True)
        else:
            # The initial episode may already be running in the same job session.
            while True:
                path = directory / 'masks/run.json'
                record = json.loads(path.read_text()) if path.exists() else {'status': 'running'}
                if record['status'] != 'running':
                    break
                time.sleep(5)
            if record['status'] != 'completed':
                raise RuntimeError(f'ep{episode} mask operation stopped: {record}; no automatic retry')
        if not (directory / 'mask-gates.json').exists():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
            subprocess.run(['nice', '-n', '10', *command, 'mask-gates', '--episode', str(episode)],
                           cwd=ROOT, env=env, check=True)
        result = json.loads((directory / 'mask-gates.json').read_text())
        with ledger.open('a') as stream:
            stream.write(json.dumps(dict(time_utc=now_utc(), episode=episode,
                         phase='mask-gates', passed=result['passed'], gates=result['gates']))+'\n')
        if result['passed']:
            event(episode, 'mask screen passed; camera solve is next; later episodes remain queued')
            return
    event('all', 'ordered mask screen finished; see per-episode gates-final.json and result.md')


if __name__ == '__main__':
    main()
