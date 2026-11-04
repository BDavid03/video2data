"""Train a Stage-0-passed video task once M1's sealed pass admits it (`docs/chain/moving-m1-sealed.md`, clause 2)."""

import argparse
import json
import os
import subprocess
import time

from . import handler as h
from . import video

SEALED = h.ROOT / 'data/outputs/moving_reference/ep39/sealed/sealed.json'


def admit(stage, episode):
    """Refuse unless the one sealed pass passed on 39 and this task still uses its sealed reference."""
    sealed = json.loads(SEALED.read_text())
    if not (sealed['reference_passed'] and sealed['allowance_passed']):
        raise PermissionError('The sealed pass did not admit M2v')
    source = stage['reference']
    if h.digest(h.ROOT / source['source_path']) != source['source_sha256']:
        raise ValueError('The reference changed after Stage 0')
    if episode == 39 and source['source_sha256'] != sealed['reference_sha256']:
        raise ValueError('This task does not use the sealed reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True, choices=(38, 39))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = h.ROOT / args.output
    video.select(args.episode, output)
    if not json.loads((h.FEATURE / 'runtime.json').read_text())['video_train_authorized']:
        raise PermissionError('video_train_authorized is false; the lead sets it')
    stage = json.loads((output / 'stage0.json').read_text())
    if stage['status'] != 'PASS':
        raise PermissionError('Stage 0 failed; this episode stopped')
    h.verify_scientific_inputs(stage)
    admit(stage, args.episode)
    command = json.loads((output / 'commands.json').read_text())['train']
    record_path = output / 'train.run.json'
    with record_path.open('x') as stream:
        json.dump(dict(status='running', operation='train', pid=os.getpid(), started_unix=time.time(),
                       stage0_sha256=h.digest(output / 'stage0.json'), admission=str(SEALED.relative_to(h.ROOT))), stream, indent=2)
    record = json.loads(record_path.read_text())
    from src.gpu import GpuLease
    try:
        with GpuLease(h.ROOT, 1024, 6000, stage='moving-video-train') as lease:
            env = {**h.environment(cpu=False), 'DYNAMIC_FEASIBILITY_DEADLINE': str(time.time() + 4500)}
            with (output / 'train.log').open('w') as log:
                subprocess.run(command, cwd=h.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            record['gpu'] = lease.report()
        record.update(status='completed', finished_unix=time.time())
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        raise
    finally:
        h.write(record_path, record)


if __name__ == '__main__':
    main()
