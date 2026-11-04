"""Continue the ordered spoon camera screen through the fixed M1 table fit."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import camera, full_c, spoon, spoon_reference
from ._final_gates import FINAL_LIMITS
from .gates import _amended_check


def table_stage(episode, directory, result):
    """Reuse candidate C's precontact table fit and its unchanged final limits."""
    selected = spoon.inputs(episode)
    full_c.cache_frames(selected, directory)
    with np.load(directory/'camera-ba.npz') as arrays:
        cameras = arrays['world_T_camera'].copy()
    gauge, table = full_c.table_fit(directory, cameras)
    cameras = np.einsum('ij,tjk->tik', gauge, cameras)
    write_json(directory/'table.json', table)
    np.savez_compressed(directory/'camera-table.npz', world_T_camera=cameras)
    measured = dict(result['gates'])
    for name, value in [('table_residual_rms_m', table['residual_rms_m']),
                        ('table_disjoint_patch_normal_max_deg', table['patch_normal_max_deg'])]:
        comparison, limit = FINAL_LIMITS[name]
        measured[name] = _amended_check(value, limit, comparison)
    failed = next((name for name in FINAL_LIMITS if name in measured and name not in spoon_reference.WAIVED
                   and not measured[name]['passed']), None)
    result = dict(result, status='STOP' if failed else 'TABLE PASS; OBJECT PENDING',
                  first_failed_gate=failed, gates=measured, completed_utc=now_utc(),
                  evidence=str((directory/'table-final.json').relative_to(spoon.ROOT)))
    write_json(directory/'table-final.json', result)
    return result


def main():
    cv2.setNumThreads(2)
    cv2.setRNGSeed(31)
    queue = json.loads(Path(__file__).with_name('spoon_queue.json').read_text())
    if queue['episodes'] != [35, 36, 37]:
        raise ValueError('Spoon order must stay 35, 36, 37')
    results = []
    for episode in queue['episodes']:
        directory = spoon.ROOT/f'data/outputs/moving_reference/ep{episode}/reference-mask-waiver'
        directory.mkdir(parents=True, exist_ok=True)
        camera_path = directory/'camera-final.json'
        if camera_path.exists():
            result = json.loads(camera_path.read_text())
        else:
            spoon.event(episode, 'mask-usability waiver: camera solve started on existing masks; CPU')
            result = spoon_reference.camera_stage(episode, directory)
            write_json(directory/'camera-completion.json', dict(status='completed', result=result))
            spoon.event(episode, f'{result["status"]}: {result["first_failed_gate"]}; '
                        f'{result["gates"].get(result["first_failed_gate"], {})}; {result["evidence"]}')
        if not result['first_failed_gate']:
            path = directory/'table-final.json'
            if path.exists():
                result = json.loads(path.read_text())
            else:
                spoon.event(episode, 'camera PASS; precontact table audit started; CPU; unchanged candidate C')
                write_json(directory/'table-started.json', dict(started_utc=now_utc(),
                           sources={str(Path(p).relative_to(spoon.ROOT)): sha256_file(p)
                                    for p in (__file__, full_c.__file__, camera.__file__)}))
                result = table_stage(episode, directory, result)
                spoon.event(episode, f'{result["status"]}: {result["first_failed_gate"]}; '
                            f'{result["gates"].get(result["first_failed_gate"], {})}; {result["evidence"]}')
        results.append(result)
        spoon_reference.summary(results)
        with (spoon_reference.POLICY/'spoon-reference-table-ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        if not result['first_failed_gate']:
            return
    spoon.event('all', 'reference queue finished: all stopped before object tracking; no Stage 0 or training')


if __name__ == '__main__':
    main()
