"""Re-score preserved cup tracks under declared symmetry without tracking again."""

import json

import numpy as np

from src.provenance import write_json,sha256_file
from src.runs import now_utc
from src.features.moving_reference._final_gates import compose_final
from . import two_object_screen as screen
from .two_object_rotation import amend
from .two_object_task import ROSTER

ROOT=screen.ROOT
OUTPUT=ROOT/'data/outputs/moving_policy/two-object'


def run(episode,directory):
    source=OUTPUT/'campaign'/f'ep{episode}/candidate-c'
    role=json.loads((source.parent/'roles.json').read_text())
    names=json.loads(ROSTER.read_text())['episodes'][str(episode)]
    active=role['active_slot']
    initialization=json.loads((source/'initialization.json').read_text())
    axis=initialization['symmetry_axis_model']
    files=['tracking.npz','reference.npz','object-diagnostics.json','gates.json','gates-amended.json','initialization.json']
    hashes={name:sha256_file(source/name) for name in files}
    with np.load(source/'tracking.npz') as saved:
        poses=saved['world_T_asset'].copy();valid=saved['valid'].copy()
    with np.load(source/'reference.npz') as saved:
        times=saved['time'].copy()
    rows=json.loads((source/'object-diagnostics.json').read_text())
    gates=json.loads((source/'gates-amended.json').read_text())
    gates,report=amend(gates,poses,times,rows,names[active],axis)
    report.update(episode=episode,time_utc=now_utc(),saved_inputs=hashes,no_retracking=True,
                  original_valid_frames=int(valid.sum()),new_valid_frames=int(valid.sum()),frames=len(valid))
    write_json(directory/'rotation-symmetry-rescore.json',report)
    sealed=json.loads((ROOT/'data/outputs/moving_reference/ep39/sealed/sealed.json').read_text())
    final=compose_final(gates,sealed)
    waived={'mask_usable_fraction','mask_longest_unusable_interval_frames'}
    failed=[(name,g) for name,g in final['gates'].items() if name not in waived and not g['passed']]
    passed=not failed
    write_json(directory/'rescored-gates.json',dict(final,passed=passed,waived_limits=sorted(waived),source=str(source.relative_to(ROOT))))
    if any(sha256_file(source/name)!=digest for name,digest in hashes.items()):
        raise ValueError('Saved-track rescore modified a preserved input')
    if passed:
        raise RuntimeError('Saved-track rescore needs remaining unmeasured table audits before Stage 0; no training authorized yet')
    reason='; '.join(f'{name}: {g["value"]} {g["comparison"]} {g["threshold"]}' for name,g in failed)
    screen.event(2,f'ep{episode}: saved-track symmetry re-score; {report["measurements"]}; restored valid=0; STOP {reason}')
    return dict(episode=episode,A=names[active],B=names[1-active],active_slot=active,passed=False,
                stage='reference symmetry re-score',reason=reason,gates=final['gates'],symmetry_rescore=report)
