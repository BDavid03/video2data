"""G3 kit-interpreter scoring with the frozen G2 computation and v5 packed baseline."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from src.paths import PROJECT as ROOT, project_path
from src.provenance import input_file, sha256_file, write_json
from src.runs import runtime_flag
from . import g2_cpu
from .g3_math import gates


def development(job):
    """Reuse G2's matched view/seed scores, then apply only G3's development gates."""
    queue = json.loads((ROOT/job['queue']).read_text())
    admission = json.loads((ROOT/job['admissions']).read_text())
    has_new = any(r.get('g3_new') and r['source_split'] == 'public' and r['decision'] == 'generated'
                  for r in admission['pairs'])
    if not has_new:
        write_json(ROOT/job['output'], dict(status='completed', verdict='FAIL',
                   gates=dict(new_pair=False), reason='no newly admitted public pair; stopped before scan scoring',
                   per_object_mean_cm=None, pair_weighted_cm=None, objects=[], folds={}, trials=[]))
        return
    g2_cpu.development(job)
    result = json.loads((ROOT/job['output']).read_text())
    result.update(**gates(result['per_object_mean_cm'], result['pair_weighted_cm'],
                         [r['change_cm'] for r in result['objects']], queue['development_limits'], has_new))
    result.update(limits=queue['development_limits'], disclosure='G3 pot shapes on reused scans; G1 rack failure and G2 post-pilot selection remain disclosed.')
    write_json(ROOT/job['output'], result)


def packed(job):
    """Compare exact v5/v6 packed geometry through the existing mesh_eval pair loop."""
    import numpy as np
    import pandas as pd
    from src.features.mesh_eval import local_cdo
    from src.features.mesh_eval import pairs
    from src.features.mesh_eval._metrics import aggregate
    from v2dlb import object_mesh as om

    queue = json.loads((ROOT/job['queue']).read_text())
    production = json.loads((ROOT/job['production']).read_text())
    directory = ROOT/job['directory']
    directory.mkdir(parents=True, exist_ok=True)
    old, new = pd.read_parquet(ROOT/job['v5']), pd.read_parquet(ROOT/job['candidate'])
    if not old.row_id.equals(new.row_id) or list(old.columns) != list(new.columns):
        raise ValueError('Packed row order/schema changed')
    parts = old.row_id.str.extract(om.ROW_ID_PATTERN)
    changed = {(r['episode'], r['object']) for r in production['pairs'] if r['changed']}
    unchanged = np.array([(int(e), n) not in changed for e, n in zip(parts[0], parts[1])])
    if not np.array_equal(old.loc[unchanged, ['x','y','z']], new.loc[unchanged, ['x','y','z']]):
        raise ValueError('Unchanged packed coordinates differ from v5')
    references, scans, roster_pairs = {}, {}, []
    roster = json.loads((ROOT/queue['roster']).read_text())['episodes']
    for ep, names in sorted(roster.items(), key=lambda item:int(item[0])):
        if int(ep) in (41, 43, 44):
            raise PermissionError('Excluded roster episode')
        for slot, name in enumerate(names):
            if name not in scans:
                try:
                    path = local_cdo.scan_path(name)
                    scans[name] = input_file(ROOT, path)
                    references[name] = local_cdo.budgeted(path)
                except FileNotFoundError:
                    scans[name] = None
            roster_pairs.append(dict(episode=int(ep), slot=slot, object=name,
                                     seed=local_cdo.CDO._object_seed(int(ep), slot),
                                     status='covered' if name in references else 'no_public_scan'))
    request = dict(files=dict(v5=job['v5'], candidate=job['candidate']), comparison=['v5','candidate'],
                   max_trials=52, time_budget_seconds=7000)
    record = dict(kind='mesh_eval.pairs', status='running', request=request, pairs=roster_pairs,
                  inputs=[input_file(ROOT, ROOT/p) for p in (job['v5'], job['candidate'], job['queue'], job['production'])], scans=scans)
    record['identity'] = pairs.identity(record)
    write_json(directory/'run.json', record)
    start = time.perf_counter()
    rows = pairs.run_queue(SimpleNamespace(**request), record, references, local_cdo.CDO, directory/'ledger.jsonl', start)
    record.update(status='completed', wall_seconds=time.perf_counter()-start)
    pairs.write_summary(directory, record, rows)
    write_json(directory/'run.json', record)
    result = aggregate(rows, roster_pairs, ('v5', 'candidate'))
    baseline = json.loads((ROOT/queue['v5_veto']).read_text())['means']
    if any(abs(result['means']['v5'][k]-baseline[k]) > 1e-12 for k in ('pair_weighted_cm','per_object_cm')):
        raise ValueError('Packed v5 baseline differs from frozen veto')
    means = result['means']['candidate']
    verdict = gates(means['per_object_cm'], means['pair_weighted_cm'],
                    [r['candidate']-r['v5'] for r in result['objects']], queue['packed_limits'], bool(changed))
    write_json(ROOT/job['output'], dict(status='completed', **verdict, means=means,
               baseline_means=result['means']['v5'], objects=result['objects'], limits=queue['packed_limits'],
               changed_pairs=len(changed), unchanged_packed_rows=int(unchanged.sum()), unchanged_geometry_identical=True,
               summary=str((directory/'summary.json').relative_to(ROOT))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('budget', 'development', 'packed'))
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    runtime_flag(Path(__file__).parent, 'g3_authorized', 'G3 CPU work')
    if Path(sys.executable).resolve() != project_path(ROOT, 'data/.runtime/kit-env/bin/python'):
        raise RuntimeError('Use the kit interpreter')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    job = json.loads((ROOT/args.request).read_text())
    {'budget':g2_cpu.budget, 'development':development, 'packed':packed}[args.operation](job)


if __name__ == '__main__':
    main()
