"""Kit-environment conversion, development scoring and placeholder production operations."""

import argparse
import json
import os
from pathlib import Path
import sys

from track3.paths import PROJECT as ROOT, project_path
from track3.provenance import input_file, sha256_file, write_json
from track3.runs import runtime_flag


def budget(job):
    import numpy as np
    from scripts.track3 import local_cdo
    arrays = {}
    for prefix, path in (('',job['mesh']),('v4_',job['v4'])):
        vertices, faces = local_cdo.budgeted(ROOT/path)
        arrays[prefix+'vertices'], arrays[prefix+'faces'] = vertices, faces
    np.savez_compressed(ROOT/job['output'],**arrays)


def development(job):
    import numpy as np
    from scripts.track3 import local_cdo
    from track3.features.mesh_extents._rule_scores import FOLDS
    from ._metrics import development_metrics
    admission = json.loads((ROOT/job['admissions']).read_text())
    assert admission['status']=='completed'
    queue = json.loads((ROOT/job['queue']).read_text())
    frozen = json.loads((ROOT/queue['development_baseline']).read_text())
    incumbent = frozen['incumbent']
    trial_rows = []
    for name,before in incumbent['per_object_cm'].items():
        views = [r for r in admission['pairs'] if r['source_split']=='public' and r['object']==name]
        if not views: continue
        reference_path = local_cdo.scan_path(name)
        reference = local_cdo.budgeted(reference_path)
        baseline_mesh = local_cdo.budgeted(ROOT/job['v4_objects']/f'{name}.ply')
        base_values = [local_cdo.CDO._object_chamfer_cm(reference,baseline_mesh,local_cdo.CDO._object_seed(e,0)) for e in (1,3,5)]
        if abs(np.mean(base_values)-before)>1e-12: raise ValueError('Development baseline differs from matched v4')
        for row in views:
            candidate = local_cdo.budgeted(ROOT/row['selected_mesh'])
            values = (base_values if row['decision']=='v4' else
                      [local_cdo.CDO._object_chamfer_cm(reference,candidate,local_cdo.CDO._object_seed(e,0)) for e in (1,3,5)])
            trial_rows.append(dict(key=row['key'],object=name,seeds=[1000003,3000009,5000015],values_cm=values,
                                   v4_values_cm=base_values,scan=input_file(ROOT,reference_path),
                                   selected_mesh_sha256=sha256_file(ROOT/row['selected_mesh'])))
    metrics=development_metrics(incumbent,trial_rows,admission['pairs'],FOLDS)
    result=dict(status='completed',**metrics,trials=trial_rows,
                baseline_per_object_cm=incumbent['per_object_mean_cm'],baseline_pair_weighted_cm=incumbent['pair_weighted_mean_cm'],
                no_adjustable_parameters=True,admission_sha256=sha256_file(ROOT/job['admissions']),
                interpretation='Public development views per object (missing masks retain v4), averaged within seed, then equal-object / fixed 26-pair weights. Evaluation views are separate.',
                disclosure='Pot/pitcher selected after G1 on reused scans; no untouched validation.')
    write_json(ROOT/job['output'],result)



def packed(job):
    import time
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from scripts.track3 import local_cdo
    from track3.features.mesh_eval import pairs
    from ._metrics import packed_metrics
    from v2dlb import object_mesh as om
    queue=json.loads((ROOT/job['queue']).read_text())
    roster=json.loads((ROOT/queue['roster']).read_text())['episodes']
    directory=ROOT/job['directory']; directory.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(ROOT/job['v4']); new=pd.read_parquet(ROOT/job['candidate'])
    if not old.row_id.equals(new.row_id) or list(old.columns)!=list(new.columns): raise ValueError('Packed row order/schema changed')
    parts=old.row_id.str.extract(om.ROW_ID_PATTERN)
    production=json.loads((ROOT/job['production']).read_text())
    changed={(r['episode'],r['object']) for r in production['pairs'] if r['changed']}
    unchanged=np.array([(int(e),n) not in changed for e,n in zip(parts[0],parts[1])])
    if not np.array_equal(old.loc[unchanged,['x','y','z']],new.loc[unchanged,['x','y','z']]): raise ValueError('Unchanged packed geometry differs')
    references,scans,roster_pairs={},{},[]
    for ep,names in sorted(roster.items(),key=lambda x:int(x[0])):
        if int(ep) in (41,43,44): raise PermissionError('Excluded roster episode')
        for slot,name in enumerate(names):
            if name not in scans:
                try:
                    path=local_cdo.scan_path(name)
                    scans[name]=input_file(ROOT,path); references[name]=local_cdo.budgeted(path)
                except FileNotFoundError: scans[name]=None
            roster_pairs.append(dict(episode=int(ep),slot=slot,object=name,seed=local_cdo.CDO._object_seed(int(ep),slot),
                                     status='covered' if name in references else 'no_public_scan'))
    request=dict(files=dict(v4=job['v4'],candidate=job['candidate']),comparison=['v4','candidate'],max_trials=52,time_budget_seconds=7000)
    record=dict(kind='mesh_eval.pairs',status='running',request=request,pairs=roster_pairs,
                inputs=[input_file(ROOT,ROOT/p) for p in (job['v4'],job['candidate'],job['queue'],job['production'])],scans=scans)
    record['identity']=pairs.identity(record)
    write_json(directory/'run.json',record); start=time.perf_counter()
    rows=pairs.run_queue(SimpleNamespace(**request),record,references,local_cdo.CDO,directory/'ledger.jsonl',start)
    record.update(status='completed',wall_seconds=time.perf_counter()-start)
    pairs.write_summary(directory,record,rows); write_json(directory/'run.json',record)
    result=packed_metrics(rows,roster_pairs); means=result['means']['candidate']
    baseline=json.loads((ROOT/'outputs/mesh_eval/pairs/mesh-round3-veto/summary.json').read_text())['means']['candidate']
    if any(abs(result['means']['v4'][k]-baseline[k])>1e-12 for k in ('pair_weighted_cm','per_object_cm')): raise ValueError('Packed v4 baseline differs')
    write_json(ROOT/job['output'],dict(status='completed',verdict=result['verdict'],gates=result['gates'],
        means=means,baseline_means=result['means']['v4'],objects=result['objects'],
        unchanged_packed_rows=int(unchanged.sum()),unchanged_geometry_identical=True,
        summary=str((directory/'summary.json').relative_to(ROOT))))


def main():
    p=argparse.ArgumentParser(); p.add_argument('operation',choices=('budget','development','packed')); p.add_argument('--request',required=True)
    args=p.parse_args(); runtime_flag(Path(__file__).parent,'g2_authorized','G2 CPU work')
    if Path(sys.executable).resolve()!=project_path(ROOT,'.runtime/kit-env/bin/python'): raise RuntimeError('Use the kit interpreter')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    job=json.loads((ROOT/args.request).read_text())
    {'budget':budget,'development':development,'packed':packed}[args.operation](job)


if __name__=='__main__': main()
