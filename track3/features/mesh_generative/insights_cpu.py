"""Score diagnostic meshes with the installed kit and its unchanged CD-O path."""

import argparse
import json
import os
from pathlib import Path
import sys

from track3.paths import PROJECT as ROOT, project_path
from track3.provenance import input_file, write_json
from track3.runs import runtime_flag


def score(job: dict) -> None:
    """Budget both meshes and the public scan, then use the recorded scoring seeds."""
    import numpy as np
    from track3.features.mesh_eval import local_cdo

    reference_path = local_cdo.scan_path(job['object'])
    reference = local_cdo.budgeted(reference_path)
    generated = local_cdo.budgeted(ROOT/job['mesh'])
    incumbent = local_cdo.budgeted(ROOT/job['v4'])
    values = [local_cdo.CDO._object_chamfer_cm(reference, generated, seed) for seed in job['seeds']]
    baseline = [local_cdo.CDO._object_chamfer_cm(reference, incumbent, seed) for seed in job['seeds']]
    np.savez_compressed(ROOT/job['arrays'], vertices=generated[0], faces=generated[1])
    write_json(ROOT/job['output'], dict(status='completed', seeds=job['seeds'], values_cm=values,
               v4_values_cm=baseline, generated_cm=float(np.mean(values)), v4_cm=float(np.mean(baseline)),
               change_cm=float(np.mean(values)-np.mean(baseline)),
               inputs=[input_file(ROOT, p) for p in (reference_path, ROOT/job['mesh'], ROOT/job['v4'])]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    runtime_flag(Path(__file__).parent, 'insights_authorized', 'Mesh insights CPU scoring')
    if Path(sys.executable).resolve() != project_path(ROOT, '.runtime/kit-env/bin/python'):
        raise RuntimeError('Use the kit interpreter')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    score(json.loads(project_path(ROOT, args.request).read_text()))
