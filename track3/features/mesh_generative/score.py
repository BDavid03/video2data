"""Budget and score one generated object with the existing matched CD-O entry point."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from scripts.track3 import local_cdo
    from v2dlb import object_mesh as om
    from track3.features.mesh_eval._metrics import solution_frame
    from track3.features.mesh_eval.pairs import score_pair
    from track3.provenance import input_file, sha256_file
    from track3.paths import PROJECT as ROOT

    baseline_path = ROOT/'outputs/mesh_eval/pairs/mesh-round3-veto/ledger.jsonl'
    baseline = [r for line in baseline_path.read_text().splitlines()
                if (r := json.loads(line))['file'] == 'candidate' and r['object'] == args.object]
    reference_path = local_cdo.scan_path(args.object)
    reference = local_cdo.budgeted(reference_path)
    candidate = local_cdo.budgeted(args.mesh)
    baseline_run = json.loads((baseline_path.parent/'run.json').read_text())
    packed_path = ROOT/baseline_run['request']['files']['candidate']
    pinned_pack = next(r for r in baseline_run['inputs'] if r['path'] == baseline_run['request']['files']['candidate'])
    if sha256_file(packed_path) != pinned_pack['sha256']:
        raise ValueError('Incumbent packed file differs from v4 comparison record')
    packed = pd.read_parquet(packed_path)
    first = baseline[0]
    solution = solution_frame(first['episode'], first['slot'], args.object, reference, om.row_id, om.ARRAYS)
    merged = solution.merge(packed, on='row_id', validate='one_to_one')
    v4 = local_cdo.CDO._to_meshes(merged, local_cdo.CDO._POINT_COLUMNS)[first['slot']]
    matched = local_cdo.CDO._object_chamfer_cm(reference, v4, first['seed'])
    equivalence_error = abs(matched - first['cd_o_cm'])
    if equivalence_error > 1e-12:
        raise ValueError(f'V4 matched scoring differs: {equivalence_error}')
    np.savez_compressed(Path(args.output).with_suffix('.npz'), vertices=candidate[0], faces=candidate[1],
                        scan_vertices=reference[0], scan_faces=reference[1],
                        v4_vertices=v4[0], v4_faces=v4[1])
    scores = []
    for before in baseline:
        episode, slot = before['episode'], before['slot']
        assert episode not in (41, 43, 44)
        pair = {k: before[k] for k in ('episode','slot','object','seed')}
        solution = solution_frame(episode, slot, args.object, reference, om.row_id, om.ARRAYS)
        predicted = solution_frame(episode, slot, args.object, candidate, om.row_id, om.ARRAYS)
        submission = predicted[['row_id','ref_x','ref_y','ref_z']].rename(
            columns={'ref_x':'x','ref_y':'y','ref_z':'z'})
        submission['code_commit_url'] = packed['code_commit_url'].iloc[0]
        row = score_pair('G1', pair, solution, submission, local_cdo.CDO,
                         sha256_file(args.mesh), sha256_file(baseline_path))
        row['v4_cm'] = before['cd_o_cm']
        scores.append(row)
        print(f"{args.object} seed={pair['seed']} G1={row['cd_o_cm']:.8f} v4={row['v4_cm']:.8f}",flush=True)
    record = dict(rows=scores, inputs=[input_file(ROOT,p) for p in (args.mesh,reference_path,baseline_path,packed_path)],
                  v4_equivalence_error_cm=equivalence_error, conversion='v2dlb.mesh_budget.budget_mesh(path, 4096, 4096)',
                  scoring='mesh_eval.pairs.score_pair -> unmodified CD-O.score',
                  scorer_envelope='Incumbent commit URL only satisfies local scorer schema; G1 working sources are separately hashed. No submission is built.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
