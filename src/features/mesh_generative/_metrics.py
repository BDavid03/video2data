"""Pure G2 view/seed aggregation, fold means and frozen regression gates."""

import numpy as np


def _gates(object_mean, pair_mean, changes, object_limit, pair_limit):
    return dict(equal_object_mean=object_mean<object_limit, pair_weighted_mean=pair_mean<=pair_limit,
                no_object_regresses=all(change<=0.15 for change in changes))


def development_metrics(incumbent, trials, admissions, folds):
    """Average views within each seed, then compute object, weighted and fold results."""
    objects=[]
    for name,before in incumbent['per_object_cm'].items():
        views=[r for r in admissions if r['source_split']=='public' and r['object']==name]
        row=dict(object=name,v4_cm=before,g2_cm=before,change_cm=0.0,
                 seed_std_cm=incumbent['seed_std_cm'][name],pairs=incumbent['pair_counts'][name],
                 fold='A' if name in folds['A'] else 'B',unchanged=True)
        if views:
            per_seed=np.mean([r['values_cm'] for r in trials if r['object']==name],axis=0)
            after=float(per_seed.mean())
            row.update(g2_cm=after,change_cm=after-before,seed_std_cm=float(per_seed.std()),
                       seed_values_cm=per_seed.tolist(),development_views=len(views),
                       unchanged=all(r['decision']=='v4' for r in views))
        objects.append(row)
    mean=float(np.mean([r['g2_cm'] for r in objects]))
    weighted=sum(r['g2_cm']*r['pairs'] for r in objects)/sum(r['pairs'] for r in objects)
    gates=_gates(mean,weighted,[r['change_cm'] for r in objects],0.62270304,0.66288090)
    return dict(verdict='PASS' if all(gates.values()) else 'FAIL',per_object_mean_cm=mean,
                pair_weighted_cm=weighted,gates=gates,objects=objects,
                folds={f:float(np.mean([r['g2_cm'] for r in objects if r['fold']==f])) for f in ('A','B')})


def packed_metrics(rows, pairs):
    """Reuse the matched-pair aggregation and apply the frozen packed thresholds."""
    from src.features.mesh_eval._metrics import aggregate
    result=aggregate(rows,pairs,('v4','candidate'))
    means=result['means']['candidate']
    gates=_gates(means['per_object_cm'],means['pair_weighted_cm'],
                 [r['candidate']-r['v4'] for r in result['objects']],0.62338396,0.66292148)
    return dict(**result,gates=gates,verdict='PASS' if all(gates.values()) else 'FAIL')
