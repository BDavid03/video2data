"""Pure descriptive statistics for reused-scan mesh diagnostics."""

import numpy as np
from scipy.stats import spearmanr


def admission_ratio(residual: dict, baseline: float) -> float:
    """Compare the generated fit with its pair's recorded v4 residual."""
    return residual['median_residual_m']/baseline


def correlation(rows: list[dict]) -> dict:
    """Describe ranks without treating views of reused objects as independent evidence."""
    rho = spearmanr([r['ratio'] for r in rows], [r['score']['change_cm'] for r in rows])
    return dict(rho=float(rho.statistic), pvalue=float(rho.pvalue), n=len(rows))


def seed_spread(rows: list[dict], crops: list[str]) -> list[dict]:
    """Summarize five generation seeds per crop using sample standard deviations."""
    result = []
    for key in crops:
        group = [r for r in rows if r['key'] == key]
        item = dict(key=key, n=len(group))
        for name, values in [('cdo_cm', [r['score']['generated_cm'] for r in group]),
                             ('ratio', [r['ratio'] for r in group])]:
            item[name] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                              min=float(np.min(values)), max=float(np.max(values)))
        result.append(item)
    return result


def counterfactuals(original: list[dict], packed: dict) -> dict:
    """Aggregate recorded pair scores without budgeting or rescoring any geometry."""
    pairs = [dict(r) for r in packed['pairs']]
    for pair in pairs:
        row = next((r for r in original if r['source_split'] == 'evaluation' and
                    r['episode'] == pair['episode'] and r['object'] == pair['object']), None)
        pair['all_generated'] = row['score']['generated_cm'] if row else pair['candidate']
        pair['combined_admission'] = (row['score']['generated_cm'] if row['historical_decision'] == 'generated'
                                     else row['score']['v4_cm']) if row else pair['candidate']
    methods = ('v4', 'candidate', 'combined_admission', 'all_generated')
    objects = [dict(object=name, **{m: float(np.mean([r[m] for r in pairs if r['object'] == name]))
                                   for m in methods}) for name in sorted({r['object'] for r in pairs})]
    means = {m: dict(pair_weighted_cm=float(np.mean([r[m] for r in pairs])),
                     equal_object_cm=float(np.mean([r[m] for r in objects]))) for m in methods}
    return dict(counterfactual_pairs=pairs, objects=objects, means=means)


def resources(rows: list[dict], wall_seconds: float) -> dict:
    """Summarize the saved generation telemetry without querying any device."""
    generations = [r['generation'] for r in rows if r['generation']]
    return dict(wall_seconds=wall_seconds, new_generations=len(generations),
                peak_allocated_mib=max(g['peak_allocated_mib'] for g in generations),
                peak_reserved_mib=max(g['peak_reserved_mib'] for g in generations),
                peak_sampled_device_mib=max(g['gpu_lease']['peak_sampled_total_device_used_mib']
                                            for g in generations))


def summarize(rows: list[dict], queue: dict, packed: dict, resource_values: dict,
              protected_identical: bool) -> dict:
    """Compute the original statistics plus pot-only ranks and explicit mask sources."""
    original = [r for r in rows if r['generation_seed'] == 12345]
    pots = [r for r in original if r['object'] == 'white_pot']
    spreads = seed_spread(rows, queue['seed_crops'])
    within = float(np.mean([r['cdo_cm']['sd']**2 for r in spreads if 'white_pot' in r['key']]))
    between = float(np.var([r['score']['generated_cm'] for r in pots], ddof=1))
    return dict(disclosure=queue['disclosure'], spearman=correlation(original),
                spearman_pot=correlation(pots), seed_spread=spreads,
                seed_sd_convention='sample SD (ddof=1), five generation seeds per unchanged crop',
                pot_between_pair_variance_cm2=between, pot_pooled_within_crop_variance_cm2=within,
                seed_to_between_pair_variance_ratio=within/between,
                mask_sources={r['key']: r['mask_source'] for r in queue['pairs']},
                resources=resource_values, protected_identical=protected_identical,
                **counterfactuals(original, packed))


def visibility_counts(mask: np.ndarray, valid: np.ndarray) -> dict:
    """Count frame-edge and invalid-remap contacts in an original input mask."""
    from .g3_math import mask_checks

    rule = dict(boundary_margin_pixels=2, minimum_pixels=0, maximum_pixels=mask.size,
                maximum_border_fraction=1, minimum_observed_points=0)
    measured = mask_checks(mask, valid, valid, rule)
    edge = np.zeros_like(mask)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return dict(frame_border_mask_pixels=int((mask & edge).sum()),
                remap_invalid_mask_pixels=int((mask & ~valid).sum()),
                near_invalid_boundary_pixels=measured['near_invalid_boundary_pixels'],
                boundary_pixels=measured['boundary_pixels'], mask_pixels=int(mask.sum()))
