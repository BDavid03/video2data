"""Array operations for frozen image crops and pilot decisions."""

import numpy as np


def masked_square(rgb, mask):
    """Whiten outside pixels and pad the mask bounds by ten percent on each side."""
    ys, xs = np.nonzero(mask)
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 1.2))
    left = int(np.floor((xs.min() + xs.max() + 1 - side) / 2))
    top = int(np.floor((ys.min() + ys.max() + 1 - side) / 2))
    white = np.where(mask[..., None], rgb, 255).astype(np.uint8)
    crop = np.full((side, side, 3), 255, dtype=np.uint8)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(rgb.shape[1], left + side), min(rgb.shape[0], top + side)
    crop[y0 - top:y1 - top, x0 - left:x1 - left] = white[y0:y1, x0:x1]
    return crop, [left, top, left + side, top + side]


def pilot_metrics(rows):
    """Compare equal-object means and contributions over the original 26 covered pairs."""
    objects = []
    for name in sorted({r['object'] for r in rows}):
        group = [r for r in rows if r['object'] == name]
        before = np.array([r['v4_cm'] for r in group])
        after = np.array([r['cd_o_cm'] for r in group])
        objects.append(dict(object=name, pairs=len(group), v4_cm=float(before.mean()),
                            generated_cm=float(after.mean()), gain_cm=float((before - after).mean()),
                            seed_std_cm=float(after.std()), seed_min_cm=float(after.min()),
                            seed_max_cm=float(after.max()), paired_gain_std_cm=float((before-after).std())))
    gain = float(np.mean([r['gain_cm'] for r in objects]))
    before = float(sum(r['v4_cm'] * r['pairs'] for r in objects) / 26)
    after = float(sum(r['generated_cm'] * r['pairs'] for r in objects) / 26)
    gates = dict(mean_gain_gt_0_10=gain > 0.10, pair_contribution_improves=after < before,
                 no_object_regresses_gt_0_15=all(r['gain_cm'] >= -0.15 for r in objects))
    return dict(objects=objects, mean_gain_cm=gain, v4_contribution_cm=before,
                generated_contribution_cm=after, contribution_gain_cm=before-after,
                gates=gates, verdict='PASS' if all(gates.values()) else 'FAIL')
