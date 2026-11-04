"""Markdown and neutral figures for saved mesh diagnostic statistics."""

from pathlib import Path

from track3.provenance import write_json


def admission_table(rows: list[dict], summary: dict) -> list[str]:
    """Render every original shape with its mask source and recorded admission."""
    lines = ['| Pair | Mask source | Border px | Remap-invalid px | Near-invalid boundary px | Handle note | Ratio | Decision | Generated cm | V4 cm | Change cm |',
             '|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|']
    for r in rows:
        v, s = r['visibility'], r['score']
        note = v['handle_note'].replace('|', '/')
        lines.append(f"| {r['key']} | {summary['mask_sources'][r['key']]} | {v['frame_border_mask_pixels']} | "
                     f"{v['remap_invalid_mask_pixels']} | {v['near_invalid_boundary_pixels']}/{v['boundary_pixels']} | "
                     f"{note} | {r['ratio']:.5f} | {r['historical_decision']} | {s['generated_cm']:.8f} | "
                     f"{s['v4_cm']:.8f} | {s['change_cm']:+.8f} |")
    return lines


def aggregate_tables(summary: dict) -> list[str]:
    """Render the fixed evaluation-pair and equal-object counterfactuals."""
    lines = ['| All 26 covered evaluation pairs / 10 objects | Pair mean cm | Equal-object mean cm |',
             '|---|---:|---:|']
    lines += [f"| {name} | {r['pair_weighted_cm']:.8f} | {r['equal_object_cm']:.8f} |"
              for name, r in summary['means'].items()]
    lines += ['', 'candidate is shipped v5; combined_admission is the failed G3 candidate. '
              'all_generated admits every generated evaluation shape. Missing-generation pairs and other objects keep v5.', '',
              '| Object | V4 cm | V5 cm | Combined admission cm | All generated cm |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['object']} | {r['v4']:.8f} | {r['candidate']:.8f} | "
              f"{r['combined_admission']:.8f} | {r['all_generated']:.8f} |" for r in summary['objects']]
    return lines


def seed_table(summary: dict) -> list[str]:
    """Render sample spread and describe the variance comparison as a ratio."""
    lines = ['| Unchanged crop (five generation seeds) | CD-O mean cm | SD cm | Min cm | Max cm | Ratio mean | SD | Min | Max |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['seed_spread']:
        c, a = r['cdo_cm'], r['ratio']
        lines.append(f"| {r['key']} | {c['mean']:.8f} | {c['sd']:.8f} | {c['min']:.8f} | {c['max']:.8f} | "
                     f"{a['mean']:.5f} | {a['sd']:.5f} | {a['min']:.5f} | {a['max']:.5f} |")
    lines += ['', f"Within-crop seed / between-pair pot variance ratio = {summary['seed_to_between_pair_variance_ratio']:.5f}. "
              f"Numerator {summary['pot_pooled_within_crop_variance_cm2']:.8f} cm²; "
              f"denominator {summary['pot_between_pair_variance_cm2']:.8f} cm². "
              'This is a descriptive ratio, not a causal fraction explained; only two pot crops were replicated.',
              'Scoring seeds stay fixed within each crop; different views also differ in scoring seeds.']
    return lines


def ranked_seeds(rows: list[dict], queue: dict) -> list[str]:
    """List seed outcomes descriptively without selecting a replacement mesh."""
    lines = ['| Crop / generation seed, ranked within crop by CD-O | CD-O cm | Change from v4 cm | Residual ratio | Would pass 1.5 rule? |',
             '|---|---:|---:|---:|---|']
    for key in queue['seed_crops']:
        for r in sorted([r for r in rows if r['key'] == key], key=lambda r:r['score']['generated_cm']):
            lines.append(f"| {r['key']} / {r['generation_seed']} | {r['score']['generated_cm']:.8f} | "
                         f"{r['score']['change_cm']:+.8f} | {r['ratio']:.5f} | "
                         f"{'yes' if r['hypothetical_admitted'] else 'no'} |")
    return lines


def conclusions(original: list[dict], summary: dict) -> tuple[str, str, str]:
    """Describe the saved counterfactual effects and public-pot seed mean."""
    pots = {r['episode']: r for r in summary['counterfactual_pairs'] if r['object'] == 'white_pot'}
    changes = {episode: pots[episode]['all_generated'] - pots[episode]['combined_admission']
               for episode in (28, 3, 8)}
    rejection = {episode: 'helped' if change > 0 else 'hurt' if change < 0 else 'did not change'
                 for episode, change in changes.items()}
    means = summary['means']
    difference = means['all_generated']['pair_weighted_cm'] - means['combined_admission']['pair_weighted_cm']
    count = len(summary['counterfactual_pairs'])
    other_effects = (f"{rejection[3]} pots 3 and 8" if rejection[3] == rejection[8]
                     else f"{rejection[3]} pot 3 and {rejection[8]} pot 8")
    pair_effects = (f"The {difference:+.8f} cm all-generated minus combined-admission pair-mean difference comes from "
                   f"pot 28 ({changes[28]:+.8f} cm / {count}), offset by pots 3 and 8 "
                   f"({changes[3]:+.8f} and {changes[8]:+.8f} cm / {count}). "
                   f"Rejection {rejection[28]} pot 28 but {other_effects}; this is not validated protection.")
    pot6 = next(r for r in original if r['source_split'] == 'evaluation' and r['episode'] == 6
                and r['object'] == 'white_pot')
    admitted = 'passed' if pot6['historical_decision'] == 'generated' else 'failed'
    worsening = pots[6]['combined_admission'] - pots[6]['candidate']
    effect = 'hurt' if worsening > 0 else 'helped' if worsening < 0 else 'did not change'
    admission = (f"Pot 6 {admitted} video admission and {effect} G3; the packed numerical veto kept v5. "
                 "No diagnostic seed or counterfactual is selected for a pack.")
    spread = next(r for r in summary['seed_spread'] if r['key'] == 'public-000042-white_pot')
    mean = spread['cdo_cm']['mean']
    reference = next(r['score']['v4_cm'] for r in original if r['key'] == spread['key'])
    comparison = 'worse than' if mean > reference else 'better than' if mean < reference else 'the same as'
    seed_mean = ("Pitcher replication covers public 21 only, not shipped evaluation 20. "
                 f"Public pot 42 five-seed mean {mean:.5f} cm is {comparison} its v4 reference {reference:.5f} cm.")
    return pair_effects, admission, seed_mean


def markdown(rows: list[dict], queue: dict, summary: dict) -> str:
    """Explain the dependent evidence, mask confound and per-pair admission effects."""
    original = [r for r in rows if r['generation_seed'] == 12345]
    all_rho, pot_rho = summary['spearman'], summary['spearman_pot']
    pair_effects, admission, seed_mean = conclusions(original, summary)
    lines = ['# Pretrained-shape diagnostics', '', summary['disclosure'], '',
             'Lower CD-O is better; change = generated minus v4. Public views use the three recorded development scoring seeds; evaluation views use their exact packed pair seed.',
             'Mask contact uses original frame coordinates before square padding; near-invalid uses the two-pixel rule. Handle pixels are unmeasured; notes are recorded assignments.', '',
             *admission_table(original, summary), '',
             f"All-shape Spearman rho = {all_rho['rho']:.6f}, n = {all_rho['n']}, nominal p = {all_rho['pvalue']:.6f}. "
             f"Within pots: rho = {pot_rho['rho']:.6f}, n = {pot_rho['n']}, nominal p = {pot_rho['pvalue']:.6f}.",
             'No validated predictive signal: nine dependent observations. Reused scans and pilot selection prevent independent-sample inference.',
             'The three handle-in-view pot crops used cluster masks; the other four used SAM. Mask source is confounded with visibility, and seed spread exceeds the apparent gap; no handle explanation is established.', '',
             *aggregate_tables(summary), '',
             pair_effects,
             admission, '',
             *seed_table(summary), '',
             seed_mean,
             f"Saved campaign resources: {summary['resources']}. Original protected-identical flag: {summary['protected_identical']}; current preservation evidence is in rerender.json.", '',
             *ranked_seeds(rows, queue), '',
             '![Admission and CD-O](admission_vs_cdo.png)', '', '![Generation seed spread with v4 references](seed_spread.png)']
    return '\n'.join(lines)+'\n'


def admission_figure(directory: Path, original: list[dict], summary: dict) -> None:
    """Label each original shape and keep the limit label clear of pitcher labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for r in original:
        admitted = r['historical_decision'] == 'generated'
        ax.scatter(r['ratio'], r['score']['change_cm'], color='0.25' if admitted else '0.6', marker='o' if admitted else 'x')
        label = f"{'Public' if r['source_split']=='public' else 'Eval'} {r['episode']} {'pitcher' if r['object']=='water_pitcher' else 'pot'}"
        offset = {('evaluation', 20): (-104, 8), ('public', 21): (5, -16),
                  ('public', 26): (5, -16)}.get((r['source_split'], r['episode']), (5, 7))
        ax.annotate(label, (r['ratio'], r['score']['change_cm']), xytext=offset, textcoords='offset points', fontsize=9)
    ax.axhline(0, color='0.7', linewidth=0.8)
    ax.axvline(1.5, color='0.6', linewidth=0.8, linestyle='--')
    ax.text(1.5, 0.98, 'Admission limit = 1.5', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=9, color='0.35')
    ax.set(xlabel='Video residual ratio: generated / v4 (unitless)', ylabel='CD-O change from v4 (cm; lower is better)',
           title=f"Video fit has no validated predictive signal (all-shape ρ={summary['spearman']['rho']:.2f}; pot ρ={summary['spearman_pot']['rho']:.2f}).")
    ax.text(0.99, 0.02, 'Circles: admitted    Crosses: rejected\nReused public scans; nine dependent observations',
            transform=ax.transAxes, ha='right', fontsize=9)
    ax.margins(x=0.18, y=0.2)
    fig.savefig(directory/'admission_vs_cdo.png', dpi=180)
    plt.close(fig)


def seed_figure(directory: Path, rows: list[dict], queue: dict) -> None:
    """Show seed outcomes against each crop's unchanged v4 reference."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    labels = []
    for x, key in enumerate(queue['seed_crops']):
        group = sorted([r for r in rows if r['key'] == key], key=lambda r:r['generation_seed'])
        first = group[0]
        labels.append(f"{'Public' if first['source_split']=='public' else 'Evaluation'} {first['episode']}\n"
                      f"{'pitcher' if first['object']=='water_pitcher' else 'pot'}")
        for ax, values, baseline in zip(axes, ([r['score']['generated_cm'] for r in group], [r['ratio'] for r in group]),
                                        (first['score']['v4_cm'], 1.0)):
            for offset, r, value in zip(np.linspace(-0.14, 0.14, len(group)), group, values):
                ax.scatter(x+offset, value, color='0.25', s=20)
                ax.annotate(str(r['generation_seed']), (x+offset, value), xytext=(0, 7), textcoords='offset points',
                            fontsize=8, rotation=90, ha='center')
            ax.scatter(x-0.23, baseline, marker='D', facecolors='white', edgecolors='0.3', s=35)
            ax.annotate('v4', (x-0.23, baseline), xytext=(-3, -13), textcoords='offset points', ha='right', fontsize=9)
            ax.set_xlabel('Unchanged input crop (five generation seeds each)')
            ax.margins(x=0.25, y=0.16)
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels)
    axes[0].set_ylabel('Public-scan CD-O (cm; lower is better)')
    axes[1].set_ylabel('Video residual ratio: generated / v4 (unitless)\nDashed line: admission limit = 1.5')
    axes[1].axhline(1.5, color='0.6', linewidth=0.8, linestyle='--')
    fig.suptitle('Generation seeds vary for each unchanged crop; open diamonds mark the v4 reference.')
    fig.savefig(directory/'seed_spread.png', dpi=180)
    plt.close(fig)


def publish(directory: Path, rows: list[dict], queue: dict, summary: dict) -> None:
    """Write only derived summaries and figures from previously recorded numbers."""
    import matplotlib
    matplotlib.use('Agg')

    write_json(directory/'summary.json', summary)
    (directory/'summary.md').write_text(markdown(rows, queue, summary))
    admission_figure(directory, [r for r in rows if r['generation_seed'] == 12345], summary)
    seed_figure(directory, rows, queue)
