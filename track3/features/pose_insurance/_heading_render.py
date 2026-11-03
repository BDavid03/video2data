"""Markdown tables and figures for the recorded heading diagnostic."""
import json
from pathlib import Path


def render(summary: dict, directory: Path, queue: dict, metric_names: list) -> None:
    """Write ranked metrics, all episode scores and prior tables."""
    episodes, variants = queue['episodes'], queue['variants']
    lines = ['# Recorded heading diagnostic', '',
             'Train labels only; Stage A oracles are ceilings. U1 was chosen after label inspection; '
             'LOO is the gate, full-train is descriptive. Scan ADD uses 1,000 seed-zero surface samples.', '',
             'O2-cups and O2-cups-up are descriptive coverage ceilings with no gate. '
             'O2-cups-up applies full-train priors to uncovered objects before the coverage oracle. '
             f"Cup coverage touches episodes {summary['cup_coverage_episodes']}; other episodes in O2-cups-up equal U1-full.", '',
             'Stage B registration failed under the frozen protocol; the historical Stage A trigger below is not a new authorization.', '',
             f"Executed {summary['executed']}; resumed {summary['resumed']}. "
             f"T1 reproduction: {summary['reproduction']['passed']}. "
             f"Stage A passes: {summary['stage_a']['passed']}. U1-LOO passes: {summary['u1_gate']['passed']}.", '',
             f"Changes since previous report: `{json.dumps(summary['mean_changes_since_previous'], sort_keys=True)}`.", '',
             '| Variant | AUC | RPE cm | MPPE cm | SP-SR | MP-SR |', '|---|---:|---:|---:|---:|---:|']
    for variant in summary['ranking']:
        lines.append('| ' + variant + ' | ' + ' | '.join(f'{summary["means"][variant][k]:.9f}' for k in metric_names) + ' |')
    lines += ['', '## Per-episode metrics', '', '| Episode | Variant | AUC | RPE cm | MPPE cm | SP-SR | MP-SR |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for ep in episodes:
        for variant in variants:
            metrics = summary['results'][variant][ep]['metrics']
            lines.append(f'| {ep} | {variant} | ' + ' | '.join(f'{metrics[k]:.9f}' for k in metric_names) + ' |')
    lines += ['', '## Full-train priors', '',
              '| Type | Episodes | N | Body-up observations | Max angle deg | Mean norm | Prior up | Prior? |',
              '|---|---|---:|---|---:|---:|---|---|']
    for name, row in summary['priors_full'].items():
        angle = '-' if row['max_pairwise_angle_deg'] is None else f'{row["max_pairwise_angle_deg"]:.6f}'
        vectors = '; '.join(', '.join(f'{x:.6f}' for x in v) for v in row['up_vectors'])
        up = '-' if row['up'] is None else ', '.join(f'{x:.6f}' for x in row['up'])
        lines.append(f'| {name} | {row["episodes"]} | {row["occurrences"]} | {vectors} | {angle} | '
                     f'{row["mean_norm"]:.9f} | {up} | {row["prior"]} |')
    lines += ['', 'All LOO fold memberships, observations and consistency statistics are in summary.json and each U1 trial.', '',
              '## U1-LOO gate', '', f"Mean changes: `{json.dumps(summary['u1_gate']['changes'], sort_keys=True)}`.", '',
              f"Worst episode AUC loss: {summary['u1_gate']['worst_auc_loss']:.9f}.", '',
              *[f'- {key}: {value}' for key, value in summary['u1_gate']['checks'].items()], '',
              'No scored policy, geometry counter-rotation, physical check or pack is part of this diagnostic.', '',
              '![Per-episode AUC](auc.png)', '']
    (directory / 'summary.md').write_text('\n'.join(lines))


def figure(summary: dict, directory: Path, queue: dict) -> None:
    """Plot every protocol variant's train-episode ADD AUC."""
    episodes, variants = queue['episodes'], queue['variants']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for variant in variants:
        ax.plot(range(len(episodes)), [summary['results'][variant][ep]['metrics']['AUC'] for ep in episodes],
                marker='o', markersize=3, label=variant)
    ax.set(xticks=range(len(episodes)), xticklabels=episodes, xlabel='Train episode', ylabel='ADD AUC', ylim=(0, 1))
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'auc.png', dpi=140)
    plt.close(fig)
