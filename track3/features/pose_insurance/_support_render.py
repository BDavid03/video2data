"""Render support diagnostic values and decoded image maps."""

import json
import math
from pathlib import Path


def serializable(value):
    """Convert arrays and explicit infinite errors into strict JSON values."""
    import numpy as np
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('Unexpected NaN outside saved image maps')
        return 'Infinity' if value > 0 else '-Infinity'
    return value


def format_number(value):
    """Format a nullable centimetre quantity without hiding missingness."""
    return 'missing' if value is None else f'{value:.4f}'


def markdown(report: dict, clean_episodes: tuple, validation_episodes: tuple) -> str:
    """Present attribution, scale gaps, held-out fits and frozen readings."""
    lines = ['# Identical-support pose diagnostic', '',
        f"This invocation: **{report['executed']} new episodes; {report['resumed']} resumed episodes**.", '',
        'Descriptive only. Stereo is a calibrated estimate, not measured depth truth. No candidate or pack follows.',
        'All gaps and contributions below are signed horizontal lengths in cm. Depth = g_I; g_F = depth + support.', '']
    for title, roster in (('Clean train episodes', clean_episodes),
                          ('Spent public validation', validation_episodes)):
        lines += [f'## {title}', '', '| Episode | g_F | g_I / depth | Support | MoGe support | Stereo support | MoGe×s full gap | MoGe×s common gap | MoGe×r_I full gap | MoGe×r_I common gap |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in report['episodes']:
            if row['episode'] not in roster:
                continue
            gaps = row['gaps']
            values = [gaps['native'][key] if gaps.get('native') else None
                      for key in ('g_F_cm', 'g_I_cm', 'support_cm', 'moge_support_cm', 'stereo_support_cm')]
            values += [gaps[name][key] if gaps.get(name) else None
                       for name in ('recorded', 'ratio') for key in ('g_F_cm', 'g_I_cm')]
            lines.append(f"| {row['episode']} | " + ' | '.join(map(format_number, values)) + ' |')
        lines.append('')
    lines += ['## Missing independent-object measurements', '']
    lines += [f"- Episode {row['episode']}: {row['missing_reason']}." for row in report['episodes'] if row['missing_reason']]
    lines += ['', '## Episode 12 assignment contrast', '',
              'Native MoGe full support and the recorded s are held fixed. Only default slot-0 xy is replaced.', '',
              '```json', json.dumps(serializable(report['assignment_episode_12']), indent=2), '```', '',
              '## Table models: spatially held-out depth MAE (cm)', '',
              '| Episode | Subset | Scale | Affine depth | Affine inverse | Plane MoGe AbsRel | Plane stereo AbsRel |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for row in report['episodes']:
        models = row['table']['models']
        values = [models[name]['mae_cm'] if name in models else None for name in ('scale', 'affine_depth', 'affine_inverse')]
        values += [models[name]['absolute_relative'] if name in models else None for name in ('plane_moge', 'plane_stereo')]
        lines.append(f"| {row['episode']} | {row['subset']} | " + ' | '.join(map(format_number, values)) + ' |')
    lines += ['', '## Frozen descriptive readings', '', '| Reading | Count / denominator | Required | Qualified |', '|---|---:|---:|---|']
    for name, reading in report['readings'].items():
        count = reading.get('count', int(reading['qualified']))
        lines.append(f"| {name} | {count}/{reading['denominator']} | {reading['required']} | {reading['qualified']} |")
    lines += ['', '## Spent validation readings (descriptive; no confirmation verdict)', '',
              '| Reading | ' + ' | '.join(f'Episode {episode}' for episode in validation_episodes) + ' | Count / denominator |', '|---|---|---|---|---:|']
    for name, reading in report['spent_validation_readings'].items():
        flags = ' | '.join(str(reading['episodes'].get(str(episode), 'missing')) for episode in validation_episodes)
        lines.append(f"| {name} | {flags} | {reading['count']}/{reading['denominator']} |")
    lines += ['', '## Support counts', '', '| Episode | Mask pixels 0/1 | Common pixels 0/1 | MoGe-full pixels 0/1 | Stereo-full pixels 0/1 |', '|---:|---|---|---|---|']
    for row in report['episodes']:
        counts = None if row['support'] is None else row['support']['counts']
        lines.append(f"| {row['episode']} | " + ' | '.join(str(counts[name]) if counts else 'missing'
                    for name in ('mask', 'common', 'moge_full', 'stereo_full')) + ' |')
    lines += ['', '## Limits and artifacts', '',
        '- Masks and table bands are MoGe-conditioned. Spent validation is reported separately and is not fresh confirmation.',
        '- Per-pixel differences are along their own rays. Cross-ray centroid shifts do not establish a separate lateral warp.',
        '- Reference vector errors are conditional on the guessed S1 body frame; validation vector-error columns are missing.',
        '- Figure ratio colours saturate outside 0.5–1.5 and residual colours outside ±5 cm; numeric errors remain unclipped.',
        '- Missing values retain their original denominators. Infinite predictions are encoded as "Infinity" in strict JSON.',
        '- summary.json contains continuous aggregates, all vectors/errors, ratios, normal sensitivity, fold coefficients and eligibility.',
        '- Episode records and maps are immutable; ledger.jsonl is append-only. reports/ preserves each invocation.', '',
        '![Support attribution, coverage and table maps](support.png)', '', 'Reproduce:', '', '```bash', report['reproduce'], '```', '']
    return '\n'.join(lines)


def figure(rows: list, maps: list, path: Path, clean_episodes: tuple,
           validation_episodes: tuple) -> None:
    """Render attribution, coverage and table residual maps in one labelled figure."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 3 + 1.8 * len(rows)), layout='constrained')
    grid = fig.add_gridspec(len(rows) + 1, 5)
    ax = fig.add_subplot(grid[0, :2])
    clean = [row for row in rows if row['episode'] in clean_episodes]
    x = np.arange(len(clean))
    for shift, key, label in ((-.25, 'g_F_cm', 'Whole-support gap'), (0, 'depth_cm', 'Identical-pixel depth'), (.25, 'support_cm', 'Net support')):
        ax.bar(x + shift, [row['gaps']['native'][key] if row['gaps'].get('native') else np.nan for row in clean], width=.25, label=label)
    ax.set(xticks=x, xticklabels=[row['episode'] for row in clean], ylabel='Signed gap (cm)', title='Clean train: native MoGe minus stereo')
    ax.axhline(0, color='black', linewidth=.6)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[0, 2:])
    supported = [row for row in rows if row['support'] is not None]
    x = np.arange(len(supported))
    for slot, shift in ((0, -.18), (1, .18)):
        ax.bar(x + shift, [row['support']['fractions']['common'][slot] for row in supported], .35, label=f'Slot {slot}')
    ax.set(xticks=x, xticklabels=[row['episode'] for row in supported], ylim=(0, 1), ylabel='Common / mask pixels', title='Identical support; ' + '/'.join(map(str, validation_episodes)) + ' are spent validation')
    ax.legend(fontsize=8)
    columns = [('ratio_map', 'Stereo / MoGe depth', .5, 1.5, 'viridis'),
               ('scale_residual_cm', 'Scale residual (cm)', -5, 5, 'coolwarm'),
               ('affine_depth_residual_cm', 'Affine depth residual (cm)', -5, 5, 'coolwarm'),
               ('affine_inverse_residual_cm', 'Affine inverse residual (cm)', -5, 5, 'coolwarm'),
               ('table_mask', 'Retained table pixels', 0, 1, 'gray')]
    for index, (row, arrays) in enumerate(zip(rows, maps), 1):
        for col, (key, title, low, high, cmap) in enumerate(columns):
            ax = fig.add_subplot(grid[index, col])
            if key in arrays:
                artist = ax.imshow(arrays[key], vmin=low, vmax=high, cmap=cmap)
                if index == 1:
                    fig.colorbar(artist, ax=ax, shrink=.65)
            else:
                ax.text(.5, .5, 'Missing: frozen fold support', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Episode {row['episode']}: {title}", fontsize=9)
            ax.set_axis_off()
    fig.suptitle('Descriptive support diagnostic; table support is MoGe-conditioned; residual = predicted stereo minus stereo', fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


