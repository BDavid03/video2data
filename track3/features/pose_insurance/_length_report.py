"""Presentation of train and confirmation length rows, without loading data or gates."""


def markdown(report: dict, presentation: dict) -> str:
    """Render the shared length rows with the request's available comparison sections."""
    def fmt(value):
        return 'missing' if value is None else f'{value:.3f}'
    video = presentation['video']
    ray = presentation['ray_errors']
    quantities = presentation['quantities']
    verdict = report['authorization']['verdict'] if ray else report['verdict']
    lines = [presentation['heading'], '', verdict, '']
    lines += [line.format(**report) for line in presentation['introduction']]
    lines += ['', presentation['length_header'], presentation['length_separator']]
    for row in report['episodes']:
        values = [str(row['episode'])]
        if not ray:
            values += [str(row['counted']), str(row['merged_cluster'])]
        values += [fmt(row['lengths_cm'][name]) for name in quantities]
        if not ray:
            values += [str(row['cluster_pixels']), str(row['stereo_valid_pixels']), fmt(row['scale'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    if ray:
        lines += ['', 'Ratios to M_o (unitless):', '', '| Episode | ' + ' | '.join(quantities[1:]) + ' |',
                  '|---|' + '---:|' * (len(quantities) - 1)]
        for row in report['episodes']:
            lines.append(f"| {row['episode']} | " + ' | '.join(fmt(row['ratios_to_M_o'][k]) for k in quantities[1:]) + ' |')
        lines += ['', '| Subset | Video construction | Count | MAE vs M_o (cm) | MAE vs M_c (cm) |', '|---|---|---:|---:|---:|']
        for group, entries in report['means'].items():
            for name in video:
                r = entries[name]
                lines.append(f"| {group} | {name} | {r['count']} | {fmt(r['mae_vs_M_o_cm'])} | {fmt(r['mae_vs_M_c_cm'])} |")
        lines += ['', '| Episode | M_c − M_o (cm) | Absolute origin length difference (cm) | V_inc − M_c (cm) | Pixels slot 0/1 | Stereo valid slot 0/1 | Merged | Default slots 0/1 | 0.5 m fallback |', '|---|---:|---:|---:|---|---|---|---|---|']
        for r in report['episodes']:
            v = r['decomposition_cm']['V_inc']
            lines.append(f"| {r['episode']} | {fmt(v['origin'])} | {fmt(v['absolute_origin'])} | {fmt(v['video_vs_centre'])} | {r['cluster_pixels']} | {r['stereo_valid_pixels']} | {r['merged_cluster']} | {r['default_slots']} | {r['default_half_m_offset']} |")
        lines += ['', 'Signed decomposition is exact: V − M_o = (M_c − M_o) + (V − M_c); absolute terms do not generally add.', 'The JSON retains signed and absolute length decompositions for every video construction.', '',
                  '| Subset | Mean absolute M_c − M_o (cm) |', '|---|---:|']
        lines += [f"| {g} | {fmt(v['origin_offset_mae_cm'])} |" for g, v in report['means'].items()]
        lines += ['', 'Along/cross-ray signed offset error (cm), after the kit slot-0 rigid alignment:', '',
                  '| Episode | Construction | Along vs M_o | Cross vs M_o | Along vs M_c | Cross vs M_c |', '|---|---|---:|---:|---:|---:|']
        for row in report['episodes']:
            for name, r in row['ray']['errors'].items():
                vals = [r[ref][key] for ref in ('M_o', 'M_c') for key in ('along_cm', 'cross_cm')] if r else [None]*4
                lines.append(f"| {row['episode']} | {name} | " + ' | '.join(map(fmt, vals)) + ' |')
        lines += ['', '| Fixed candidate | Clean count | MAE gain vs V_inc (cm) | Episodes lower | Pass |', '|---|---:|---:|---:|---|']
        for name, r in report['authorization']['checks'].items():
            lines.append(f"| {name} | {r['count']} | {fmt(r['gain_cm'])} | {r['lower_episodes']} | {r['passed']} |")
        lines += ['', '## Definitions and limits', '', *report['notes'], '',
                  'S1 maximum absolute position reproduction difference: ' + str(report['s1_max_position_difference_m']) + ' m.', '',
                  '![Per-episode lengths and ray error](lengths.png)', '', 'Reproduce:', '', '```bash', report['reproduce'], '```', '']
    else:
        lines += ['', '| Construction | Available / counted | Counted MAE vs M_o cm | Available-subset MAE vs M_o cm | Available-subset MAE vs M_c cm |',
                  '|---|---:|---:|---:|---:|']
        for name,r in report['means'].items():
            lines.append(f"| {name} | {r['count']}/{r['expected_count']} | {fmt(r['counted_mae_vs_M_o_cm'])} | {fmt(r['available_subset_mae_vs_M_o_cm'])} | {fmt(r['available_subset_mae_vs_M_c_cm'])} |")
        lines += ['', '| Amendment 1 check | Pass |', '|---|---|']
        lines += [f'| {key} | {value} |' for key,value in report['checks'].items()]
        lines += ['', 'The counted-set MAEs and gain are missing when a mandatory measurement is missing; available-subset means do not satisfy the registered test.',
                  '', 'V_inc and incumbent V_hull are undefined here: no S1 scene exists. V_s is the raw MoGe centroid distance times the exact existing episode-scale estimator; invalid scale is never replaced.',
                  'Every measured stereo cluster needs >=300 valid pixels. Identities/unions were frozen from frame-0 RGB overlays before any confirmation mocap was loaded.',
                  'M_c uses exact scan AABB centres transformed by mocap poses; all scalar lengths use the same horizontal definitions as the train diagnostic.',
                  'There is no incumbent body anchor for a kit-aligned ray-error split; no camera-to-mocap yaw calibration is inferred.',
                  '', '![Confirmation lengths and scalar errors](lengths.png)', '', 'Reproduce:', '', '```bash',
                  report['reproduce'], '```', '']
    return '\n'.join(lines)


def figure(rows: list, path, presentation: dict) -> None:
    """Plot available lengths and either aligned ray components or scalar errors."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ray = presentation['ray_errors']
    quantities, video = presentation['quantities'], presentation['video']
    fig, axes = plt.subplots(2, 1, figsize=(14, 9) if ray else (11, 7), constrained_layout=True)
    x = np.arange(len(rows))
    width = .11 if ray else .15
    for index, name in enumerate(quantities):
        values = [np.nan if row['lengths_cm'][name] is None else row['lengths_cm'][name] for row in rows]
        axes[0].bar(x + (index-(len(quantities)-1)/2)*width, values, width, label=name)
    for index, name in enumerate(video):
        for component, hatch in ((('along_cm', ''), ('cross_cm', '//')) if ray else (('scalar', ''),)):
            if ray:
                values = [abs(row['ray']['errors'][name]['M_o'][component]) if row['ray']['errors'][name] else np.nan for row in rows]
                position = x + (index-2)*.17 + (-.04 if not hatch else .04)
            else:
                values = [abs(row['lengths_cm'][name]-row['lengths_cm']['M_o']) if row['lengths_cm'][name] is not None else np.nan for row in rows]
                position = x + (index-1)*.2
            axes[1].bar(position, values, .08 if ray else .2, label=f'{name} {component[:-3]}' if ray else name,
                        color=f'C{index+2}', **dict(hatch=hatch) if ray else {})
    for ax, title in zip(axes, presentation['titles']):
        ax.set(title=title, xlabel=presentation['xlabel'], ylabel='cm', xticks=x,
               xticklabels=[str(row['episode'])+(' (excluded)' if row.get('counted') is False else '') for row in rows])
        if ray:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
        ax.legend(ncol=5, **dict(fontsize=8) if ray else {})
        ax.grid(axis='y', alpha=.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
