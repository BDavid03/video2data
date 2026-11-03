"""Render the historical R10 report and its fitted-profile figure."""


def markdown(results, killed, recipe_sha256):
    document = {'decision': 'R10 dropped' if killed else 'pooled gates completed', 'results': results,
                'provenance': 'video_only', 'scans_opened': False, 'cdo_called': False, 'cdo_ledgers_read': False,
                'excluded_episodes': [41, 43, 44], 'recipe_sha256': recipe_sha256,
                'limitations': 'MoGe2 depth is unanchored pseudo-geometry. Interior floor points remain in the residual/coverage denominator. Three nearly adjacent frames are weak independent evidence. No claims about CD-O.'}
    lines = ['# R10 pooled vessel profiles — Phase A', '', '**' + document['decision'] + '.**', '',
             'CPU only. No scans, CD-O scorer, CD-O ledger, GPU, or excluded episodes used.', '',
             '## Predeclared choices on tuning video', '',
             'The episode 42 opening overlay shows an open interior and a one-sided handle. Before fitting,',
             '`choices.json` fixed soft-L1 (0.05 reference radius), linear knots at 0/0.5/1, the 90° handle',
             'sector exclusion, original v4 frame admission plus successful local plane refit and >=300 retained points,',
             'alternating frame subsets, and all definitions and constants in `recipe.json`. No alternatives were searched.',
             'Frame height is its 98th percentile; centre and radial scale are fitted separately per frame.',
             'Shared knot radii have arithmetic mean one, fixing the profile/scale ambiguity.', '',
             'On omitted frames, even retained raster pixels calibrate nuisance centre/scale; odd pixels are scored.',
             'Residual = equal-frame mean absolute radial distance divided by the fixed reference radius.',
             'Coverage = equal-frame fraction within 0.10 reference radius, including interior/floor observations.',
             'The constant-radius model uses the same observations, nuisance fits, robust loss and evaluation.', '',
             '| Trial | Frames | a: maximum knot difference / R (<0.05) | b: residual reduction (>=10%) | c: profile / constant coverage | Height | Decision |',
             '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for result in results:
        gate = result.get('gate')
        if gate and 'pooled_fit' in gate:
            lines.append(f"| {result['trial']['name']} | {len(result['admitted_frames'])} | {max(gate['knot_differences_radius']):.6f} | {gate['residual_reduction']:.3%} | {gate['coverage']['profile']:.3%} / {gate['coverage']['constant']:.3%} | {gate['checks']['height']} | {result['decision']} |")
        else:
            lines.append(f"| {result['trial']['name']} | {len(result['admitted_frames'])} | — | — | mask/table only | — | {result['decision']} |")
    for result in results:
        lines.extend(['', '## ' + result['trial']['name'], '', 'Admitted: ' + ', '.join(result['admitted_frames']) + '.', ''])
        gate = result.get('gate')
        if gate and 'pooled_fit' in gate:
            for i, (fit, support) in enumerate(zip(gate['subset_fits'], gate['height_support'])):
                lines.append(f"Subset {i}: {gate['subset_ids'][i]}; radii {fit['radii']}; height span {support['p02_p98_span']:.6f}; occupied fraction {support['occupied_fraction']:.1%}; knot-window counts {support['knot_window_counts']}; height pass {support['pass']}.")
                lines.append('')
            lines.extend([f"Held-out radial residual / R: profile {gate['profile_residual_radius']:.8f}, constant {gate['constant_residual_radius']:.8f}.", '',
                          f"Gate flags: `{gate['checks']}`. Pooled radii: {gate['pooled_fit']['radii']}.", '',
                          f"![Normalized video points and fitted profile]({result['profile_figure']})", ''])
        lines.extend(['| Frame | Admitted | Mask pixels | Retained | Local plane | Geometry equivalence error (m) |',
                      '| --- | --- | ---: | ---: | --- | ---: |'])
        for row in result['frames']:
            lines.append(f"| {row.get('id', row['frame'])} | {row['admitted']} | {row.get('mask_pixels', '—')} | {row.get('retained_points', '—')} | {row.get('local_plane', '—')} | {row.get('geometry_max_abs_error_m', '—')} |")
    if killed:
        lines.extend(['', 'R10 is dropped. All four vessels retain v4. No evaluation-episode profile data was opened,',
                      'no candidate PLYs were constructed, and no three-view candidate renders exist. Other vessel figures',
                      'are deliberately absent because the kill test stopped the job. The saved recipe is a kill-test',
                      'record, not an adopted/frozen production recipe.'])
    lines.extend(['', '## Limits and reproduction', '', document['limitations'], '',
                  '`PYTHONPATH=. CUDA_VISIBLE_DEVICES=\'\' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .runtime/linux-cuda/bin/python -B .runtime/tmp/round4-profiles/run.py`', '',
                  'The editable queue is `.runtime/tmp/round4-profiles/queue.json`; completed trials resume by recipe/source identity.',
                  'Each completed trial appends `ledger.jsonl` and stores all input hashes and detailed numbers in `trials/`.',
                  'The driver reconstructs original table masks, object masks and local extents, requiring agreement to 1e-10 m.', ''])
    return document, '\n'.join(lines)


def profile_figure(name, frames, gate, recipe, directory):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    fit = gate['pooled_fit']
    for frame, nuisance in zip(frames, fit['nuisance']):
        points = frame['points']
        centre, scale = np.array(nuisance[:2]), nuisance[2]
        radial = np.linalg.norm(points[:, :2] - centre, axis=1) / scale
        sample = np.linspace(0, len(points) - 1, min(2000, len(points)), dtype=int)
        ax.scatter(points[sample, 2], radial[sample], s=2, alpha=.14, label=frame['id'])
    ax.plot([0, .5, 1], fit['radii'], 'k-o', lw=2, label='Pooled three-knot profile')
    ax.axhline(1, color='crimson', ls='--', label='Constant radius (separately fitted scales)')
    ax.set(xlabel='Height / frame height (98th percentile)', ylabel='Radial distance / fitted frame scale',
           title=f'{name}: video-only fit; gate {gate["decision"]}', ylim=(0, 2.4))
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = directory / f'{name}-profile.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target.name
