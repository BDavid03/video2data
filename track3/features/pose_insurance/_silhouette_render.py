"""S1 Markdown and candidate-outline presentation from saved plain values."""

FIGURE_CASES = ((18, 'mini_sweeper'), (19, 'mini_sweeper'), (23, 'wooden_piece_1'))

def prediction_markdown(rows: list) -> str:
    """Format all candidate scores, margins and abstention reasons without labels."""
    lines = ['# S1 predictions', '', 'IoU variants: original / eroded / dilated. Chamfer units: pixels.', '',
             '| Episode/object | Search | Candidate/index | Loss ratio | IoU O/E/D | Chamfer O/E/D (px) |',
             '|---|---:|---|---:|---|---|']
    for row in rows:
        for search, candidates in enumerate(row['searches']):
            inverse = {local: primary for primary, local in enumerate(row['mappings'][search])}
            for local, c in enumerate(candidates):
                iou = '/'.join('NA' if x is None else f'{x:.6f}' for x in c['iou'])
                chamfer = '/'.join('NA' if x is None else f'{x:.4f}' for x in c['contour_chamfer_px'])
                label = f"c{inverse[local]}" if local in inverse else 'unmatched'
                lines.append(f"| {row['episode']} {row['name']} | {search} | {label}/{c['index']} | "
                             f"{c['loss_ratio']:.6f} | {iou} | {chamfer} |")
    lines += ['', '| Episode/object | Role | Selected identity | Minimum margin | Abstain reasons |',
              '|---|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['episode']} {row['name']} | {row['role']} | {row['decision']['selected']} | "
                     f"{row['decision']['minimum_margin']} | {'; '.join(row['abstain_reasons'])} |")
    return '\n'.join(lines) + '\n'


def assessment_markdown(result: dict) -> str:
    """Present the known-answer cases, control completeness and withheld application."""
    lines = ['# S1 assessment', '', f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**. "
             f"Episode 19: **{result['episode_19']}**.", '',
             f"Predictions completed: {result['predictions_finished_utc']}; "
             f"assessment started: {result['assessment_started_utc']}.", '',
             '| Episode/object | Testable | Correct identities | Selected | Verdict/reason |',
             '|---|---|---|---|---|']
    for case in result['known_cases']:
        lines.append(f"| {case['episode']} {case['name']} | {case['testable']} | {case['correct_candidates']} | "
                     f"{case['selected']} | {case['verdict']}: {case['reason']} |")
    lines += ['', '| Control | Complete | Per-search selected identities | Veto | Reasons |',
              '|---|---|---|---|---|']
    for control in result['controls']:
        lines.append(f"| {control['episode']} {control['name']} | {control['complete']} | "
                     f"{control['search_selections']} | {control['veto']} | {'; '.join(control['reasons'])} |")
    lines += ['', f"Known-answer pass: {result['known_passed']}; controls complete: {result['controls_complete']}; "
              f"control veto: {result['control_veto']}.", '',
              'No wooden decision establishes wooden-flip resolution. Controls test consistency with depth, '
              'not truth. H3 and submissions are unchanged.', '']
    return '\n'.join(lines)


def figure(rows: list, renders: dict):
    """Overlay search-zero candidate outlines for the three requested example objects."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, key in zip(axes, FIGURE_CASES):
        row = next(r for r in rows if (r['episode'], r['name']) == key)
        arrays = renders[key]
        ax.imshow(arrays['mask'], cmap='gray', vmin=0, vmax=1)
        for local, candidate in enumerate(row['searches'][0]):
            contours, _ = cv2.findContours(arrays[f'search_0_candidate_{local}'].astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            color = plt.get_cmap('tab10')(local % 10)
            for contour in contours:
                xy = contour[:, 0]
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=.8)
            ax.plot([], [], color=color, label=f"c{local}, loss +{100*(candidate['loss_ratio']-1):.2f}%, "
                    f"IoU {candidate['iou'][0]:.4f}" if candidate['iou'][0] is not None else f'c{local}: invalid')
        ax.set_title(f'Episode {key[0]}: {key[1]}')
        ax.set_xlabel('camera A u (px)')
        ax.set_ylabel('camera A v (px)')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=8)
    fig.suptitle('S1 saved registration basins — original mask IoU (unitless)')
    fig.text(.5, .015, 'masks are depth-derived partial regions', ha='center')
    fig.tight_layout(rect=(0, .14, 1, .95))
    return fig

