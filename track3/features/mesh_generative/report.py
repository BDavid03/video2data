"""Publish the fixed pilot gate and a labelled comparison figure."""

from pathlib import Path

GATE = ('**Pilot continuation gate:** mean CD-O gain over v4 on the three objects > 0.10 cm, '
        'their pair-weighted contribution\nimproves, and no object regresses > 0.15 cm. '
        'Passing authorises only the full panel below; failing ends G1 and v4 stays.')


def publish(directory: Path, summary: dict, objects: list[dict]):
    """Write the decision and four-column geometry comparison without altering any mesh."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lines = ['# Frozen G1 pilot', '', GATE, '', f"**Verdict: {summary['verdict']}**. Stopped at the pilot gate; no full-panel generation or packing.",
             '', 'Reused public scans: the three objects were selected using known v4 errors. This is not untouched validation.',
             '', '| Object | Scoring pairs | v4 cm | G1 cm | Gain cm | G1 seed SD / range cm | Generation s (load excluded) | Peak allocated / reserved MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    by_name = {o['object']: o for o in objects}
    for row in sorted(summary['objects'], key=lambda r:r['gain_cm'], reverse=True):
        gen = by_name[row['object']]['generation']
        lines.append(f"| {row['object']} | {row['pairs']} | {row['v4_cm']:.8f} | {row['generated_cm']:.8f} | "
                     f"{row['gain_cm']:+.8f} | {row['seed_std_cm']:.8f} / {row['seed_min_cm']:.8f}–{row['seed_max_cm']:.8f} | "
                     f"{gen['generation_seconds']:.2f} | {gen['peak_allocated_mib']:.0f} / {gen['peak_reserved_mib']:.0f} |")
    lines += ['', f"Equal-object mean gain: **{summary['mean_gain_cm']:+.8f} cm** (>0.10 required).",
              f"Contribution to the 26-pair mean: **{summary['v4_contribution_cm']:.8f} → {summary['generated_contribution_cm']:.8f} cm** "
              f"(gain {summary['contribution_gain_cm']:+.8f} cm).",
              '', 'Seed SD is descriptive spread over the incumbent’s existing object-specific seeds, not generation seeds or a confidence interval. Pitcher has one scoring seed; its SD is zero by construction.',
              '', 'One frame-0 crop and one mesh per object; seed 12345, 30 steps, octree 380, ordinary marching cubes. '
              'White RGB inputs use the pinned upstream image processor unchanged. Crop margin is 10% per side. '
              'No scan dimensions, geometry or score entered generation.',
              '', 'Conversion: unmodified kit `budget_mesh(path, 4096, 4096)`; matched `mesh_eval.pairs.score_pair` and original v4 scan/seed pairs. '
              'Budgeted generated, v4 and scan arrays are in each `scores.npz`; raw generation is `raw.ply`. '
              'The 5+1+6 scoring rows reuse the three generated shapes; they do not generate the full panel.',
              '', '![Crop, generated mesh, v4 mesh, scan](comparison.png)',
              '', 'Figure uses budgeted geometry with independent centered axis scales; generated Y-up is displayed Z-up. '
              'It illustrates shape, not shared metric size or registered pose.', '']
    (directory/'summary.md').write_text('\n'.join(lines))
    fig = plt.figure(figsize=(16,12), constrained_layout=True)
    for i, job in enumerate(objects):
        ax = fig.add_subplot(3,4,4*i+1)
        ax.imshow(Image.open(directory/job['object']/'crop.png'))
        ax.axis('off'); ax.set_title(job['object']+'\nFrame-0 masked crop',fontsize=10)
        with np.load(directory/job['object']/'scores.npz', allow_pickle=False) as arrays:
            for j, (prefix,title) in enumerate((('', 'Generated (budgeted)'),('v4_', 'V4 (packed)'),('scan_', 'Scan (budgeted)')),2):
                vertices = arrays[prefix+'vertices'].copy()
                faces = arrays[prefix+'faces'].astype(int)
                if prefix == '':
                    vertices = vertices[:,[0,2,1]] * [1,-1,1]
                triangles = vertices[faces]
                valid = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>0
                triangles = triangles[valid]
                bounds = np.array([triangles.reshape(-1,3).min(axis=0),triangles.reshape(-1,3).max(axis=0)])
                center, radius = bounds.mean(axis=0), np.ptp(bounds,axis=0).max()*0.55
                ax = fig.add_subplot(3,4,4*i+j,projection='3d')
                ax.add_collection3d(Poly3DCollection(triangles,facecolor='#81b8ce',edgecolor='#41697d',linewidth=0.05))
                ax.set(xlim=center[0]+np.array([-radius,radius]),ylim=center[1]+np.array([-radius,radius]),
                       zlim=center[2]+np.array([-radius,radius]))
                ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho'); ax.view_init(25,-60)
                ax.set_axis_off(); ax.set_title(title)
    fig.suptitle('G1: one frozen generation per object · independent display scales · public scans for scoring only')
    fig.savefig(directory/'comparison.png',dpi=150)
    plt.close(fig)
