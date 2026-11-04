"""G2 admission, gate and resource reporting with fixed evaluation examples."""

import json
from pathlib import Path

from src.paths import PROJECT as ROOT


def publish(directory,admissions,development,packed):
    """Publish all choices, both gate outcomes and three predeclared comparison rows."""
    lines=['# G2: pot and pitcher pretrained shapes','',
        '**Disclosure:** G2 selected pot and pitcher after their G1 results on these same public scans. '
        'This is development evidence on reused scans, not untouched validation. G1 remains FAIL.','',
        f"**Development: {development['verdict']}. Packed veto: {packed['verdict'] if packed else 'NOT RUN (development failed)'}.**",'',
        'Development uses public pot 0/26/42 and pitcher 21. Missing-mask views retain v4; admitted views '
        'are averaged within scoring seed 1/3/5 before object averaging. The other eight objects retain matched v4 '
        'development results. Pair weights are the fixed 26-pair roster counts. Evaluation meshes are generated separately.',
        '', '| Development gate | Measured | Threshold | Verdict |','|---|---:|---|---|',
        f"| Equal-object mean (10 scanned objects), cm | {development['per_object_mean_cm']:.8f} | < 0.62270304 | {'PASS' if development['gates']['equal_object_mean'] else 'FAIL'} |",
        f"| Pair-weighted mean, cm | {development['pair_weighted_cm']:.8f} | <= 0.66288090 | {'PASS' if development['gates']['pair_weighted_mean'] else 'FAIL'} |",
        f"| Largest object regression, cm | {max(r['change_cm'] for r in development['objects']):+.8f} | <= 0.15 | {'PASS' if development['gates']['no_object_regresses'] else 'FAIL'} |",'',
        f"Fold A / B equal-object means: {development['folds']['A']:.8f} / {development['folds']['B']:.8f} cm. "
        "No adjustable parameter or scan-driven fitting choice is selected.",'',
        '| Object | Fold | V4 cm | G2 cm | Change cm | Seed SD cm |','|---|---|---:|---:|---:|---:|']
    for r in development['objects']:
        lines.append(f"| {r['object']} | {r['fold']} | {r['v4_cm']:.8f} | {r['g2_cm']:.8f} | {r['change_cm']:+.8f} | {r['seed_std_cm']:.8f} |")
    lines += ['', '**Video-only admission:** after one Sim(3) fit to frame-0 observed depth points, generated '
        'median point-to-surface residual must be <= 1.5 x v4 fitted the same way. All decisions were recorded '
        'in `ledger.jsonl` and sealed in `admissions.json` before any new scan score.', '',
        '| Pair | Generated median | V4 median | Ratio | Decision / reason |',
        '|---|---:|---:|---:|---|']
    for r in admissions['pairs']:
        a,b=r['generated_residual_m'],r['v4_residual_m']
        values=f'{a:.8f} | {b:.8f} | {a/b:.5f}' if a is not None and b else 'n/a | n/a | n/a'
        lines.append(f"| {r['key']} | {values} | {r['decision']}: {r['reason']} |")
    lines += ['', 'Residual units are the episode’s unanchored MoGe depth units (nominal metres); compare only '
        'within pair. Fit settings are frozen in `g2_queue.json`: deterministic PCA initialization, '
        'one 40-iteration proper Sim(3) ICP, 4096 observed fit points / 8192 sampled surface points. '
        'Final medians use exact triangle distances on every valid observed mask point. No scan enters admission.','']
    if packed:
        m=packed['means']; g=packed['gates']
        lines += ['| Packed veto | Measured | Threshold | Verdict |','|---|---:|---|---|',
            f"| Equal-object mean, cm | {m['per_object_cm']:.8f} | < 0.62338396 | {'PASS' if g['equal_object_mean'] else 'FAIL'} |",
            f"| Pair-weighted mean, cm | {m['pair_weighted_cm']:.8f} | <= 0.66292148 | {'PASS' if g['pair_weighted_mean'] else 'FAIL'} |",
            f"| Largest object regression, cm | {max(r['candidate']-r['v4'] for r in packed['objects']):+.8f} | <= 0.15 | {'PASS' if g['no_object_regresses'] else 'FAIL'} |",'',
            f"Unchanged packed rows: {packed['unchanged_packed_rows']}; geometry identical. See `../g2-packed-check/production.json` for every PLY hash.",
            'Kit preflight was run for both the v4 manifest’s `track_3` and reconstruction `track_3_mesh`, with Kaggle skipped. '
            'The kit preflight checks environment/sample availability; the packer and additional row/geometry comparisons check the placeholder file.','']
    lines += ['Lead repack command (requires both gates PASS; this job does not run it):','',
        '```bash','bash scripts/reproduce/d1_meshes_v5.sh "https://github.com/<owner>/video2data/commit/<real-full-sha>" --run',
        '```','', '| Pair | New generation? | Load s | Generate s | Peak allocated / reserved MiB | Device sampled peak MiB |',
        '|---|---|---:|---:|---:|---:|']
    for r in admissions['pairs']:
        if 'generation' in r:
            g=r['generation']; new=not bool(r.get('reuse_pilot'))
            lines.append(f"| {r['key']} | {'yes' if new else 'pilot reuse'} | {g['load_seconds']:.2f} | {g['generation_seconds']:.2f} | {g['peak_allocated_mib']:.2f} / {g['peak_reserved_mib']:.0f} | {g['gpu_lease']['peak_sampled_total_device_used_mib']} |")
    lines += ['', '![Fixed evaluation examples: crop, generated, v4, scan](comparison.png)', '',
        'Display scales and orientations are independent. Generated Y-up is displayed Z-up. '
        'The figure does not alter candidate geometry or the admission decision.', '']
    (directory/'summary.md').write_text('\n'.join(lines))
    figure(directory,admissions)


def figure(directory,admissions):
    """Render the fixed evaluation 5, 8 and 20 examples after admission and scoring."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    jobs=[r for r in admissions['pairs'] if r['source_split']=='evaluation' and r['episode'] in (5,8,20) and 'generated_mesh' in r]
    fig=plt.figure(figsize=(16,4*len(jobs)),constrained_layout=True)
    for i,r in enumerate(jobs):
        ax=fig.add_subplot(len(jobs),4,4*i+1); ax.imshow(Image.open(directory/r['key']/'crop.png')); ax.axis('off')
        ax.set_title(f"{r['key']}\nSelected: {r['decision']}",fontsize=9)
        with np.load(directory/r['key']/'budgeted.npz') as a,np.load(ROOT/f"data/outputs/mesh_generative/g1-pilot/{r['object']}/scores.npz") as scan:
            shapes=[(a['vertices'],a['faces'],'Generated'),(a['v4_vertices'],a['v4_faces'],'V4'),(scan['scan_vertices'],scan['scan_faces'],'Scan')]
            for j,(v,f,title) in enumerate(shapes,2):
                if j==2: v=v[:,[0,2,1]]*[1,-1,1]
                triangles=v[f.astype(int)]; triangles=triangles[np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>0]
                points=triangles.reshape(-1,3); center=(points.min(0)+points.max(0))/2; radius=np.ptp(points,axis=0).max()*.55
                ax=fig.add_subplot(len(jobs),4,4*i+j,projection='3d')
                ax.add_collection3d(Poly3DCollection(triangles,facecolor='#81b8ce',edgecolor='#41697d',linewidth=.05))
                ax.set(xlim=center[0]+np.array([-radius,radius]),ylim=center[1]+np.array([-radius,radius]),zlim=center[2]+np.array([-radius,radius]))
                ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho'); ax.view_init(25,-60); ax.set_axis_off(); ax.set_title(title)
    fig.suptitle('G2 fixed evaluation examples · independent display scales · scans for measurement only')
    fig.savefig(directory/'comparison.png',dpi=150); plt.close(fig)
