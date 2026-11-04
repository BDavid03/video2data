"""Reader-facing G3 decisions, stop reasons, gate tables and neutral-material figures."""

import json

from src.paths import PROJECT as ROOT
from src.provenance import write_json
from .g2 import key


def candidate_rows(queue, prepared, decisions, record):
    """Describe every candidate, including pairs skipped by a preceding gate."""
    by_key = {r['key']:r for r in decisions['pairs']} if decisions else {}
    rows = []
    for pair in queue['pairs']:
        ident = key(pair)
        row = by_key.get(ident, {})
        mask = row.get('mask', {})
        if prepared[ident]['lid']:
            reason = 'lid rule: white_pot_lid in episode object list'
            decision = 'v4'
        else:
            reason = row.get('reason', 'not run: '+record.get('stop_reason', record.get('error', 'preceding stage incomplete')))
            decision = row.get('decision', 'v4 retained; not inferred')
        a, b = row.get('generated_residual_m'), row.get('v4_residual_m')
        rows.append(dict(key=ident, box_score=mask.get('box_score'), mask_pixels=mask.get('mask_pixels'),
                         border_fraction=mask.get('border_fraction'), observed_points=mask.get('observed_points'),
                         mask_checks=mask.get('checks'), residual_ratio=a/b if a is not None and b else None,
                         decision=decision, reason=reason, generated_mesh=row.get('generated_mesh'),
                         mask=mask, generation=row.get('generation')))
    return rows


def resources(controls, candidates):
    """Collect measured subprocess peaks without confusing allocation and device sampling."""
    reports = []
    for row in [*controls, *candidates]:
        mask = row.get('mask', {})
        for name in ('detection', 'segmentation'):
            if mask.get(name):
                reports.append(dict(key=row['key'], stage=name, **mask[name]))
        if row.get('generation'):
            reports.append(dict(key=row['key'], stage='generation', **row['generation']))
    return reports


def publish(directory, queue, prepared, controls, decisions, development, packed, record):
    """Publish all six choices and both score gates even when the control stops execution."""
    rows = candidate_rows(queue, prepared, decisions, record)
    reports = resources(controls, rows)
    summary = dict(status=record['status'], control_verdict=record['control_verdict'],
                   development_verdict=record['development_verdict'], packed_verdict=record['packed_verdict'],
                   stop_reason=record.get('stop_reason', record.get('error')), candidates=rows, controls=controls,
                   development=development, packed=packed, wall_seconds_before_reporting=record['wall_seconds'],
                   peak_allocated_mib=max((r['peak_allocated_mib'] for r in reports), default=0),
                   peak_reserved_mib=max((r['peak_reserved_mib'] for r in reports), default=0),
                   peak_device_sampled_mib=max((r['gpu_lease']['peak_sampled_total_device_used_mib'] for r in reports), default=0))
    write_json(directory/'summary.json', summary)
    lines = ['# G3: automatic frame-zero white-pot masks', '',
        f"**Control {record['control_verdict']}; development {record['development_verdict']}; packed veto {record['packed_verdict']}.** "
        + ('D1 v6 passes.' if record['packed_verdict'] == 'PASS' else 'D1 v5 stands.'), '',
        '**Disclosure:** These are pot shapes evaluated on reused public scans, not untouched validation. '
        'G1 failed because its rack regressed by 0.23158184 cm. G2 selected pots and pitcher after seeing their G1 '
        'results on the same scans. Every pot pair uses the same public pot scan. G3 uses one frozen prompt and '
        'no retries, repairs or selection after scan scoring.', '',
        'Incumbent: real-URL D1 v5 SHA256 `bcb93d57d86b4dac06ed21af9af84ce39edffe52b3d9c35e28928391e47e4e58`. '
        'All settings and pair identities are in `g3_queue.json` and the committed G3 protocol.', '']
    if summary['stop_reason']:
        lines += [f"**Stop:** {summary['stop_reason']}.", '']
    lines += ['| Control (recorded cluster reference) | IoU | Required | Verdict |', '|---|---:|---:|---|']
    for pair in queue['controls']:
        found = next((r for r in controls if r['key'] == key(pair)), None)
        value = f"{found['iou']:.8f}" if found else 'not run'
        verdict = ('PASS' if found['passed'] else 'FAIL') if found else 'NOT RUN'
        lines.append(f"| {key(pair)} | {value} | ≥ 0.70 | {verdict} |")
    lines += ['', '| Candidate | Box score | Mask px | Boundary near invalid/border | Valid depth points | Mask checks | Residual ratio | Decision / reason |',
              '|---|---:|---:|---:|---:|---|---:|---|']
    def number(value, digits=5):
        return 'not run' if value is None else f'{value:.{digits}f}'
    for row in rows:
        checks = ', '.join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in (row['mask_checks'] or {}).items()) or 'not run'
        border = 'not run' if row['border_fraction'] is None else f"{row['border_fraction']:.3%}"
        lines.append(f"| {row['key']} | {number(row['box_score'])} | {row['mask_pixels'] if row['mask_pixels'] is not None else 'not run'} | "
                     f"{border} | {row['observed_points'] if row['observed_points'] is not None else 'not run'} | {checks} | "
                     f"{number(row['residual_ratio'])} | {row['decision']}: {row['reason']} |")
    lines += ['', 'Mask limits: 10,000–45,000 pixels inclusive; at most 10% of inner boundary pixels within '
        'Euclidean distance 2 pixels of the image border or remap-invalid pixels; at least 2,000 valid depth points. '
        'Components and inner boundary use eight-neighbour connectivity. All choices enter the ledger before any scan score. '
        'Residuals use G2’s same Sim(3), budget and every observed masked depth point; maximum ratio is 1.5.', '',
        '| Development gate | Measured | Threshold | Verdict |', '|---|---:|---|---|']
    if development:
        for name, value, limit in (
            ('new_pair', sum(r['decision']=='generated' and r['key'].startswith('public-') for r in rows), '≥ 1 newly admitted public pair'),
            ('equal_object_mean', development['per_object_mean_cm'], '< 0.56244822 cm'),
            ('pair_weighted_mean', development['pair_weighted_cm'], '< 0.61972984 cm'),
            ('no_object_regresses', max((r['change_cm'] for r in development['objects']), default=None), '≤ 0.15 cm versus v4')):
            passed = development['gates'].get(name)
            lines.append(f"| {name} | {number(value,8)} | {limit} | {'PASS' if passed else 'FAIL' if passed is False else 'NOT RUN'} |")
        if development.get('reason'):
            lines += ['', development['reason']]
        if development['objects']:
            lines += ['', '| Object | Fold | V4 cm | G3 cm | Change cm | Seed SD cm |', '|---|---|---:|---:|---:|---:|']
            for row in development['objects']:
                lines.append(f"| {row['object']} | {row['fold']} | {row['v4_cm']:.8f} | {row['g2_cm']:.8f} | {row['change_cm']:+.8f} | {row['seed_std_cm']:.8f} |")
            lines += ['', f"Fold A/B means: {development['folds']['A']:.8f} / {development['folds']['B']:.8f} cm."]
    else:
        lines.append('| All development gates | NOT RUN | Control must pass | NOT RUN |')
    lines += ['', '| Packed veto | Measured | Threshold | Verdict |', '|---|---:|---|---|']
    if packed and packed.get('means'):
        for name, value, limit in (
            ('new_pair', packed['changed_pairs'], '≥ 1 changed evaluation pair'),
            ('equal_object_mean', packed['means']['per_object_cm'], '< 0.56443301 cm'),
            ('pair_weighted_mean', packed['means']['pair_weighted_cm'], '< 0.62080727 cm'),
            ('no_object_regresses', max(r['candidate']-r['v5'] for r in packed['objects']), '≤ 0.15 cm versus v5')):
            lines.append(f"| {name} | {number(value,8)} | {limit} | {'PASS' if packed['gates'][name] else 'FAIL'} |")
        lines += ['', '| Object | V5 cm | G3 cm |', '|---|---:|---:|']
        for row in packed['objects']:
            lines.append(f"| {row['object']} | {row['v5']:.8f} | {row['candidate']:.8f} |")
        lines += ['', f"Unchanged packed rows: {packed['unchanged_packed_rows']}; coordinates byte-identical to v5. "
                  'Production PLY hashes are in `../g3-packed-check/production.json`. '
                  + ('Both kit preflights ran with Kaggle skipped.' if packed.get('preflight') else 'Preflights not run: numeric packed veto failed.')]
    else:
        lines.append(f"| All packed gates | {'FAIL: '+packed['reason'] if packed else 'NOT RUN'} | Development must pass | {packed['verdict'] if packed else 'NOT RUN'} |")
    lines += ['', f"Wall time before reporting: {record['wall_seconds']:.2f} s; final wall time is in `run.json`. "
        f"Maximum torch allocation/reservation: {summary['peak_allocated_mib']:.2f}/{summary['peak_reserved_mib']:.2f} MiB. "
        f"Device-wide sampled peak: {summary['peak_device_sampled_mib']} MiB (includes other applications; sampling can miss peaks).", '',
        '| Pair | Stage | Wall s | Allocation / reservation MiB |', '|---|---|---:|---:|']
    for report in reports:
        wall = report.get('wall_seconds', report.get('total_seconds', 0))
        lines.append(f"| {report['key']} | {report['stage']} | {wall:.2f} | {report['peak_allocated_mib']:.2f} / {report['peak_reserved_mib']:.2f} |")
    lines += ['', '![Every G3 candidate: masked crop, generated mesh and v4](comparison.png)', '',
              'A missing mask or mesh is labelled with its stop reason. Neutral material, independent display scales '
              'and orientations; generated Y-up is displayed Z-up. No scan geometry is used in this figure.', '']
    (directory/'summary.md').write_text('\n'.join(lines))
    figure(directory, rows, decisions)


def figure(directory, rows, decisions):
    """Render all six candidates, retaining explicit blank panels for stopped stages."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from PIL import Image
    import trimesh

    baseline = json.loads((ROOT/'data/outputs/mesh_generative/g2/admissions.json').read_text())
    v4_path = ROOT/next(r['v4_mesh'] for r in baseline['pairs'] if r['object'] == 'white_pot')
    v4 = trimesh.load(v4_path, force='mesh', process=False)
    fig = plt.figure(figsize=(12, 3.4*len(rows)), constrained_layout=True)
    for i, row in enumerate(rows):
        item = directory/row['key']
        ax = fig.add_subplot(len(rows), 3, i*3+1)
        ax.imshow(Image.open(item/'rgb.png'))
        if (item/'mask.npz').exists():
            with np.load(item/'mask.npz') as a:
                rgba = np.zeros((*a['mask'].shape,4))
                rgba[...,0] = 1
                rgba[...,3] = a['mask']*.38
                ax.imshow(rgba)
            bounds = row['mask'].get('crop_bounds')
            if bounds:
                ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[3],bounds[1])
        ax.set_title(f"{row['key']}\n{row['decision']}", fontsize=9)
        ax.set_axis_off()
        for j in (2,3):
            ax = fig.add_subplot(len(rows),3,i*3+j,projection='3d')
            if j == 2 and not row['generated_mesh']:
                ax.text2D(.05,.5,'Not generated\n'+row['reason'],transform=ax.transAxes,fontsize=8,wrap=True)
                ax.set_axis_off()
                continue
            shape = trimesh.load(ROOT/row['generated_mesh'],force='mesh',process=False) if j == 2 else v4
            if j == 2 and (item/'budgeted.npz').exists():
                with np.load(item/'budgeted.npz') as a:
                    vertices, faces = a['vertices'], a['faces']
            else:
                vertices, faces = np.asarray(shape.vertices), np.asarray(shape.faces)
            if j == 2:
                vertices = vertices[:,[0,2,1]]*[1,-1,1]
            triangles = vertices[faces.astype(int)]
            triangles = triangles[np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>0]
            points = triangles.reshape(-1,3)
            center = (points.min(0)+points.max(0))/2
            radius = np.ptp(points,axis=0).max()*.55
            ax.add_collection3d(Poly3DCollection(triangles,facecolor='#acb2b6',edgecolor='#626970',linewidth=.05))
            span = np.array([-radius,radius])
            ax.set(xlim=center[0]+span,ylim=center[1]+span,zlim=center[2]+span)
            ax.set_box_aspect((1,1,1)); ax.set_proj_type('ortho'); ax.view_init(25,-60)
            ax.set_axis_off(); ax.set_title('Generated' if j==2 else 'V4 incumbent')
    fig.suptitle('G3: frame-zero pot candidates · mask in red · neutral material · independent display scales')
    fig.savefig(directory/'comparison.png',dpi=140)
    plt.close(fig)
