"""Run the initialization defect fix and public-only hybrid tracker queue on CPU."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from src.paths import project_path, output_path
from src.provenance import sha256_file, write_json
from src.runs import now_utc
from . import depth_track as h, _depth_track as fit, _depth_hybrid as hybrid
from . import camera, full_c, track, candidate_c, camera_ba
from ._object import render, iou

ROOT = h.ROOT
OUTPUT = h.OUTPUT/'hybrid-v5'
PROTOCOL = h.PROTOCOL
CONFIG = dict(revision=5, depth_trim_fraction=.8, robust_scale=.1,
              distance_transform=True, continuity_weight=.5, rotation_grid=False)


def read(path):
    return json.loads(path.read_text())


def sources():
    paths = [Path(__file__), Path(hybrid.__file__), Path(h.__file__), Path(fit.__file__),
             Path(full_c.__file__), Path(track.__file__), Path(candidate_c.__file__), Path(camera_ba.__file__),
             Path(__file__).with_name('_object.py'), Path(__file__).with_name('model_track.py'),
             Path(__file__).with_name('camera.py'), Path(__file__).with_name('gates.py')]
    directory = ROOT/'src/features/moving_policy'
    paths += [directory/f'{n}.py' for n in ('two_object_facing', '_two_object_facing',
              '_two_object_symmetry', 'two_object_rotation', '_two_object_rotation', 'two_object_table', '_two_object_table')]
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in paths}


def admit(episode):
    h.authorize(episode)
    output_path(ROOT, OUTPUT)
    if episode not in h.ALLOWED:
        raise PermissionError('Only the fixed public train/tuning population')


def record_protocol(directory, title, detail):
    directory.mkdir(parents=True, exist_ok=True)
    copy = directory/'protocol.md'
    if not copy.exists():
        with PROTOCOL.open('a') as stream:
            stream.write(f'\n## {title} — {now_utc()}\n\n{detail}\n')
        copy.write_bytes(PROTOCOL.read_bytes())
        write_json(directory/'protocol-seal.json',dict(utc=now_utc(), sha256=sha256_file(copy), before_tracking=True))


def identity(directory, item, extra=()):
    paths = [directory/'protocol.md', project_path(ROOT,item['mesh']['path']),
             project_path(ROOT,item['mesh']['trace']), project_path(ROOT,item['masks'])/'request.json']
    cache = project_path(ROOT,item['cache'])
    paths += [cache/'camera-ba.npz', cache/'camera-diagnostics.json', cache/'table.json']
    paths += [cache/'frames'/f'frame_{i:06d}.npz' for i in range(item['frames'])]
    paths += [project_path(ROOT,item['masks'])/'frames'/f'frame_{i:06d}.npz' for i in range(item['frames'])]
    paths += list(extra)
    return dict(item=item, configuration=CONFIG, sources=sources(),
        inputs={str(p.relative_to(ROOT)):sha256_file(p) for p in paths}, labels_read=False, gpu=False)


def seal(directory):
    files = {str(p.relative_to(ROOT)):sha256_file(p) for p in directory.iterdir()
             if p.is_file() and p.name not in ('run.json','seal.json') and not p.name.startswith('evaluation')}
    write_json(directory/'seal.json',dict(utc=now_utc(), files=files, before_labels=True))


def check(directory):
    for name, digest in read(directory/'seal.json')['files'].items():
        if sha256_file(project_path(ROOT,name)) != digest:
            raise ValueError(f'Sealed file changed: {name}')


def load(item):
    admit(item['episode'])
    cache = project_path(ROOT,item['cache'])
    with np.load(cache/'camera-ba.npz') as saved:
        cameras = saved['world_T_camera'].copy()
    gauge = np.asarray(read(cache/'table.json')['world_new_T_world_initial'])
    cameras = gauge[None]@cameras
    camera_rows = read(cache/'camera-diagnostics.json')
    _,k,_,_ = camera.rectification(read(h.snapshot()/'track_3/public/meta/camera_calibration.json'))
    mesh = trimesh.load(project_path(ROOT,item['mesh']['path']),force='mesh',process=False)
    model = fit.geometry(mesh.vertices,mesh.faces,item['object'])
    model['configuration'] = CONFIG
    times = np.asarray(read(project_path(ROOT,item['masks'])/'request.json')['timestamps'])
    return cache,cameras,camera_rows,k,mesh,model,times


def evaluate(directory, item):
    """Read public mocap only after this track's output seal, using M1 correspondence."""
    import pyarrow.parquet as pq
    admit(item['episode']); check(directory)
    result_path = directory/'evaluation-m1.json'
    if result_path.exists():
        return read(result_path)
    reference = directory/'reference.npz'
    write_json(directory/'evaluation-request.json',dict(utc=now_utc(), output_seal_sha256=sha256_file(directory/'seal.json'),
        reference_sha256=sha256_file(reference), evaluator='M1 exact frame correspondence and one first-pose alignment',
        source_sha256=sources()))
    label = h.snapshot()/f"track_3/public/data/chunk-000/episode_{item['episode']:06d}.parquet"
    rows = pq.read_table(label,columns=['observation.objects']).to_pylist()
    objects = [next(o for o in r['observation.objects'] if o['name']==item['object']) for r in rows]
    mocap = np.asarray([o['pose'] for o in objects],float)
    available = np.asarray([o['visible'] for o in objects]) & np.isfinite(mocap).all(1)
    available &= abs(np.linalg.norm(mocap[:,3:],axis=1)-1)<=.001
    with np.load(reference) as saved:
        p=saved['position'];q=saved['quaternion_xyzw']
    if len(p)!=len(mocap) or not available[0]:
        raise ValueError('M1 requires equal frame counts and a valid initial label')
    model = read(directory/'mesh-identity.json')
    errors = fit.m1_errors(p,q,mocap[:,:3],mocap[:,[4,5,6,3]],np.asarray(model['axis_model']),model['symmetry'])
    pos,angle = [e[available] for e in errors]
    result=dict(episode=item['episode'],object=item['object'],frames=len(p),label_coverage=float(available.mean()),
        position_lt_3cm_fraction=float(np.mean(pos<.03)),angle_lt_30deg_fraction=float(np.mean(angle<30)),
        joint_lt_3cm_30deg_fraction=float(np.mean((pos<.03)&(angle<30))),
        median_position_cm=float(np.median(pos)*100),median_angle_deg=float(np.median(angle)),
        mean_position_cm=float(np.mean(pos)*100),mean_angle_deg=float(np.mean(angle)),
        reference_sha256=sha256_file(reference),label_sha256=sha256_file(label),development_only=True)
    gates=read(directory/'gates.json')
    quality=all(gates['gates'][n]['passed'] for n in h.QUALITY)
    result['passed']=bool(quality and result['label_coverage']>=.9 and result['position_lt_3cm_fraction']>=.9 and
        result['angle_lt_30deg_fraction']>=.9 and result['joint_lt_3cm_30deg_fraction']>=.9 and
        result['median_position_cm']<=2 and result['median_angle_deg']<=15)
    write_json(result_path,result)
    np.savez_compressed(directory/'evaluation-errors.npz',position_m=errors[0],angle_deg=errors[1],label_valid=available)
    return result


def init_fix():
    """Repeat revision three with only C's precise frame-zero initializer/convention."""
    item=next(i for i in read(h.OUTPUT/'queue.json')['clips'] if i['episode']==39)
    directory=OUTPUT/'rev3-init-fix'
    detail=("Defect fix, not a tracker revision: retain revision 3's mesh, optimizer and full-cloud validity. "
        "Use full_c.initialize on C's original canonical procedural mesh and full frame-zero stereo cloud, "
        "including C's fixed base height. Map that exact canonical pose through the campaign mesh's recorded "
        "coordinate_rotation; no new scan or labels. Do not optimize or select another start at frame zero. "
        "Verify the canonical pose equals C's saved forward frame zero before tracking. Subsequent frames use "
        "revision 3 unchanged. C's reference stays the baseline; all pass bars unchanged.")
    record_protocol(directory,'Defect fix: rev 3 + init fix',detail)
    cdir=ROOT/'data/outputs/moving_reference/ep39/candidate-c'
    extra=[cdir/'asset/mini_dust_pan.obj',cdir/'object-forward.npz']
    request=identity(directory,item,extra)
    if (directory/'seal.json').exists():
        check(directory);return evaluate(directory,item)
    write_json(directory/'request.json',request)
    write_json(directory/'run.json',dict(status='running',utc=now_utc(),gpu=False,labels_read=False))
    started=time.perf_counter();cache,cameras,cr,k,mesh,model,times=load(item)
    cmesh=trimesh.load(extra[0],force='mesh',process=False)
    trace=read(project_path(ROOT,item['mesh']['trace']))['rows']
    binding=next(r for r in trace if r['episode']==item['mesh']['episode'] and r['object']==item['object'])
    initial,canonical=hybrid.c_start(full_c.read_frame(cache,0),cameras[0],cmesh,k,np.asarray(binding['coordinate_rotation']))
    with np.load(extra[1]) as saved:
        difference=float(np.max(abs(canonical-saved['world_T_asset'][0])))
    if difference>1e-10:
        raise ValueError(f'C frame-zero initializer differs from saved C: {difference}')
    write_json(directory/'initialization.json',dict(world_T_campaign_asset=initial.tolist(),world_T_C_canonical=canonical.tolist(),
        maximum_difference_to_C_forward_frame0=difference,asset_convention='C canonical x-long y-short z-up; recorded campaign coordinate_rotation',scan_reads=[]))
    poses=[];valid=[];details=[];previous=initial
    for i in range(len(times)):
        frame=full_c.read_frame(cache,i)
        if i==0:
            co=np.linalg.inv(cameras[i])@initial
            row=dict(fit.fit_metrics(co,model,fit.evidence(frame,k),k),fitted=True,evaluations=0,starts=1)
        else:
            co,row=h.track_frame(frame,k,model,cameras[i],previous,details[-1])
        good=h.frame_valid(row,cr[i]['observed'],co) and row['fitted']
        previous=cameras[i]@co;poses.append(previous);valid.append(good)
        details.append(dict(row,frame_index=i,observed=good,keyframe_refined=False))
        if i%40==0:print(json.dumps(dict(stage='init-fix',frame=i,total=len(times))),flush=True)
    h.save_track(directory,item,cameras,np.asarray(poses),times,np.asarray(valid),details,k,model,cr)
    if identity(directory,item,extra)!=request:raise ValueError('Inputs changed during defect-fix run')
    seal(directory)
    write_json(directory/'run.json',dict(status='completed',gpu=False,labels_read=False,wall_seconds=time.perf_counter()-started))
    return evaluate(directory,item)


def regression39():
    """Resume sealed C unchanged and confirm the hybrid is an exact no-op on its valid frames."""
    item=next(i for i in read(h.OUTPUT/'queue.json')['clips'] if i['episode']==39)
    directory=OUTPUT/'ep39-mini_dust_pan'/'hybrid'
    record_protocol(directory,'Revision 5 hybrid regression on tuning 39',
        "Resume the existing sealed C reference on 39, unchanged; it is 100% valid. Select C's exact pose/validity "
        "where valid, otherwise fit from C's last valid pose, with cups solving only position and axis. "
        "No fit may override a valid C frame; bridge results never alter C's tracker state/keyframes. "
        "No last-valid C pose means fallback-alone from C's frame-zero initializer, disclosed. "
        "Keep the frozen full-cloud residual validity and pass bars. Confirm equality of every array and reference bytes on 39. "
        "Train uses the unchanged campaign two_object_facing.reference_tracks; saved 39 instead uses its own historical C model tracker. "
        "These are two disclosed C implementations, not a claim that 39 tests missing-keyframe recovery.")
    cdir=ROOT/'data/outputs/moving_reference/ep39/candidate-c'
    expected=read(ROOT/'data/outputs/moving_reference/ep39/sealed/sealed.json')['reference_sha256']
    if sha256_file(cdir/'reference.npz')!=expected:raise ValueError('C differs from sealed baseline')
    if (directory/'seal.json').exists():check(directory);return evaluate(directory,item)
    with np.load(cdir/'reference.npz') as saved:
        arrays={key:saved[key].copy() for key in saved.files}
    if not arrays['valid'].all():raise ValueError('39 regression expects the saved all-valid C')
    selected=np.asarray([hybrid.select_c(p,True,np.zeros((4,4)),False)[0] for p in arrays['world_T_object']])
    if not np.array_equal(selected,arrays['world_T_object']):raise ValueError('Hybrid changed valid C')
    shutil.copyfile(cdir/'reference.npz',directory/'reference.npz')
    shutil.copyfile(cdir/'gates-final.json',directory/'gates.json')
    write_json(directory/'mesh-identity.json',dict(symmetry=None,axis_model=[0.,0.,1.],source='unchanged C sealed reference'))
    write_json(directory/'regression.json',dict(frames=len(selected),C_valid_frames=int(arrays['valid'].sum()),bridge_frames=0,
        exact_array_equality=True,exact_reference_bytes=sha256_file(directory/'reference.npz')==expected,reference_sha256=expected,
        baseline_method='Historical C on 39; resumed existing sealed output, not overwritten',sources=sources(),utc=now_utc()))
    seal(directory)
    return evaluate(directory,item)


def selection(episode,name,masks):
    job=read(masks/'request.json')
    return dict(job=job,masks=masks,paths=[h.snapshot()/f'track_3/public/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4' for key in ('c','b')],
        mask_files=[masks/'frames'/f'frame_{i:06d}.npz' for i in range(job['frame_count'])],
        calibration=h.snapshot()/'track_3/public/meta/camera_calibration.json',episode=episode)


def recover_cpu(episode,name,masks):
    """Complete CPU caches for finished masks whose original prep failed at summary metadata."""
    directory=OUTPUT/'recovered-inputs'/f'ep{episode}-{name}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'completed.json').exists():return directory
    selected=selection(episode,name,masks)
    if not (directory/'frame0.npz').exists():candidate_c.prepare_frame0(selected,directory)
    full_c.cache_frames(selected,directory)
    if not (directory/'camera-ba.npz').exists():
        data=camera_ba.observations(selected,directory)
        cameras,ids,landmarks,report=camera_ba.bundle_adjust(data)
        np.savez_compressed(directory/'camera-ba.npz',world_T_camera=cameras,landmark_ids=ids,landmarks=landmarks)
        write_json(directory/'bundle-adjustment.json',report)
        write_json(directory/'camera-diagnostics.json',camera_ba.heldout_rows(data,cameras))
    if not (directory/'table.json').exists():
        with np.load(directory/'camera-ba.npz') as saved:_,table=full_c.table_fit(directory,saved['world_T_camera'])
        write_json(directory/'table.json',table)
    write_json(directory/'completed.json',dict(utc=now_utc(),source_masks=str(masks.relative_to(ROOT)),gpu=False,
        reason='Original GPU mask stage completed; missing CPU stages recovered after wall_seconds summary defect'))
    return directory


def ready_items():
    items=[]
    for entry in read(h.OUTPUT/'inventory.json')['episodes']:
        ep=entry['episode'];name=entry['object']
        if ep==39:continue
        parent=h.OUTPUT/f'inputs/ep{ep}-{name}';masks=parent/'masks'
        if not (masks/'run.json').exists() or read(masks/'run.json')['status']!='completed':continue
        job=read(masks/'request.json')
        if len(list((masks/'frames').glob('frame_*.npz')))!=job['frame_count']:continue
        prep=read(parent/'run.json') if (parent/'run.json').exists() else {}
        if prep.get('status')=='completed':cache=parent/'cache';recovered=False
        elif prep.get('status')=='failed' and 'wall_seconds' in prep.get('error',''):
            cache=None;recovered=True
        else:continue
        item=dict(entry,masks=str(masks.relative_to(ROOT)),cache=None if cache is None else str(cache.relative_to(ROOT)),
                  recovered=recovered,frames=job['frame_count'])
        items.append(item)
    order=[(11,'beige_cup'),(11,'blue_cup'),(12,'blue_cup'),(21,'blue_cup'),
           (0,'white_pot'),(0,'white_pot_lid'),(21,'water_pitcher'),(26,'white_pot'),
           (42,'white_pot'),(18,'mini_sweeper'),(18,'mini_dust_pan'),(31,'mini_sweeper'),
           (12,'plastic_dish_rack'),(26,'plastic_dish_rack'),(31,'plastic_dish_rack'),
           (23,'wooden_piece_1'),(23,'wooden_piece_2')]
    return sorted(items,key=lambda item:order.index((item['episode'],item['object'])))


def disclose_c(directory, method):
    gates=read(directory/'gates.json')
    gates.update(method=method,object_features_used=True,campaign_admitted=False)
    gates['gates']['object_reference_valid_fraction']['note']=(
        'Original campaign C validity' if method=='C-alone' else
        'C validity when C is valid; otherwise frozen full-cloud fit-residual validity')
    write_json(directory/'gates.json',gates)


def c_track(item,directory):
    """Call the campaign's reference tracker unchanged, with immutable prepared inputs."""
    from src.features.moving_policy.two_object_facing import reference_tracks
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'seal.json').exists():check(directory);return
    cache=project_path(ROOT,item['cache']);selected=selection(item['episode'],item['object'],project_path(ROOT,item['masks']))
    if not (directory/'frames').exists():(directory/'frames').symlink_to(cache/'frames',target_is_directory=True)
    record_protocol(directory,f"Revision 5 C baseline public {item['episode']} {item['object']}",
        'Call campaign two_object_facing.reference_tracks unchanged, including its original initializer, features, '
        'keyframes, symmetry/facing decisions and smoothing. No gate edits, no labels before output sealing. '
        'Independent camera/table/support admission remains outside this experiment.')
    request=identity(directory,item);write_json(directory/'request.json',request)
    started=time.perf_counter();_,_,cr,_,mesh,model,times=load(item)
    maps,k,q,_=camera.rectification(read(selected['calibration']))
    with np.load(cache/'camera-ba.npz') as saved:cameras=saved['world_T_camera'].copy()
    write_json(directory/'run.json',dict(status='running',utc=now_utc(),labels_read=False,gpu=False))
    tracked=reference_tracks(selected,directory,mesh,item['object'],maps,k,q,cameras,cr)
    write_json(directory/'initialization.json',tracked['initialization'])
    cameras=tracked['cameras'];poses=tracked['filtered'];valid=tracked['valid'];rows=tracked['object_rows']
    for i,row in enumerate(rows):
        observed=fit.evidence(full_c.read_frame(cache,i),k)
        row.update(iou=float(tracked['scores'][i]) if np.isfinite(tracked['scores'][i]) else
            fit.fit_metrics(np.linalg.inv(cameras[i])@poses[i],model,observed,k)['iou'],
            stereo_supported_pixels=observed['stereo_pixels'])
    h.save_track(directory,item,cameras,poses,times,valid,rows,k,model,cr)
    disclose_c(directory,'C-alone')
    if identity(directory,item)!=request:raise ValueError('C inputs/source changed during run')
    seal(directory);write_json(directory/'run.json',dict(status='completed',wall_seconds=time.perf_counter()-started,gpu=False,labels_read=False))


def bridge_track(item,cdir,directory,c_error=None):
    """Use C exactly where valid and independent depth fits elsewhere without altering C state."""
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'seal.json').exists():check(directory);return
    record_protocol(directory,f"Revision 5 hybrid public {item['episode']} {item['object']}",
        'C pose/validity are preserved exactly on C-valid frames. On other frames fit mask+depth, initialized '
        'from the last C-valid pose (never feed the bridge into C). If C has no valid pose yet, use its own '
        'frame-zero initializer and disclose fallback-alone. Frozen fit residual validity, same mesh and '
        'unchanged quality/accuracy bars. Cups solve only position and directed axis. No new scored or holdout access.')
    cache,cameras,cr,k,mesh,model,times=load(item)
    extra=[] if c_error is not None else [cdir/n for n in
        ('reference.npz','tracking.npz','object-diagnostics.json','initialization.json','seal.json')]
    request=identity(directory,item,extra);write_json(directory/'request.json',dict(request,C_unavailable=c_error))
    cposes=None;cvalid=np.zeros(len(times),bool);crows=[]
    if c_error is None:
        check(cdir)
        with np.load(cdir/'reference.npz') as saved:cposes=saved['world_T_object'];cvalid=saved['valid']
        with np.load(cdir/'tracking.npz') as saved:cameras=saved['world_T_camera']
        crows=read(cdir/'object-diagnostics.json')
        initialization=read(cdir/'initialization.json')
        initial=np.asarray(initialization['initial_world_T_asset'])
        # C's initializer uses the pre-gauge world; obtain its written first-frame pose in C's final gauge.
        initial=cposes[0].copy()
    else:
        frame=full_c.read_frame(cache,0)
        co,initialization=track.initial_pose(frame['xyz'],frame['visible'],frame['object_mask'],frame['hands'],cameras[0],mesh.vertices,mesh.faces,k)
        initial=cameras[0]@co
    write_json(directory/'initialization.json',dict(C_error=c_error,initial_world_T_asset=initial.tolist(),no_labels=True))
    poses=[];valid=[];rows=[];last_c=None
    started=time.perf_counter()
    for i in range(len(times)):
        frame=full_c.read_frame(cache,i);observed=fit.evidence(frame,k)
        if cvalid[i]:
            pose=cposes[i].copy();good=True;last_c=pose.copy()
            detail=fit.fit_metrics(np.linalg.inv(cameras[i])@pose,model,observed,k)
            source='C';attempt=False
        else:
            anchor=last_c if last_c is not None else initial
            co,detail=h.track_frame(frame,k,model,cameras[i],anchor)
            pose=cameras[i]@co
            good=h.frame_valid(detail,cr[i]['observed'],co) and detail['fitted']
            source='depth' if last_c is not None else 'fallback-alone';attempt=True
        poses.append(pose);valid.append(good)
        rows.append(dict(detail,frame_index=i,observed=good,keyframe_refined=False,source=source,
            C_valid=bool(cvalid[i]),C_keyframe_refined=crows[i].get('keyframe_refined') if crows else None,
            fit_attempted=attempt,anchor_available=last_c is not None))
        if i%40==0:print(json.dumps(dict(stage='hybrid',episode=item['episode'],object=item['object'],frame=i,source=source)),flush=True)
    poses=np.asarray(poses);valid=np.asarray(valid,bool)
    h.save_track(directory,item,cameras,poses,times,valid,rows,k,model,cr)
    disclose_c(directory,'hybrid')
    exact=bool(cposes is not None and np.array_equal(poses[cvalid],cposes[cvalid]))
    if cposes is not None and not exact:raise ValueError('Hybrid changed valid C poses')
    write_json(directory/'selection.json',dict(C_valid_frames=int(cvalid.sum()),fallback_frames=int((~cvalid).sum()),
        fallback_valid_frames=int((valid&~cvalid).sum()),C_valid_poses_exact=exact,C_unavailable=c_error,
        last_C_anchor_missing_frames=sum(not r['anchor_available'] and r['fit_attempted'] for r in rows)))
    if identity(directory,item,extra)!=request:raise ValueError('Hybrid inputs/source changed during run')
    seal(directory);write_json(directory/'run.json',dict(status='completed',wall_seconds=time.perf_counter()-started,gpu=False,labels_read=False))


def summary():
    lines=['## Hybrid revision 5 and initialization defect fix','',
        'M1 frame correspondence and initial alignment; all rows are public train/tuning development. '
        'Validity and accuracy bars unchanged. C39 is its historical model-based C; train C is the unchanged '
        'campaign feature/keyframe implementation. These routes differ and are disclosed. No scored episode is tracked.', '',
        '| Clip/object/method | Frames | Valid | IoU median | IoU p10 | Coverage | Max m/s | Max deg/s | <3cm | <30deg | Both | Median cm / deg | Result |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    baseline=read(h.OUTPUT/'evaluation-m1/candidate-c.json')
    sources_rows=[('39 / pan / saved C',baseline,ROOT/'data/outputs/moving_reference/ep39/candidate-c/gates-final.json')]
    for d in sorted(OUTPUT.rglob('evaluation-m1.json')):
        e=read(d);sources_rows.append((str(d.parent.relative_to(OUTPUT)),e,d.parent/'gates.json'))
    for name,e,g in sources_rows:
        corrected=g.with_name('campaign-instruments.json')
        if corrected.exists():g=corrected
        gates=read(g)['gates'];values=[gates[n]['value'] for n in ('object_reference_valid_fraction','reprojection_iou_median',
            'reprojection_iou_p10','reprojection_coverage_fraction','object_speed_max_m_s','object_angular_rate_max_deg_s')]
        values += [e['position_lt_3cm_fraction'],e['angle_lt_30deg_fraction'],e['joint_lt_3cm_30deg_fraction']]
        name='rev 3 + init fix' if name=='rev3-init-fix' else name
        passed=gates.get('assessed_passed',e.get('passed'))
        lines.append(f"| {name} | {e['frames']} | "+' | '.join('—' if v is None else f'{v:.4f}' for v in values)+
            f" | {e['median_position_cm']:.3f} / {e['median_angle_deg']:.3f} | "+('baseline' if 'saved C' in name else 'PASS' if passed else 'FAIL')+' |')
    lines += ['', 'The preparation summary wall_seconds defect was fixed without stopping the live job. '
        'The first failed objects with completed SAM2 stages may receive CPU-only cache recovery in recovered-inputs/. '
        'Original preparation directories and records are not rewritten by that recovery.', '',
        'No independent camera/table/support admission or campaign training is performed. Missing/uncompleted inputs '
        'and C exceptions are in queue-ledger.jsonl and pending.json. Resume with the command below after more preparation:', '',
        '```bash', "CUDA_VISIBLE_DEVICES='' PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \\",
        '  data/.runtime/linux-cuda/bin/python -B -m src.features.moving_reference.depth_hybrid queue --until <future-UTC-ISO8601>', '```']
    notes=OUTPUT/'report-notes.md'
    if notes.exists():lines += ['',notes.read_text()]
    text='\n'.join(lines)+'\n';(OUTPUT/'summary.md').write_text(text)
    original=h.OUTPUT/'summary.md';base=original.read_text().split('\n## Hybrid revision 5 and initialization defect fix')[0]
    original.write_text(base+'\n'+text)


def run_queue(until):
    deadline=datetime.fromisoformat(until).timestamp();done=set()
    ledger=OUTPUT/'queue-ledger.jsonl'
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row=json.loads(line)
            if row['status']=='completed':done.add((row['episode'],row['object']))
    while time.time()<deadline:
        ready=[i for i in ready_items() if (i['episode'],i['object']) not in done]
        if not ready:
            write_json(OUTPUT/'pending.json',dict(utc=now_utc(),until=until,waiting=True,
                completed=[list(k) for k in sorted(done)],prep_log='data/outputs/moving_reference/depth-track/gpu-prep.log'))
            print(json.dumps(dict(stage='waiting-for-prep',utc=now_utc(),until=until,completed=len(done))),flush=True)
            time.sleep(min(20,max(0,deadline-time.time())));continue
        item=ready[0];key=(item['episode'],item['object']);parent=OUTPUT/f"ep{key[0]}-{key[1]}"
        try:
            if item['recovered']:
                cache=recover_cpu(*key,project_path(ROOT,item['masks']));item['cache']=str(cache.relative_to(ROOT))
            cdir=parent/'C-alone';error=None
            try:
                c_track(item,cdir)
                evaluate(cdir,item)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                write_json(parent/'C-unavailable.json',dict(error=error,utc=now_utc(),missing_stage='see exact exception',prep_recovered=item['recovered']))
            bridge_track(item,cdir,parent/'hybrid',error)
            evaluate(parent/'hybrid',item)
            row=dict(episode=key[0],object=key[1],status='completed',C_error=error,recovered=item['recovered'],utc=now_utc())
        except Exception as exc:
            row=dict(episode=key[0],object=key[1],status='failed',error=f'{type(exc).__name__}: {exc}',utc=now_utc())
        with ledger.open('a') as stream:stream.write(json.dumps(row)+'\n')
        done.add(key);summary();print(json.dumps(row),flush=True)
    write_json(OUTPUT/'pending.json',dict(utc=now_utc(),until=until,waiting=False,completed=[list(k) for k in sorted(done)],
        remaining=[dict(episode=e['episode'],object=e['object']) for e in read(h.OUTPUT/'inventory.json')['episodes']
                   if e['episode']!=39 and (e['episode'],e['object']) not in done]))
    summary()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['init-fix','regression39','queue','summary'])
    parser.add_argument('--until')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);cv2.setRNGSeed(31)
    if args.operation=='init-fix':print(json.dumps(init_fix()),flush=True);summary()
    elif args.operation=='regression39':print(json.dumps(regression39()),flush=True);summary()
    elif args.operation=='summary':summary()
    else:
        if not args.until:parser.error('queue requires --until, at most 30 minutes of waiting in this job')
        run_queue(args.until)


if __name__=='__main__':main()
