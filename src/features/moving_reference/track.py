"""Run the sealed, CPU-only candidate-A queue and persist auditable gate evidence."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.paths import project_path
from src.provenance import sha256_file, write_json
from . import camera
from ._object import body_axes, initial_pose, iou, object_step, render, smooth
from .gates import evaluate

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
OUTPUT = ROOT/'data/outputs/moving_reference'
SETTINGS = dict(width=960, stereo='SGBM 128 disparities, block 5, LR <=1px, depth 0.1..5m',
                seed=31, object_erosion_px=3, hand_dilation_px=7, background_dilation_px=21,
                keyframe_interval=10, retained_keyframes='frame 0 and two latest retained frames',
                smoothing='One centered Savitzky-Golay window 5, polynomial 2, interp edges; translation and continuous rotation vectors; valid contiguous segments only',
                opposite_facing_iou_margin=.02, protocol='docs/chain/moving-m1-protocol.md',
                zero_action_support='CPU evaluator; one second, 20 steps, static frame-0 reference; hands cleared',
                queue=[39,38], no_labels=True, gpu=False)


def sources():
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in
            [FEATURE/'__main__.py', FEATURE/'camera.py', FEATURE/'_object.py', FEATURE/'track.py', FEATURE/'gates.py', FEATURE/'support.py',
             ROOT/'src/features/mesh_extents/_shapes.py', ROOT/'src/paths.py', ROOT/'src/provenance.py']}


def inputs(episode):
    if episode not in (39,38):
        raise PermissionError('Only sealed episodes 39 and 38 are authorized')
    masks = OUTPUT/f'ep{episode}'/('masks' if episode==39 else 'masks-smallest')
    if not (masks/'summary.json').exists():
        raise FileNotFoundError(f'Required masks not ready: {masks}/summary.json; do not use ep38/masks')
    summary=json.loads((masks/'summary.json').read_text())
    if summary['status']!='completed':
        raise ValueError('Mask producer is incomplete')
    job=json.loads((masks/'request.json').read_text())
    source=json.loads((ROOT/'data/configs/sources.lock.json').read_text())['dataset']
    snapshot=project_path(ROOT,Path(source['cache_path'])/'snapshots'/source['revision'])
    split='public' if episode==39 else 'evaluation'
    paths=[snapshot/f'track_3/{split}/videos/chunk-000/observation.images.ego_cam_{key}/episode_{episode:06d}.mp4' for key in ('c','b')]
    calibration=snapshot/'track_3/public/meta/camera_calibration.json'
    if sha256_file(calibration)!=json.loads((ROOT/'data/configs/calibration/ego.json').read_text())['source_sha256']:
        raise ValueError('Calibration differs from audited source')
    mask_files=[masks/'frames'/f'frame_{i:06d}.npz' for i in range(job['frame_count'])]
    queue=json.loads((ROOT/'src/features/moving_policy/queue.json').read_text())
    mesh_inputs=[project_path(ROOT,queue[key]) for key in ('measurement','attributes')]
    files=[*paths,calibration,masks/'request.json',masks/'summary.json',*mask_files,*mesh_inputs,
           ROOT/'data/configs/sources.lock.json',ROOT/'data/configs/calibration/ego.json',
           ROOT/'data/reports/data-audit/videos.json',ROOT/SETTINGS['protocol'],
           ROOT/'src/features/mesh_extents/_shapes.py',ROOT/'src/features/moving_policy/queue.json',
           ROOT/'src/paths.py', ROOT/'src/provenance.py']
    hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in files}
    clocks=json.loads((ROOT/'data/reports/data-audit/videos.json').read_text())
    rows=[]
    for path in paths:
        relative=str(path.relative_to(snapshot))
        row,=[r for r in clocks['videos'] if r['path']==relative]
        rows.append(row)
    if any(r['decoded_frame_count']!=job['frame_count'] or r['pts_seconds']!=job['timestamps'] for r in rows):
        raise ValueError('Stereo and mask clocks differ')
    return dict(paths=paths, calibration=calibration, masks=masks, job=job, hashes=hashes,
                mesh_inputs=mesh_inputs, mask_files=mask_files)


def asset(selected):
    from src.features.mesh_extents._shapes import thin_tool_profile
    measure,attributes=[json.loads(p.read_text()) for p in selected['mesh_inputs']]
    dims=measure['objects']['mini_dust_pan']['median_m']
    profile=attributes['objects']['mini_dust_pan']['profile']
    return thin_tool_profile(dims,profile)


def masks_at(selected,index,maps,shape):
    with np.load(selected['mask_files'][index],allow_pickle=False) as arrays:
        raw=arrays['masks'].astype(np.uint8)
        if int(arrays['frame_index'])!=index or abs(float(arrays['timestamp_seconds'])-selected['job']['timestamps'][index])>1e-6:
            raise ValueError('Mask frame or clock mismatch')
    width,height=selected['job']['source_resolution_width_height']
    # The remap is in original pixels; SAM inputs used half-pixel resize coordinates.
    mx=(maps[0][0]+.5)*raw.shape[2]/width-.5
    my=(maps[0][1]+.5)*raw.shape[1]/height-.5
    masks=np.stack([cv2.remap(m,mx,my,cv2.INTER_NEAREST).astype(bool) for m in raw])
    obj=masks[0]; hands=masks[1:].any(axis=0)
    dilated=cv2.dilate(hands.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    visible=cv2.erode(obj.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~dilated
    excluded=cv2.dilate(masks.any(axis=0).astype(np.uint8),np.ones((21,21),np.uint8))
    oy,ox=np.nonzero(obj)
    center=np.array([np.median(ox),np.median(oy)]) if len(ox) else np.array(shape[::-1])/2
    # Forearms are not segmented: conservatively exclude the hand-to-border ray.
    for hand in masks[1:]:
        yy,xx=np.nonzero(hand)
        if len(xx):
            start=np.array([np.median(xx),np.median(yy)])
            direction=start-center
            direction/=max(np.linalg.norm(direction),1)
            end=start+direction*max(shape)*2
            thickness=max(35,round(1.5*np.sqrt(len(xx))))
            cv2.line(excluded,tuple(np.rint(start).astype(int)),tuple(np.rint(end).astype(int)),1,thickness)
    return obj,hands,visible,~excluded.astype(bool)


def captures(paths):
    result=[cv2.VideoCapture(str(p),cv2.CAP_FFMPEG) for p in paths]
    for capture in result:
        if not capture.isOpened():
            raise RuntimeError('Cannot open stereo stream')
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
    return result


def read_pair(streams,maps):
    images=[]
    for stream,mapping in zip(streams,maps):
        ok,image=stream.read()
        if not ok:
            raise RuntimeError('Stereo stream ended early')
        images.append(cv2.remap(image,*mapping,cv2.INTER_LINEAR))
    gray=[cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]
    return images,gray


def body_convention(episode,selected,directory):
    constant=OUTPUT/'ep39/body-convention.json'
    if episode==38:
        result=json.loads(constant.read_text())
        write_json(directory/'body-convention.json',result)
        return np.asarray(result['scan_T_asset']),result
    if constant.exists():
        result=json.loads(constant.read_text())
        write_json(directory/'body-convention.json',result)
        return np.asarray(result['scan_T_asset']),result
    import trimesh
    snapshot=selected['calibration'].parents[3]
    scan=snapshot/'track_3/public/mesh/mini_dust_pan/mini_dust_pan.glb'
    if not scan.exists():
        scan=scan.with_name('mini_dust_pan_visual.glb')
    # Exactly one object, exactly one load; never split its disconnected triangles.
    mesh=trimesh.load(scan,force='mesh',process=False)
    bounds_transform,extents=trimesh.bounds.oriented_bounds(mesh)
    samples,_=trimesh.sample.sample_surface(mesh,10000,seed=31)
    transform,detail=body_axes(bounds_transform,extents,samples)
    result=dict(scan_T_asset=transform.tolist(),asset_T_scan=np.linalg.inv(transform).tolist(),
                method='Oriented bounding box, longest in-plane axis toward narrower end (+x handle), thin axis upward from area-dense floor, right-handed +y; origin at bounding center. No scan scale or vertices enter reconstruction or tracking.',
                scan_input=dict(path=str(scan.relative_to(ROOT)),sha256=sha256_file(scan)),
                details=detail, applies_identically_to_episodes=[39,38],trajectory_used=False)
    write_json(constant,result); write_json(directory/'body-convention.json',result)
    return transform,result


def track_frames(selected,directory,maps,k,q,mesh):
    count=selected['job']['frame_count']
    camera_poses=[]; object_poses=[]; camera_rows=[]; object_rows=[]; usable=[]
    streams=captures(selected['paths'])
    keyframes=[]; previous=None; previous_pose=None; state=None; table=None; initialization=None
    try:
        for index in range(count):
            images,gray=read_pair(streams,maps)
            xyz,valid_depth=camera.stereo(*gray,q)
            obj,hands,visible,static=masks_at(selected,index,maps,gray[0].shape)
            points=camera.corners(gray[0],visible)
            _,supported=camera.lift(points,xyz)
            usable.append(bool(supported.sum()>=30))
            if index==0:
                world_T_camera,table=camera.define_world(xyz,static&valid_depth,obj)
                state=camera.camera_start(gray[0],xyz,static,world_T_camera)
                held=state['pixels']; regions=(held[:,0]>=gray[0].shape[1]/2).astype(int)+2*(held[:,1]>=gray[0].shape[0]/2)
                row=dict(observed=True,inliers=len(state['map_world']),heldout_count=len(held),heldout_regions=len(np.unique(regions)),
                         heldout_reprojection_px640=[0.]*len(held),heldout_world_residual_m=[0.]*len(held))
                pose,initialization=initial_pose(xyz,visible,obj,hands,world_T_camera,mesh.vertices,mesh.faces,k)
                detail=dict(correspondences=int(supported.sum()),backward_correspondences=0,keyframe_correspondences=int(supported.sum()),
                            keyframes=[0],keyframe_refined=True,fb_translation_m=None,fb_rotation_deg=None,observed=True)
                write_json(directory/'world.json',dict(world_T_rectified_camera0=world_T_camera.tolist(),
                           heading='First camera viewing direction projected on table',origin='Nearest table point to first camera',**table))
                write_json(directory/'initialization.json',initialization)
                keyframes.append((index,gray[0].copy(),xyz.copy(),visible.copy(),pose.copy()))
            else:
                world_T_camera,row=camera.camera_step(state,gray[0],xyz,static,k)
                pose,detail=object_step(previous,(gray[0],xyz,visible),keyframes,previous_pose)
                if index%10==0 and detail['observed']:
                    keyframes.append((index,gray[0].copy(),xyz.copy(),visible.copy(),pose.copy()))
                    keyframes=keyframes[:1]+keyframes[-2:] if len(keyframes)>3 else keyframes
            detail.update(frame_index=index,stereo_texture_points=int(supported.sum()))
            camera_rows.append(dict(frame_index=index,**row)); object_rows.append(detail)
            camera_poses.append(world_T_camera.copy()); object_poses.append(pose.copy())
            previous=(gray[0],xyz,visible); previous_pose=pose
            if index%20==0 or index==count-1:
                print(json.dumps(dict(frame=index,total=count,camera=row['observed'],heldout=row['heldout_count'],
                                     object_pairs=detail['correspondences'],object_keyframe=detail['keyframe_correspondences'])),flush=True)
    finally:
        for stream in streams:
            stream.release()
    return np.asarray(camera_poses),np.asarray(object_poses),camera_rows,object_rows,usable,table,initialization


def reprojection(selected,directory,maps,k,mesh,cameras,world_poses,valid):
    count=len(cameras); selected_frames=set(np.rint(np.linspace(0,count-1,12)).astype(int))
    streams=captures(selected['paths']); scores=[]; panels=[]
    try:
        for index in range(count):
            images,gray=read_pair(streams,maps)
            obj,hands,_,_=masks_at(selected,index,maps,gray[0].shape)
            camera_T_asset=np.linalg.inv(cameras[index])@world_poses[index]
            prediction=render(mesh.vertices,mesh.faces,camera_T_asset,k,obj.shape)
            score=iou(prediction,obj,hands)
            scores.append(score if valid[index] else np.nan)
            if index in selected_frames:
                observed=images[0].copy(); overlay=observed.copy()
                for mask,color in ((obj,(0,220,0)),(prediction,(0,80,255))):
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay,contours,-1,color,2)
                overlay[hands]=(overlay[hands]*.5+np.array([180,70,0])*.5).astype(np.uint8)
                label=f'frame {index}  IoU {score:.2f}  {"VALID" if valid[index] else "INVALID"}'
                for panel in (observed,overlay):
                    cv2.putText(panel,label,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                # Each sample has raw image above reference overlay, twelve columns.
                panels.append(np.vstack([cv2.resize(observed,(320,200)),cv2.resize(overlay,(320,200))]))
    finally:
        for stream in streams:
            stream.release()
    cv2.imwrite(str(directory/'overlay-strip.png'),np.hstack(panels))
    return scores


def run(episode,attempt='candidate-a'):
    started=time.perf_counter(); cv2.setNumThreads(2); cv2.setRNGSeed(31)
    if '/' in attempt or attempt.startswith('.'):
        raise ValueError('Attempt must be a simple output directory name')
    selected=inputs(episode)
    identity=dict(settings=SETTINGS,sources=sources(),inputs=selected['hashes'])
    if episode==38:
        freeze=json.loads((OUTPUT/'ep39/code-freeze.json').read_text())
        if freeze['sources']!=identity['sources'] or freeze['settings']!=SETTINGS:
            raise ValueError('Code or settings changed after episode 39 freeze')
        if sha256_file(OUTPUT/'ep39/body-convention.json')!=freeze['body_convention_sha256']:
            raise ValueError('Body convention changed after the freeze')
        if sha256_file(ROOT/SETTINGS['protocol'])!=freeze['protocol_sha256']:
            raise ValueError('Protocol changed after the freeze')
        if (OUTPUT/'ep38/tracker-once.json').exists():
            raise FileExistsError('Episode 38 runs only once')
        write_json(OUTPUT/'ep38/tracker-once.json',identity)
    directory=OUTPUT/f'ep{episode}'/attempt
    directory.mkdir(parents=True,exist_ok=True)
    request=directory/'request.json'
    if request.exists():
        if json.loads(request.read_text())!=identity:
            raise ValueError('Existing attempt has changed inputs or code; preserve it under its original name')
        if (directory/'gates.json').exists():
            print((directory/'gates.json').read_text()); return
    write_json(request,identity)
    record=dict(status='running',episode=episode,started_utc=datetime.now(timezone.utc).isoformat(),gpu=None,label_access=False)
    write_json(directory/'run.json',record)
    try:
        calibration=json.loads(selected['calibration'].read_text())
        maps,k,q,r1=camera.rectification(calibration,SETTINGS['width'])
        mesh=asset(selected); mesh.export(directory/'canonical-asset.obj')
        cameras,relative,camera_rows,object_rows,usable,table,initialization=track_frames(selected,directory,maps,k,q,mesh)
        valid=np.asarray([a['observed'] and b['observed'] and usable[i] and initialization['opposite_resolved']
                          for i,(a,b) in enumerate(zip(camera_rows,object_rows))],bool)
        world=np.einsum('tij,tjk->tik',cameras,relative)
        filtered=smooth(world,valid)
        np.savez_compressed(directory/'tracking.npz',world_T_camera=cameras,camera_T_asset=relative,
                            world_T_asset_raw=world,world_T_asset=filtered,valid=valid,intrinsics=k,R1=r1)
        write_json(directory/'camera-diagnostics.json',camera_rows)
        write_json(directory/'object-diagnostics.json',object_rows)
        scores=reprojection(selected,directory,maps,k,mesh,cameras,filtered,valid)
        base=float(camera.apply(filtered[0],mesh.vertices)[:,2].min())
        times=np.asarray(selected['job']['timestamps'],float)
        from .support import run as support_check
        support=support_check(directory,filtered[0],mesh)
        gates=evaluate(table,camera_rows,object_rows,times,filtered,valid,scores,usable,base,support)
        # Body convention is intentionally read only after tracking and reprojection.
        convention,body=body_convention(episode,selected,directory)
        reference=np.einsum('tij,jk->tik',filtered,np.linalg.inv(convention))
        quaternion=Rotation.from_matrix(reference[:,:3,:3]).as_quat()  # kit xyzw
        for i in range(1,len(quaternion)):
            if quaternion[i]@quaternion[i-1]<0:
                quaternion[i]*=-1
        np.savez_compressed(directory/'reference.npz',position=reference[:,:3,3],quaternion=quaternion,
                            quaternion_xyzw=quaternion,frame_index=np.arange(len(times)),time=times,valid=valid,
                            world_T_object=reference,pose=np.c_[reference[:,:3,3],quaternion],
                            gates_passed=np.asarray(gates['passed']))
        np.savez_compressed(directory/'reprojection.npz',iou=np.asarray(scores),valid=valid)
        gates.update(episode=episode,initial_facing_resolved=initialization['opposite_resolved'],
                     quaternion_order='xyzw (kit)',smoothing=SETTINGS['smoothing'])
        write_json(directory/'gates.json',gates)
        for name in ('reference.npz','gates.json','body-convention.json','overlay-strip.png'):
            (OUTPUT/f'ep{episode}'/name).write_bytes((directory/name).read_bytes())
        record.update(status='completed',passed=gates['passed'],wall_seconds=time.perf_counter()-started,
                      reference_sha256=sha256_file(directory/'reference.npz'),
                      body_convention_sha256=sha256_file(directory/'body-convention.json'))
        record['artifacts']={str(p.relative_to(directory)):sha256_file(p) for p in sorted(directory.rglob('*'))
                             if p.is_file() and 'cache' not in p.parts and p.name!='run.json'}
        write_json(directory/'run.json',record)
        with (OUTPUT/f'ep{episode}'/'tracker-ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(attempt=attempt,**record))+'\n')
        print(json.dumps(dict(episode=episode,passed=gates['passed'],wall_seconds=record['wall_seconds'],reference=str(directory/'reference.npz'))),flush=True)
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}',wall_seconds=time.perf_counter()-started)
        write_json(directory/'run.json',record)
        raise


def freeze():
    result=json.loads((OUTPUT/'ep39/gates.json').read_text())
    record=dict(sources=sources(),settings=SETTINGS,episode39_passed=result['passed'],
                reference_sha256=sha256_file(OUTPUT/'ep39/reference.npz'),
                body_convention_sha256=sha256_file(OUTPUT/'ep39/body-convention.json'),
                protocol_sha256=sha256_file(ROOT/SETTINGS['protocol']),
                frozen_utc=datetime.now(timezone.utc).isoformat())
    path=OUTPUT/'ep39/code-freeze.json'
    if path.exists():
        raise FileExistsError('Code freeze already exists')
    write_json(path,record)
    print(json.dumps(record),flush=True)
