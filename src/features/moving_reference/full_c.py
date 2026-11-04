"""Execute and record the frozen camera/table/model-tracking candidate C."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from src.provenance import sha256_file,write_json
from . import camera,track,camera_ba,model_track
from ._object import pair,smooth,iou,render
from .gates import evaluate

ROOT=track.ROOT
FEATURE=Path(__file__).parent
SETTINGS=dict(smoothing='One Savitzky-Golay pass, window 5 polynomial 2, translation and continuous rotvec, contiguous observed segments',
    table='Pre-contact prefix before first dilated hand/object overlap, maximum 12 frames; original four patch predicates unchanged',
    object='Forward/backward SE3 silhouette bidirectional chamfer and robust stereo-to-own-surface; previous world pose and KLT prediction; weak fits include x/z 180 degree starts',
    correspondence_gate='Unchanged A RANSAC-Kabsch KLT inlier count. Stereo pixels separately reported, not substituted.',
    support='S2 and exactly one S3 >=10mm collision comparison, each 60 control steps; visual mesh unchanged')


def source_hashes():
    names=['full_c.py','camera_ba.py','model_track.py','candidate_c.py','support_drop.py','camera.py','_object.py','gates.py','track.py','support.py']
    return {str((FEATURE/name).relative_to(ROOT)):sha256_file(FEATURE/name) for name in names}


def cache_frames(selected,directory):
    target=directory/'frames';target.mkdir(exist_ok=True)
    count=selected['job']['frame_count']
    if len(list(target.glob('frame_*.npz')))==count:
        return
    maps,k,q,_=camera.rectification(json.loads(selected['calibration'].read_text()))
    streams=track.captures(selected['paths'])
    sampled=set(np.rint(np.linspace(0,count-1,12)).astype(int))
    try:
        for index in range(count):
            images,gray=track.read_pair(streams,maps)
            path=target/f'frame_{index:06d}.npz'
            if path.exists():
                continue
            xyz,valid=camera.stereo(*gray,q)
            obj,hands,visible,static=track.masks_at(selected,index,maps,gray[0].shape)
            np.savez_compressed(path,gray=gray[0],xyz=xyz.astype(np.float32),object_mask=obj,hands=hands,
                                visible=visible,static=static,depth_valid=valid)
            if index in sampled:
                cv2.imwrite(str(target/f'image_{index:06d}.jpg'),images[0])
            if index%60==0:
                print(json.dumps(dict(stage='cache',frame=index)),flush=True)
    finally:
        for stream in streams:
            stream.release()


def read_frame(directory,index):
    with np.load(directory/'frames'/f'frame_{index:06d}.npz') as arrays:
        return {key:arrays[key] for key in arrays.files}


def table_fit(directory,cameras):
    first=read_frame(directory,0);yy,xx=np.indices(first['gray'].shape)
    oy,ox=np.nonzero(first['object_mask']);mx,my=np.median(ox),np.median(oy)
    nearby=(abs(xx-mx)<first['gray'].shape[1]*.3)&(yy>my-first['gray'].shape[0]*.1)
    patch_masks=[nearby & ((yy>my)==yh)&((xx>mx)==xh) for yh in (False,True) for xh in (False,True)]
    groups=[[] for _ in range(4)];used=[]
    for index in range(min(12,len(cameras))):
        frame=read_frame(directory,index)
        hand=cv2.dilate(frame['hands'].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        if index and (hand&frame['object_mask']).any():
            break
        valid=frame['static']&frame['depth_valid']
        for group,patch in zip(groups,patch_masks):
            points=frame['xyz'][valid&patch]
            if len(points)>2000:
                points=points[np.linspace(0,len(points)-1,2000).astype(int)]
            group.extend(camera.apply(cameras[index],points).tolist())
        used.append(index)
    points=np.concatenate([np.asarray(group).reshape(-1,3) for group in groups])
    # Rotate into the existing plane fitter's y-up optical convention.
    basis=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
    normal,offset,inliers=camera.plane(points@basis.T)
    normal=normal@basis
    distance=abs(points@normal+offset)
    patch_normals=[]
    for group in groups:
        p=np.asarray(group).reshape(-1,3);p=p[abs(p@normal+offset)<.01]
        if len(p)<100:
            continue
        _,_,vt=np.linalg.svd(p-p.mean(0),full_matrices=False);n=vt[-1]*np.sign(vt[-1]@normal);patch_normals.append(n)
    angles=[np.rad2deg(np.arccos(np.clip(a@b,-1,1))) for i,a in enumerate(patch_normals) for b in patch_normals[i+1:]]
    rotation=Rotation.align_vectors(np.array([[0.,0.,1.]]),normal[None])[0].as_matrix()
    gauge=np.eye(4);gauge[:3,:3]=rotation;gauge[2,3]=offset
    residual=distance[distance<.01]
    report=dict(residual_rms_m=float(np.sqrt(np.mean(residual**2))),patch_normal_max_deg=float(max(angles)) if len(patch_normals)==4 else None,
                precontact_frames=used,normal_before_gauge=normal.tolist(),offset_before_gauge_m=float(offset),
                world_new_T_world_initial=gauge.tolist(),patch_point_counts=list(map(len,groups)),patch_definition='A frame-0 image medians and +/-0.3 width, y > median-0.1 height; unchanged predicates')
    return gauge,report


def geometry(directory):
    asset=track.OUTPUT/'ep39/candidate-c/asset'
    mesh=trimesh.load(asset/'mini_dust_pan.obj',force='mesh',process=False)
    parameters=json.loads((asset/'parameters.json').read_text())
    with np.load(track.OUTPUT/'ep39/candidate-c/frame0.npz') as arrays:
        frame={key:arrays[key] for key in arrays.files}
    with np.load(asset/'placement.npz') as arrays:
        pose=arrays['world_T_asset']
    model=model_track.tracking_geometry(frame['object_mask'],frame['k'],frame['world_T_camera'],pose,
                    parameters['thickness_m'],mesh.vertices,mesh.faces)
    return mesh,model,parameters


def initialize(frame,camera_pose,mesh,k):
    # The same fixed asset gets an independent frame-0 pose on each episode.
    pose,_=model_track.footprint(frame['object_mask'],frame['xyz'],frame['visible'],k,camera_pose,
                                known_thickness=np.ptp(mesh.vertices[:,2]))
    pose[2,3]=-mesh.vertices[:,2].min()
    return pose


def object_pass(directory,cameras,k,mesh,model,direction,forward=None):
    count=len(cameras);indices=list(range(count)) if direction=='forward' else list(range(count-1,-1,-1))
    poses=np.tile(np.eye(4),(count,1,1));details=[None]*count
    previous=None;previous_co=None;previous_world=None
    for position,index in enumerate(indices):
        frame=read_frame(directory,index)
        evidence=model_track.image_evidence(frame['object_mask'],frame['hands'],frame['xyz'],frame['visible'])
        if position==0:
            world=initialize(frame,cameras[index],mesh,k) if direction=='forward' else forward[index].copy()
            co=np.linalg.inv(cameras[index])@world
            detail=dict(iou=iou(model_track.fast_render(model,co,k,frame['gray'].shape),frame['object_mask'],frame['hands']),
                        cost=0.,starts=1,weak_fit=False,opposite_resolved=True,opposite_iou_margin=None)
            corners=camera.corners(frame['gray'],frame['visible'])
            _,supported=camera.lift(corners,frame['xyz'])
            count_pairs=int(supported.sum()) if direction=='forward' else 0
        else:
            motion,count_pairs=pair(previous['gray'],frame['gray'],previous['xyz'],frame['xyz'],previous['visible'],frame['visible'])
            base=np.linalg.inv(cameras[index])@previous_world
            starts=[base]
            if motion is not None:
                starts.append(motion@previous_co)
            co,detail=model_track.select_pose(starts,evidence,model,k)
            world=cameras[index]@co
        poses[index]=world
        detail.update(frame_index=index,correspondences=int(count_pairs),stereo_supported_pixels=evidence['stereo_pixels'],
                      observed=bool(len(evidence['cloud'])>=6 and detail['opposite_resolved']))
        details[index]=detail;previous=frame;previous_co=co;previous_world=world
        if position%20==0 or position==count-1:
            print(json.dumps(dict(stage=direction,frame=index,iou=detail['iou'],klt=count_pairs,stereo=evidence['stereo_pixels'])),flush=True)
    np.savez_compressed(directory/f'object-{direction}.npz',world_T_asset=poses)
    write_json(directory/f'object-{direction}.json',details)
    return poses,details


def plots(directory,selected,cameras,poses,valid,ious,details,reference=None):
    import os
    os.environ['MPLCONFIGDIR']=str(directory/'figure-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count=len(poses);sample=np.rint(np.linspace(0,count-1,12)).astype(int)
    mesh,model,_=geometry(directory);panels=[]
    for index in sample:
        frame=read_frame(directory,index);image=cv2.imread(str(directory/'frames'/f'image_{index:06d}.jpg'))
        overlay=image.copy();prediction=render(mesh.vertices,mesh.faces,np.linalg.inv(cameras[index])@poses[index],selected['k'],frame['gray'].shape)
        for mask,color in [(frame['object_mask'],(0,220,0)),(prediction,(0,80,255))]:
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,color,2)
        overlay[frame['hands']]=(overlay[frame['hands']]*.5+np.array([180,70,0])*.5).astype(np.uint8)
        for img in (image,overlay):
            cv2.putText(img,f'{index} IoU {ious[index]:.2f} {"VALID" if valid[index] else "INVALID"}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        panels.append(np.vstack([cv2.resize(image,(320,200)),cv2.resize(overlay,(320,200))]))
    strip=np.hstack(panels);cv2.imwrite(str(directory/'overlay-strip.png'),strip)
    cv2.imwrite(str(directory/'overlay-detail.png'),np.hstack([strip[:,320*i:320*(i+1)] for i in (0,3,7,10)]))
    times=np.asarray(selected['times']);position=(poses if reference is None else reference)[:,:3,3]
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for ax,h,v,title in [(axes[0],0,1,'Top view'),(axes[1],0,2,'Side view')]:
        ax.plot(position[:,h],position[:,v],color='gray',alpha=.7);dots=ax.scatter(position[:,h],position[:,v],c=times,s=9,cmap='viridis')
        ax.scatter(position[0,h],position[0,v],marker='s',color='red',label='Start');ax.set(xlabel='World x (m)',ylabel=f'World {"y" if v==1 else "z"} (m)',title=title);ax.axis('equal');ax.legend(fontsize=8)
    fig.colorbar(dots,ax=axes,label='Video time (s)');fig.suptitle(f'Episode {selected["episode"]}: video-derived reference, no mocap; camera gate may fail')
    fig.savefig(directory/'world-trajectory.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(9,6),sharex=True,constrained_layout=True)
    axes[0].plot(times,ious);axes[0].axhline(.6,color='red',linestyle='--');axes[0].set(ylabel='Mesh/mask IoU')
    axes[1].plot(times,[row['correspondences'] for row in details],label='A KLT inliers');axes[1].plot(times,[row['stereo_supported_pixels'] for row in details],label='Stereo pixels',alpha=.6);axes[1].axhline(30,color='red',linestyle='--');axes[1].set(ylabel='Support count',yscale='symlog');axes[1].legend()
    axes[2].step(times,valid.astype(int));axes[2].set(ylabel='Observed validity',xlabel='Video time (s)',yticks=[0,1]);fig.savefig(directory/'tracking-diagnostic.png',dpi=160);plt.close(fig)


def run(episode):
    cv2.setNumThreads(2);cv2.setRNGSeed(31);started=time.perf_counter()
    directory=track.OUTPUT/f'ep{episode}/candidate-c';directory.mkdir(parents=True,exist_ok=True)
    selected=track.inputs(episode)
    if episode==38:
        freeze=json.loads((track.OUTPUT/'ep39/candidate-c/code-freeze.json').read_text())
        if freeze['sources']!=source_hashes() or freeze['settings']!=SETTINGS:
            raise RuntimeError('C code/settings differ from 39 freeze')
        if (directory/'once.json').exists():
            raise FileExistsError('C episode 38 is one shot')
        write_json(directory/'once.json',dict(sources=source_hashes(),started_utc=datetime.now(timezone.utc).isoformat()))
    write_json(directory/'request-full.json',dict(episode=episode,sources=source_hashes(),inputs=selected['hashes'],settings=SETTINGS))
    write_json(directory/'run-full.json',dict(status='running',episode=episode,gpu=False))
    if not (directory/'camera-ba.npz').exists():
        camera_ba.run(episode)
    with np.load(directory/'camera-ba.npz') as arrays:
        cameras=arrays['world_T_camera'].copy()
    with np.load(directory/'klt-observations.npz') as arrays:
        data={key:arrays[key] for key in arrays.files}
    k=data['k'];cache_frames(selected,directory)
    gauge,table=table_fit(directory,cameras);cameras=np.einsum('ij,tjk->tik',gauge,cameras)
    write_json(directory/'table.json',table)
    mesh,model,parameters=geometry(directory)
    forward_path=directory/'object-forward.npz'
    if forward_path.exists():
        with np.load(forward_path) as arrays:
            forward=arrays['world_T_asset']
        details=json.loads((directory/'object-forward.json').read_text())
    else:
        forward,details=object_pass(directory,cameras,k,mesh,model,'forward')
    backward_path=directory/'object-backward.npz'
    if backward_path.exists():
        with np.load(backward_path) as arrays:
            backward=arrays['world_T_asset']
        back_details=json.loads((directory/'object-backward.json').read_text())
    else:
        backward,back_details=object_pass(directory,cameras,k,mesh,model,'backward',forward)
    camera_rows=camera_ba.heldout_rows(data,cameras)
    valid=np.asarray([a['observed'] and b['observed'] for a,b in zip(camera_rows,details)],bool)
    # Camera gate failure is preserved globally; valid means observed, not admitted.
    filtered=smooth(forward,valid)
    ious=[];usable=[]
    for index in range(len(cameras)):
        frame=read_frame(directory,index)
        prediction=render(mesh.vertices,mesh.faces,np.linalg.inv(cameras[index])@filtered[index],k,frame['gray'].shape)
        ious.append(iou(prediction,frame['object_mask'],frame['hands']))
        corners=camera.corners(frame['gray'],frame['visible']);_,supported=camera.lift(corners,frame['xyz']);usable.append(bool(supported.sum()>=30))
        detail=details[index]
        detail['fb_translation_m']=float(np.linalg.norm(forward[index,:3,3]-backward[index,:3,3]))
        detail['fb_rotation_deg']=float(np.rad2deg(Rotation.from_matrix(forward[index,:3,:3].T@backward[index,:3,:3]).magnitude()))
    support_result=json.loads((track.OUTPUT/'ep39/candidate-c/support-drop/S3/result.json').read_text())
    support_values=dict(status='measured 60 steps; shared asset S3 collision',maximum_translation_m=support_result['maximum_translation_m'],maximum_rotation_deg=support_result['maximum_rotation_deg'])
    base=float(camera.apply(filtered[0],mesh.vertices)[:,2].min());times=np.asarray(selected['job']['timestamps'])
    body_path=track.OUTPUT/'ep39/candidate-c/asset/body-convention.json';body=json.loads(body_path.read_text())
    reference=filtered@np.linalg.inv(np.asarray(body['scan_T_asset']))
    gates=evaluate(table,camera_rows,details,times,reference,valid,[value if valid[i] else np.nan for i,value in enumerate(ious)],usable,base,support_values)
    gates.update(episode=episode,correspondence_definition=SETTINGS['correspondence_gate'],
                 stereo_pixel_30_fraction=float(np.mean([row['stereo_supported_pixels']>=30 for row in details])),
                 camera_gates_passed=all(row['passed'] for key,row in gates['gates'].items() if key.startswith('camera_')))
    for key in ('zero_action_support_translation_m','zero_action_support_rotation_deg'):
        gates['gates'][key]['note']='Shared visual asset with S3 >=10mm collision: 60 CPU control steps, zero actions, hands clear, no resets; full result retained.'
    gates['gates']['object_speed_max_m_s']['note']='All written frames, measured at submitted scan-convention body origin after the single frozen smoothing pass.'
    gates['gates']['object_reference_valid_fraction']['note']='Observed camera and model fit with resolved tested opposite hypotheses; whole-reference admission is separately false when any global gate fails.'
    quat=Rotation.from_matrix(reference[:,:3,:3]).as_quat()
    for index in range(1,len(quat)):
        if quat[index]@quat[index-1]<0:quat[index]*=-1
    np.savez_compressed(directory/'reference.npz',position=reference[:,:3,3],quaternion=quat,quaternion_xyzw=quat,
        frame_index=np.arange(len(times)),time=times,valid=valid,world_T_object=reference,pose=np.c_[reference[:,:3,3],quat],gates_passed=np.asarray(gates['passed']))
    np.savez_compressed(directory/'tracking.npz',world_T_camera=cameras,world_T_asset=filtered,world_T_asset_forward=forward,world_T_asset_backward=backward,valid=valid,intrinsics=k)
    np.savez_compressed(directory/'reprojection.npz',iou=np.asarray(ious),valid=valid)
    write_json(directory/'gates.json',gates);write_json(directory/'object-diagnostics.json',details);write_json(directory/'camera-diagnostics-final.json',camera_rows)
    (directory/'body-convention.json').write_bytes(body_path.read_bytes())
    plots(directory,dict(k=k,times=times,episode=episode),cameras,filtered,valid,ious,details,reference)
    record=dict(status='completed',episode=episode,passed=gates['passed'],wall_seconds=time.perf_counter()-started,
                reference_sha256=sha256_file(directory/'reference.npz'),sources=source_hashes(),label_access=False,gpu=False)
    write_json(directory/'run-full.json',record)
    print(json.dumps(record),flush=True)


def freeze():
    directory=track.OUTPUT/'ep39/candidate-c'
    if not (directory/'reference.npz').exists():raise RuntimeError('39 full reference required before freeze')
    write_json(directory/'code-freeze.json',dict(sources=source_hashes(),settings=SETTINGS,
        reference_sha256=sha256_file(directory/'reference.npz'),asset_sha256=sha256_file(directory/'asset/mini_dust_pan.obj'),
        body_sha256=sha256_file(directory/'asset/body-convention.json'),frozen_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['run','freeze']);parser.add_argument('--episode',type=int,choices=[39,38]);args=parser.parse_args()
    freeze() if args.operation=='freeze' else run(args.episode)
