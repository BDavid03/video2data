"""Measure Amendment 1 on the unchanged C camera and sealed references."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from src.provenance import sha256_file,write_json
from . import camera,gates,track

ROOT=track.ROOT
SETTINGS=dict(table_ellipse_center_xy=[.52,.72],table_ellipse_radii_xy=[.17,.25],
    exclusion_pixels=21,stationary_translation_m=.001,stationary_rotation_deg=.2,
    noise_explanation='At least 100 same-track consecutive pairs over >=3 near-stationary frames; control median 3D >= old median 3D and median along-ray >=3 times perpendicular. Otherwise retain old instrument.',
    camera_revision='None retained: no justified calibration change; same tracks and camera matrices preserved')


def plane(points):
    if len(points)<30:return None,None
    p=np.asarray(points)
    for _ in range(3):
        center=np.median(p,axis=0);_,_,vt=np.linalg.svd(p-center,full_matrices=False);n=vt[-1]
        if n[2]<0:n=-n
        d=-float(np.median(p@n));keep=abs(p@n+d)<.015
        if keep.sum()<30:break
        p=p[keep]
    return n,d


def body_overlay(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    asset=directory/'asset';points=np.load(asset/'body-audit-points.npz')['scan_in_asset']
    mesh=trimesh.load(asset/'mini_dust_pan.obj',force='mesh',process=False)
    floor=points[(points[:,0]<-.01)&(abs(points[:,1])<.02)]
    rim=points[(points[:,0]<-.01)&(abs(points[:,1])>.05)]
    correct=bool(np.median(rim[:,2])>np.median(floor[:,2]))
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    labels=[('Handle',np.array([.065,0,.005])),('Rim',np.array([-.06,.06,.01])),('Underside',np.array([-.04,0,-.014]))]
    for ax,(h,v,title) in zip(axes,[(0,1,'Top'),(0,2,'Side'),(1,2,'End')]):
        ax.scatter(points[::3,h],points[::3,v],s=.4,alpha=.35,label='Public scan under constant')
        ax.scatter(mesh.vertices[:,h],mesh.vertices[:,v],s=.5,alpha=.4,label='Tracking asset')
        for name,p in labels:ax.annotate(name,(p[h],p[v]),fontsize=8)
        ax.set(title=title,xlabel=f'{"xyz"[h]} (m)',ylabel=f'{"xyz"[v]} (m)');ax.axis('equal')
    axes[0].legend(fontsize=7);fig.suptitle('Body convention audit only: no scan geometry used in tracking')
    fig.savefig(asset/'body-overlay.png',dpi=160);plt.close(fig)
    result=dict(constant_changed=False,floor_median_z_m=float(np.median(floor[:,2])),rim_median_z_m=float(np.median(rim[:,2])),
                rim_above_floor=correct,heuristic='Area-median sign alone is insufficient; independent central-floor versus outer-rim regions confirm the current up sign. Handle is +x in the top overlay.')
    write_json(asset/'body-audit.json',result)
    if not correct:raise RuntimeError('Body constant requires correction; do not admit before review')


def run(episode):
    import cv2
    directory=track.OUTPUT/f'ep{episode}/candidate-c';root39=track.OUTPUT/'ep39/candidate-c'
    if episode==38:
        freeze=json.loads((root39/'amendment-freeze.json').read_text())
        if freeze['sources']!=sources():raise RuntimeError('Amendment code differs from 39 freeze')
        if (directory/'amendment-once.json').exists():raise FileExistsError('Amendment 38 already ran')
        write_json(directory/'amendment-once.json',dict(sources=sources()))
    old_path=directory/'gates.json';old_hash=sha256_file(old_path);old=json.loads(old_path.read_text())
    write_json(directory/'amendment-request.json',dict(settings=SETTINGS,sources=sources(),frozen_gates_sha256=old_hash,
        protocol_sha256=sha256_file(ROOT/'docs/chain/moving-m1-protocol.md')))
    with np.load(directory/'tracking.npz') as arrays:cameras=arrays['world_T_camera']
    with np.load(directory/'klt-observations.npz') as arrays:data={key:arrays[key] for key in arrays.files}
    with np.load(directory/'frames/frame_000000.npz') as arrays:mask=arrays['object_mask']|arrays['hands']
    mask=cv2.dilate(mask.astype(np.uint8),np.ones((21,21),np.uint8)).astype(bool)
    height,width=mask.shape;k=data['k'];all_rows=data['rows'];all_ids=all_rows[:,1].astype(int)
    first=all_rows[(all_rows[:,0]==0)&~data['heldout'][all_ids]]
    def region(uv):
        x,y=uv[:,0],uv[:,1]
        ellipse=((x/width-.52)/.17)**2+((y/height-.72)/.25)**2<=1
        ij=np.rint(uv).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<width)&(ij[:,1]>=0)&(ij[:,1]<height)
        return ellipse&inside&~mask[np.clip(ij[:,1],0,height-1),np.clip(ij[:,0],0,width-1)]
    training=camera.apply(cameras[0],first[region(first[:,2:4]),6:9]);normal,offset=plane(training)
    if normal is None:normal=np.array([0.,0.,1.]);offset=0.;plane_available=False
    else:plane_available=True
    rows=all_rows[data['heldout'][all_ids]];anchors=np.full_like(data['landmarks'],np.nan);is_table=np.zeros(len(anchors),bool)
    previous={};per_frame=[];norms=[];perps=[];along=[];controls=[];control_frames=[];spatial=[];late=[]
    for frame,pose in enumerate(cameras):
        observed=rows[rows[:,0]==frame];ids=observed[:,1].astype(int);points=camera.apply(pose,observed[:,6:9]);fresh=np.isnan(anchors[ids,0])
        anchors[ids[fresh]]=points[fresh]
        if fresh.any():
            uv0=camera.project(camera.apply(np.linalg.inv(cameras[0]),points[fresh]),k);is_table[ids[fresh]]=region(uv0)
        error=points-anchors[ids];rays=points-pose[:3,3];rays/=np.linalg.norm(rays,axis=1)[:,None]
        parallel=np.einsum('ij,ij->i',error,rays);perpendicular=np.linalg.norm(error-parallel[:,None]*rays,axis=1)
        norms.extend(np.linalg.norm(error,axis=1));perps.extend(perpendicular);along.extend(abs(parallel))
        projected=camera.project(camera.apply(np.linalg.inv(pose),anchors[ids]),k);reprojection=np.linalg.norm(projected-observed[:,2:4],axis=1)*640/width
        spatial.extend(np.c_[observed[:,2]/width,observed[:,3]/height,observed[:,8],frame-data['born'][ids],reprojection].tolist())
        if frame:
            delta=np.linalg.norm(cameras[frame,:3,3]-cameras[frame-1,:3,3]);angle=np.rad2deg(Rotation.from_matrix(cameras[frame-1,:3,:3].T@pose[:3,:3]).magnitude())
            if delta<=.001 and angle<=.2:
                used=0
                for index,p,ray in zip(ids,points,rays):
                    if int(index) in previous:
                        e=p-previous[int(index)];a=float(e@ray);q=np.linalg.norm(e-a*ray);controls.append((np.linalg.norm(e),abs(a),q));used+=1
                if used:control_frames.append(frame)
        previous={int(i):p for i,p in zip(ids,points)}
        table=is_table[ids];p=points[table];diff=error[table];count=len(p)
        if count>=30 and plane_available:
            med=float(np.median(abs(p@normal+offset)));in_plane=diff-(diff@normal)[:,None]*normal;displacement=float(np.median(np.linalg.norm(in_plane,axis=1)))
            if frame>=int(.8*len(cameras)):late.extend(p.tolist())
        else:med=None;displacement=None
        per_frame.append(dict(frame=frame,heldout_table_points=count,offset_median_m=med,in_plane_displacement_median_m=displacement))
    control=np.asarray(controls).reshape(-1,3)
    control_report=dict(pairs=len(control),frames=len(control_frames),frame_indices=control_frames,
        norm_median_m=float(np.median(control[:,0])) if len(control) else None,
        along_median_m=float(np.median(control[:,1])) if len(control) else None,
        perpendicular_median_m=float(np.median(control[:,2])) if len(control) else None)
    late_normal,_=plane(np.asarray(late).reshape(-1,3));angle=float(np.rad2deg(np.arccos(np.clip(normal@late_normal,-1,1)))) if late_normal is not None and plane_available else None
    offsets=[r['offset_median_m'] if r['offset_median_m'] is not None else np.nan for r in per_frame];displacements=[r['in_plane_displacement_median_m'] if r['in_plane_displacement_median_m'] is not None else np.nan for r in per_frame]
    fraction=float(np.isfinite(offsets).mean());amended={key:dict(value) for key,value in old['gates'].items() if not key.startswith('zero_action_support')}
    amended.update(gates.amended_reprojection_gate(old['gates']['camera_heldout_reprojection_median_px640']['value'],old['gates']['camera_heldout_reprojection_p95_px640']['value']))
    amended['table_disjoint_patch_normal_max_deg']=gates.amended_patch_normal_gate(old['gates']['table_disjoint_patch_normal_max_deg']['value'])
    amended['camera_heldout_world_residual_median_m']=gates.perpendicular_world_gate(float(np.median(perps)),float(np.median(norms)),control_report)
    objects=json.loads((directory/'object-diagnostics.json').read_text())
    amended['object_30_stereo_correspondence_fraction']=gates.stereo_pixel_gate([r['stereo_supported_pixels'] for r in objects],[r['correspondences'] for r in objects])
    amended.update(gates.table_drift_gate(offsets,angle,fraction));amended.update(gates.long_baseline_gate(displacements,fraction))
    spatial=np.asarray(spatial);tail=spatial[spatial[:,4]>old['gates']['camera_heldout_reprojection_p95_px640']['value']]
    diagnosis=dict(tail_observations=len(tail),tail_median_xy_normalized=np.median(tail[:,:2],axis=0).tolist(),tail_median_depth_m=float(np.median(tail[:,2])),tail_median_age_frames=float(np.median(tail[:,3])),
        tail_edge_fraction=float(np.mean((tail[:,0]<.1)|(tail[:,0]>.9)|(tail[:,1]<.1)|(tail[:,1]>.9))),
        all_edge_fraction=float(np.mean((spatial[:,0]<.1)|(spatial[:,0]>.9)|(spatial[:,1]<.1)|(spatial[:,1]>.9))),camera_change='None: unchanged camera and held-out selection; no post-error track removal')
    result=dict(passed=all(r['passed'] for r in amended.values()),episode=episode,gates=amended,frozen_rule=old,
        frozen_gates_sha256=old_hash,noise_floor=control_report,camera_diagnosis=diagnosis,
        table_region=SETTINGS,frame0_training_table_points=len(training),frame0_plane_normal=normal.tolist(),frame0_plane_offset_m=offset,
        late_normal=late_normal.tolist() if late_normal is not None else None,table_rows=per_frame,camera_unchanged=True)
    with np.load(directory/'reference.npz') as arrays:reference={key:arrays[key] for key in arrays.files}
    if bool(reference['gates_passed'])!=result['passed']:
        backup=directory/'reference-before-amendment.npz'
        if not backup.exists():backup.write_bytes((directory/'reference.npz').read_bytes())
        reference['gates_passed']=np.asarray(result['passed']);np.savez_compressed(directory/'reference.npz',**reference)
    result['reference_sha256']=sha256_file(directory/'reference.npz')
    write_json(directory/'gates-amended.json',result)
    if sha256_file(old_path)!=old_hash:raise RuntimeError('Frozen gates unexpectedly changed')
    print(json.dumps(dict(episode=episode,passed=result['passed'],gates={key:(value['value'],value['passed']) for key,value in amended.items()},noise_floor=control_report,camera_diagnosis=diagnosis,reference_sha256=result['reference_sha256'])),flush=True)


def sources():return {name:sha256_file(Path(__file__).parent/name) for name in ('amendment.py','gates.py')}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['run','freeze','body']);parser.add_argument('--episode',type=int,choices=[39,38]);args=parser.parse_args()
    directory=track.OUTPUT/'ep39/candidate-c'
    os.environ['MPLCONFIGDIR']=str(directory/'figure-cache')
    if args.operation=='body':body_overlay(directory)
    elif args.operation=='freeze':write_json(directory/'amendment-freeze.json',dict(sources=sources(),settings=SETTINGS,reference_sha256=sha256_file(directory/'reference.npz')))
    else:run(args.episode)
