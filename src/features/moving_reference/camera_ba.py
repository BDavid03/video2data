"""Dense KLT stereo observations and sparse metric bundle adjustment for M1 C."""

import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from src.provenance import sha256_file,write_json
from . import camera,track
from .candidate_c import identity


def stereo_pixels(left,right,pixels,xyz,k,baseline):
    """Refine independently matched stereo pixels, checking epipolar and reverse flow."""
    if not len(pixels):
        return np.empty((0,2)),np.empty((0,3)),np.zeros(0,bool)
    points,valid=camera.lift(pixels,xyz)
    dense_valid=valid.copy()
    initial=np.asarray(pixels,np.float32).copy()
    initial[:,0]-=k[0,0]*baseline/np.maximum(points[:,2],.1)
    initial[~valid]=pixels[~valid]
    dense_right=initial.copy()
    options=dict(winSize=(15,15),maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005),
                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    source=np.ascontiguousarray(pixels,np.float32).reshape(-1,1,2)
    guess=np.ascontiguousarray(initial,np.float32).reshape(-1,1,2)
    matched,status,_=cv2.calcOpticalFlowPyrLK(left,right,source,guess,**options)
    back,reverse,_=cv2.calcOpticalFlowPyrLK(right,left,matched,source.copy(),**options)
    matched=matched[:,0];back=back[:,0]
    valid &= status.ravel().astype(bool)&reverse.ravel().astype(bool)
    valid &= np.linalg.norm(back-pixels,axis=1)<=.75
    valid &= abs(matched[:,1]-pixels[:,1])<=.5
    disparity=pixels[:,0]-matched[:,0]
    valid &= disparity>1
    z=k[0,0]*baseline/np.maximum(disparity,1)
    refined=np.c_[(pixels[:,0]-k[0,2])*z/k[0,0],(pixels[:,1]-k[1,2])*z/k[1,1],z]
    valid &= (z>.1)&(z<5)&np.isfinite(refined).all(1)
    # LR-consistent calibrated SGBM remains a metric observation where subpixel
    # photometric refinement is unavailable; it is not replaced by mono depth.
    matched[~valid]=dense_right[~valid]
    refined[~valid]=points[~valid]
    return matched,refined,dense_valid


def grid(shape,static,occupied,stride=16):
    y,x=np.mgrid[8:shape[0]-8:stride,8:shape[1]-8:stride]
    pixels=np.c_[x.ravel(),y.ravel()].astype(np.float32)
    good=static[y.ravel(),x.ravel()]
    if len(occupied):
        exclude=np.zeros(shape,np.uint8)
        for u,v in occupied:
            cv2.circle(exclude,(round(float(u)),round(float(v))),stride//2,1,-1)
        good &= ~exclude[y.ravel(),x.ravel()].astype(bool)
    return pixels[good],((x.ravel()//stride+2*(y.ravel()//stride))%4==0)[good]


def observations(selected,directory):
    calibration=json.loads(selected['calibration'].read_text())
    maps,k,q,_=camera.rectification(calibration)
    baseline=1/q[3,2]
    count=selected['job']['frame_count']
    streams=track.captures(selected['paths'])
    active_ids=np.zeros(0,int);active_uv=np.empty((0,2),np.float32)
    landmarks=[];roles=[];born=[];rows=[];poses=[];camera_ok=[];bad_counts=[]
    previous=None;previous_static=None;initial_world=None
    try:
        for frame in range(count):
            images,gray=track.read_pair(streams,maps)
            xyz,depth_valid=camera.stereo(*gray,q)
            obj,hands,visible,static=track.masks_at(selected,frame,maps,gray[0].shape)
            if frame==0:
                with np.load(directory/'frame0.npz') as arrays:
                    pose=arrays['world_T_camera'].copy()
                initial_world=pose.copy();observed=True
            else:
                nxt,good=camera.flow(previous,gray[0],active_uv,previous_static,static)
                active_ids=active_ids[good];active_uv=nxt[good]
                right_uv,points,supported=stereo_pixels(*gray,active_uv,xyz,k,baseline)
                fit=supported & ~np.asarray(roles,bool)[active_ids]
                ids=active_ids[fit]
                pose,n_inliers=camera.pnp(np.asarray(landmarks)[ids],active_uv[fit],k)
                observed=pose is not None
                if pose is None:
                    pose=poses[-1].copy()
                # Only training tracks define the independent moving-region mask.
                predicted=camera.project(camera.apply(np.linalg.inv(pose),np.asarray(landmarks)[ids]),k)
                error=np.linalg.norm(predicted-active_uv[fit],axis=1)
                bad=error>3
                moving=np.zeros(static.shape,np.uint8)
                for index,(u,v),is_bad in zip(ids,active_uv[fit],bad):
                    bad_counts[index]+=int(is_bad)
                    if is_bad:
                        cv2.circle(moving,(round(float(u)),round(float(v))),18,1,-1)
                ij=np.rint(active_uv).astype(int)
                dynamic=moving[np.clip(ij[:,1],0,static.shape[0]-1),np.clip(ij[:,0],0,static.shape[1]-1)].astype(bool)
                keep=supported & ~dynamic
                for index,uv,rv,p in zip(active_ids[keep],active_uv[keep],right_uv[keep],points[keep]):
                    rows.append((frame,index,*uv,*rv,*p))
                active_ids=active_ids[keep];active_uv=active_uv[keep]
            if frame%5==0:
                new_pixels,new_roles=grid(gray[0].shape,static,active_uv)
                if len(new_pixels):
                    right_uv,points,supported=stereo_pixels(*gray,new_pixels,xyz,k,baseline)
                    new_pixels,right_uv,points,new_roles=(a[supported] for a in (new_pixels,right_uv,points,new_roles))
                    new_ids=np.arange(len(landmarks),len(landmarks)+len(new_pixels))
                    landmarks.extend(camera.apply(pose,points).tolist());roles.extend(new_roles.tolist())
                    born.extend([frame]*len(new_ids));bad_counts.extend([0]*len(new_ids))
                    for index,uv,rv,p in zip(new_ids,new_pixels,right_uv,points):
                        rows.append((frame,index,*uv,*rv,*p))
                    active_ids=np.r_[active_ids,new_ids];active_uv=np.vstack([active_uv,new_pixels])
            poses.append(pose);camera_ok.append(observed)
            previous=gray[0];previous_static=static
            if frame%20==0 or frame==count-1:
                print(json.dumps(dict(stage='KLT',frame=frame,tracks=len(active_ids),landmarks=len(landmarks),camera=observed)),flush=True)
    finally:
        for stream in streams:
            stream.release()
    result=dict(rows=np.asarray(rows,float),landmarks=np.asarray(landmarks),heldout=np.asarray(roles,bool),
                born=np.asarray(born),bad_counts=np.asarray(bad_counts),world_T_camera=np.asarray(poses),
                camera_observed=np.asarray(camera_ok),k=k,baseline=np.asarray(baseline))
    np.savez_compressed(directory/'klt-observations.npz',**result)
    return result


def bundle_adjust(data,maximum_points=1400,maximum_evaluations=200):
    rows=data['rows'];frames=rows[:,0].astype(int);ids=rows[:,1].astype(int)
    counts=np.bincount(ids,minlength=len(data['landmarks']))
    eligible=np.flatnonzero(~data['heldout']&(counts>=8)&(data['bad_counts']<=2))
    # Uniform ID subsampling preserves the ordered spatial grid and birth times.
    if len(eligible)>maximum_points:
        eligible=eligible[np.linspace(0,len(eligible)-1,maximum_points).astype(int)]
    point_map=np.full(len(data['landmarks']),-1,int);point_map[eligible]=np.arange(len(eligible))
    fit=point_map[ids]>=0
    selected=rows[fit];frame=frames[fit];point=point_map[ids[fit]]
    initial=np.linalg.inv(data['world_T_camera'])
    count=len(initial)
    extrinsics=np.c_[Rotation.from_matrix(initial[:,:3,:3]).as_rotvec(),initial[:,:3,3]]
    camera_size=6*(count-1)
    x0=np.r_[extrinsics[1:].ravel(),data['landmarks'][eligible].ravel()]
    n=len(selected);k=data['k'];baseline=float(data['baseline'])

    def unpack(values):
        cameras=np.vstack([extrinsics[:1],values[:camera_size].reshape(-1,6)])
        return cameras,values[camera_size:].reshape(-1,3)

    def residual(values):
        cameras,points=unpack(values)
        rotation=Rotation.from_rotvec(cameras[:,:3]).as_matrix()
        xyz=np.einsum('nij,nj->ni',rotation[frame],points[point])+cameras[frame,3:]
        depth=np.maximum(xyz[:,2],.02)
        u=k[0,0]*xyz[:,0]/depth+k[0,2]
        v=k[1,1]*xyz[:,1]/depth+k[1,2]
        return np.c_[u-selected[:,2],v-selected[:,3],u-k[0,0]*baseline/depth-selected[:,4],v-selected[:,5]].ravel()

    residual_rows=np.repeat(np.arange(n*4),3)
    point_columns=np.tile(np.arange(3),n*4)+np.repeat(camera_size+point*3,12)
    present=frame>0
    cam_rows=np.repeat((np.flatnonzero(present)[:,None]*4+np.arange(4)).ravel(),6)
    cam_columns=np.tile(np.arange(6),int(present.sum())*4)+np.repeat((frame[present]-1)*6,24)
    sparsity=coo_matrix((np.ones(len(residual_rows)+len(cam_rows)),
                        (np.r_[residual_rows,cam_rows],np.r_[point_columns,cam_columns])),shape=(n*4,len(x0))).tocsr()
    started=time.perf_counter()
    result=least_squares(residual,x0,jac_sparsity=sparsity,loss='huber',f_scale=1.,
                         x_scale='jac',tr_solver='lsmr',max_nfev=maximum_evaluations,
                         ftol=1e-5,xtol=1e-6,gtol=1e-5,verbose=2)
    cameras,points=unpack(result.x)
    matrices=np.tile(np.eye(4),(count,1,1));matrices[:,:3,:3]=Rotation.from_rotvec(cameras[:,:3]).as_matrix();matrices[:,:3,3]=cameras[:,3:]
    report=dict(landmarks=len(eligible),observations=n,variables=len(x0),residuals=n*4,
                success=bool(result.success),message=result.message,nfev=result.nfev,cost=float(result.cost),
                optimality=float(result.optimality),wall_seconds=time.perf_counter()-started,
                initial_median_px=float(np.median(np.abs(residual(x0)))),final_median_px=float(np.median(np.abs(residual(result.x)))),
                loss='Huber',f_scale_px=1.,heldout_used=False,first_camera_fixed=True,
                metric_constraint='Calibrated left and right pinhole reprojection; no monocular scale prior')
    return np.linalg.inv(matrices),eligible,points,report


def heldout_rows(data,poses):
    observations=data['rows'];ids=observations[:,1].astype(int)
    selected=observations[data['heldout'][ids]]
    anchors={}
    rows=[]
    for frame,pose in enumerate(poses):
        observed=selected[selected[:,0]==frame]
        residual=[];world_error=[];pixels=[]
        for obs in observed:
            index=int(obs[1]);point=obs[6:9];uv=obs[2:4]
            if index not in anchors:
                anchors[index]=camera.apply(pose,point[None])[0]
            predicted=camera.project(camera.apply(np.linalg.inv(pose),anchors[index][None]),data['k'])[0]
            residual.append(float(np.linalg.norm(predicted-uv)*640/960))
            world_error.append(float(np.linalg.norm(camera.apply(pose,point[None])[0]-anchors[index])))
            pixels.append(uv)
        pixels=np.asarray(pixels).reshape(-1,2)
        regions=(pixels[:,0]>=480).astype(int)+2*(pixels[:,1]>=300)
        rows.append(dict(frame_index=frame,observed=bool(data['camera_observed'][frame]),
                         heldout_count=len(pixels),heldout_regions=len(np.unique(regions)),
                         heldout_reprojection_px640=residual,heldout_world_residual_m=world_error))
    return rows


def camera_gates(rows):
    # The exact existing gate evaluator owns definitions and thresholds.
    from .gates import evaluate
    n=len(rows);poses=np.tile(np.eye(4),(n,1,1))
    dummy=[dict(correspondences=0,fb_translation_m=None,fb_rotation_deg=None)]*n
    result=evaluate(dict(residual_rms_m=0,patch_normal_max_deg=0),rows,dummy,np.arange(n)/20,poses,
                    np.zeros(n,bool),np.full(n,np.nan),np.zeros(n,bool),0,
                    dict(maximum_translation_m=None,maximum_rotation_deg=None,status='unmeasured'))
    gates={key:value for key,value in result['gates'].items() if key.startswith('camera_')}
    return dict(passed=all(row['passed'] for row in gates.values()),gates=gates,definitions='Unmodified gates.py')


def run(episode=39):
    cv2.setNumThreads(2);cv2.setRNGSeed(31)
    directory=track.OUTPUT/f'ep{episode}/candidate-c'
    directory.mkdir(parents=True,exist_ok=True)
    selected=track.inputs(episode)
    if not (directory/'frame0.npz').exists():
        from .candidate_c import prepare_frame0
        prepare_frame0(selected,directory)
    write_json(directory/'request-camera.json',identity(selected))
    path=directory/'klt-observations.npz'
    if path.exists():
        with np.load(path) as arrays:
            data={key:arrays[key] for key in arrays.files}
    else:
        data=observations(selected,directory)
    before=heldout_rows(data,data['world_T_camera'])
    write_json(directory/'camera-before-ba.json',camera_gates(before))
    poses,ids,landmarks,report=bundle_adjust(data)
    rows=heldout_rows(data,poses)
    gates=camera_gates(rows)
    np.savez_compressed(directory/'camera-ba.npz',world_T_camera=poses,landmark_ids=ids,landmarks=landmarks)
    write_json(directory/'bundle-adjustment.json',report)
    write_json(directory/'camera-diagnostics.json',rows)
    write_json(directory/'camera-gates.json',gates)
    print(json.dumps(gates,indent=2),flush=True)
