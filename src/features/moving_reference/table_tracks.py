"""One fixed independent table-measurement batch on sealed C references."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.provenance import sha256_file,write_json
from . import camera,camera_ba,gates,track
from ._table_audit import inside,robust_plane,normal_angle
from ._table_tracks import projected_region,point_metrics
from .table_audit import SEALED

SETTINGS=dict(stride_px=8,seed_interval_frames=5,occupied_radius_px=4,
    exclusions='Unchanged cached static mask: 21-pixel pan/hand dilation and original hand-to-border forearm rays',
    flow='Existing camera.flow: pyramidal KLT, forward-backward <=0.75px, previous/current allowed masks',
    stereo='Existing camera_ba.stereo_pixels: calibrated epipolar/reverse stereo LK, same LR-consistent SGBM fallback',
    table_footprint='Exact corrected table-region-corrected.png, carried by the training plane homography',
    anchors='Every fresh ID is lifted once at first accepted sight. Existing IDs and anchors never reset at seeding. Dead tracks are not revived.',
    ransac_tolerance_m=.015,ransac_trials=1000,ransac_seed=0,minimum_points=30,minimum_coverage=.8,
    independence='All tracks are new after camera BA and never enter any camera fit or map')


def sources():
    names=['table_tracks.py','_table_tracks.py','_table_audit.py','camera.py','camera_ba.py','gates.py']
    return {name:sha256_file(Path(__file__).parent/name) for name in names}


def run(episode, frozen_sources):
    directory=track.OUTPUT/f'ep{episode}/candidate-c'
    assert sha256_file(directory/'reference.npz')==SEALED[episode]
    preserved={name:sha256_file(directory/name) for name in ['reference.npz','tracking.npz','gates.json','gates-amended.json','gates-amended-2.json','table-region-corrected.png']}
    if (directory/'table-tracks-started.json').exists():
        raise FileExistsError('One attempt already started; do not rerun or tune')
    assert sources()==frozen_sources
    old=json.loads((directory/'gates-amended-2.json').read_text())
    request=json.loads((directory/'request-camera.json').read_text())['inputs']
    video_paths=[track.ROOT/name for name in request if '/videos/' in name and name.endswith('.mp4')]
    video_paths=sorted(video_paths,key=lambda p:0 if 'ego_cam_c' in str(p) else 1)
    calibration_path,=[track.ROOT/name for name in request if name.endswith('camera_calibration.json')]
    assert len(video_paths)==2
    for path in video_paths+[calibration_path]:assert sha256_file(path)==request[str(path.relative_to(track.ROOT))]
    maps,k,q,_=camera.rectification(json.loads(calibration_path.read_text()))
    baseline=1/q[3,2]
    with np.load(directory/'tracking.npz') as a:cameras=a['world_T_camera'].copy()
    with np.load(directory/'reference.npz') as a:times=a['time'].copy()
    region=cv2.imread(str(directory/'table-region-corrected.png'),cv2.IMREAD_GRAYSCALE)>0
    normal=np.array(old['frame0_normal']);offset=old['frame0_offset_m']
    write_json(directory/'table-tracks-started.json',dict(settings=SETTINGS,sources=frozen_sources,preserved_inputs=preserved,
        video_inputs={str(p.relative_to(track.ROOT)):sha256_file(p) for p in video_paths+[calibration_path]},started_utc=datetime.now(timezone.utc).isoformat()))
    streams=track.captures(video_paths)
    active_ids=np.empty(0,int);active_uv=np.empty((0,2),np.float32)
    anchors=[];born=[];observations=[];records=[];late=[];cache_hashes={}
    previous=None;previous_allowed=None
    snapshots={};sampled=set(np.rint(np.linspace(0,len(cameras)-1,12)).astype(int))
    try:
        for frame,pose in enumerate(cameras):
            images,gray=track.read_pair(streams,maps)
            cache_path=directory/f'frames/frame_{frame:06d}.npz'
            cache_hashes[str(cache_path.relative_to(track.ROOT))]=sha256_file(cache_path)
            with np.load(cache_path) as a:
                xyz=a['xyz'];static=a['static'];cached_gray=a['gray']
            assert np.array_equal(gray[0],cached_gray),'Video/cache rectification mismatch'
            footprint=projected_region(region,k,pose,cameras[0],normal,offset)
            allowed=footprint&static
            flow_before=len(active_ids)
            if frame:
                nxt,good=camera.flow(previous,gray[0],active_uv,previous_allowed,allowed)
                active_ids=active_ids[good];active_uv=nxt[good]
            flow_survivors=len(active_ids)
            stereo_dropped=0
            if len(active_ids):
                right,points,supported=camera_ba.stereo_pixels(*gray,active_uv,xyz,k,baseline)
                stereo_dropped=int((~supported).sum())
                active_ids=active_ids[supported];active_uv=active_uv[supported]
                right=right[supported];points=points[supported]
            else:
                right=np.empty((0,2));points=np.empty((0,3))
            added=0
            if frame%SETTINGS['seed_interval_frames']==0:
                seeds,_=camera_ba.grid(gray[0].shape,allowed,active_uv,stride=SETTINGS['stride_px'])
                stereo_uv,lifted,supported=camera_ba.stereo_pixels(*gray,seeds,xyz,k,baseline)
                seeds=seeds[supported];stereo_uv=stereo_uv[supported];lifted=lifted[supported]
                ids=np.arange(len(anchors),len(anchors)+len(seeds));added=len(ids)
                anchors.extend(camera.apply(pose,lifted).tolist());born.extend([frame]*added)
                active_ids=np.r_[active_ids,ids];active_uv=np.vstack([active_uv,seeds])
                right=np.vstack([right,stereo_uv]);points=np.vstack([points,lifted])
            world=camera.apply(pose,points);anchor=np.asarray(anchors).reshape(-1,3)[active_ids]
            med,displacement=point_metrics(world,anchor,normal,offset)
            ages=frame-np.asarray(born,int)[active_ids]
            count=len(active_ids);used=count>=30
            if used and frame>=int(.8*len(cameras)):late.extend(world.tolist())
            capacity,_=camera_ba.grid(gray[0].shape,allowed,np.empty((0,2)),stride=SETTINGS['stride_px'])
            _,dense_valid=camera.lift(capacity,xyz)
            available=int(allowed.sum());total=int(footprint.sum())
            record=dict(frame=frame,time_s=float(times[frame]),points=count,used=used,
                offset_median_m=med if used else None,in_plane_displacement_median_m=displacement if used else None,
                diagnostic_offset_median_m=med,diagnostic_displacement_median_m=displacement,
                fresh_tracks=added,flow_dropped=flow_before-flow_survivors,stereo_dropped=stereo_dropped,
                median_track_age_frames=float(np.median(ages)) if count else None,
                tracks_at_least_20_frames=int((ages>=20).sum()),tracks_at_least_100_frames=int((ages>=100).sum()),
                footprint_pixels=total,visible_pixels=available,visible_fraction=available/max(total,1),
                visible_grid_capacity=len(capacity),dense_stereo_grid_capacity=int(dense_valid.sum()),
                insufficient_visible_grid=len(capacity)<30,insufficient_dense_stereo_grid=int(dense_valid.sum())<30)
            records.append(record)
            observations.extend((frame,int(i),*uv,*rv,*p) for i,uv,rv,p in zip(active_ids,active_uv,right,points))
            # Only diagnostic images are retained; camera and reference are immutable.
            draw=images[0].copy();tint=draw.copy();tint[footprint&~static]=(0,0,180)
            draw=cv2.addWeighted(draw,.8,tint,.2,0)
            contours=cv2.findContours(footprint.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(draw,contours,-1,(255,255,0),1)
            for uv,age in zip(active_uv,ages):cv2.circle(draw,tuple(np.rint(uv).astype(int)),2,(255,100,0) if age==0 else (0,255,0),-1)
            cv2.putText(draw,f'ep{episode} f{frame} n={count} age={record["median_track_age_frames"]} visible={record["visible_fraction"]:.2f}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            cv2.putText(draw,'Blue=new; green=retained; red=masked; cyan=footprint',(8,44),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            if frame in sampled:snapshots[frame]=draw.copy()
            # Keep a compact set of extrema without retaining every full image.
            scores=[-count,med or 0,displacement or 0]
            for name,score in zip(['fewest','offset','displacement'],scores):
                if name not in snapshots or score>snapshots[name][0]:snapshots[name]=(score,frame,draw.copy())
            previous=gray[0];previous_allowed=allowed
            if frame%40==0 or frame==len(cameras)-1:print(json.dumps(dict(episode=episode,frame=frame,points=count,visible_grid=len(capacity))),flush=True)
    finally:
        for stream in streams:stream.release()
    late_normal,late_offset,inliers=robust_plane(np.asarray(late).reshape(-1,3),.015,1000)
    medians=[r['offset_median_m'] if r['used'] else np.nan for r in records]
    displacements=[r['in_plane_displacement_median_m'] if r['used'] else np.nan for r in records]
    coverage=float(np.mean([r['used'] for r in records]));result_gates={key:dict(value) for key,value in old['gates'].items()}
    result_gates.update(gates.table_drift_gate(medians,normal_angle(normal,late_normal),coverage))
    result_gates.update(gates.long_baseline_gate(displacements,coverage))
    result=dict(episode=episode,passed=all(v['passed'] for v in result_gates.values()),gates=result_gates,
        previous_table_gates={key:v for key,v in old['gates'].items() if key.startswith('table_')},settings=SETTINGS,sources=frozen_sources,
        preserved_inputs=preserved,reference_sha256=SEALED[episode],camera_refitted=False,reference_rewritten=False,
        frame0_normal=normal.tolist(),frame0_offset_m=offset,late_normal=late_normal.tolist() if late_normal is not None else None,
        late_points=len(late),late_inliers=int(inliers.sum()),track_count=len(anchors),
        insufficient_visibility_frames=[r['frame'] for r in records if r['insufficient_visible_grid']],
        insufficient_dense_stereo_frames=[r['frame'] for r in records if r['insufficient_dense_stereo_grid']],
        uncovered_frames=[r['frame'] for r in records if not r['used']],table_rows=records)
    write_json(directory/'table-tracks-inputs.json',dict(frame_caches=cache_hashes))
    result['frame_cache_hashes_sha256']=sha256_file(directory/'table-tracks-inputs.json')
    np.savez_compressed(directory/'table-tracks-observations.npz',rows=np.asarray(observations,float),anchors=np.asarray(anchors),born=np.asarray(born),k=k)
    write_json(directory/'gates-amended-3.json',result)
    ordered=[snapshots[f] for f in sorted(sampled)]
    cv2.imwrite(str(directory/'table-tracks-overlay.png'),np.vstack([np.hstack(ordered[:6]),np.hstack(ordered[6:])]))
    cv2.imwrite(str(directory/'table-tracks-worst.png'),np.vstack([snapshots[name][2] for name in ['fewest','offset','displacement']]))
    plot(directory,records,old)
    for name,digest in preserved.items():assert sha256_file(directory/name)==digest,name
    assert sources()==frozen_sources
    print(json.dumps(dict(episode=episode,passed=result['passed'],gates={k:v for k,v in result_gates.items() if k.startswith('table_')},reference_sha256=SEALED[episode])),flush=True)
    return result


def plot(directory,records,old):
    os.environ['MPLCONFIGDIR']=str(directory/'figure-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=[r['time_s'] for r in records]
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for key,scale,ax,label,limit in [('offset_median_m',1000,axes[0],'Frame median offset (mm)',10),('in_plane_displacement_median_m',1000,axes[1],'In-plane median displacement (mm)',10)]:
        ax.plot(times,[r[key]*scale if r[key] is not None else np.nan for r in old['table_rows']],label='Previous tracks')
        ax.plot(times,[r[key]*scale if r[key] is not None else np.nan for r in records],label='Fresh independent tracks')
        ax.axhline(limit,c='r',ls='--');ax.set_ylabel(label);ax.legend()
    axes[2].plot(times,[r['points'] for r in records],label='Accepted observations')
    axes[2].plot(times,[r['visible_grid_capacity'] for r in records],alpha=.5,label='Visible grid capacity')
    axes[2].axhline(30,c='r',ls='--');axes[2].set(xlabel='Video time (s)',ylabel='Point count');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Independent table measurement: fixed camera, footprint and first-sight anchors')
    fig.savefig(directory/'table-tracks-diagnostic.png',dpi=140);plt.close(fig)


def main(queue_path):
    cv2.setNumThreads(2)
    queue=json.loads(queue_path.read_text());assert queue['episodes']==[39,38] and queue['settings']==SETTINGS
    frozen=sources();root=track.OUTPUT/'ep39/candidate-c';summary=[]
    for episode in queue['episodes']:
        path=track.OUTPUT/f'ep{episode}/candidate-c/gates-amended-3.json'
        if path.exists():
            result=json.loads(path.read_text());assert result['settings']==SETTINGS and result['sources']==frozen
        else:
            result=run(episode,frozen)
            with (root/'table-tracks-ledger.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(episode=episode,passed=result['passed'],reference_sha256=result['reference_sha256'],gates_sha256=sha256_file(path)))+'\n')
        summary.append(dict(episode=episode,passed=result['passed'],table={key:value for key,value in result['gates'].items() if key.startswith('table_')}))
    write_json(root/'table-tracks-summary.json',dict(settings=SETTINGS,sources=frozen,completed_utc=datetime.now(timezone.utc).isoformat(),episodes=summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',type=Path,required=True)
    main(parser.parse_args().queue)
