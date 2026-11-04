"""Explain saved validity predicates and separate quality diagnostics on CPU."""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from src.provenance import sha256_file,write_json
from src.runs import now_utc
from src.features.moving_reference._object import render,iou
from .two_object_screen import ROOT
from ._two_object_rotation import orientation_degrees

OUTPUT=ROOT/'data/outputs/moving_policy/two-object'
ALLOWED={1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33}


def ranges(mask):
    ids=np.flatnonzero(mask)
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [(int(g[0]),int(g[-1])) for g in groups if len(g)]


def latest():
    found={}
    for parent in OUTPUT.glob('campaign*'):
        if not parent.is_dir():continue
        for path in parent.glob('ep*/candidate-c/gates-final.json'):
            episode=int(path.parents[1].name[2:])
            if episode not in ALLOWED or not (path.parents[1]/'result.json').exists():continue
            if episode not in found or path.stat().st_mtime_ns>found[episode].stat().st_mtime_ns:found[episode]=path
    return found


def analyze(path,target):
    directory=path.parent;episode=int(directory.parent.name[2:])
    request=json.loads((directory/'request-camera.json').read_text())
    masks=ROOT/request['mask_directory'];job=json.loads((masks/'request.json').read_text());n=job['frame_count']
    initialization=json.loads((directory/'initialization.json').read_text()) if (directory/'initialization.json').exists() else None
    rows=json.loads((directory/'object-diagnostics.json').read_text()) if (directory/'object-diagnostics.json').exists() else None
    camera=json.loads((directory/'camera-diagnostics.json').read_text()) if (directory/'camera-diagnostics.json').exists() else None
    track_path=directory/'tracking.npz'
    if not track_path.exists():track_path=directory/'object-tracking-raw.npz'
    arrays={}
    if track_path.exists():
        with np.load(track_path) as saved:arrays={k:saved[k].copy() for k in saved.files}
    valid=arrays.get('valid')
    if initialization and not initialization['opposite_resolved']:valid=np.zeros(n,bool)
    actual={};diagnostics={}
    if initialization and not initialization['opposite_resolved']:actual['unresolved initial facing']=np.ones(n,bool)
    if camera and len(camera)==n:actual['camera not observed']=~np.asarray([r['observed'] for r in camera])
    if rows and len(rows)==n:
        observed=np.asarray([r['observed'] for r in rows]);refined=np.asarray([r['keyframe_refined'] for r in rows])
        actual['keyframe rigid refinement unavailable']=~refined
        actual['consecutive rigid fit unavailable while keyframe fit exists']=~observed&refined
        diagnostics['stereo correspondences <30']=np.asarray([r['correspondences']<30 for r in rows])
        for field,key,limit in [('fb_rotation_deg','FB rotation >5deg',5),('fb_translation_m','FB translation >1cm',.01),
                               ('stereo_supported_pixels','supported stereo pixels <30',30)]:
            values=[r.get(field) for r in rows]
            if any(v is not None for v in values):
                value=np.asarray([np.nan if v is None else v for v in values])
                diagnostics[key]=np.isfinite(value)&(value<limit if 'pixels' in field else value>limit)
    empty=np.zeros(n,bool);hand_overlap=np.zeros(n,bool);mask_missing=np.zeros(n,bool)
    for index in range(n):
        file=masks/'frames'/f'frame_{index:06d}.npz'
        if not file.exists():mask_missing[index]=True;continue
        with np.load(file) as saved:seg=saved['masks']
        empty[index]=not seg[0].any();hand_overlap[index]=(seg[0]&seg[1:3].any(axis=0)).any()
    diagnostics.update({'mask file missing':mask_missing,'object mask empty':empty,'hand-mask overlap':hand_overlap})
    overlap=np.flatnonzero(hand_overlap);phase=np.full(n,'no hand overlap observed',object)
    if len(overlap):
        phase[:overlap[0]]='before first hand overlap';phase[overlap[0]:overlap[-1]+1]='between first/last hand overlap';phase[overlap[-1]+1:]='after last hand overlap'
    positions=arrays.get('world_T_asset')
    raw_iou=np.full(n,np.nan)
    role_path=directory.parent/'roles.json'
    if positions is not None and role_path.exists():
        role=json.loads(role_path.read_text());names=json.loads((ROOT/'data/.runtime/kit-20261102/v2d_submission_kit/data/track_3_evaluation_objects.json').read_text())['episodes'][str(episode)]
        name=names[role['active_slot']]
        times=np.asarray(job['timestamps']);speed=np.linalg.norm(np.diff(positions[:,:3,3],axis=0),axis=1)/np.diff(times)
        axis=initialization.get('symmetry_axis_model',[0.,0.,1.]) if initialization else [0.,0.,1.]
        rotations=Rotation.from_matrix(positions[:,:3,:3]).as_quat()
        rate=orientation_degrees(rotations[1:],rotations[:-1],name,axis)/np.diff(times)
        diagnostics['speed >1.5m/s']=np.r_[False,speed>1.5];diagnostics['angular rate >540deg/s']=np.r_[False,rate>540]
        mesh_path=directory/'asset'/f'{name}.obj'
        if mesh_path.exists() and 'intrinsics' in arrays:
            mesh=trimesh.load(mesh_path,force='mesh',process=False);k=arrays['intrinsics']
            for index in range(n):
                frame=directory/'frames'/f'frame_{index:06d}.npz'
                if not frame.exists():continue
                with np.load(frame) as cached:
                    obj=cached['object_mask'];hands=cached['hands'];shape=cached['gray'].shape
                relative=np.linalg.inv(arrays['world_T_camera'][index])@positions[index]
                raw_iou[index]=iou(render(mesh.vertices,mesh.faces,relative,k,shape),obj,hands)
            diagnostics['raw silhouette IoU <0.30']=np.isfinite(raw_iou)&(raw_iou<.30)
            diagnostics['raw silhouette IoU <0.60']=np.isfinite(raw_iou)&(raw_iou<.60)
    invalid=~valid if valid is not None else None
    counts={key:int((value&invalid).sum()) if invalid is not None else None for key,value in actual.items()}
    counts={k:v for k,v in counts.items() if v is None or v>0}
    if invalid is not None:
        explained=np.zeros(n,bool)
        for value in actual.values():explained|=value
        unknown=invalid&~explained
        if unknown.any():counts['unexplained by saved predicates']=int(unknown.sum())
    dominant=max((k for k in counts if counts[k] is not None),key=lambda k:counts[k],default='not measured / unavailable')
    if invalid is not None and not invalid.any():dominant='none (all frames valid)'
    result=dict(episode=episode,directory=str(directory.relative_to(ROOT)),frames=n,
        invalid_frames=int(invalid.sum()) if invalid is not None else None,actual_invalid_causes=counts,dominant=dominant,
        diagnostics={k:dict(all_frames=int(v.sum()),invalid_frames=int((v&invalid).sum()) if invalid is not None else None) for k,v in diagnostics.items()},
        phase_invalid_counts={p:int((invalid&(phase==p)).sum()) if invalid is not None else None for p in dict.fromkeys(phase)},
        invalid_frame_ranges=ranges(invalid) if invalid is not None else [],raw_iou_frames=int(np.isfinite(raw_iou).sum()))
    with (target.with_suffix('.csv')).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','time_s','invalid','phase_proxy','actual_invalid_reasons','quality_diagnostics','raw_silhouette_iou'])
        for i in range(n):writer.writerow([i,job['timestamps'][i],bool(invalid[i]) if invalid is not None else 'unavailable',phase[i],
            '; '.join(k for k,v in actual.items() if v[i]),'; '.join(k for k,v in diagnostics.items() if v[i]),float(raw_iou[i]) if np.isfinite(raw_iou[i]) else 'unavailable'])
    write_json(target,result)
    return result


def run():
    cache=OUTPUT/'validity-diagnosis';cache.mkdir(exist_ok=True);results=[]
    for episode,path in sorted(latest().items()):
        inputs=[path,*[p for p in (path.parent/'tracking.npz',path.parent/'object-tracking-raw.npz',path.parent/'object-diagnostics.json',path.parent/'initialization.json') if p.exists()]]
        identity='-'.join(sha256_file(p)[:10] for p in inputs)+'-'+sha256_file(Path(__file__))[:8]
        target=cache/f'ep{episode}-{identity}.json'
        result=json.loads(target.read_text()) if target.exists() else analyze(path,target)
        results.append((result,target))
    lines=['# Saved-reference validity diagnosis','',f'CPU-only snapshot {now_utc()}. No gates or tracks changed.','',
      'Actual validity is resolved initial facing AND observed camera AND observed object (consecutive rigid fit plus keyframe refinement). Mask usability is waived. Mask/depth/correspondence/FB/speed/reprojection rows below are diagnostics, not additional per-frame rejection rules. Counts may overlap. Missing fields remain unavailable.','',
      'Phase bins use first/last overlap of the original object and hand SAM masks (B excluded). They approximate before contact / carrying / after placement; these physical events and actual hand occlusion are not labelled. The table reports the measured overlap proxy, not verified contact.','',
      '| Episode | Invalid / frames | Actual invalid causes (nonexclusive) | Diagnostics on invalid frames (all frames) | Where: frame ranges; phase proxy |',
      '| ---: | --- | --- | --- | --- |']
    for r,path in results:
        counts='; '.join(f'{k}: {v}' for k,v in r['actual_invalid_causes'].items()) or ('none (all frames valid)' if r['invalid_frames']==0 else 'No saved full object-validity outcome')
        diag='; '.join(f'{k}: {v["invalid_frames"]} ({v["all_frames"]})' for k,v in r['diagnostics'].items())
        spans=', '.join(f'{a}-{b}' for a,b in r['invalid_frame_ranges'][:12]) or 'unavailable / none'
        phases='; '.join(f'{k}: {v}' for k,v in r['phase_invalid_counts'].items())
        lines.append(f'| {r["episode"]} | {r["invalid_frames"]} / {r["frames"]} | {counts} | {diag} | {spans}; {phases} |')
    lines+=['','Most common saved cause per episode:','']
    for r,path in results:
        lines.append(f'- Episode {r["episode"]}: {r["dominant"]}. Per-frame reasons and every interval: [{path.with_suffix(".csv").name}](validity-diagnosis/{path.with_suffix(".csv").name}).')
    lines+=['','Raw silhouette IoUs, where available, were recomputed from saved poses/cameras, cached video masks and our own procedural mesh; no re-tracking, scan or scored label was used. A global facing stop dominates all frames even when local correspondence fits existed. An unavailable saved diagnostic is not a zero failure count.']
    (OUTPUT/'validity-diagnosis.md').write_text('\n'.join(lines)+'\n')
    return len(results)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign-pid',type=int);args=parser.parse_args()
    cv2.setNumThreads(1)
    while True:
        print(json.dumps(dict(time_utc=now_utc(),episodes=run())),flush=True)
        if args.campaign_pid is None:break
        stat=Path(f'/proc/{args.campaign_pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':break
        time.sleep(15)


if __name__=='__main__':main()
