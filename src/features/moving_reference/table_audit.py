"""Audit and correct table instruments without rewriting sealed references."""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.provenance import sha256_file, write_json
from . import amendment, camera, gates, track
from ._table_audit import robust_plane, footprint, inside, normal_angle, table_view_pixels

SETTINGS = dict(ransac_tolerance_m=.015, ransac_trials=1000, ransac_seed=0,
    rule='Keep the original frame-0 ellipse as a proposal. RANSAC on its training tracks only identifies the table plane. The convex hull of those training inlier pixels defines the fixed frame-0 footprint. Track that footprint through the training plane homography, using image rays rather than held-out stereo depths, at first sight and each visible frame. Exclude frame-0 and current dilated pan/hand masks. First-sight anchors never re-anchor. No held-out error selects table membership.',
    minimum_points=30, late_fraction=.2,
    tolerance_source='15 mm, the existing Amendment 1 plane inlier tolerance; fixed before corrected measurements on either clip')
SEALED = {39:'44cd03d9da3a9b017610eab00439662eb8d7bc83ff00bd0a386cff618d30b3e1',
          38:'c5da9fbe4802a62ccfeca8627f2b9aa69a2acf14eaeecbc565b3a728fb275f12'}


def fit(points):
    return robust_plane(points, SETTINGS['ransac_tolerance_m'], SETTINGS['ransac_trials'])


def sources():
    return {name:sha256_file(Path(__file__).parent/name) for name in ('table_audit.py','_table_audit.py','gates.py','amendment.py')}


def run(episode):
    directory = track.OUTPUT/f'ep{episode}/candidate-c'
    os.environ['MPLCONFIGDIR'] = str(directory/'figure-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert sha256_file(directory/'reference.npz') == SEALED[episode]
    preserved = {name:sha256_file(directory/name) for name in ('reference.npz','gates.json','gates-amended.json','tracking.npz','klt-observations.npz')}
    old = json.loads((directory/'gates-amended.json').read_text())
    with np.load(directory/'tracking.npz') as a:
        cameras = a['world_T_camera'].copy()
    with np.load(directory/'reference.npz') as a:
        times = a['time'].copy()
    with np.load(directory/'klt-observations.npz') as a:
        data = {key:a[key] for key in a.files}
    with np.load(directory/'frames/frame_000000.npz') as a:
        exclusion = a['object_mask'] | a['hands']
        shape = a['gray'].shape
    exclusion = cv2.dilate(exclusion.astype(np.uint8), np.ones((21,21),np.uint8)).astype(bool)
    yy, xx = np.indices(shape)
    proposal = (((xx/shape[1]-.52)/.17)**2+((yy/shape[0]-.72)/.25)**2 <= 1) & ~exclusion
    def original_region(uv):
        x,y=uv[:,0],uv[:,1]
        ellipse=((x/shape[1]-.52)/.17)**2+((y/shape[0]-.72)/.25)**2<=1
        return ellipse & inside(~exclusion,uv)
    rows = data['rows']; ids = rows[:,1].astype(int)
    first = rows[(rows[:,0] == 0) & ~data['heldout'][ids]]
    first = first[original_region(first[:,2:4])]
    training = camera.apply(cameras[0], first[:,6:9])
    normal, offset, inliers = fit(training)
    assert normal is not None
    region = footprint(shape, first[inliers,2:4]) & proposal
    cv2.imwrite(str(directory/'table-region-corrected.png'), region.astype(np.uint8)*255)
    held = rows[data['heldout'][ids]]
    anchors = {}; old_member = {}; member = {}
    records = []; snapshots = []; late = []; old_late = []
    old_normal = np.array(old['frame0_plane_normal']); old_offset = old['frame0_plane_offset_m']
    for frame, pose in enumerate(cameras):
        observed = held[held[:,0] == frame]; indices = observed[:,1].astype(int)
        points = camera.apply(pose, observed[:,6:9])
        table_uv,forward=table_view_pixels(observed[:,2:4],data['k'],pose,cameras[0],normal,offset)
        on_table=inside(region,table_uv)&forward
        for index, point, eligible in zip(indices, points,on_table):
            if int(index) not in anchors:
                anchors[int(index)] = point.copy()
                uv = camera.project(camera.apply(np.linalg.inv(cameras[0]),point[None]),data['k'])
                old_member[int(index)] = bool(original_region(uv)[0])
                member[int(index)] = bool(eligible)
        old_pick = np.array([old_member[int(i)] for i in indices],bool)
        new_pick = np.array([member[int(i)] for i in indices],bool)&on_table
        with np.load(directory/f'frames/frame_{frame:06d}.npz') as a:
            foreground = cv2.dilate((a['object_mask']|a['hands']).astype(np.uint8),np.ones((21,21),np.uint8)).astype(bool)
            new_pick &= inside(~foreground, observed[:,2:4])
        p = points[new_pick]; old_p = points[old_pick]
        count = len(p); usable = count >= 30
        fitted, _, _ = fit(p) if usable else (None,None,None)
        old_fitted, _ = amendment.plane(old_p)
        diffs = p-np.asarray([anchors[int(i)] for i in indices[new_pick]]).reshape(-1,3)
        in_plane = diffs-(diffs@normal)[:,None]*normal
        med = float(np.median(abs(p@normal+offset))) if count else None
        displacement = float(np.median(np.linalg.norm(in_plane,axis=1))) if count else None
        if frame >= int(.8*len(cameras)):
            if usable:late.extend(p.tolist())
            if len(old_p)>=30:old_late.extend(old_p.tolist())
        records.append(dict(frame=frame,time_s=float(times[frame]),points=count,used=usable,
            offset_median_m=med if usable else None,diagnostic_offset_median_m=med,
            in_plane_displacement_median_m=displacement if usable else None,
            normal_change_deg=normal_angle(normal,fitted),old_normal_change_deg=normal_angle(old_normal,old_fitted),
            old_offset_median_m=old['table_rows'][frame]['offset_median_m']))
        snapshots.append((observed,old_pick,new_pick,points))
    late_normal, late_offset, late_inliers = fit(np.asarray(late).reshape(-1,3))
    old_late_ransac, _, _ = fit(np.asarray(old_late).reshape(-1,3))
    medians = [r['offset_median_m'] if r['offset_median_m'] is not None else np.nan for r in records]
    displacements = [r['in_plane_displacement_median_m'] if r['in_plane_displacement_median_m'] is not None else np.nan for r in records]
    used_fraction = float(np.mean([r['used'] for r in records]))
    amended = {key:dict(value) for key,value in old['gates'].items()}
    amended.update(gates.table_drift_gate(medians,normal_angle(normal,late_normal),used_fraction))
    amended.update(gates.long_baseline_gate(displacements,used_fraction))
    result = dict(episode=episode,passed=all(v['passed'] for v in amended.values()),gates=amended,
        previous_gates={key:value for key,value in old['gates'].items() if key.startswith('table_')},
        settings=SETTINGS,sources=sources(),preserved_inputs=preserved,reference_sha256=SEALED[episode],
        original_frame0_normal=old_normal.tolist(),original_frame0_normal_to_world_up_deg=normal_angle(old_normal,np.array([0.,0.,1.])),
        frame0_normal=normal.tolist(),frame0_offset_m=offset,training_points=len(training),training_inliers=int(inliers.sum()),
        late_normal=late_normal.tolist() if late_normal is not None else None,late_offset_m=late_offset,
        late_points=len(late),late_inliers=int(late_inliers.sum()),
        same_original_points_ransac_normal_change_deg=normal_angle(normal,old_late_ransac),
        table_rows=records,reference_rewritten=False,
        diagnosis='Contaminated image proposal extends below the table onto the lap; initial unseeded SVD/clipping fits a steep mixture plane. Corrected training-only planar footprint and RANSAC do not use held-out errors to select points. Coverage remains a gate, not a reason to enlarge the footprint.')
    write_json(directory/'gates-amended-2.json',result)
    fig,axes = plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
    axes[0].plot(times,[r['old_offset_median_m']*1000 if r['old_offset_median_m'] is not None else np.nan for r in records],label='Before')
    axes[0].plot(times,np.asarray(medians)*1000,label='After');axes[0].axhline(10,c='r',ls='--');axes[0].set_ylabel('Median absolute offset (mm)')
    axes[1].plot(times,[r['old_normal_change_deg'] for r in records],label='Before SVD')
    axes[1].plot(times,[r['normal_change_deg'] for r in records],label='After RANSAC');axes[1].axhline(2,c='r',ls='--');axes[1].set_ylabel('Per-frame normal change (deg)')
    axes[1].axhline(old['gates']['table_late_normal_change_deg']['value'],c='tab:blue',ls=':',label='Before pooled late normal')
    axes[1].axhline(normal_angle(normal,late_normal),c='tab:orange',ls=':',label='After pooled late normal')
    axes[2].plot(times,[r['points'] for r in records]);axes[2].axhline(30,c='r',ls='--');axes[2].set(xlabel='Video time (s)',ylabel='Held-out table points')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend();axes[1].legend();fig.suptitle(f'Episode {episode}: independent table audit; sealed reference unchanged')
    fig.savefig(directory/'table-drift-before-after.png',dpi=150);plt.close(fig)
    worst_offset=max(range(len(records)),key=lambda f:records[f]['old_offset_median_m'] or -1)
    worst_normal=max(range(len(records)),key=lambda f:records[f]['old_normal_change_deg'] or -1)
    selected=list(dict.fromkeys([0,worst_offset,worst_normal,int(.9*len(cameras)),len(cameras)-1]))
    tiles=[]
    for f in selected:
        with np.load(directory/f'frames/frame_{f:06d}.npz') as a:image=cv2.cvtColor(a['gray'],cv2.COLOR_GRAY2BGR)
        observed,old_pick,new_pick,points=snapshots[f]
        for row,old_yes,new_yes,p in zip(observed,old_pick,new_pick,points):
            if old_yes:
                color=(0,255,0) if new_yes else (0,0,255)
                cv2.circle(image,tuple(np.rint(row[2:4]).astype(int)),4,color,-1)
        contour=cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        if f==0:cv2.drawContours(image,contour,-1,(255,255,0),1)
        cv2.putText(image,f'ep{episode} f{f}: green retained, red excluded; n={records[f]["points"]}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
        tiles.append(image)
    cv2.imwrite(str(directory/'table-points-audit.png'),np.vstack(tiles))
    np.savez_compressed(directory/'table-audit-points.npz',frame=np.concatenate([np.full(len(s[0]),f) for f,s in enumerate(snapshots)]),
        rows=np.concatenate([s[0] for s in snapshots]),original_table=np.concatenate([s[1] for s in snapshots]),corrected_table=np.concatenate([s[2] for s in snapshots]))
    for name,digest in preserved.items():assert sha256_file(directory/name)==digest,name
    print(json.dumps(dict(episode=episode,table={k:v for k,v in amended.items() if k.startswith('table_')},reference_sha256=SEALED[episode],training_inliers=int(inliers.sum()),late_inliers=int(late_inliers.sum()),worst_frames=selected)),flush=True)
    return result


if __name__ == '__main__':
    for episode in (39,38):
        path=track.OUTPUT/f'ep{episode}/candidate-c/gates-amended-2.json'
        if path.exists():
            raise FileExistsError(f'Preserve existing instrument audit: {path}')
        run(episode)
