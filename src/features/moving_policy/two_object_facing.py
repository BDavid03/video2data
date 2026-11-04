"""Retain original initialization candidates and evidence for clip-facing decisions."""

from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from src.features.moving_reference import _object,track
from src.provenance import write_json,sha256_file
from ._two_object_facing import best_opposite,mesh_flip,choose_clip
from ._two_object_symmetry import SYMMETRIES,resolve_facing,mesh_axis
from . import two_object_rotation


def captured_initial(captured):
    original=track.initial_pose
    def initialize(*args):
        fits=[]
        minimize=_object.minimize
        def save_fit(*a,**kw):
            result=minimize(*a,**kw);fits.append(result.x.copy());return result
        _object.minimize=save_fit
        try:
            pose,detail=original(*args)
        finally:
            _object.minimize=minimize
        xyz,visible,obj,hands,world_camera,vertices,faces,k=args
        camera_world=np.linalg.inv(world_camera)
        for values in fits:
            world=np.eye(4);world[:3,:3]=Rotation.from_euler('z',values[0]).as_matrix();world[:3,3]=values[1:]
            relative=camera_world@world
            score=_object.iou(_object.render(vertices,faces,relative,k,obj.shape),obj,hands)
            captured.append(dict(iou=score,world_T_asset=world.tolist(),camera_T_asset=relative.tolist()))
        return pose,detail
    return initialize


def track_candidate(selected,directory,maps,k,q,mesh,name,seed=None):
    captured=[];original_initial=track.initial_pose;original_step=track.object_step
    if seed is None:
        track.initial_pose=captured_initial(captured)
    else:
        track.initial_pose=lambda *args:(np.asarray(seed['camera_T_asset']),dict(seed['initialization']))
    if name in SYMMETRIES:
        track.object_step=two_object_rotation.object_step(mesh_axis(mesh.face_normals,mesh.area_faces),name)
    try:
        result=track.track_frames(selected,directory,maps,k,q,mesh)
    finally:
        track.initial_pose=original_initial;track.object_step=original_step
    return result,captured


def resolve_mesh(initialization,mesh):
    if initialization['opposite_resolved']:
        return initialization
    check=mesh_flip(mesh.vertices,mesh.faces,initialization['initial_world_T_asset'])
    result=dict(initialization,own_mesh_flip=check)
    if check['passed']:
        result.update(opposite_resolved=True,resolved_by='own-mesh flip symmetry')
    return result


def opposite_seed(initialization,candidates):
    selected=best_opposite(candidates,initialization['initial_world_T_asset'])
    if selected is None:
        return None
    detail=dict(initialization,initial_world_T_asset=selected['world_T_asset'],opposite_resolved=False,
                frame0_hypothesis='best opposite')
    return dict(camera_T_asset=selected['camera_T_asset'],initialization=detail)


def compare_clip(directory,initializations,poses,valids,scores):
    result=choose_clip(scores)
    artifacts=[]
    for name,pose,valid,values in zip(('winner','opposite'),poses,valids,scores):
        target=directory/'hypotheses'/name
        target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/'comparison.npz',world_T_asset=pose,valid=valid,iou=values)
        artifacts.append(dict(path=str((target/'comparison.npz').relative_to(track.ROOT)),sha256=sha256_file(target/'comparison.npz')))
    result.update(hypotheses=artifacts,initializations=initializations,instrument='existing track.reprojection finite IoU median on observed camera/object frames')
    write_json(directory/'whole-clip-facing.json',result)
    if result['opposite_resolved']:
        initialization=dict(initializations[result['selected']],opposite_resolved=True,resolved_by='whole-clip reprojection',
                            whole_clip_medians=result['medians'],whole_clip_margin=result['margin'],whole_clip_selected=result['selected'])
    else:
        initialization=dict(initializations[0],whole_clip_medians=result['medians'],whole_clip_margin=result['margin'])
    return result,initialization


def verify(directory,mesh,initialization):
    reason=initialization.get('resolved_by')
    if reason=='own-mesh flip symmetry':
        measured=mesh_flip(mesh.vertices,mesh.faces,initialization['initial_world_T_asset'])
        if not measured['passed'] or measured!=initialization['own_mesh_flip']:
            raise ValueError('Own-mesh facing certificate changed')
    elif reason=='whole-clip reprojection':
        report=json.loads((directory/'whole-clip-facing.json').read_text());scores=[];poses=[]
        for item in report['hypotheses']:
            path=track.ROOT/item['path']
            if sha256_file(path)!=item['sha256']:raise ValueError('Facing comparison input changed')
            with np.load(path) as saved:scores.append(saved['iou'].copy());poses.append(saved['world_T_asset'].copy())
        measured=choose_clip(scores)
        if any(measured[k]!=report[k] for k in measured) or not measured['opposite_resolved']:
            raise ValueError('Whole-clip facing comparison no longer resolves')
        with np.load(directory/'reference.npz') as ref:
            if not np.array_equal(ref['world_T_object'],poses[measured['selected']]):
                raise ValueError('Reference differs from selected whole-clip hypothesis')


def reference_tracks(selected,directory,mesh,name,maps,k,q,cameras,camera_rows):
    """Track the winner, resolve a tie if needed, and retain each comparison input."""
    from src.features.moving_reference import full_c
    from .two_object_table import fit
    first,candidates=track_candidate(selected,directory,maps,k,q,mesh,name)
    initialization=resolve_facing(name,first[6],mesh.face_normals,mesh.area_faces)
    initialization=resolve_mesh(initialization,mesh)
    runs=[first];initializations=[initialization]
    seed=opposite_seed(initialization,candidates) if not initialization['opposite_resolved'] else None
    if seed:
        write_json(directory/'initial-candidates.json',candidates)
        opposite=directory/'hypotheses/opposite';opposite.mkdir(parents=True)
        second,_=track_candidate(selected,opposite,maps,k,q,mesh,name,seed)
        runs.append(second);initializations.append(second[6])
    full_c.cache_frames(selected,directory)
    gauge,table=fit(directory,cameras)
    cameras=np.einsum('ij,tjk->tik',gauge,cameras)
    worlds=[];valids=[];scores=[]
    for index,run in enumerate(runs):
        valid=np.asarray([a['observed'] and b['observed'] for a,b in zip(camera_rows,run[3])],bool)
        world=_object.smooth(np.einsum('tij,tjk->tik',cameras,run[1]),valid)
        target=directory if len(runs)==1 else directory/'hypotheses'/('winner' if index==0 else 'opposite')
        target.mkdir(parents=True,exist_ok=True)
        projection=np.asarray(track.reprojection(selected,target,maps,k,mesh,cameras,world,valid))
        write_json(target/'object-diagnostics.json',run[3])
        worlds.append(world);valids.append(valid);scores.append(projection)
    chosen=0
    if len(runs)==2:
        decision,initialization=compare_clip(directory,initializations,worlds,valids,scores)
        if decision['opposite_resolved']:chosen=decision['selected']
        import shutil
        shutil.copyfile(directory/'hypotheses'/('winner' if chosen==0 else 'opposite')/'overlay-strip.png',directory/'overlay-strip.png')
    valid=valids[chosen] if initialization['opposite_resolved'] else np.zeros(len(valids[chosen]),bool)
    write_json(directory/'object-diagnostics.json',runs[chosen][3])
    np.savez_compressed(directory/'object-tracking-raw.npz',camera_T_asset=runs[chosen][1],world_T_asset=worlds[chosen],
        world_T_camera=cameras,valid=valid,intrinsics=k,object_observed=np.asarray([r['observed'] for r in runs[chosen][3]]),
        camera_observed=np.asarray([r['observed'] for r in camera_rows]))
    return dict(relative=runs[chosen][1],object_rows=runs[chosen][3],usable=runs[chosen][4],initialization=initialization,
                cameras=cameras,filtered=worlds[chosen],valid=valid,scores=scores[chosen],table=table)
