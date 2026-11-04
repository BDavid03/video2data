"""Capture forward/backward closure rotations without changing M1 pose fits."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.features.moving_reference import _object
from src.features.moving_reference.gates import _amended_check
from ._two_object_symmetry import SYMMETRIES
from ._two_object_rotation import rotation_degrees,reference_metrics


def object_step(axis,name):
    original=_object.object_step
    def measure(previous,current,keyframes,previous_pose):
        pairs=[]
        original_pair=_object.pair
        def capture(*args):
            result=original_pair(*args)
            pairs.append(result[0])
            return result
        _object.pair=capture
        try:
            pose,detail=original(previous,current,keyframes,previous_pose)
        finally:
            _object.pair=original_pair
        detail['fb_rotation_raw_deg']=detail['fb_rotation_deg']
        if len(pairs)==2 and all(p is not None for p in pairs):
            closure=pairs[1][:3,:3]@pairs[0][:3,:3]
            body=previous_pose[:3,:3].T@closure@previous_pose[:3,:3]
            detail['fb_closure_rotation_body']=body.tolist()
            detail['fb_rotation_deg']=float(rotation_degrees(body,axis,SYMMETRIES[name]))
        return pose,detail
    return measure


def amend(source,poses,times,rows,name,axis):
    if name not in SYMMETRIES:
        return source,dict(applied=False,restored_valid_frames=0)
    measurements=reference_metrics(poses,times,rows,name,axis)
    changes={}
    for key,value in measurements.items():
        old=source['gates'][key]
        changes[key]=dict(old=old['value'],new=value)
        source['gates'][key]=_amended_check(value,old['threshold'],old['comparison'])
    source['angular_all_frames_max_deg_s']=measurements['object_angular_rate_max_deg_s']
    return source,dict(applied=True,symmetry=SYMMETRIES[name],axis_model=np.asarray(axis).tolist(),measurements=changes,
        restored_valid_frames=0,validity_rule='M1 observed flags reflect rigid/keyframe correspondence; no rotation-spike rejection flag exists, so no flags are restored',
        fb_closure_matrices=sum('fb_closure_rotation_body' in r for r in rows),
        scalar_only_fb_closures=sum(r.get('fb_rotation_deg') is not None and 'fb_closure_rotation_body' not in r for r in rows),
        scalar_only_rule='Retain original conservative angle; scalar magnitude cannot identify twist')


def mesh_axes(paths):
    import trimesh
    from ._two_object_symmetry import mesh_axis
    meshes=[trimesh.load(path,force='mesh',process=False) for path in paths]
    return [mesh_axis(mesh.face_normals,mesh.area_faces) for mesh in meshes]
