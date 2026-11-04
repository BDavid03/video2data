"""Measure object orientation differences modulo declared axial symmetry."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.features.pose_insurance._moving import tracking_metrics
from ._two_object import resting_metrics
from ._two_object_symmetry import SYMMETRIES


def rotation_degrees(relative, axis, symmetry):
    """Remove continuous twist, or reduce lid twist modulo pi, retaining swing."""
    rotation=Rotation.from_matrix(np.asarray(relative))
    q=rotation.as_quat()
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    projection=np.einsum('...i,i->...',q[...,:3],axis)
    twist=np.concatenate([projection[...,None]*axis,q[...,3:]],axis=-1)
    norm=np.linalg.norm(twist,axis=-1,keepdims=True)
    twist=np.divide(twist,norm,out=np.zeros_like(twist),where=norm>1e-12)
    twist[...,3]=np.where(norm[...,0]>1e-12,twist[...,3],1.)
    swing=rotation*Rotation.from_quat(twist).inv()
    if symmetry=='order 2':
        angle=2*np.arctan2(projection,q[...,3])
        angle=(angle+np.pi/2)%np.pi-np.pi/2
        swing=swing*Rotation.from_rotvec(angle[...,None]*axis)
    return np.degrees(swing.magnitude())


def orientation_degrees(achieved,reference,name,axis):
    a=Rotation.from_quat(achieved);r=Rotation.from_quat(reference)
    relative=(r.inv()*a).as_matrix()
    if name not in SYMMETRIES:
        return np.degrees(Rotation.from_matrix(relative).magnitude())
    return rotation_degrees(relative,axis,SYMMETRIES[name])


def tracking(achieved,reference,names,axes):
    result=tracking_metrics(achieved,reference)
    for slot,name in enumerate(names):
        if name not in SYMMETRIES:
            continue
        a,r=achieved[:,slot],reference[:,slot]
        angle=orientation_degrees(a[:,3:],r[:,3:],name,axes[slot])
        static=orientation_degrees(np.repeat(r[:1,3:],len(r),axis=0),r[:,3:],name,axes[slot])
        result[slot].update(raw_Theta_deg=result[slot]['Theta_deg'],raw_first_orientation_deg=result[slot]['first_orientation_deg'],
                            Theta_deg=float(angle.mean()),Theta_S_deg=float(static.mean()),first_orientation_deg=float(angle[0]),
                            symmetry=SYMMETRIES[name],symmetry_axis_model=np.asarray(axes[slot]).tolist())
    return result


def resting(poses,start,name,axis):
    result=resting_metrics(poses,start)
    if name not in SYMMETRIES:
        return result
    target=np.asarray(start)[[4,5,6,3]]
    angle=orientation_degrees(poses[:,3:],target,name,axis)
    drift=orientation_degrees(poses[:,3:],poses[0,3:],name,axis)
    result.update(raw_rotation_metrics={k:v for k,v in result.items() if 'rotation' in k},
        first_rotation_deg=float(angle[0]),maximum_rotation_from_start_deg=float(angle.max()),
        maximum_rotation_drift_deg=float(drift.max()),symmetry=SYMMETRIES[name])
    result['passed']=bool(result['first_translation_m']<=.01 and result['first_rotation_deg']<=2 and
        result['maximum_translation_from_start_m']<.01 and result['maximum_rotation_from_start_deg']<2 and
        result['maximum_translation_drift_m']<.01 and result['maximum_rotation_drift_deg']<2)
    return result


def reference_metrics(poses,times,rows,name,axis):
    relative=poses[:-1,:3,:3].transpose(0,2,1)@poses[1:,:3,:3]
    angular=rotation_degrees(relative,axis,SYMMETRIES[name])/np.diff(times)
    fb=[row['fb_rotation_deg'] for row in rows[1:] if row.get('fb_rotation_deg') is not None]
    return dict(object_angular_rate_max_deg_s=float(angular.max()),
                object_fb_rotation_median_deg=float(np.median(fb)) if fb else None)
