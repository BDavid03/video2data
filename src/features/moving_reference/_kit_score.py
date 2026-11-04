"""Pure array conversion and temporal diagnostics for local kit scoring."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._depth_track import m1_errors,symmetry_degrees


def pose_in_world_frame(pose, shared_T_world):
    result=pose.copy()
    result[...,:3]=pose[...,:3]@shared_T_world[:3,:3].T+shared_T_world[:3,3]
    result[...,3:]=(Rotation.from_matrix(shared_T_world[:3,:3])*Rotation.from_quat(pose[...,3:])).as_quat()
    return result


def pose_in_mesh_frame(pose, body_T_mesh):
    rotation=Rotation.from_quat(pose[...,3:]).as_matrix()
    result=pose.copy()
    result[...,:3]+=np.einsum('...ij,j->...i',rotation,body_T_mesh[:3,3])
    result[...,3:]=Rotation.from_matrix(rotation@body_T_mesh[:3,:3]).as_quat()
    return result


def first_run(mask,length=1):
    hits=np.flatnonzero(np.convolve(np.asarray(mask,int),np.ones(length,int),mode='valid')==length)
    return int(hits[0]) if len(hits) else None


def timing(values):
    values=np.asarray(values,float);dt=np.diff(values)
    return dict(first=float(values[0]),last=float(values[-1]),duration_s=float(values[-1]-values[0]),
        mean_rate_hz=float((len(values)-1)/(values[-1]-values[0])),
        interval_median_s=float(np.median(dt)),interval_min_s=float(dt.min()),interval_max_s=float(dt.max()))


def temporal_diagnostic(prediction,truth,valid,video_time,label_time,axis,symmetry):
    errors,angle=m1_errors(prediction[:,:3],prediction[:,3:],truth[:,:3],truth[:,3:],axis,symmetry)
    displacement=np.linalg.norm(truth[:,:3]-truth[0,:3],axis=1)
    relative=(Rotation.from_quat(truth[0,3:]).inv()*Rotation.from_quat(truth[:,3:])).as_matrix()
    rotation=symmetry_degrees(relative,axis,symmetry)
    turn=Rotation.from_quat(truth[0,3:])*Rotation.from_quat(prediction[0,3:]).inv()
    aligned=turn.apply(prediction[:,:3]-prediction[0,:3])+truth[0,:3]
    vt=np.asarray(video_time)-video_time[0];lt=np.asarray(label_time)-label_time[0]
    inside=(vt>=lt[0])&(vt<=lt[-1])
    interpolated=np.column_stack([np.interp(vt,lt,truth[:,i]) for i in range(3)])
    timed=np.linalg.norm(aligned-interpolated,axis=1)
    good=(errors<.03)&(angle<30)
    result=dict(first_error_gt_3cm=first_run(errors>.03),first_sustained_error_gt_3cm=first_run(errors>.03,3),
        mocap_translation_start=first_run(displacement>.005,3),mocap_rotation_start=first_run(rotation>5,3),
        frame_index_median_error_cm=float(np.median(errors)*100),
        fixed_timestamp_interpolation_median_cm=float(np.median(timed[inside])*100),
        fixed_timestamp_overlap_frames=int(inside.sum()),success_count=int(good.sum()),
        successful_C_valid_frames=int((good&valid).sum()),successful_C_invalid_frames=int((good&~valid).sum()),
        successful_frames=np.flatnonzero(good).tolist())
    return result,dict(error_m=errors,angle_deg=angle,mocap_displacement_m=displacement,mocap_rotation_deg=rotation,
        fixed_timestamp_error_m=timed,success=good,aligned_position=aligned)
