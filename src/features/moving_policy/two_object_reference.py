"""Apply existing M1 tracking and its unchanged final instruments to a joint candidate."""

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.provenance import write_json, sha256_file
from src.features.moving_reference import (track, camera, camera_ba, candidate_c, full_c,
    amendment, table_audit, table_tracks)
from src.features.moving_reference._object import smooth
from src.features.moving_reference._final_gates import FINAL_LIMITS, compose_final
from src.features.moving_reference.gates import evaluate, _amended_check
from .two_object_task import assets
from ._two_object_symmetry import resolve_facing,mesh_axis,SYMMETRIES
from . import two_object_rotation as symmetry_rotation
from ._two_object_camera import compose_joint,provisional_camera,REGIONS
from .two_object_facing import reference_tracks


def stopped(episode, directory, reason, measured):
    """Retain measured failures and mark all later gates unmeasured."""
    gates = {name:dict(value=None,comparison=op,threshold=limit,passed=False,status='not_run_after_first_failure')
             for name,(op,limit) in FINAL_LIMITS.items()}
    gates.update(measured)
    result = dict(episode=episode,passed=False,reason=reason,gates=gates,labels_read=False,scan_reads=[])
    write_json(directory/'gates-final.json',result)
    return result


def with_companion_occlusion(selected, other_directory, output):
    """Exclude B from the static camera background and treat it as an occluder."""
    target = output/'occlusion-masks'
    target.mkdir()
    paths = []
    for index, path in enumerate(selected['mask_files']):
        with np.load(path,allow_pickle=False) as saved:
            arrays = {k:saved[k] for k in saved.files}
        with np.load(other_directory/'frames'/f'frame_{index:06d}.npz',allow_pickle=False) as saved:
            other = saved['masks'][0]
        arrays['masks'] = np.concatenate([arrays['masks'],other[None]],axis=0)
        destination = target/f'frame_{index:06d}.npz'
        np.savez_compressed(destination,**arrays)
        paths.append(destination)
    return dict(selected,mask_files=paths)



def run_audit(operation,args,directory,record_name):
    """Keep completed measurements if an optional plot rejects a missing normal."""
    try:
        operation(*args)
    except TypeError as error:
        record_path=directory/record_name
        if 'NoneType' not in str(error) or not record_path.exists():
            raise
        record=json.loads(record_path.read_text())
        for name,digest in record['preserved_inputs'].items():
            if sha256_file(directory/name)!=digest:
                raise ValueError('Audit input changed before its plot failed') from error
        write_json(directory/(record_path.stem+'-plot-warning.json'),
                   dict(error=str(error),measurements_preserved=True,record=record_name,
                        gate_changes=False,measurement_retry=False))


def run(episode, selected, directory, other_directory, camera_source=None):
    """Run a single reference attempt and stop at its first measured failure."""
    directory.mkdir(parents=True,exist_ok=False)
    # The existing audit handlers take a root and append this fixed episode layout.
    if directory.name != 'candidate-c' or directory.parent.name != f'ep{episode}':
        raise ValueError('Reference directory must follow the existing M1 audit layout')
    original_root = track.OUTPUT
    previous_audit_seal, previous_track_seal = table_audit.SEALED, table_tracks.SEALED
    track.OUTPUT = directory.parents[1]
    try:
        selected = with_companion_occlusion(selected,other_directory,directory)
        inventory = [*selected['paths'],selected['calibration'],selected['masks']/'request.json',
                     selected['masks']/'summary.json',*selected['mask_files']]
        selected['hashes'] = {str(p.relative_to(track.ROOT)):sha256_file(p) for p in inventory}
        write_json(directory/'request-camera.json',dict(inputs=selected['hashes'],mask_directory=str(selected['masks'].relative_to(track.ROOT)),
                   method='Existing M1 stereo KLT, metric BA, object keyframes and final audits; own procedural body frame'))
        write_json(directory/'sources.json', {str(p.relative_to(track.ROOT)):sha256_file(p)
            for p in [Path(__file__),*[Path(__file__).with_name(name+'.py') for name in ('_two_object_symmetry','_two_object_rotation','two_object_rotation','_two_object_camera','_two_object_facing','two_object_facing','_two_object_table','two_object_table')], *track.FEATURE.glob('*.py')]})
        candidate_c.prepare_frame0(selected,directory)
        with np.load(directory/'frame0.npz',allow_pickle=False) as frame:
            if not (frame['visible'] & frame['depth_valid']).any():
                return stopped(episode,directory,
                    'No frame-zero stereo object cloud after companion occlusion; M1 initialization is undefined', {})
        if camera_source is not None:
            previous=json.loads((camera_source/'request-camera.json').read_text())
            if previous['mask_directory']!=str(selected['masks'].relative_to(track.ROOT)):
                raise ValueError('Reused camera solve has a different active mask source')
            for name,digest in previous['inputs'].items():
                if sha256_file(track.ROOT/name)!=digest:raise ValueError('Reused camera input changed')
            names=['camera-ba.npz','klt-observations.npz','bundle-adjustment.json','camera-diagnostics.json']
            for name in names:shutil.copyfile(camera_source/name,directory/name)
            with np.load(directory/'camera-ba.npz') as saved:cameras=saved['world_T_camera'].copy()
            with np.load(directory/'klt-observations.npz') as saved:data={key:saved[key].copy() for key in saved.files}
            rows=json.loads((directory/'camera-diagnostics.json').read_text())
            write_json(directory/'camera-solve-reused.json',dict(source=str(camera_source.relative_to(track.ROOT)),
                       files={name:sha256_file(directory/name) for name in names},inputs_verified=True))
            frames=camera_source/'frames'
            if len(list(frames.glob('frame_*.npz')))==selected['job']['frame_count']:
                (directory/'frames').symlink_to(frames.resolve(),target_is_directory=True)
        else:
            data = camera_ba.observations(selected,directory)
            cameras, ids, landmarks, report = camera_ba.bundle_adjust(data)
            rows = camera_ba.heldout_rows(data,cameras)
            np.savez_compressed(directory/'camera-ba.npz',world_T_camera=cameras,landmark_ids=ids,landmarks=landmarks)
            write_json(directory/'bundle-adjustment.json',report)
            write_json(directory/'camera-diagnostics.json',rows)
        residual = [v for row in rows for v in row['heldout_reprojection_px640']]
        measured = dict(
            camera_heldout_reprojection_median_px640=_amended_check(float(np.median(residual)) if residual else None,.5),
            camera_heldout_reprojection_p95_px640=_amended_check(float(np.percentile(residual,95)) if residual else None,2),
            camera_heldout_tracks_min=_amended_check(min(r['heldout_count'] for r in rows),30,'ge'),
            camera_heldout_regions_min=_amended_check(min(r['heldout_regions'] for r in rows),4,'ge'),
            camera_coverage_fraction=_amended_check(float(np.mean([r['observed'] for r in rows])),.98,'ge'))
        if not provisional_camera(measured):
            return stopped(episode,directory,'camera BA failed unchanged M1 gate',measured)
        if not measured[REGIONS]['passed']:
            measured[REGIONS]['status']='not_run_after_first_failure'
            measured[REGIONS]['note']='Three regions provisional; final world-residual prerequisite pending'
            write_json(directory/'camera-regions-provisional.json',dict(measured=measured,world_residual_pending=True,passed=False))
        mesh_output = directory/'asset'
        mesh_output.mkdir()
        names, binding, meshes = assets(episode,mesh_output)
        role = json.loads((directory.parent/'roles.json').read_text())
        mesh = meshes[role['active_slot']]
        maps,k,q,_ = camera.rectification(json.loads(selected['calibration'].read_text()))
        name=names[role['active_slot']]
        axis=mesh_axis(mesh.face_normals,mesh.area_faces) if name in SYMMETRIES else np.array([0.,0.,1.])
        tracked=reference_tracks(selected,directory,mesh,name,maps,k,q,cameras,rows)
        initialization=tracked['initialization']
        write_json(directory/'initialization.json',initialization)
        if not initialization['opposite_resolved']:
            measured['object_reference_valid_fraction'] = _amended_check(0.,.9,'ge')
            return stopped(episode,directory,'Initial facing remains unresolved under declared, own-mesh and whole-clip rules',measured)
        cameras=tracked['cameras'];relative=tracked['relative'];object_rows=tracked['object_rows'];usable=tracked['usable']
        filtered=tracked['filtered'];valid=tracked['valid'];scores=tracked['scores'];table=tracked['table']
        rows = camera_ba.heldout_rows(data,cameras)
        np.savez_compressed(directory/'tracking.npz',world_T_camera=cameras,world_T_asset=filtered,
                            camera_T_asset=relative,valid=valid,intrinsics=k)
        for index,detail in enumerate(object_rows):
            frame = full_c.read_frame(directory,index)
            detail['stereo_supported_pixels'] = int((frame['visible']&frame['depth_valid']).sum())
        times = np.asarray(selected['job']['timestamps'],float)
        base = float(camera.apply(filtered[0],mesh.vertices)[:,2].min())
        source = evaluate(table,rows,object_rows,times,filtered,valid,scores,usable,base,
                          dict(status='Stage 0 follows reference admission',maximum_translation_m=None,maximum_rotation_deg=None))
        source,symmetry_report=symmetry_rotation.amend(source,filtered,times,object_rows,name,axis)
        write_json(directory/'rotation-symmetry.json',symmetry_report)
        source.update(episode=episode,label_free=True)
        write_json(directory/'gates.json',source)
        write_json(directory/'object-diagnostics.json',object_rows)
        quaternion = Rotation.from_matrix(filtered[:,:3,:3]).as_quat()
        for index in range(1,len(quaternion)):
            if quaternion[index]@quaternion[index-1]<0:
                quaternion[index] *= -1
        np.savez_compressed(directory/'reference.npz',position=filtered[:,:3,3],quaternion_xyzw=quaternion,
            frame_index=np.arange(len(times)),time=times,valid=valid,world_T_object=filtered,
            pose=np.c_[filtered[:,:3,3],quaternion],gates_passed=np.asarray(False))
        amendment.run(episode)
        sealed=json.loads((original_root/'ep39/sealed/sealed.json').read_text())
        interim=compose_joint(json.loads((directory/'gates-amended.json').read_text()),sealed)
        waived={'mask_usable_fraction','mask_longest_unusable_interval_frames'}
        stable={name:g for name,g in interim['gates'].items()
                if name not in waived and not name.startswith(('table_offset_','table_landmark_','table_drift_','table_late_'))}
        failed=next((name for name,g in stable.items() if not g['passed']),None)
        if failed:
            g=stable[failed]
            reason=f'{failed}: {g["value"]} {g["comparison"]} {g["threshold"]}'
            return stopped(episode,directory,reason,stable)
        reference_hash = sha256_file(directory/'reference.npz')
        table_audit.SEALED = table_tracks.SEALED = {episode:reference_hash}
        run_audit(table_audit.run,(episode,),directory,'gates-amended-2.json')
        run_audit(table_tracks.run,(episode,table_tracks.sources()),directory,'gates-amended-3.json')
        sealed = json.loads((original_root/'ep39/sealed/sealed.json').read_text())
        final = compose_joint(json.loads((directory/'gates-amended-3.json').read_text()),sealed)
        final.update(episode=episode,reference_sha256=reference_hash,
                     reason='all unchanged M1 gates passed' if final['passed'] else 'unchanged M1 reference gate failed',
                     labels_read=False,scan_reads=[],asset=binding['objects'][role['active_slot']],
                     evidence={p.name:sha256_file(p) for p in directory.glob('*.json')})
        write_json(directory/'gates-final.json',final)
        return final
    finally:
        track.OUTPUT = original_root
        table_audit.SEALED, table_tracks.SEALED = previous_audit_seal, previous_track_seal
