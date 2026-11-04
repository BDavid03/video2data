"""Bind a completed joint policy to its reference, physics, export and report evidence."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import trimesh

from src.provenance import sha256_file
from src.features.pose_insurance import pack_moving as m
from src.features.pose_insurance._moving import tracking_metrics, pose_matrices, reprojection_summary
from src.features.moving_reference._final_gates import FINAL_LIMITS, compose_final
from src.features.moving_reference import camera, track
from src.features.moving_reference._object import render, iou
from ._two_object import resting_metrics
from ._two_object_symmetry import resolve_facing
from .two_object_rotation import mesh_axes
from ._two_object_rotation import tracking as symmetry_tracking,resting as symmetry_resting
from . import two_object_screen as screen
from . import handler as h
from .two_object_facing import verify as verify_facing
from ._two_object_camera import compose_joint,REGIONS
from .two_object_contract import require_stage0,require_resting_start,checkpoint_recipe,evaluator_record
from ._two_object_contract import require_source_reference

REPORT_SHARED = [
 "When the first frame cannot tell an object's facing, both facings are tracked and the one that fits the whole clip better is kept.",
 "The table-flatness check uses three table patches when four are not visible.",
 "When an object's own mesh is unchanged by the 180-degree turn, either facing gives the same rollout, so the tie is resolved.",
 "The camera check accepts three image regions when held-out reprojection accuracy passes.",
 'When frame-zero boxes overlap, joint detection searches every fifth frame and seeds bidirectional segmentation at the first separated pair.',
 "Rotation checks use swing-twist decomposition modulo the organisers' declared cup and lid symmetry.",
 "For cups and the pot lid, a tie between opposite facings is resolved by the organisers' declared symmetry.",
 'Both objects are detected jointly as distinct instances (box IoU below 0.5) before segmentation.',
 'Each added two-object episode exports both objects on the full official grid from one learned checkpoint and one unedited official evaluator rollout.',
 'Training uses FlashSAC, 1024 worlds, seed 0, update ratio 2, 66763776 transitions, the episode-38 curriculum, and the scheduled final checkpoint.',
 'The mask-usability gate (usable fraction and longest unusable gap) was waived; the stereo-support gate was retained.',
 'Object B is an unactuated free rigid body using our own procedural reconstruction, initialized at its v5 static pose; its start error and full-rollout drift must pass the 1 cm and 2 degree physical limits.',
 'These references are video-derived, use no scored labels or scan geometry, and have no labelled validation for these scored clips; policy self-tracking does not establish reference accuracy.']


def disclosure(path,episode,names,active):
    path=m.path(path)
    if path!=h.ROOT/'docs/report/track3/main.tex':
        raise ValueError('Joint admission must bind the promoted report')
    text=' '.join(line for line in path.read_text().splitlines() if not line.lstrip().startswith('%'))
    text=re.sub(r'\s+',' ',text.replace('\\_','_'))
    required=[*REPORT_SHARED,f'Episode {episode} moves {names[active]} while {names[1-active]} rests.']
    missing=[sentence for sentence in required if sentence not in text]
    if missing:
        raise ValueError('Report disclosure missing: '+' | '.join(missing))
    return dict(**m.entry(path),required_strings=required)


def reference_gates(directory,episode):
    decision=h.ROOT/'data/outputs/moving_policy/two-object/fraction-waiver-decision.json'
    waived=json.loads(decision.read_text())['final_M1_waived'] if decision.exists() else []
    if any(name not in ('mask_usable_fraction','mask_longest_unusable_interval_frames') for name in waived):
        raise ValueError('Only the two explicitly authorized usability measures may be waived')
    final_path=directory/'campaign-gates-final.json' if waived else directory/'gates-final.json'
    final=m.read(final_path)
    source=m.read(directory/'gates-amended-3.json')
    seal=h.ROOT/'data/outputs/moving_reference/ep39/sealed/sealed.json'
    expected=compose_joint(source,m.read(seal))
    if final['gates']!=expected['gates'] or final.get('episode')!=episode:
        raise ValueError('Final reference gates differ from unchanged M1 instruments')
    for name,(comparison,limit) in FINAL_LIMITS.items():
        g=final['gates'][name]
        expected_limit=expected['gates'][name]['threshold'] if name==REGIONS else limit
        if g['comparison']!=comparison or g['threshold']!=expected_limit:
            raise ValueError(f'Reference threshold changed: {name}')
        if name not in waived and not g['passed']:
            raise ValueError(f'Reference failed {name}')
    if final.get('passed') is not True or final.get('reference_sha256')!=sha256_file(directory/'reference.npz'):
        raise ValueError('Reference is not a hash-bound passing result')
    for name,digest in source['preserved_inputs'].items():
        if sha256_file(directory/name)!=digest:
            raise ValueError('Preserved M1 reference instrument changed')
    return dict(gates=m.entry(final_path),source=m.entry(directory/'reference.npz'),sealed=m.entry(seal),
                waived_limits=waived,waiver_decision=m.entry(decision) if decision.exists() else None,
                camera_rule=m.entry(h.ROOT/'src/features/moving_policy/_two_object_camera.py'),
                joint_protocol=m.entry(h.ROOT/'docs/chain/moving-two-object-protocol.md'))


def inspect(directory,episode,report='docs/report/track3/main.tex',require_disclosure=True):
    review=m.read(h.ROOT/'data/outputs/moving_policy/two-object/review-fixes-status.json')
    if review.get('passed') is not True:
        raise PermissionError('Joint review fixes have not passed or loaded; admission/training held')
    directory=m.path(directory)
    stage=m.read(directory/'stage0.json')
    require_stage0(stage,directory,episode)
    binding=stage['binding'];active=stage['active_slot'];resting=1-active
    names=[o['name'] for o in binding['objects']]
    roster,objects=m.official_roster()
    if episode not in roster or names!=objects[episode].object_names or len(names)!=2:
        raise ValueError('Joint official roster or slot order differs')
    refdir=m.path(binding['reference']['path']).parent
    if sha256_file(refdir/'reference.npz')!=binding['reference']['sha256']:
        raise ValueError('Reference changed after joint construction')
    ref=reference_gates(refdir,episode)
    task=directory/f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
    row=pq.ParquetFile(task).read().to_pylist()[0]
    if row['object_body_names']!=names or len(row['object_urdf_paths'])!=2:
        raise ValueError('Task must contain both official objects')
    for slot,obj in enumerate(binding['objects']):
        if (sha256_file(row['object_mesh_paths'][slot])!=obj['mesh_sha256'] or
                sha256_file(row['object_urdf_paths'][slot])!=obj['urdf_sha256']):
            raise ValueError('Actual simulated asset changed')
    positions=np.asarray(row['object_body_position']);quaternions=np.asarray(row['object_body_wxyz'])
    reference=np.concatenate([positions,quaternions[...,[1,2,3,0]]],axis=-1)
    start=np.asarray(binding['objects'][resting]['pose_wxyz'])
    if not (np.array_equal(positions[:,resting],np.tile(start[:3],(len(positions),1))) and
            np.array_equal(quaternions[:,resting],np.tile(start[3:],(len(positions),1)))):
        raise ValueError('B reference does not stay at its v5 start')
    with np.load(refdir/'reference.npz') as source:
        source_errors=require_source_reference(reference,np.c_[source['position'],source['quaternion_xyzw']],
            source['frame_index'],source['time'],float(row['fps']),binding['translation_m'],active)
    packed_binding=require_resting_start(stage,task)
    checkpoint=directory/'training/policy_66763776.safetensors'
    learned=m.inspect_learning(directory,checkpoint,task)
    recipe=checkpoint_recipe(checkpoint)
    _,code=m.evaluator_inputs()
    trajectory=directory/'score-66763776/trajectory.parquet'
    frame,achieved=m.inspect_export(trajectory,sha256_file(checkpoint),episode,roster[episode],names,code)
    official_file=directory/'score-66763776/official_metrics.json'
    evaluated=evaluator_record(directory,checkpoint,trajectory,official_file,directory/'stage0.json')
    task_reference=m.reference_poses(directory/'reference-task.npz',frame,2)
    from src.features.pose_insurance._moving import translation_errors
    export_errors=translation_errors(reference,task_reference,np.zeros(3))
    if export_errors['maximum_translation_error_m']>2e-6 or export_errors['maximum_orientation_error_deg']>1e-4:
        raise ValueError('Evaluator reference columns differ from actual joint task')
    official=m.read(official_file)
    m.require_completion(official,sha256_file(checkpoint),len(positions))
    if official.get('environment_steps',0)<m.FINAL_STEPS:
        raise ValueError('Official evaluator report predates the frozen transition total')
    axes=mesh_axes(row["object_mesh_paths"])
    tracking=symmetry_tracking(achieved,reference,names,axes)
    m.require_tracking(tracking[active],f'Joint episode {episode} moving slot {active}')
    m.require_first(tracking[active])
    rest=symmetry_resting(achieved[:,resting],start,names[resting],axes[resting])
    if not rest['passed']:
        raise ValueError(f'Resting B failed physical limits: {rest}')
    lease_worker=h.ROOT/'src/features/moving_policy/two_object_train_worker.py'
    train=m.read(directory/'train.run.json')
    if (train.get('passive_body_worker_sha256')!=sha256_file(lease_worker) or
        'src.features.moving_policy.two_object_train_worker' not in train['argv'] or
        train['argv'][train['argv'].index('--active-slot')+1]!=str(active)):
        raise ValueError('Training does not bind the unactuated B controller')
    with np.load(refdir/'tracking.npz') as arrays:
        cameras=arrays['world_T_camera'].copy();valid=arrays['valid'].copy();k=arrays['intrinsics'].copy()
    mask_directory=h.ROOT/m.read(refdir/'request-camera.json')['mask_directory']
    selected=screen.selected_inputs(episode,mask_directory)
    selected['mask_files']=[refdir/'occlusion-masks'/f'frame_{i:06d}.npz' for i in range(len(positions))]
    maps,_,_,_=camera.rectification(json.loads(selected['calibration'].read_text()))
    mesh=trimesh.load(row['object_mesh_paths'][active],force='mesh',process=False)
    initialization=m.read(refdir/'initialization.json')
    verify_facing(refdir,mesh,initialization)
    if initialization.get('resolved_by')=='declared scorer symmetry':
        original=dict(initialization,opposite_resolved=initialization['original_opposite_resolved'])
        rechecked=resolve_facing(names[active],original,mesh.face_normals,mesh.area_faces)
        if (rechecked.get('resolved_by')!='declared scorer symmetry' or
                rechecked['symmetry_axis_table_angle_deg']>10. or
                not np.isclose(rechecked['symmetry_axis_table_angle_deg'],initialization['symmetry_axis_table_angle_deg'],atol=1e-7)):
            raise ValueError('Declared symmetry axis applicability changed')
    poses=pose_matrices(achieved[:,active],np.asarray(binding['translation_m']))
    values=[]
    for index in range(len(poses)):
        # The saved frame grid owns the image shape; principal points need not be centered.
        with np.load(refdir/'frames'/f'frame_{index:06d}.npz') as arrays:
            shape=arrays['gray'].shape
        obj,hands,_,_=track.masks_at(selected,index,maps,shape)
        prediction=render(mesh.vertices,mesh.faces,np.linalg.inv(cameras[index])@poses[index],k,shape)
        values.append(iou(prediction,obj,hands))
    projection=reprojection_summary(np.asarray(values),valid)
    m.require_reprojection(projection,'Joint achieved')
    result=dict(kind='moving_policy.joint_admission',schema_version=1,episode=episode,active_slot=active,
        names=names,trajectory=m.entry(trajectory),checkpoint=m.entry(checkpoint),task=m.entry(task),
        stage0=m.entry(directory/'stage0.json'),learning=learned,reference=ref,tracking=tracking[active],
        evaluator_run=evaluated,official_metrics=m.entry(official_file),resolved_recipe=recipe,source_pose_errors=source_errors,
        export_reference_errors=export_errors,reference_task=m.entry(directory/'reference-task.npz'),resting_start_source=packed_binding,
        resting=rest,achieved_reprojection=projection,policy_directory=str(directory.relative_to(h.ROOT)),
        abstract_disclosure=disclosure(report,episode,names,active) if require_disclosure else None,
        admitted=bool(require_disclosure),labels_read=False,scan_reads=[])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=int,required=True)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--abstract-disclosure',default='docs/report/track3/main.tex')
    parser.add_argument('--out',required=True)
    parser.add_argument('--physics-only',action='store_true')
    args=parser.parse_args()
    target=m.path(args.out)
    if target.exists():
        raise FileExistsError('Use fresh immutable admission evidence')
    result=inspect(args.policy,args.episode,args.abstract_disclosure,not args.physics_only)
    m.write(target,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
