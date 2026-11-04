"""Reuse the frozen M2 checks and add the resting object's joint-scene support check."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

from . import handler as h
from .two_object_task import build as build_joint
from .geometry import matrix,motion_summary,cylinder_support,orientation_errors


def worker(operation,task,output):
    command=[str(h.PYTHON),'-B','-m','src.features.moving_policy.two_object_cpu',operation,
             '--task',str(task),'--output',str(output)]
    with (output/f'joint-{operation}.log').open('x') as log:
        subprocess.run(command,cwd=h.ROOT,env=h.environment(),stdout=log,stderr=subprocess.STDOUT,check=True)


def single_active(row,slot):
    row=dict(row)
    for field in ('object_body_names','safe_object_body_names','object_mesh_paths','object_urdf_paths','object_mesh_radius'):
        row[field]=[row[field][slot]]
    row['object_name']=row['safe_object_name']=row['object_body_names'][0]
    for field in ('object_body_position','object_body_wxyz'):
        row[field]=np.asarray(row[field])[:,slot:slot+1].tolist()
    row['object_root_position']=np.asarray(row['object_body_position'])[:,0].tolist()
    row['object_root_axis_angle']=__import__('scipy').spatial.transform.Rotation.from_matrix(
        matrix(np.asarray(row['object_body_wxyz'])[:,0])).as_rotvec().tolist()
    for side in ('left','right'):
        ids=np.asarray(row[f'mano_{side}_object_contact_part_ids'])
        row[f'mano_{side}_object_contact_part_ids']=np.where(ids>0,1,0).tolist()
    return row


def prepare(episode,active,reference,output):
    binding=build_joint(episode,active,reference,output)
    with np.load(reference) as source:
        times=source['time'].copy()
    template_file=next(h.TEMPLATE.glob('*.parquet'))
    template=pq.ParquetFile(template_file).read().to_pylist()[0]
    candidates=[]
    for candidate in binding['candidates']:
        task=h.ROOT/candidate['task']
        table=pq.ParquetFile(task).read();row=table.to_pylist()[0]
        single=single_active(row,active)
        directory=output/candidate['pair']/'active-only'
        directory.mkdir()
        h.OUTPUT=directory
        h.TASK=directory/f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
        h.TASK.parent.mkdir(parents=True)
        single_table=pa.Table.from_pylist([single],schema=table.schema)
        pq.write_table(single_table,h.TASK)
        shutil.copyfile(row['object_mesh_paths'][active],directory/'mini_dust_pan.obj')
        mesh=trimesh.load(row['object_mesh_paths'][active],force='mesh',process=False)
        poses=np.c_[np.asarray(single['object_body_position'])[:,0],np.asarray(single['object_body_wxyz'])[:,0]]
        build=dict(episode=episode,frames=len(times),fps=candidate['fps'],acquisition_end_frame=candidate['acquisition_end_frame'],
            bounds=mesh.bounds.tolist(),motion=motion_summary(poses[:,:3],times),task=str(h.TASK.relative_to(h.ROOT)))
        h.write(directory/'build.json',build)
        h.cpu('inspect',h.TASK)
        checks=h.geometry_checks(template,single_table,poses,times,build,body_transform=np.eye(4))
        with np.load(directory/'task-fk.npz') as fk:
            angular=orientation_errors(fk['quaternions'][:,0,[1,2,3,0]],poses[:,3:])
        checks['6_truth']['passed'] &= bool(angular.max()<=1e-4)
        checks['6_truth']['maximum_orientation_error_deg']=float(angular.max())
        eligible=checks['2_closure']['passed'] and checks['3_penetration']['passed']
        candidates.append(dict(**candidate,checks=checks,eligible=eligible,
            peak_wrist_speed_m_s=max(checks['4a_finger_limits']['maximum_wrist_speed_m_s'].values()),
            active_directory=str(directory.relative_to(h.ROOT))))
    eligible=[c for c in candidates if c['eligible']]
    chosen=min(eligible or candidates,key=lambda c:(c['peak_wrist_speed_m_s'],c['pair']))
    checks=chosen['checks']
    failed=next((n for n,c in checks.items() if not c['passed']),None)
    if not failed:
        h.OUTPUT=h.ROOT/chosen['active_directory']
        h.TASK=h.OUTPUT/f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
        checks['4b_reachability']=h.amended_worker('reachability')
        failed=None if checks['4b_reachability']['passed'] else '4b_reachability'
    if not failed:
        h.OUTPUT=output
        h.TASK=output/f'motion/sequence_id=episode_{episode:06d}/robot_name=sharpa_wave/task.parquet'
        h.TASK.parent.mkdir(parents=True)
        shutil.copyfile(h.ROOT/chosen['task'],h.TASK)
        build=dict(episode=episode,frames=chosen['frames'],fps=chosen['fps'],acquisition_end_frame=chosen['acquisition_end_frame'],
                   task=str(h.TASK.relative_to(h.ROOT)),active_slot=active,resting_slot=1-active)
        h.write(output/'build.json',build)
        with np.load(output/'reference-task.npz') as saved:reference_arrays={key:saved[key].copy() for key in saved.files}
        chosen_row=pq.ParquetFile(h.TASK).read().to_pylist()[0]
        reference_arrays['pose']=np.concatenate([np.asarray(chosen_row['object_body_position']),np.asarray(chosen_row['object_body_wxyz'])[...,[1,2,3,0]]],axis=-1)
        np.savez_compressed(output/'reference-task.npz',**reference_arrays)
        worker('inspect',h.TASK,output)
        info=json.loads((output/'joint-fk.json').read_text())
        support=[]
        with np.load(output/'joint-fk.npz') as fk:
            positions=np.asarray(pq.ParquetFile(h.TASK).read().to_pylist()[0]['object_body_position'])
            quaternions=np.asarray(pq.ParquetFile(h.TASK).read().to_pylist()[0]['object_body_wxyz'])
            for slot,object_info in enumerate(info['objects']):
                pieces=[fk[f'object_{slot}_piece_{i}'] for i in range(object_info['pieces'])]
                center,radius,top=cylinder_support(pieces,positions[0,slot],matrix(quaternions[0,slot]))
                support.append((center,radius,top))
            truth=float(np.abs(fk['positions']-positions).max())
            angles=orientation_errors(fk['quaternions'].reshape(-1,4)[:,[1,2,3,0]],quaternions.reshape(-1,4))
        checks['6_joint_truth']=dict(passed=truth<=1e-6 and float(angles.max())<=1e-4,
                                    maximum_translation_error_m=truth,maximum_orientation_error_deg=float(angles.max()))
        target=output/f'reconstructed_stage/episode_{episode:06d}_sharpa_wave_support.usda'
        target.parent.mkdir()
        cylinders=[]
        for slot,(center,radius,top) in enumerate(support):
            cylinders.append(f' def Cylinder "table_{slot}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{\n'
                f'  uniform token axis = "Z"\n  double height = 0.01\n  double radius = {radius}\n'
                f'  double3 xformOp:translate = ({center[0]}, {center[1]}, {top-.005})\n'
                '  uniform token[] xformOpOrder = ["xformOp:translate"]\n }\n')
        target.write_text('#usda 1.0\n(defaultPrim = "support_surfaces"\n metersPerUnit = 1\n upAxis = "Z")\n'
                          'def Xform "support_surfaces" {\n'+''.join(cylinders)+'}\n')
        worker('support',h.TASK,output)
        checks['5_support']=json.loads((output/'joint-support.json').read_text())
    failed=next((n for n,c in checks.items() if not c['passed']),None)
    status='PASS' if not failed else 'STOP'
    stage=dict(status=status,episode=episode,checks=checks,selected_grasp=chosen['pair'],grasp_candidates=candidates,
        binding=binding,active_slot=active,reason='PASS' if not failed else f'{failed}: {checks[failed]}',
        artifacts=h.input_inventory([p for p in output.rglob('*') if p.is_file() and 'cache' not in p.parts]),
        gpu_used=False,labels_read=False,scan_reads=[])
    h.write(output/'stage0.json',stage)
    return stage
