"""Run the fixed S2 and thicker-collision 60-step hands-cleared CPU comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation
from src.provenance import sha256_file,write_json
from . import support

ROOT=Path(__file__).resolve().parents[3]
DIRECTORY=ROOT/'data/outputs/moving_reference/ep39/candidate-c'


def run():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    queue=[dict(name='S2',minimum_collision_thickness_m=0.),dict(name='S3',minimum_collision_thickness_m=.01)]
    write_json(DIRECTORY/'support-drop-queue.json',queue)
    for item in queue:
        target=DIRECTORY/'support-drop'/item['name'];target.mkdir(parents=True,exist_ok=True)
        if (target/'result.json').exists():
            continue
        mesh=trimesh.load(DIRECTORY/'asset/mini_dust_pan.obj',force='mesh',process=False)
        original_thickness=np.ptp(mesh.vertices[:,2])
        if item['name']=='S3':
            mesh.vertices[:,2]*=max(1.,.01/original_thickness)
        mesh.export(target/'collision.obj')
        mesh.export(target/'canonical-asset.obj')
        stage=target/'work';stage.mkdir(exist_ok=True)
        with np.load(DIRECTORY/'asset/placement.npz') as arrays:
            pose=arrays['world_T_asset'].copy()
        task,request=support.prepare(stage,pose,mesh)
        table=pq.ParquetFile(task).read();row=table.to_pylist()[0]
        row={key:np.repeat(np.asarray(value)[:1],81,axis=0).tolist() if isinstance(value,list) and len(value)==21 else value for key,value in row.items()}
        urdf=stage/'mini_dust_pan.urdf'
        text=urdf.read_text();lo=text.index('<inertial>');hi=text.index('</inertial>')+len('</inertial>')
        text=text[:lo]+'<inertial><mass value="0.3"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>'+text[hi:]
        text=text.replace(str(target/'canonical-asset.obj'),str(target/'collision.obj'));urdf.write_text(text)
        pq.write_table(pa.Table.from_pylist([row],schema=table.schema),task)
        np.savez_compressed(target/'placement.npz',world_T_asset=pose)
        env={**os.environ,'PYTHONPATH':os.pathsep.join((str(ROOT),str(support.FC/'src'))),'CUDA_VISIBLE_DEVICES':'',
             'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PXR_WORK_THREAD_LIMIT':'1'}
        for key in ('WARP_CACHE_PATH','FLASH_CHORD_CACHE_DIR','XDG_CACHE_HOME','TMPDIR','MPLCONFIGDIR'):
            cache=DIRECTORY/'support/cache'/key.lower();cache.mkdir(parents=True,exist_ok=True);env[key]=str(cache)
        command=[str(support.PYTHON),'-B','-m','src.features.moving_reference.support_drop','worker','--directory',str(target),'--task',str(task)]
        write_json(target/'request.json',dict(**item,steps=60,hands_cleared=True,argv=command,visual_unchanged=True,
                    source_sha256=sha256_file(Path(__file__)),task_sha256=sha256_file(task),collision_sha256=sha256_file(target/'collision.obj')))
        with (target/'worker.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Support worker failed: {target}')
        with (DIRECTORY/'support-drop-ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(**item,result=json.loads((target/'result.json').read_text())))+'\n')


def worker(directory,task):
    import warp as wp
    wp.config.enable_cuda=False;wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH'];wp.set_device('cpu')
    import newton
    from flash_chord.data.loader import load_reference
    from flash_chord.scene.objects import spawn_objects
    from flash_chord.envs.base import BaseEnv
    from src.features.moving_policy.geometry import cylinder_support
    from src.features.moving_policy.diagnose import compose_environment,contact_snapshot
    reference=load_reference(str(task),control_fps=20.,motion_speed=1.)
    builder=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    spawn_objects(builder,reference,decompose_collision=True)
    pieces=[]
    for i,flags in enumerate(builder.shape_flags):
        if int(flags)&int(newton.ShapeFlags.COLLIDE_SHAPES):
            points=np.asarray(builder.shape_source[i].vertices,float)*np.asarray(builder.shape_scale[i]);t=np.asarray(builder.shape_transform[i])
            pieces.append(Rotation.from_quat(t[3:]).apply(points)+t[:3])
    with np.load(directory/'placement.npz') as arrays:
        pose=arrays['world_T_asset']
    center,radius,top=cylinder_support(pieces,pose[:3,3],pose[:3,:3])
    fixture=directory/'work/reconstructed_stage/video_reference_sharpa_wave_support.usda'
    fixture.write_text(f'''#usda 1.0
(defaultPrim = "support_surfaces"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "support_surfaces" {{
 def Cylinder "table" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
  uniform token axis = "Z"
  double height = 0.01
  double radius = {radius}
  double3 xformOp:translate = ({center[0]}, {center[1]}, {top-.005})
  uniform token[] xformOpOrder = ["xformOp:translate"]
 }}
}}
''')
    setup,env,config=compose_environment(task,20.)
    body=setup.scene.objects[0].root.body_id
    first=env.state_0.body_q.numpy()[body].copy();initial=Rotation.from_quat(first[3:])
    records=[];substep_contacts=[];original=env.solver.step
    def observe(state_in,state_out,control,contacts,dt):
        original(state_in,state_out,control,contacts,dt)
        snap=contact_snapshot(env,body,len(substep_contacts),dt)
        substep_contacts.append(snap)
    env.solver.step=observe
    action=wp.zeros(env.action.action_dim,dtype=wp.float32,device='cpu')
    for step in range(61):
        substep_contacts=[]
        if step:
            env.timestep.assign(np.asarray([step-1],np.int32));BaseEnv.step(env,action)
        current=env.state_0.body_q.numpy()[body].copy();rotation=Rotation.from_quat(current[3:])
        vector=(rotation*initial.inv()).as_rotvec()
        records.append(dict(step=step,time_s=step/20,pose_xyzw=current.tolist(),translation_m=float(np.linalg.norm(current[:3]-first[:3])),
                  rotation_deg=float(np.rad2deg(np.linalg.norm(vector))),rotation_vector_world_deg=np.rad2deg(vector).tolist(),
                  tilt_deg=float(np.rad2deg(np.arccos(np.clip((rotation.apply([0,0,1]))@[0,0,1],-1,1)))),
                  yaw_delta_deg=float(np.rad2deg(vector[2])),contacts=substep_contacts))
    result=dict(status='measured',device='cpu',steps=60,reference_frames=reference.num_frames,hull_count=len(pieces),
                mass_kg=float(env.model.body_mass.numpy()[body]),inertia_diagonal_kg_m2=[.001]*3,hull_gap_m=.001,
                maximum_translation_m=max(row['translation_m'] for row in records),maximum_rotation_deg=max(row['rotation_deg'] for row in records),
                records=records,voc_scale=config['env']['voc_scale'],no_resets=True)
    result['passed']=result['maximum_translation_m']<=.01 and result['maximum_rotation_deg']<=2
    write_json(directory/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['run','worker'])
    parser.add_argument('--directory',type=Path);parser.add_argument('--task',type=Path);args=parser.parse_args()
    run() if args.operation=='run' else worker(args.directory,args.task)
