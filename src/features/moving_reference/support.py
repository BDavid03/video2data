"""Prepare and run a CPU evaluator check of the reconstructed frame-0 support."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from src.provenance import sha256_file, write_json

ROOT=Path(__file__).resolve().parents[3]
FC=ROOT/'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord'
TEMPLATE=ROOT/'data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave'
PYTHON=ROOT/'data/.runtime/flashchord/venv/bin/python'


def prepare(directory, world_T_asset, mesh):
    import pyarrow as pa
    import pyarrow.parquet as pq
    template,=TEMPLATE.glob('*.parquet')
    table=pq.ParquetFile(template).read()
    original=table.to_pylist()[0]
    count=21
    row={key:np.repeat(np.asarray(value)[:1],count,axis=0).tolist()
         if isinstance(value,list) and len(value)==390 else value for key,value in original.items()}
    position=world_T_asset[:3,3]
    rotation=Rotation.from_matrix(world_T_asset[:3,:3])
    xyzw=rotation.as_quat(); wxyz=xyzw[[3,0,1,2]]
    mesh_path=directory.parent/'canonical-asset.obj'
    urdf=directory/'mini_dust_pan.urdf'
    dims=np.ptp(mesh.vertices,axis=0)
    inertia=.1/12*(np.sum(dims**2)-dims**2)
    urdf.write_text(f'''<robot name="mini_dust_pan"><link name="mini_dust_pan">
<inertial><mass value="0.1"/><origin xyz="0 0 0"/><inertia ixx="{inertia[0]}" ixy="0" ixz="0" iyy="{inertia[1]}" iyz="0" izz="{inertia[2]}"/></inertial>
<visual><geometry><mesh filename="{mesh_path}"/></geometry></visual>
<collision><geometry><mesh filename="{mesh_path}"/></geometry></collision>
</link></robot>''')
    row.update(raw_motion_file='M1 frame-0 stereo pose; static support diagnostic; no episode labels',fps=20.,
               object_name='mini_dust_pan',safe_object_name='mini_dust_pan',object_body_names=['mini_dust_pan'],
               safe_object_body_names=['mini_dust_pan'],object_mesh_paths=[str(mesh_path)],object_urdf_paths=[str(urdf)],
               object_mesh_radius=[float(np.linalg.norm(mesh.vertices,axis=1).max())],
               object_root_position=np.tile(position,(count,1)).tolist(),
               object_root_axis_angle=np.tile(rotation.as_rotvec(),(count,1)).tolist(),
               object_body_position=np.tile(position,(count,1,1)).tolist(),
               object_body_wxyz=np.tile(wxyz,(count,1,1)).tolist(),object_articulation=np.zeros(count).tolist())
    # The shipped hand template supplies only evaluator robot joint structure.
    for side in ('left','right'):
        offset=np.array([5.,0.,5.])
        for key in (f'robot_{side}_wrist_position',f'mano_{side}_trans'):
            row[key]=(np.asarray(row[key])+offset).tolist()
        for key in (f'robot_{side}_frames',f'mano_{side}_joints'):
            value=np.asarray(row[key]); value[...,:3]+=offset
            row[key]=value.tolist()
        for kind in ('link','object'):
            for field in ('positions','normals'):
                key=f'mano_{side}_{kind}_contact_{field}'
                row[key]=np.zeros_like(row[key]).tolist()
        key=f'mano_{side}_object_contact_part_ids'
        row[key]=np.zeros_like(row[key]).tolist()
    task=directory/'motion/sequence_id=video_reference/robot_name=sharpa_wave/task.parquet'
    task.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row],schema=table.schema),task)
    support=directory/'reconstructed_stage/video_reference_sharpa_wave_support.usda'
    support.parent.mkdir(parents=True,exist_ok=True)
    support.write_text('''#usda 1.0
(defaultPrim = "support_surfaces"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "support_surfaces" {
 def Cylinder "table" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
  uniform token axis = "Z"
  double height = 0.01
  double radius = 2.0
  double3 xformOp:translate = (0, 0, -0.005)
  uniform token[] xformOpOrder = ["xformOp:translate"]
 }
}
''')
    return task,dict(template=dict(path=str(template.relative_to(ROOT)),sha256=sha256_file(template)),
                     task_sha256=sha256_file(task),support_sha256=sha256_file(support),urdf_sha256=sha256_file(urdf),
                     geometry_sha256=sha256_file(mesh_path),horizon_seconds=1.,steps=20,device='cpu',
                     hands='Shipped hand template repeated at frame 0 and translated by [5,0,5] m; no contacts',
                     table='Level plane z=0 represented by a 2 m radius, 1 cm thick cylinder; geometric extent is a support-check fixture, not a reconstruction')


def run(directory,world_T_asset,mesh):
    directory=directory/'support'
    directory.mkdir(exist_ok=True)
    task,request=prepare(directory,world_T_asset,mesh)
    environment={**os.environ,'PYTHONPATH':os.pathsep.join((str(ROOT),str(FC/'src'))),
                 'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu',
                 'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','WANDB_MODE':'disabled',
                 'PXR_WORK_THREAD_LIMIT':'1','PYTHONUNBUFFERED':'1'}
    for key in ('WARP_CACHE_PATH','FLASH_CHORD_CACHE_DIR','JAX_COMPILATION_CACHE_DIR','XDG_CACHE_HOME','MPLCONFIGDIR','TMPDIR'):
        path=directory/'cache'/key.lower(); path.mkdir(parents=True,exist_ok=True)
        environment[key]=str(path)
    command=[str(PYTHON),'-B','-m','src.features.moving_reference.support','--task',str(task),'--output',str(directory)]
    write_json(directory/'request.json',dict(**request,argv=command))
    with (directory/'worker.log').open('w') as log:
        process=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    result=directory/'support.json'
    if result.exists():
        return json.loads(result.read_text())
    report=dict(status='unmeasured',maximum_translation_m=None,maximum_rotation_deg=None,
                reason=f'CPU evaluator worker exit {process.returncode}; see worker.log')
    write_json(result,report)
    return report


def worker(task,output):
    result=dict(status='unmeasured',maximum_translation_m=None,maximum_rotation_deg=None,device='cpu')
    try:
        import warp as wp
        wp.config.enable_cuda=False
        wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH']
        wp.set_device('cpu')
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from flash_chord.configuration import instantiate_typed
        from flash_chord.embodiments.base import Embodiment
        from flash_chord.envs.base import BaseEnv
        from flash_chord.envs.rl import RLEnv,RLEnvConfig
        from flash_chord.evaluation.harness import apply_evaluation_composition
        from flash_chord.scene.setup import setup_scene
        from flash_chord.scene.collision import CollisionPolicy
        import flash_chord
        base=Path(flash_chord.__file__).parent
        with initialize_config_dir(config_dir=str(base/'configs'),version_base=None):
            cfg=compose(config_name='train_flash_sac',overrides=[f'task.parquet={json.dumps(str(task))}',
                        'sim.fps=20','task.motion_speed=1.0','scene.world_count=1'])
        config=OmegaConf.to_container(cfg,resolve=True)
        apply_evaluation_composition(config,world_count=1,reset_mode='explicit')
        write_json(output/'evaluator-config.json',config)
        cfg=OmegaConf.create(config)
        setup=setup_scene(parquet=str(task),control_fps=20.,motion_speed=1.,
                          embodiment=instantiate_typed(cfg.embodiment,Embodiment),
                          collision=instantiate_typed(cfg.collision,CollisionPolicy),world_count=1,
                          include_support=True,decompose_objects=True)
        env=RLEnv(setup.scene,setup.reference,config=instantiate_typed(cfg.env,RLEnvConfig))
        env.reset(frame_id=0)
        body=setup.scene.objects[0].root.body_id
        first=env.state_0.body_q.numpy()[body].copy()
        action=wp.zeros(env.action.action_dim,dtype=wp.float32,device='cpu')
        poses=[first.tolist()]
        for step in range(20):
            env.timestep.assign(np.asarray([step],np.int32))
            BaseEnv.step(env,action)
            poses.append(env.state_0.body_q.numpy()[body].tolist())
        poses=np.asarray(poses)
        translation=np.linalg.norm(poses[:,:3]-first[:3],axis=1)
        angle=np.rad2deg((Rotation.from_quat(poses[:,3:])*Rotation.from_quat(first[3:]).inv()).magnitude())
        result.update(status='measured',maximum_translation_m=float(translation.max()),maximum_rotation_deg=float(angle.max()),
                      pose_xyzw=poses.tolist(),steps_completed=20,horizon_seconds=1.,model_device=str(env.model.device),
                      voc_scale=config['env']['voc_scale'],no_resets=True,zero_actions=True)
    except Exception as error:
        import traceback
        result.update(reason=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    write_json(output/'support.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    worker(arguments.task,arguments.output)
