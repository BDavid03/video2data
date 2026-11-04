"""Measure imported joint geometry and both-body support with the installed evaluator physics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .cpu import initialize
from .handler import write
from ._two_object_rotation import orientation_degrees
from .two_object_rotation import mesh_axes


def inspect(task,output):
    from flash_chord.scene.setup import setup_scene
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.configuration import instantiate_typed
    from hydra import compose, initialize_config_dir
    import flash_chord
    import newton
    build=json.loads((output/'build.json').read_text())
    with initialize_config_dir(config_dir=str(Path(flash_chord.__file__).parent/'configs'),version_base=None):
        cfg=compose(config_name='train_flash_sac')
    setup=setup_scene(parquet=str(task),control_fps=build['fps'],motion_speed=1.,embodiment=SharpaHands(),
        collision=instantiate_typed(cfg.collision,CollisionPolicy),world_count=1,include_support=False,decompose_objects=True)
    model=setup.scene.model
    transforms=model.shape_transform.numpy()
    scales=model.shape_scale.numpy()
    arrays={};objects=[]
    body_poses=model.body_q.numpy()
    for slot,binding in enumerate(setup.scene.objects):
        pieces=[]
        for shape in binding.shapes.ids():
            if not int(model.shape_flags.numpy()[shape])&int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            source=model.shape_source[shape]
            if source is None:
                continue
            p=np.asarray(source.vertices,float)*scales[shape]
            p=Rotation.from_quat(transforms[shape,3:]).apply(p)+transforms[shape,:3]
            pieces.append(p)
            arrays[f'object_{slot}_piece_{len(pieces)-1}']=p
        objects.append(dict(slot=slot,name=binding.name,root_body_id=binding.root.body_id,
                            initial_pose_xyzw=body_poses[binding.root.body_id].tolist(),pieces=len(pieces)))
    arrays['positions']=setup.reference.object_body_pos_w()
    arrays['quaternions']=setup.reference.object_body_quat_w()
    np.savez_compressed(output/'joint-fk.npz',**arrays)
    write(output/'joint-fk.json',dict(objects=objects,frames=setup.reference.num_frames,
        source='Installed setup_scene with decompose_objects=True, no support during geometry import',device=str(model.device)))


def support(task,output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import warp as wp
    from flash_chord.envs.base import BaseEnv
    from .diagnose import compose_environment,shifted_hand
    build=json.loads((output/'build.json').read_text())
    sequence=f'episode_{build["episode"]:06d}'
    row=pq.ParquetFile(task).read().to_pylist()[0]
    names=row['object_body_names'];axes=mesh_axes(row['object_mesh_paths'])
    results={}
    for mode,steps in [('trajectories',49),('hands-cleared',60)]:
        active=task
        if mode=='hands-cleared':
            active=output/f'hands-cleared/motion/sequence_id={sequence}/robot_name=sharpa_wave/task.parquet'
            active.parent.mkdir(parents=True,exist_ok=True)
            table=pq.ParquetFile(task).read();row=table.to_pylist()[0]
            for side in ('left','right'):
                row=shifted_hand(row,side,np.tile([0.,0.,1.],(build['frames'],1)))
            pq.write_table(pa.Table.from_pylist([row],schema=table.schema),active)
            target=output/f'hands-cleared/reconstructed_stage/{sequence}_sharpa_wave_support.usda'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((output/f'reconstructed_stage/{sequence}_sharpa_wave_support.usda').read_bytes())
        setup,env,config=compose_environment(active,build['fps'])
        ids=[o.root.body_id for o in setup.scene.objects]
        if len(ids)!=2 or config['env']['voc_scale']!=0:
            raise ValueError('Joint support requires both free bodies and zero evaluator VOC')
        first=env.state_0.body_q.numpy()[ids].copy()
        poses=[first.tolist()]
        action=wp.zeros(env.action.action_dim,dtype=wp.float32,device='cpu')
        for step in range(steps):
            env.timestep.assign(np.asarray([step],np.int32))
            BaseEnv.step(env,action)
            poses.append(env.state_0.body_q.numpy()[ids].copy().tolist())
        values=np.asarray(poses)
        objects=[]
        horizon=build['acquisition_end_frame'] if mode=='trajectories' else steps
        for slot in (0,1):
            translation=np.linalg.norm(values[:,slot,:3]-first[slot,:3],axis=1)
            angle=np.degrees((Rotation.from_quat(values[:,slot,3:])*Rotation.from_quat(first[slot,3:]).inv()).magnitude())
            raw_turn=float(angle[:horizon+1].max())
            angle=orientation_degrees(values[:,slot,3:],first[slot,3:],names[slot],axes[slot])
            maximum=float(translation[:horizon+1].max());turn=float(angle[:horizon+1].max())
            objects.append(dict(slot=slot,maximum_translation_m=maximum,maximum_rotation_deg=turn,raw_maximum_rotation_deg=raw_turn,
                                passed=maximum<=.01 and turn<=2.))
        results[mode]=dict(passed=all(o['passed'] for o in objects),objects=objects,poses_xyzw=poses,
                           gated_steps=horizon,voc_scale=0.,reset_count=0,device=str(env.model.device))
        write(output/f'joint-{mode}-steps.json',results[mode])
    write(output/'joint-support.json',dict(passed=all(r['passed'] for r in results.values()),measurements=results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['inspect','support'])
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    initialize(args.output)
    (inspect if args.operation=='inspect' else support)(args.task,args.output)


if __name__=='__main__':
    main()
