"""Installed FlashCHORD loader, FK and support checks in its isolated interpreter."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def initialize(output, gpu=False):
    import warp as wp
    # Set before Newton can initialize Warp. No CUDA context or device is requested.
    wp.config.enable_cuda = gpu
    wp.config.kernel_cache_dir = str(output / 'cache/warp')
    wp.set_device('cuda:0' if gpu else 'cpu')
    return wp


def robot(reference):
    import newton
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    builder = newton.ModelBuilder()
    embodiment = SharpaHands()
    layout = embodiment.build(builder)
    bound = embodiment.bind_reference(builder, layout, reference)
    return builder, layout, bound


def shape_vertices(builder, body_q, side):
    """All colliding Newton hand mesh solids, in the current world frame."""
    from scipy.spatial.transform import Rotation
    import newton
    result = []
    for shape, body in enumerate(builder.shape_body):
        if (not (int(builder.shape_flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES))
                or not str(builder.body_label[body]).rsplit('/', 1)[-1].startswith(side + '_')):
            continue
        source = builder.shape_source[shape]
        if int(builder.shape_type[shape]) != int(newton.GeoType.MESH):
            raise ValueError('Expected mesh-only Sharpa collision geometry')
        vertices = np.asarray(source.vertices, float) * np.asarray(builder.shape_scale[shape])
        shape_pose = np.asarray(builder.shape_transform[shape])
        vertices = Rotation.from_quat(shape_pose[3:]).apply(vertices) + shape_pose[:3]
        pose = body_q[body]
        vertices = Rotation.from_quat(pose[3:]).apply(vertices) + pose[:3]
        result.append(vertices)
    return result


def template(args, output):
    import newton
    import pyarrow.parquet as pq
    from flash_chord.data.loader import load_reference
    from flash_chord.embodiments.binding import _reference_body_q
    from .geometry import template_frame
    task = pq.ParquetFile(args.task).read().to_pylist()[0]
    frame = template_frame(task)
    reference = load_reference(args.task, control_fps=float(task['fps']), motion_speed=1.)
    builder, layout, bound = robot(reference)
    body_q = _reference_body_q(builder, bound.joint_q[frame:frame + 1], device='cpu')[0]
    arrays = {'frame': np.asarray(frame)}
    info = {'frame': frame, 'hand_piece_counts': {}, 'limits': {}}
    for hand in bound.hands:
        side = hand.side
        arrays[f'{side}_groups'] = np.concatenate((hand.wrist_pos_w[frame:frame + 1], hand.fingertip_pos_w[frame]))
        pieces = shape_vertices(builder, body_q, side)
        info['hand_piece_counts'][side] = len(pieces)
        for i, vertices in enumerate(pieces):
            arrays[f'{side}_piece_{i}'] = vertices
        h = layout.hand(side)
        limits = {}
        for name, q, dof in zip(h.finger_joint_names, h.finger_q_ids, h.finger_dof_ids, strict=True):
            limits[name] = [float(builder.joint_limit_lower[dof]), float(builder.joint_limit_upper[dof])]
        info['limits'][side] = limits
        info[f'{side}_joint_values'] = dict(zip(h.finger_joint_names, hand.finger_joint_pos[frame].tolist(), strict=True))
    np.savez_compressed(output / 'template-fk.npz', **arrays)
    (output / 'template-fk.json').write_text(json.dumps(info, indent=2) + '\n')


def inspect(args, output):
    import newton
    import pyarrow.parquet as pq
    from flash_chord.data.loader import load_reference
    from flash_chord.embodiments.binding import _reference_body_q
    from flash_chord.scene.objects import spawn_objects
    from scipy.spatial.transform import Rotation
    task = pq.ParquetFile(args.task).read().to_pylist()[0]
    control_fps = json.loads((output / 'build.json').read_text())['fps']
    reference = load_reference(args.task, control_fps=control_fps, motion_speed=1.)
    builder, layout, bound = robot(reference)
    acquisition = int(json.loads((output / 'build.json').read_text())['acquisition_end_frame'])
    body_q = _reference_body_q(builder, bound.joint_q[acquisition:acquisition + 1], device='cpu')[0]
    arrays = {'positions': reference.object_body_pos_w(), 'quaternions': reference.object_body_quat_w()}
    info = {'frames': reference.num_frames, 'fps': reference.fps, 'hand_piece_counts': {}, 'joint_limit_excess_rad': 0.}
    for hand in bound.hands:
        side = hand.side
        arrays[f'{side}_groups'] = np.concatenate((hand.wrist_pos_w[:, None], hand.fingertip_pos_w), axis=1)
        pieces = shape_vertices(builder, body_q, side)
        info['hand_piece_counts'][side] = len(pieces)
        for i, vertices in enumerate(pieces):
            arrays[f'{side}_piece_{i}'] = vertices
        h = layout.hand(side)
        values = bound.joint_q[:, h.finger_q_ids]
        lower = np.asarray(builder.joint_limit_lower)[list(h.finger_dof_ids)]
        upper = np.asarray(builder.joint_limit_upper)[list(h.finger_dof_ids)]
        excess = np.maximum(np.maximum(lower - values, values - upper), 0.)
        info['joint_limit_excess_rad'] = max(info['joint_limit_excess_rad'], float(excess.max()))
    # Import exactly as the evaluator does. Importing the OBJ into Newton rounds
    # vertices to float32 before CoACD; decomposing a separate float64 Trimesh can
    # produce a different cache key and different hulls from the same OBJ.
    objects = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(objects)
    spawn_objects(objects, reference, decompose_collision=True)
    shapes = [i for i, flags in enumerate(objects.shape_flags)
              if int(flags) & int(newton.ShapeFlags.COLLIDE_SHAPES)]
    info['object_piece_count'] = len(shapes)
    info['object_collision_source'] = 'Installed scene.objects.spawn_objects, including exact Newton mesh import and CoACD'
    for i, shape in enumerate(shapes):
        source = objects.shape_source[shape]
        vertices = np.asarray(source.vertices, float) * np.asarray(objects.shape_scale[shape])
        pose = np.asarray(objects.shape_transform[shape])
        arrays[f'object_piece_{i}'] = Rotation.from_quat(pose[3:]).apply(vertices) + pose[:3]
        arrays[f'object_faces_{i}'] = np.asarray(source.indices).reshape(-1, 3)
    np.savez_compressed(output / 'task-fk.npz', **arrays)
    (output / 'task-fk.json').write_text(json.dumps(info, indent=2) + '\n')


def support(args, output, device='cpu'):
    """The evaluator's environment composition, zero residual actions, no policy rollout."""
    import warp as wp
    from hydra import compose, initialize_config_dir
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import apply_evaluation_composition
    from flash_chord.scene.setup import setup_scene
    from flash_chord.scene.collision import CollisionPolicy
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    fc = Path(__import__('flash_chord').__file__).parent
    build = json.loads((output / 'build.json').read_text())
    result = {'passed': False, 'status': 'unproved', 'device': device, 'steps_completed': 0,
              'maximum_translation_m': None, 'maximum_rotation_deg': None}
    try:
        with initialize_config_dir(config_dir=str(fc / 'configs'), version_base=None):
            cfg = compose(config_name='train_flash_sac', overrides=[f'task.parquet={json.dumps(args.task)}',
                          f'sim.fps={build["fps"]}', 'task.motion_speed=1.0', 'scene.world_count=1'])
        config = OmegaConf.to_container(cfg, resolve=True)
        apply_evaluation_composition(config, world_count=1, reset_mode='explicit')
        cfg = OmegaConf.create(config)
        env_config = instantiate_typed(cfg.env, RLEnvConfig)
        setup = setup_scene(parquet=args.task, control_fps=build['fps'], motion_speed=1.,
                            embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                            collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=1,
                            include_support=True, decompose_objects=True)
        env = RLEnv(setup.scene, setup.reference, config=env_config)
        env.reset(frame_id=0)
        body = setup.scene.objects[0].root.body_id
        first = env.state_0.body_q.numpy()[body].copy()
        result.update(object_body_id=int(body), object_binding=setup.scene.objects[0].name,
                      initial_pose_xyzw=first.tolist(), pose_xyzw=[first.tolist()],
                      model_device=str(setup.scene.model.device),
                      voc_scale=float(config['env']['voc_scale']),
                      maximum_virtual_force=float(config['env']['voc']['max_force']),
                      maximum_virtual_torque=float(config['env']['voc']['max_torque']))
        action = wp.zeros(env.action.action_dim, dtype=wp.float32, device=device)
        translation, angles = [], []
        for step in range(build['acquisition_end_frame']):
            env.step(action)
            pose = env.state_0.body_q.numpy()[body]
            result['pose_xyzw'].append(pose.tolist())
            translation.append(float(np.linalg.norm(pose[:3] - first[:3])))
            angles.append(float(np.degrees((Rotation.from_quat(pose[3:]) * Rotation.from_quat(first[3:]).inv()).magnitude())))
            result['steps_completed'] = step + 1
        result.update(status='measured', maximum_translation_m=max(translation), maximum_rotation_deg=max(angles))
        result['passed'] = max(translation) <= .01 and max(angles) <= 2.
    except Exception as error:
        import traceback
        result.update(reason=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    (output / f'support-{device}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['template', 'inspect', 'support', 'support-gpu'])
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    initialize(output, gpu=args.operation == 'support-gpu')
    if args.operation == 'support-gpu':
        support(args, output, device='cuda:0')
    else:
        globals()[args.operation](args, output)


if __name__ == '__main__':
    main()
