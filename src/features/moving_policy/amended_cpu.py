"""CPU-only evaluator physics for the frozen Amendment 1 support and wrist checks."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .cpu import initialize
from .geometry import (diagnostic_feedforward_residual, inverse_residual_input,
                       orientation_errors)
from .handler import write


def support(task, output):
    """Measure both prescribed support horizons, retaining every achieved object pose."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import warp as wp
    from flash_chord.envs.base import BaseEnv
    from .diagnose import compose_environment, contact_snapshot, shifted_hand
    build = json.loads((output / 'build.json').read_text())
    results = {}
    for mode, steps in [('trajectories', 49), ('hands-cleared', 60)]:
        active_task = task
        if mode == 'hands-cleared':
            active_task = output / 'hands-cleared/motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet'
            active_task.parent.mkdir(parents=True, exist_ok=True)
            table = pq.ParquetFile(task).read()
            row = table.to_pylist()[0]
            for side in ('left', 'right'):
                row = shifted_hand(row, side, np.tile([0., 0., 1.], (build['frames'], 1)))
            pq.write_table(pa.Table.from_pylist([row], schema=table.schema), active_task)
            support_path = output / 'hands-cleared/reconstructed_stage/episode_000039_sharpa_wave_support.usda'
            support_path.parent.mkdir(parents=True, exist_ok=True)
            support_path.write_bytes((output / 'reconstructed_stage/episode_000039_sharpa_wave_support.usda').read_bytes())
        setup, env, config = compose_environment(active_task, build['fps'])
        body = setup.scene.objects[0].root.body_id
        first = env.state_0.body_q.numpy()[body].copy()
        record = dict(status='measured', steps=[], peak_contact_force_N=0., peak_total_contact_force_N=0.,
                      voc_scale=config['env']['voc_scale'], device=str(env.model.device),
                      playback='explicit reference index each step; no resets, including after termination')
        original_step = env.solver.step
        step_peak, total_peak = 0., 0.
        def observe(state_in, state_out, control, contacts, dt):
            nonlocal step_peak, total_peak
            original_step(state_in, state_out, control, contacts, dt)
            contacts_now = contact_snapshot(env, body, 0, dt)['object_contacts']
            step_peak = max(step_peak, max((c['force_norm_N'] for c in contacts_now), default=0.))
            total_peak = max(total_peak, float(np.linalg.norm(np.sum([c['force_on_object_world_N'] for c in contacts_now], axis=0)))) if contacts_now else total_peak
        env.solver.step = observe
        action = wp.zeros(env.action.action_dim, dtype=wp.float32, device='cpu')
        for step in range(steps + 1):
            if step:
                env.timestep.assign(np.array([step - 1], np.int32))
                step_peak, total_peak = 0., 0.
                BaseEnv.step(env, action)
            pose = env.state_0.body_q.numpy()[body].copy()
            record['steps'].append(dict(step=step, reference_frame=max(0, step - 1),
                  pose_xyzw=pose.tolist(), translation_m=float(np.linalg.norm(pose[:3] - first[:3])),
                  rotation_deg=float(np.degrees((Rotation.from_quat(pose[3:]) * Rotation.from_quat(first[3:]).inv()).magnitude())),
                  peak_contact_force_N=step_peak, peak_total_contact_force_N=total_peak))
            record['peak_contact_force_N'] = max(record['peak_contact_force_N'], step_peak)
            record['peak_total_contact_force_N'] = max(record['peak_total_contact_force_N'], total_peak)
        horizon = build['acquisition_end_frame'] if mode == 'trajectories' else steps
        gated = record['steps'][:horizon + 1]
        record.update(gated_steps=horizon, maximum_translation_m=max(x['translation_m'] for x in record['steps']),
                      maximum_rotation_deg=max(x['rotation_deg'] for x in record['steps']),
                      gated_maximum_translation_m=max(x['translation_m'] for x in gated),
                      gated_maximum_rotation_deg=max(x['rotation_deg'] for x in gated))
        record['passed'] = record['gated_maximum_translation_m'] <= .01 and record['gated_maximum_rotation_deg'] <= 2.
        results[mode] = record
        write(output / f'{mode}-steps.json', record)
    write(output / 'steps.json', dict(status='measured', passed=all(r['passed'] for r in results.values()), measurements=results))


def wrist_environment(task, fps):
    """Build only the evaluator's robot and controller; no object or support exists."""
    import newton
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed
    from flash_chord.data.loader import load_reference
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.base import BaseEnv
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.evaluation.harness import apply_evaluation_composition
    from flash_chord.scene.builder import Scene
    from flash_chord.scene.collision import (CollisionPolicy, ShapeSpan, capture_collision_layout, apply_collision_policy)
    fc = Path(__import__('flash_chord').__file__).parent
    with initialize_config_dir(config_dir=str(fc / 'configs'), version_base=None):
        cfg = compose(config_name='train_flash_sac', overrides=[f'task.parquet={json.dumps(str(task))}', f'sim.fps={fps}', 'task.motion_speed=1.0', 'scene.world_count=1'])
    config = OmegaConf.to_container(cfg, resolve=True)
    apply_evaluation_composition(config, world_count=1, reset_mode='explicit')
    cfg = OmegaConf.create(config)
    reference = load_reference(task, control_fps=fps, motion_speed=1.)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    builder = newton.ModelBuilder()
    layout = embodiment.build(builder)
    bound = embodiment.bind_reference(builder, layout, reference)
    policy = instantiate_typed(cfg.collision, CollisionPolicy)
    shapes = ShapeSpan(0, builder.shape_count)
    absent = ShapeSpan(builder.shape_count, builder.shape_count)
    collision = capture_collision_layout(builder, layout, shapes, absent, absent)
    apply_collision_policy(builder, collision, policy)
    builder.request_contact_attributes('force')
    if policy.ground:
        builder.add_ground_plane()
    scene = Scene(model=builder.finalize(), layout=layout, robot_reference=bound,
                  collision_layout=collision, collision_policy=policy, objects=[], support=None,
                  object_scales=np.empty((1, 0)), object_root_position_offsets_w=np.empty((1, 0, 3)), world_count=1)
    from dataclasses import replace
    object_positions = reference.object_body_pos_w()[:, 0].copy()
    count = reference.num_frames
    reference = replace(reference, _object_body_pos_w=np.empty((count, 0, 3), np.float32),
                        _object_body_quat_w=np.empty((count, 0, 4), np.float32),
                        _object_articulation=np.empty((count, 0), np.float32),
                        _object_body_names=[], _object_mesh_paths=[], _object_urdf_paths=[],
                        _object_mesh_radius=np.empty(0, np.float32), _object_assets=())
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    env_config.contact = None
    env_config.objective = None
    env = BaseEnv(scene, reference, config=env_config)
    env.diagnostic_object_positions = object_positions
    env.reset(frame_id=0)
    return env


def reachability(task, output):
    """Run zero residual and a single fixed reference-only diagnostic, never a policy."""
    import warp as wp
    from flash_chord.embodiments.binding import _reference_body_q
    build = json.loads((output / 'build.json').read_text())
    results = {}
    for mode in ('zero-residual', 'diagnostic-feedforward'):
        env = wrist_environment(task, build['fps'])
        bound = env.scene.robot_reference
        # Palm-frame targets include the fixed local frame offset used by binding.
        from .cpu import robot
        builder, _, _ = robot(env.reference)
        target_bodies = _reference_body_q(builder, bound.joint_q, device='cpu')
        buffers = env.action.buffers
        planned = np.zeros((build['frames'], env.action.action_dim))
        for side, start in zip(buffers.layout.sides, buffers.layout.hand_action_starts, strict=True):
            hand = env.scene.layout.hand(side)
            poses = target_bodies[:, hand.wrist_body_id]
            mass = float(env.model.body_mass.numpy()[hand.wrist_body_id])
            inertia = env.model.body_inertia.numpy()[hand.wrist_body_id]
            planned[:, start:start + 6] = diagnostic_feedforward_residual(poses[:, :3], poses[:, [6, 3, 4, 5]], build['fps'], mass, inertia)
        actual = {side: [] for side in buffers.layout.sides}
        residual_peak = np.zeros(env.action.action_dim)
        input_peak = 0.
        for frame in range(build['frames']):
            if frame:
                env.timestep.assign(np.array([frame - 1], np.int32))
                desired = planned[frame - 1] if mode == 'diagnostic-feedforward' else np.zeros(env.action.action_dim)
                action = inverse_residual_input(desired, buffers.filtered_action.numpy(), buffers.scale.numpy(), buffers.clip.numpy(), buffers.ema, buffers.input_mapping, buffers.input_scale.numpy())
                input_peak = max(input_peak, float(np.max(np.abs(action))))
                env.step(wp.array(action.astype(np.float32), dtype=wp.float32, device='cpu'))
                residual_peak = np.maximum(residual_peak, np.abs(buffers.filtered_action.numpy()))
            poses = env.state_0.body_q.numpy()
            for side in buffers.layout.sides:
                actual[side].append(poses[env.scene.layout.hand(side).wrist_body_id].copy())
        report = dict(hands={}, maximum_absolute_policy_input=input_peak,
                      maximum_physical_residual=residual_peak.tolist(), physical_clip=buffers.clip.numpy().tolist(),
                      input_mapping=buffers.input_mapping, object_bodies=0, support_bodies=0,
                      diagnostic_only=True, exported=False, scored=False,
                      method='Single fixed inverse soft-wrist PD from reference derivatives; no achieved-state feedback; EMA/input-map inversion')
        arrays = {}
        for side in buffers.layout.sides:
            hand = env.scene.layout.hand(side)
            target = target_bodies[:, hand.wrist_body_id]
            achieved = np.asarray(actual[side])
            error = np.linalg.norm(achieved[:, :3] - target[:, :3], axis=1)
            theta = orientation_errors(achieved[:, 3:], target[:, [6, 3, 4, 5]])
            use = slice(build['acquisition_end_frame'], None)
            positions = env.diagnostic_object_positions
            radius = float(np.linalg.norm(target[:, :3] - positions, axis=1).max())
            allowance = error + 2 * radius * np.sin(np.radians(theta) / 2)
            report['hands'][side] = dict(mean_position_error_m=float(error[use].mean()), maximum_position_error_m=float(error[use].max()),
                 mean_orientation_error_deg=float(theta[use].mean()), maximum_orientation_error_deg=float(theta[use].max()),
                 radius_m=radius, allowance_maximum_m=float(allowance[use].max()), allowance_limit_m=.19,
                 allowance_fraction_within=float((allowance[use] <= .19).mean()))
            arrays[side + '_achieved_xyzw'] = achieved
            arrays[side + '_reference_xyzw'] = target
            arrays[side + '_position_error_m'] = error
            arrays[side + '_orientation_error_deg'] = theta
        report['passed'] = all(h['mean_position_error_m'] <= .03 and h['mean_orientation_error_deg'] <= 15. and h['maximum_position_error_m'] <= .15 for h in report['hands'].values())
        report['inside_physical_bounds'] = bool(np.all(residual_peak <= buffers.clip.numpy() + 1e-6))
        report['passed'] &= report['inside_physical_bounds']
        np.savez_compressed(output / f'{mode}-wrists.npz', **arrays)
        write(output / f'{mode}.json', report)
        results[mode] = report
    write(output / 'reachability.json', dict(status='measured', passed=results['diagnostic-feedforward']['passed'], measurements=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('support', 'reachability'))
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    initialize(args.output)
    globals()[args.operation](args.task, args.output)


if __name__ == '__main__':
    main()
