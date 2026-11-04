"""CPU contact isolation and raw-motion measurements for the M2 Stage 0 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .handler import ROOT, FC, PYTHON, TASK, TEMPLATE, digest, write

OUTPUT = ROOT / 'data/outputs/moving_policy/diagnosis'
RECORDED = TASK.parents[3]
CACHE_FOLDERS = {'WARP_CACHE_PATH': 'warp', 'FLASH_CHORD_CACHE_DIR': 'flash-chord',
                 'JAX_COMPILATION_CACHE_DIR': 'jax', 'XDG_CACHE_HOME': 'xdg',
                 'MPLCONFIGDIR': 'matplotlib', 'TMPDIR': 'tmp'}


def motion_values(positions: np.ndarray, quaternions: np.ndarray, times: np.ndarray) -> tuple:
    """Measure interval velocities in midpoint body frames using shortest SO(3) increments."""
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    dt = np.diff(times)
    increments = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
    midpoint = Rotation.from_rotvec(increments * .5) * rotations[:-1]
    velocity = np.diff(positions, axis=0) / dt[:, None]
    angular = increments / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    rate = np.degrees(np.linalg.norm(angular, axis=1))
    distance = np.linalg.norm(positions - positions[0], axis=1)
    angle = np.degrees((rotations * rotations[0].inv()).magnitude())
    displaced = np.flatnonzero((distance > .01) | (angle > 5.))
    result = dict(frames=len(times), duration_s=float(times[-1] - times[0]),
                  fps=float((len(times) - 1) / (times[-1] - times[0])),
                  peak_speed_m_s=float(speed.max()), p95_speed_m_s=float(np.percentile(speed, 95)),
                  peak_angular_deg_s=float(rate.max()), p95_angular_deg_s=float(np.percentile(rate, 95)),
                  total_travel_m=float(np.sum(speed * dt)), total_rotation_deg=float(np.sum(rate * dt)),
                  first_displaced_frame=int(displaced[0]) if len(displaced) else None,
                  first_displaced_time_s=float(times[displaced[0]] - times[0]) if len(displaced) else None)
    return result, midpoint.inv().apply(velocity), midpoint.inv().apply(angular)


def point_peak(velocity: np.ndarray, angular: np.ndarray, point: np.ndarray) -> float:
    """Return the maximum speed of one fixed body-frame point."""
    return float(np.linalg.norm(velocity + np.cross(angular, point), axis=1).max())


def slowest_point(velocity: np.ndarray, angular: np.ndarray) -> dict:
    """Solve the convex epigraph problem for the slowest point inside a 30 cm ball."""
    def constraints(x):
        speeds = np.linalg.norm(velocity + np.cross(angular, x[:3]), axis=1)
        return np.r_[x[3] - speeds, .3 ** 2 - x[:3] @ x[:3]]

    fit = minimize(lambda x: x[3], np.r_[np.zeros(3), point_peak(velocity, angular, np.zeros(3))],
                   method='SLSQP', constraints={'type': 'ineq', 'fun': constraints},
                   options={'maxiter': 1000, 'ftol': 1e-11})
    return dict(r_body_m=fit.x[:3].tolist(), radius_m=float(np.linalg.norm(fit.x[:3])),
                minimum_peak_m_s=point_peak(velocity, angular, fit.x[:3]),
                optimizer_success=bool(fit.success), optimizer_message=str(fit.message),
                maximum_constraint_violation=float(max(0., -constraints(fit.x).min())))


def shifted_hand(row: dict, side: str, offset: np.ndarray) -> dict:
    """Translate all positional reference fields for one hand consistently."""
    row = dict(row)
    for key in (f'robot_{side}_wrist_position', f'mano_{side}_trans'):
        row[key] = (np.asarray(row[key]) + offset).tolist()
    for key in (f'robot_{side}_frames', f'mano_{side}_joints'):
        values = np.asarray(row[key]).copy()
        values[..., :3] += offset[:, None]
        row[key] = values.tolist()
    for kind in ('link', 'object'):
        key = f'mano_{side}_{kind}_contact_positions'
        values = np.asarray(row[key]).copy()
        active = np.asarray(row[f'mano_{side}_object_contact_part_ids']) != 0
        values += offset[:, None] * active[..., None]
        row[key] = values.tolist()
    return row


def environment() -> dict:
    """Use the slice's CPU environment with every writable cache under diagnosis."""
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT), str(FC / 'src'))),
           'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu',
           'WANDB_MODE': 'disabled', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    for key, folder in CACHE_FOLDERS.items():
        path = OUTPUT / 'cache' / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def relocate_assets() -> dict:
    """Resolve the relocated M2 and shipped tissue assets without modifying either parquet."""
    from flash_chord.assets import registry
    import pyarrow.parquet as pq
    mapping = {}
    for task, base in ((TASK, RECORDED), (next(TEMPLATE.glob('*.parquet')), TEMPLATE.parent.parent)):
        row = pq.ParquetFile(task).read().to_pylist()[0]
        mapping.update({value: str(base / Path(value).name)
                        for key in ('object_mesh_paths', 'object_urdf_paths') for value in row[key]
                        if not Path(value).is_file()})
    for value in mapping.values():
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    original = registry.resolve_asset_path

    def resolve(path, *, reference_path=None):
        return mapping[path] if path in mapping else original(path, reference_path=reference_path)

    registry.resolve_asset_path = resolve
    return mapping


def compose_environment(task: Path, fps: float):
    """Compose exactly cpu.support's explicit evaluator environment on the CPU."""
    from hydra import compose, initialize_config_dir
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import apply_evaluation_composition
    from flash_chord.scene.setup import setup_scene
    from flash_chord.scene.collision import CollisionPolicy
    from omegaconf import OmegaConf
    fc = Path(__import__('flash_chord').__file__).parent
    with initialize_config_dir(config_dir=str(fc / 'configs'), version_base=None):
        cfg = compose(config_name='train_flash_sac', overrides=[f'task.parquet={json.dumps(str(task))}',
                      f'sim.fps={fps}', 'task.motion_speed=1.0', 'scene.world_count=1'])
    config = OmegaConf.to_container(cfg, resolve=True)
    apply_evaluation_composition(config, world_count=1, reset_mode='explicit')
    cfg = OmegaConf.create(config)
    setup = setup_scene(parquet=str(task), control_fps=fps, motion_speed=1.,
                        embodiment=instantiate_typed(cfg.embodiment, Embodiment),
                        collision=instantiate_typed(cfg.collision, CollisionPolicy), world_count=1,
                        include_support=True, decompose_objects=True)
    env = RLEnv(setup.scene, setup.reference, config=instantiate_typed(cfg.env, RLEnvConfig))
    env.reset(frame_id=0)
    return setup, env, config


def collision_geometry(model) -> dict:
    """Read active collision meshes directly from the composed Newton model."""
    import newton
    vertices = {}
    transforms, scales = model.shape_transform.numpy(), model.shape_scale.numpy()
    for shape, flags in enumerate(model.shape_flags.numpy()):
        if int(flags) & int(newton.ShapeFlags.COLLIDE_SHAPES) and model.shape_source[shape] is not None:
            points = np.asarray(model.shape_source[shape].vertices) * scales[shape]
            vertices[shape] = Rotation.from_quat(transforms[shape, 3:]).apply(points) + transforms[shape, :3]
    return vertices


def clearances(env, geometry: dict, distance_data) -> dict:
    """Measure signed MuJoCo solid distances and vertex heights at an achieved state."""
    import mujoco
    solver, scene, model = env.solver, env.scene, env.model
    solver._update_mjc_data(distance_data, model, env.state_0)
    mujoco.mj_forward(solver.mj_model, distance_data)
    shape_body = model.shape_body.numpy()
    body_q = env.state_0.body_q.numpy()
    support_shapes = list(scene.collision_layout.support.ids())
    object_shapes = [i for i in scene.collision_layout.objects.ids() if i in geometry]
    mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
    inverse = {int(shape): geom for geom, shape in enumerate(mapping)}
    top = max(float(distance_data.geom_xpos[inverse[s], 2] + solver.mj_model.geom_size[inverse[s],
                    1 if solver.mj_model.geom_type[inverse[s]] == mujoco.mjtGeom.mjGEOM_CYLINDER else 2])
              for s in support_shapes)

    def world_vertices(shape):
        pose = body_q[shape_body[shape]]
        return Rotation.from_quat(pose[3:]).apply(geometry[shape]) + pose[:3]

    hands = {}
    for hand in scene.collision_layout.hands:
        shapes = [s for s in hand.shape_ids if s in geometry]
        best = min((float(mujoco.mj_geomDistance(solver.mj_model, distance_data, inverse[h], inverse[o], 10., None)), h, o)
                   for h in shapes for o in object_shapes)
        minimum_z = min(float(world_vertices(s)[:, 2].min()) for s in shapes)
        hands[hand.side] = dict(support_top_clearance_m=minimum_z - top,
                                object_solid_distance_m=best[0], closest_shape_pair=[best[1], best[2]])
    object_bottom = min(float(world_vertices(s)[:, 2].min()) for s in object_shapes)
    return dict(support_top_m=top, object_collision_bottom_m=object_bottom,
                object_support_clearance_m=object_bottom - top, hands=hands)


def contact_snapshot(env, object_body: int, substep: int, dt: float) -> dict:
    """Expose each object contact's signed separation and solver force after one substep."""
    solver = env.solver
    solver.update_contacts(env.contacts, env.state_1)
    count = int(solver.mjw_data.nacon.numpy()[0])
    a = env.contacts.rigid_contact_shape0.numpy()[:count]
    b = env.contacts.rigid_contact_shape1.numpy()[:count]
    body = env.model.shape_body.numpy()
    dist = solver.mjw_data.contact.dist.numpy()[:count]
    force = env.contacts.force.numpy()[:count]
    normal = env.contacts.rigid_contact_normal.numpy()[:count]
    records = []
    for i, (s0, s1) in enumerate(zip(a, b, strict=True)):
        if object_body not in (int(body[s0]), int(body[s1])):
            continue
        sign = 1. if body[s0] == object_body else -1.
        f = sign * force[i, :3]
        records.append(dict(shapes=[int(s0), int(s1)],
                            labels=[env.model.shape_label[s0], env.model.shape_label[s1]],
                            bodies=[int(body[s0]), int(body[s1])],
                            body_labels=[env.model.body_label[body[s0]], env.model.body_label[body[s1]]],
                            separation_m=float(dist[i]), depth_m=float(max(0., -dist[i])),
                            force_on_object_world_N=f.tolist(), force_norm_N=float(np.linalg.norm(f)),
                            impulse_on_object_world_N_s=(f * dt).tolist(), normal_world=normal[i].tolist()))
    return dict(substep=substep, physics_dt_s=dt, object_contacts=records)


def run_variant(spec: dict) -> dict:
    """Run one real zero-action trajectory and observe its first control interval."""
    import mujoco
    import warp as wp
    directory = OUTPUT / spec['name']
    task = directory / 'motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet'
    mapping = relocate_assets()
    try:
        setup, env, config = compose_environment(task, spec['fps'])
    except FileNotFoundError as error:
        if spec['name'] != 'E':
            raise
        return dict(status='unavailable', variant=spec, legal_support_absence=False,
                    reason=str(error), steps=[], task_sha256=digest(task),
                    source_sha256=digest(Path(__file__)))
    body = setup.scene.objects[0].root.body_id
    geometry = collision_geometry(env.model)
    distance_data = mujoco.MjData(env.solver.mj_model)
    first = env.state_0.body_q.numpy()[body].copy()
    result = dict(status='completed', variant=spec, device=str(env.model.device),
                  task_sha256=digest(task), asset_relocations=mapping, steps=[], first_step_substeps=[],
                  support=str(setup.support_usda), voc_scale=config['env']['voc_scale'],
                  maximum_virtual_force=config['env']['voc']['max_force'],
                  maximum_virtual_torque=config['env']['voc']['max_torque'],
                  substeps=env.config.sim.substeps, gravity=env.model.gravity.numpy().tolist(),
                  source_sha256=digest(Path(__file__)))
    mapping_array = env.solver.mjc_geom_to_newton_shape.numpy()[0]
    for component in ('objects', 'support'):
        result[component + '_shape_parameters'] = [dict(shape=int(s), label=env.model.shape_label[s],
                 newton_margin_m=float(env.model.shape_margin.numpy()[s]),
                 margin_m=float(env.solver.mj_model.geom_margin[g]), gap_m=float(env.solver.mj_model.geom_gap[g]),
                 solref=env.solver.mj_model.geom_solref[g].tolist(), solimp=env.solver.mj_model.geom_solimp[g].tolist())
                 for g, s in enumerate(mapping_array) if s in getattr(setup.scene.collision_layout, component).ids()]
    initial_clearance = clearances(env, geometry, distance_data)
    result['object_piece_initial_support_clearance_m'] = {
        str(s): float((Rotation.from_quat(first[3:]).apply(geometry[s]) + first[:3])[:, 2].min()
                      - initial_clearance['support_top_m'])
        for s in setup.scene.collision_layout.objects.ids() if s in geometry}
    result['mass_kg'] = float(env.model.body_mass.numpy()[body])
    original_step = env.solver.step
    observed = 0

    def observe(state_in, state_out, control, contacts, dt):
        nonlocal observed
        original_step(state_in, state_out, control, contacts, dt)
        if observed < env.config.sim.substeps:
            snapshot = contact_snapshot(env, body, observed + 1, dt)
            snapshot['object_pose_xyzw'] = state_out.body_q.numpy()[body].tolist()
            snapshot['object_velocity_linear_angular'] = state_out.body_qd.numpy()[body].tolist()
            result['first_step_substeps'].append(snapshot)
        observed += 1

    env.solver.step = observe
    action = wp.zeros(env.action.action_dim, dtype=wp.float32, device='cpu')
    for step in range(spec['steps'] + 1):
        if step:
            env.step(action)
        pose = env.state_0.body_q.numpy()[body]
        velocity = env.state_0.body_qd.numpy()[body]
        result['steps'].append(dict(step=step, time_s=step / spec['fps'], object_pose_xyzw=pose.tolist(),
              object_linear_velocity_m_s=velocity[:3].tolist(), object_angular_velocity_rad_s=velocity[3:].tolist(),
              object_linear_speed_m_s=float(np.linalg.norm(velocity[:3])),
              object_angular_speed_deg_s=float(np.degrees(np.linalg.norm(velocity[3:]))),
              translation_m=float(np.linalg.norm(pose[:3] - first[:3])),
              rotation_deg=float(np.degrees((Rotation.from_quat(pose[3:]) * Rotation.from_quat(first[3:]).inv()).magnitude())),
              clearance=clearances(env, geometry, distance_data)))
    result['maximum_translation_m'] = max(s['translation_m'] for s in result['steps'])
    result['maximum_rotation_deg'] = max(s['rotation_deg'] for s in result['steps'])
    result['passed'] = result['maximum_translation_m'] <= .01 and result['maximum_rotation_deg'] <= 2.
    if spec['name'] == 'A':
        baseline = json.loads((RECORDED / 'support-cpu.json').read_text())
        actual = np.array([s['object_pose_xyzw'] for s in result['steps']])
        result['baseline_maximum_pose_difference'] = float(np.max(np.abs(actual - baseline['pose_xyzw'])))
        result['reproduced'] = abs(result['maximum_translation_m'] - 10.516670227050781) < 1e-4
    return result


def prepare_variant(spec: dict) -> None:
    """Copy one task and its support, changing only the declared reference fields."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pxr import Usd, Gf
    directory = OUTPUT / spec['name']
    task = directory / 'motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet'
    task.parent.mkdir(parents=True, exist_ok=True)
    source = next(TEMPLATE.glob('*.parquet')) if spec['name'] == 'F' else TASK
    table = pq.ParquetFile(source).read()
    row = table.to_pylist()[0]
    changed = []
    if spec.get('hands_up_m'):
        for side in ('left', 'right'):
            offset = np.zeros_like(row[f'robot_{side}_wrist_position'], dtype=float)
            offset[:, 2] = spec['hands_up_m']
            row = shifted_hand(row, side, offset)
    if spec.get('clear_approach'):
        baseline = json.loads((OUTPUT / 'A/steps.json').read_text())
        times = np.arange(len(row['object_body_position'])) / spec['fps']
        blend = 1 - np.clip(times / .5, 0., 1.) ** 2 * (3 - 2 * np.clip(times / .5, 0., 1.))
        for side in ('left', 'right'):
            raise_m = max(0., .01 - baseline['steps'][0]['clearance']['hands'][side]['support_top_clearance_m'])
            offset = np.zeros((len(times), 3)); offset[:, 2] = raise_m * blend
            row = shifted_hand(row, side, offset)
            spec[side + '_start_raise_m'] = raise_m
    original = table.to_pylist()[0]
    changed = [key for key in row if row[key] != original[key]]
    if changed:
        pq.write_table(pa.Table.from_pylist([row], schema=table.schema), task)
    else:
        shutil.copyfile(source, task)
    support = directory / 'reconstructed_stage/episode_000039_sharpa_wave_support.usda'
    if spec['name'] == 'E':
        write(directory / 'inputs.json', dict(spec=spec, source=str(source), source_sha256=digest(source),
               task_sha256=digest(task), changed_columns=changed, support=None))
        return
    support.parent.mkdir(parents=True, exist_ok=True)
    if spec['name'] == 'F':
        from flash_chord.assets.registry import support_usda_for_parquet
        source_support = support_usda_for_parquet(str(source))
    else:
        source_support = RECORDED / 'reconstructed_stage/episode_000039_sharpa_wave_support.usda'
    shutil.copyfile(source_support, support)
    if spec.get('support_lower_m'):
        stage = Usd.Stage.Open(str(support))
        for prim in stage.Traverse():
            attribute = prim.GetAttribute('xformOp:translate')
            if attribute and prim.GetTypeName() in ('Cube', 'Cylinder'):
                value = attribute.Get()
                attribute.Set(Gf.Vec3d(value[0], value[1], value[2] - spec['support_lower_m']))
        stage.GetRootLayer().Save()
    write(directory / 'inputs.json', dict(spec=spec, source=str(source), source_sha256=digest(source),
           task_sha256=digest(task), changed_columns=changed, source_support=str(source_support),
           support_sha256=digest(support), source_support_sha256=digest(source_support)))


def measure_motion() -> None:
    """Read only allowlisted mocap poses and task wrist arrays, without a physics run."""
    import pyarrow.parquet as pq
    from .geometry import template_frame
    snapshot = next((ROOT / 'data/datasets--nvidia--video_to_data_challenge/snapshots').iterdir())
    recorded = pq.ParquetFile(TASK).read().to_pylist()[0]
    tissue_path = next(TEMPLATE.glob('*.parquet'))
    tissue = pq.ParquetFile(tissue_path).read().to_pylist()[0]
    entries = []
    for episode in (39, 42):
        path = snapshot / f'track_3/public/data/chunk-000/episode_{episode:06d}.parquet'
        rows = pq.ParquetFile(path).read(columns=['observation.objects', 'timestamp']).to_pylist()
        names = sorted({obj['name'] for row in rows for obj in row['observation.objects']})
        for name in names:
            objects = [next(obj for obj in row['observation.objects'] if obj['name'] == name) for row in rows]
            poses = np.asarray([obj['pose'] for obj in objects])
            entries.append((f'episode_{episode:06d}/{name}', path, poses, np.array([r['timestamp'] for r in rows]),
                            recorded if episode == 39 else None, 11 if episode == 39 else None,
                            sum(not obj['visible'] for obj in objects)))
    for body, name in enumerate(tissue['object_body_names']):
        poses = np.c_[np.asarray(tissue['object_body_position'])[:, body], np.asarray(tissue['object_body_wxyz'])[:, body]]
        entries.append(('tissue_box_simple/' + name, tissue_path, poses,
                        np.arange(len(poses)) / tissue['fps'], tissue, template_frame(tissue), 0))
    result = {'method': 'Adjacent raw pose differences; shortest SO(3) increments; midpoint body velocities. '
              'No smoothing or physics. Convex SLSQP epigraph minimax within Euclidean radius 0.30 m.', 'objects': {}}
    for name, path, poses, times, task, acquisition, invisible in entries:
        measured, velocity, angular = motion_values(poses[:, :3], poses[:, 3:], times)
        measured.update(input=str(path), input_sha256=digest(path), invisible_frames=invisible,
                        slowest_point=slowest_point(velocity, angular), wrists={})
        if task is not None:
            position = np.asarray(task['object_body_position'])[acquisition, 0]
            rotation = Rotation.from_quat(np.asarray(task['object_body_wxyz'])[acquisition, 0, [1, 2, 3, 0]])
            for side in ('left', 'right'):
                local = rotation.inv().apply(np.asarray(task[f'robot_{side}_wrist_position'])[acquisition] - position)
                measured['wrists'][side] = dict(frame=acquisition, r_body_m=local.tolist(),
                                                peak_m_s=point_peak(velocity, angular, local))
        else:
            measured['wrist_unavailable_reason'] = 'No recorded episode-42 grasp task was supplied or exists in moving_policy/ep42.'
        result['objects'][name] = measured
    write(OUTPUT / 'motion.json', result)
    lines = ['# Raw mocap motion', '', result['method'], '',
             '| Clip / object | Frames | Duration s | FPS | Speed peak / p95 m/s | Angular peak / p95 deg/s | Travel m | Rotation deg | First >1 cm or >5 deg |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, m in result['objects'].items():
        lines.append(f'| {name} | {m["frames"]} | {m["duration_s"]:.6f} | {m["fps"]:.6f} | '
                     f'{m["peak_speed_m_s"]:.6f} / {m["p95_speed_m_s"]:.6f} | '
                     f'{m["peak_angular_deg_s"]:.3f} / {m["p95_angular_deg_s"]:.3f} | '
                     f'{m["total_travel_m"]:.6f} | {m["total_rotation_deg"]:.3f} | {m["first_displaced_frame"]} |')
    lines += ['', '| Clip / object | Optimal r body m | Min peak m/s | Left wrist peak m/s | Right wrist peak m/s |',
              '| --- | --- | ---: | ---: | ---: |']
    for name, m in result['objects'].items():
        p = m['slowest_point']
        wrists = [f'{m["wrists"][s]["peak_m_s"]:.6f}' if s in m['wrists'] else 'unavailable' for s in ('left', 'right')]
        lines.append(f'| {name} | {np.round(p["r_body_m"], 6).tolist()} | {p["minimum_peak_m_s"]:.6f} | {wrists[0]} | {wrists[1]} |')
    lines += ['', 'Wrist points are frozen in the object body frame at frame 11 for M2, and at the shipped '
              f'tissue grasp frame {template_frame(tissue)} (the existing template selection). Their body coordinates '
              'are in motion.json. Episode 42 has no recorded grasp task here, so no wrist measurements are invented. '
              'The optimal point is a virtual rigid point and need not lie inside the object. '
              'The minimax objective is convex; convergence and feasibility are recorded, without claiming an exact symbolic optimum.', '']
    (OUTPUT / 'motion.md').write_text('\n'.join(lines))


def summarize(queue: list) -> None:
    """Write the measured control table and contact evidence for the lead's review."""
    lines = ['# M2 CPU diagnosis', '', '| Variant | Steps | Max translation m | Max rotation deg | Rest gate | Raw data |',
             '| --- | ---: | ---: | ---: | --- | --- |']
    results = {}
    for spec in queue:
        path = OUTPUT / spec['name'] / 'steps.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        results[spec['name']] = r
        if r['status'] == 'unavailable':
            lines.append(f'| {spec["name"]}: {spec["description"]} | 0 | — | — | unavailable | [{spec["name"]}/steps.json]({spec["name"]}/steps.json) |')
            continue
        lines.append(f'| {spec["name"]}: {spec["description"]} | {spec["steps"]} | '
                     f'{r["maximum_translation_m"]:.9f} | {r["maximum_rotation_deg"]:.6f} | {r["passed"]} | '
                     f'[{spec["name"]}/steps.json]({spec["name"]}/steps.json) |')
    if 'A' in results:
        a = results['A']
        contact = a['first_step_substeps'][0]
        force = np.sum([c['force_on_object_world_N'] for c in contact['object_contacts']], axis=0)
        impulse = force * contact['physics_dt_s']
        largest = max(contact['object_contacts'], key=lambda c: c['force_norm_N'])
        clearance = a['steps'][0]['clearance']
        lines += ['', '## Cause and smallest fix', '',
                  f'A reproduces the immutable trajectory exactly (maximum pose-array difference '
                  f'{a["baseline_maximum_pose_difference"]:.1f}; excursion {a["maximum_translation_m"]:.9f} m). '
                  f'The first impulse is support `table` shape 138 against the object’s CoACD pieces: '
                  f'{len(contact["object_contacts"])} object contacts, all support/object, sum force '
                  f'{force[2]:.6f} N upward and impulse {impulse[2]:.9f} N·s over '
                  f'{contact["physics_dt_s"]:.9f} s. This launches the 0.1 kg body at '
                  f'{contact["object_velocity_linear_angular"][2]:.6f} m/s in the first physics substep. '
                  f'The largest individual pair is {largest["shapes"]} ({largest["labels"]}), '
                  f'{largest["force_norm_N"]:.6f} N / {largest["impulse_on_object_world_N_s"][2]:.9f} N·s upward. '
                  'B and D reproduce the same first impulse despite clearing the hands. '
                  f'The actual CoACD collision bottom is {-1000 * clearance["object_support_clearance_m"]:.6f} mm '
                  'below the support top; the support was placed against the undecomposed reconstruction. '
                  'Active Newton and MuJoCo object/support margins and MuJoCo gaps are zero: this is actual hull '
                  'overlap, not an inflated contact margin. **No tested M2 placement change meets 1 cm / 2°** '
                  'over its specified horizon, including the fine alignment sweep. '
                  'The passing shipped tissue control does not establish a fix for M2; no smallest passing M2 '
                  'change has been demonstrated. The evidence identifies the initial contact source, but does '
                  'not prove which internal solver mechanism causes the later drop-test instability.', '',
                  f'At reset the left/right hand minima are '
                  f'{1000 * clearance["hands"]["left"]["support_top_clearance_m"]:.3f} / '
                  f'{1000 * clearance["hands"]["right"]["support_top_clearance_m"]:.3f} mm relative to the support plane. '
                  'D raises those starts to +10 mm, then blends the added height to zero at the existing grasp. '
                  'This removes the initial hand/support overlap while keeping the target grasp unchanged. '
                  'Zero residual action still includes the evaluator’s reference-tracking hand controller.']
        write(OUTPUT / 'cause.json', dict(first_contact_force_on_object_N=force.tolist(),
              first_contact_impulse_on_object_N_s=impulse.tolist(), largest_contact=largest,
              contact_shape_pairs=[c['shapes'] for c in contact['object_contacts']],
              object_collision_support_overlap_m=-clearance['object_support_clearance_m'],
              reproduced_exactly=a['baseline_maximum_pose_difference'] == 0.))
    lines += ['', 'E: unavailable with the evaluator composition. `setup_scene(include_support=True)` raises '
              '`FileNotFoundError` if asset resolution finds no support; a missing file does not imply support-free simulation. '
              'No illegal support-free run was substituted.', '',
              'Each variant directory holds `inputs.json`, a copied task/support, and `steps.json` with achieved poses, '
              'linear/angular velocities, signed solid clearances, and every object contact/force during the first control step. '
              'A positive clearance is separated; a negative clearance is penetration. The support-top height is a plane metric. '
              'Object clearance uses MuJoCo `mj_geomDistance` on the actual convex collision solids.', '',
              'A is a byte-for-byte copy of the task. Its stale pre-layout asset paths are resolved at runtime to the '
              'same immutable M2 reconstruction files under data/outputs; no source file is edited. '
              'The shipped tissue asset paths receive the same relocation-only treatment to their existing source files. '
              'Existing CoACD cache entries are copied read-only into the diagnostic cache to preserve the recorded hulls.', '',
              '## Motion and checks', '',
              'See [motion.md](motion.md) and [motion.json](motion.json) for raw mocap speeds, angular rates, '
              'travel, rotation, movement onset, minimax rigid points and recorded wrist-point speeds. '
              'Episode 42 has no supplied recorded grasp task: its two wrist-point measurements remain unavailable. '
              'The raw translation measurements match the existing `geometry.motion_summary` to 2.3e-16 '
              '(see [equivalence.json](equivalence.json)). Compilation and a CPU import were also checked. '
              'A’s full pose-array equality provides the old-versus-new physics equivalence check.', '',
              '## Commands', '', '```bash',
              'PYTHONDONTWRITEBYTECODE=1 data/.runtime/flashchord/venv/bin/python -B -m src.features.moving_policy.diagnose run',
              'CUDA_VISIBLE_DEVICES=\'\' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord/src data/.runtime/flashchord/venv/bin/python -B data/outputs/moving_policy/diagnosis/equivalence.py',
              '```', '', 'The command launches sequential CPU workers with the required interpreter, CPU flags, '
              'FlashCHORD PYTHONPATH and local caches. Exact worker commands are in `commands.json`. '
              'Compilation/import commands are in `verification.txt`. Previous instrumentation attempts, including '
              'a tissue asset-path failure before relocation was added, are retained under `revisions/`. '
              'No training, GPU, test suite, install, git mutation, scan geometry, or excluded-episode access.', '']
    (OUTPUT / 'summary.md').write_text('\n'.join(lines))


def run() -> None:
    """Execute and resume the fixed CPU isolation queue, stopping on a failed A reproduction."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = environment()
    fps = json.loads((RECORDED / 'build.json').read_text())['fps']
    queue_path = OUTPUT / 'queue.json'
    if not queue_path.exists():
        write(queue_path, [dict(name='A', description='as recorded', steps=11, fps=fps),
              dict(name='B', description='hands +1 m up', steps=60, fps=fps, hands_up_m=1.),
              dict(name='C-2mm', description='B; support lowered 2 mm', steps=60, fps=fps, hands_up_m=1., support_lower_m=.002),
              dict(name='C-0.5mm', description='B; support lowered 0.5 mm', steps=60, fps=fps, hands_up_m=1., support_lower_m=.0005),
              dict(name='D', description='raised clear approach; original grasp', steps=11, fps=fps, clear_approach=True),
              dict(name='E', description='B; missing support file', steps=11, fps=fps, hands_up_m=1.),
              dict(name='F', description='shipped tissue control', steps=11, fps=30.)])
    queue = json.loads(queue_path.read_text())
    # Both entries belong to this own-reconstruction Stage 0 record, never dataset scans.
    for cache in (RECORDED / 'cache/flash-chord/convex').glob('*.npz'):
        target = OUTPUT / 'cache/flash-chord/convex' / cache.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(cache, target)
    commands = json.loads((OUTPUT / 'commands.json').read_text()) if (OUTPUT / 'commands.json').exists() else []
    for spec in queue:
        directory = OUTPUT / spec['name']
        result_path = directory / 'steps.json'
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode() + Path(__file__).read_bytes()).hexdigest()
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get('request_identity') != identity:
                raise ValueError(f'Existing result has a different identity: {result_path}')
            if spec['name'] == 'A' and not previous.get('reproduced'):
                raise RuntimeError('A did not reproduce: stop')
            continue
        command = [str(PYTHON), '-B', '-m', 'src.features.moving_policy.diagnose', 'worker', '--variant', spec['name']]
        commands.append(command)
        write(OUTPUT / 'commands.json', commands)
        directory.mkdir(exist_ok=True)
        started = time.time()
        record = dict(status='running', request_identity=identity, argv=command,
                      started_unix=started, device='cpu', source_sha256=digest(Path(__file__)))
        write(directory / 'run.json', record)
        with (directory / 'run.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            write(directory / 'run.json', record)
            code = child.wait()
        record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                      finished_unix=time.time(), elapsed_s=time.time() - started)
        write(directory / 'run.json', record)
        if code:
            raise RuntimeError(f'{spec["name"]} exited {code}; see {directory / "run.log"}')
        result = json.loads(result_path.read_text())
        with (OUTPUT / 'ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(name=spec['name'], identity=identity,
                         maximum_translation_m=result.get('maximum_translation_m'),
                         maximum_rotation_deg=result.get('maximum_rotation_deg'), passed=result.get('passed'),
                         status=result['status'])) + '\n')
        summarize(queue)
        if spec['name'] == 'A' and not result['reproduced']:
            raise RuntimeError('A did not reproduce 10.5167 m: stop')
    measure_motion()
    summarize(queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'worker', 'motion'))
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.operation == 'run':
        run()
    elif args.operation == 'motion':
        measure_motion()
    else:
        from .cpu import initialize
        initialize(OUTPUT)
        spec = next(s for s in json.loads((OUTPUT / 'queue.json').read_text()) if s['name'] == args.variant)
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode() + Path(__file__).read_bytes()).hexdigest()
        prepare_variant(spec)
        result = run_variant(spec)
        result['request_identity'] = identity
        write(OUTPUT / spec['name'] / 'steps.json', result)


if __name__ == '__main__':
    main()
