"""Build the fixed M2 task, record six Stage 0 checks, and prepare gated commands."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

from .request import Request
from .geometry import (deepest_piece_penetration, follow_object, matrix, motion_summary,
                       rim_grasp, transform_poses, wxyz)

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
OUTPUT = ROOT / 'data/outputs/moving_policy/ep39-a1'
SCORE_STEPS = (16154624, 32814080, 66763776)
FC = ROOT / 'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord'
PYTHON = ROOT / 'data/.runtime/flashchord/venv/bin/python'
TASK = OUTPUT / 'motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet'
TEMPLATE = ROOT / 'data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment(cpu=True):
    folders = {'WARP_CACHE_PATH': 'warp', 'FLASH_CHORD_CACHE_DIR': 'flash-chord',
               'JAX_COMPILATION_CACHE_DIR': 'jax', 'XDG_CACHE_HOME': 'xdg',
               'MPLCONFIGDIR': 'matplotlib', 'TMPDIR': 'tmp'}
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(ROOT), str(FC / 'src'))),
           'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '' if cpu else '0',
           'JAX_PLATFORMS': 'cpu' if cpu else 'cuda', 'WANDB_MODE': 'disabled',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    for key, folder in folders.items():
        path = ROOT / 'data/outputs/moving_policy/ep39-a1/cache' / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def cpu(operation, task):
    command = [str(PYTHON), '-B', '-m', 'src.features.moving_policy.cpu', operation,
               '--task', str(task), '--output', str(OUTPUT)]
    with (OUTPUT / f'{operation}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    return command


def load_mocap():
    # Exact allowlist path. Never discover or iterate other episode files.
    snapshot = next((ROOT / 'data/datasets--nvidia--video_to_data_challenge/snapshots').iterdir())
    path = snapshot / 'track_3/public/data/chunk-000/episode_000039.parquet'
    table = pq.ParquetFile(path).read(columns=['observation.objects', 'timestamp', 'frame_index'])
    rows = table.to_pylist()
    if not np.array_equal(table['frame_index'].to_numpy(), np.arange(len(rows))):
        raise ValueError('M2 requires the complete consecutive episode-39 frame grid')
    if any(len(r['observation.objects']) != 1 or r['observation.objects'][0]['name'] != 'mini_dust_pan'
           or not r['observation.objects'][0]['visible'] for r in rows):
        raise ValueError('M2 does not fill missing/invisible mocap')
    poses = np.asarray([r['observation.objects'][0]['pose'] for r in rows], float)
    times = table['timestamp'].to_numpy().astype(float)
    return path, poses, times


def construct(template, schema, mocap, times, mesh, template_fk, repair, pair="longitudinal", head_fraction=.6):
    """Assemble world arrays; object poses are copied, never procedurally animated."""
    from src.features.dynamic_feasibility.hand_clearance import nearest_surface
    count = len(times)
    frame = int(template_fk['frame'])
    fps = (count - 1) / float(times[-1] - times[0])
    from .geometry import acquisition_rule
    approach, approach_translation, approach_rotation = acquisition_rule(mocap, times)
    acquisition = int(np.ceil(.5 * fps)) if approach else 0
    first_rotation = matrix(mocap[0, 3:])
    rotations = matrix(mocap[:, 3:])
    row = {}
    for key, value in template.items():
        row[key] = (np.repeat(np.asarray(value)[frame:frame + 1], count, axis=0).tolist()
                    if isinstance(value, list) and len(value) == 390 else value)
    row.update(fps=fps, raw_motion_file='M2 tuning-39 mocap; object-driven robot references; not human reconstruction',
               object_name='mini_dust_pan', safe_object_name='mini_dust_pan',
               object_body_names=['mini_dust_pan'], safe_object_body_names=['mini_dust_pan'],
               object_body_position=mocap[:, None, :3].tolist(), object_body_wxyz=mocap[:, None, 3:].tolist(),
               object_root_position=mocap[:, :3].tolist(),
               object_root_axis_angle=__import__('scipy').spatial.transform.Rotation.from_matrix(rotations).as_rotvec().tolist(),
               object_articulation=np.zeros(count).tolist(), object_mesh_paths=[str(OUTPUT / 'mini_dust_pan.obj')],
               object_urdf_paths=[str(OUTPUT / 'mini_dust_pan.urdf')],
               object_mesh_radius=[float(np.linalg.norm(mesh.vertices, axis=1).max())])
    grasp = []
    for side in ('left', 'right'):
        cloud = np.concatenate([template_fk[k] for k in template_fk.files if k.startswith(side + '_piece_')])
        rotation, translation, detail = rim_grasp(template_fk[f'{side}_groups'], cloud, mesh.triangles, side, repair, pair, head_fraction)
        # Canonical reconstruction coordinates -> mocap body frame. At t=0 the
        # reconstructed pan is level. This upper-bound placement uses no scan.
        rotation = first_rotation.T @ rotation
        translation = first_rotation.T @ translation
        frames = transform_poses(template[f'robot_{side}_frames'][frame], rotation, translation)
        wrist = np.r_[template[f'robot_{side}_wrist_position'][frame], template[f'robot_{side}_wrist_wxyz'][frame]]
        wrist = transform_poses(wrist, rotation, translation)
        frame_world = follow_object(frames, mocap[:, :3], rotations)
        wrist_world = follow_object(wrist[None], mocap[:, :3], rotations)[:, 0]
        # Fixed approach from 6 cm outward, smoothly reaching the rigid grasp at .5 s.
        progress = np.clip((times - times[0]) / .5, 0., 1.)
        blend = 1 - progress * progress * (3 - 2 * progress)
        sign = -1. if side == 'left' else 1.
        offset = np.einsum('tij,j->ti', rotations, first_rotation.T @ (np.array([sign * .06, 0., 0.]) if pair == 'longitudinal' else np.array([0., sign * .06, 0.]))) * blend[:, None] * approach
        frame_world[..., :3] += offset[:, None]
        wrist_world[:, :3] += offset
        row[f'robot_{side}_frames'] = frame_world.tolist()
        row[f'robot_{side}_wrist_position'] = wrist_world[:, :3].tolist()
        row[f'robot_{side}_wrist_wxyz'] = wrist_world[:, 3:].tolist()
        # Legacy MANO fields are transformed template placeholders, never evidence
        # of recovered human motion. FlashCHORD consumes the robot fields above.
        joints = np.asarray(template[f'mano_{side}_joints'][frame]) @ rotation.T + translation
        row[f'mano_{side}_joints'] = (np.einsum('tij,kj->tki', rotations, joints) + mocap[:, None, :3] + offset[:, None]).tolist()
        joint_quat = wxyz(rotation @ matrix(template[f'mano_{side}_joints_wxyz'][frame]))
        row[f'mano_{side}_joints_wxyz'] = wxyz(rotations[:, None] @ matrix(joint_quat)[None]).tolist()
        row[f'mano_{side}_trans'] = wrist_world[:, :3].tolist()
        row[f'mano_{side}_global_orient'] = __import__('scipy').spatial.transform.Rotation.from_matrix(matrix(wrist_world[:, 3:])).as_rotvec().tolist()
        tip_canonical = template_fk[f'{side}_groups'][3] @ (first_rotation @ rotation).T + first_rotation @ translation
        close, _, normal = nearest_surface(mesh.triangles, tip_canonical[None])
        contact_local = close[0] @ first_rotation
        normal_local = normal[0] @ first_rotation
        for kind in ('link', 'object'):
            contact = np.zeros((count, 16, 3)); normals = contact.copy()
            contact[acquisition:, 9] = np.einsum('tij,j->ti', rotations[acquisition:], contact_local) + mocap[acquisition:, :3]
            normals[acquisition:, 9] = np.einsum('tij,j->ti', rotations[acquisition:], normal_local)
            row[f'mano_{side}_{kind}_contact_positions'] = contact.tolist()
            row[f'mano_{side}_{kind}_contact_normals'] = normals.tolist()
        ids = np.zeros((count, 16), dtype=int); ids[acquisition:, 9] = 1
        row[f'mano_{side}_object_contact_part_ids'] = ids.tolist()
        detail.update(template_frame=frame, active=True, rigid_after_frame=acquisition,
                      approach=approach, approach_translation_m=approach_translation, approach_rotation_deg=approach_rotation,
                      grasp_rotation_object=rotation.tolist(), grasp_translation_object_m=translation.tolist())
        grasp.append(detail)
    return pa.Table.from_pylist([row], schema=schema), grasp, acquisition, fps


def input_inventory(paths):
    paths = list(dict.fromkeys(Path(p) for p in paths))
    return [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)} for path in paths if path.is_file()]


def amended_worker(operation):
    """Run one bounded CPU worker with caches and records inside the candidate."""
    command = [str(PYTHON), '-B', '-m', 'src.features.moving_policy.amended_cpu', operation,
               '--task', str(TASK), '--output', str(OUTPUT)]
    result = OUTPUT / ('steps.json' if operation == 'support' else 'reachability.json')
    if result.exists():
        return json.loads(result.read_text())
    record = {'status': 'running', 'argv': command, 'started_unix': time.time()}
    write(OUTPUT / f'{operation}.run.json', record)
    with (OUTPUT / f'{operation}.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid
        write(OUTPUT / f'{operation}.run.json', record)
        code = child.wait()
    record.update(status='completed' if code == 0 else 'failed', exit_code=code, finished_unix=time.time())
    write(OUTPUT / f'{operation}.run.json', record)
    if code:
        raise RuntimeError(f'{operation} failed; see {OUTPUT / (operation + ".log")}')
    return json.loads(result.read_text())


def write_support(path, center, radius, top):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'''#usda 1.0
(defaultPrim = "support_surfaces"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "support_surfaces" {{
 def Cylinder "table" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
  uniform token axis = "Z"
  double height = 0.01
  double radius = {radius}
  double3 xformOp:translate = ({center[0]}, {center[1]}, {top - .005})
  uniform token[] xformOpOrder = ["xformOp:translate"]
 }}
}}
''')


def stage0():
    """Execute Amendment 1's resumable fixed queue, stopping at the first support pass."""
    import shutil
    from .geometry import cylinder_support
    global OUTPUT, TASK
    root_output, root_task = OUTPUT, TASK
    Request()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'stage0.json').exists():
        raise FileExistsError('Stage 0 is immutable once recorded')
    queue = json.loads((FEATURE / 'queue.json').read_text())
    if queue['support_queue'] != ['S1', 'S2', 'S3'] or tuple(queue['score_steps']) != SCORE_STEPS:
        raise ValueError('The brief freezes both support and checkpoint queues')
    os.environ.update(environment())
    mocap_path, mocap, times = load_mocap()
    template_path = next(TEMPLATE.glob('*.parquet'))
    parquet = pq.ParquetFile(template_path)
    template = parquet.read().to_pylist()[0]
    if not (OUTPUT / 'template-fk.npz').exists():
        cpu('template', template_path)
    measure_path = ROOT / queue['measurement']
    measure = json.loads(measure_path.read_text())
    if (measure['split'], measure['episode_index'], measure['status']) != ('public', 39, 'completed'):
        raise ValueError('Dimensions must come from completed public-39 video measurements')
    dims = np.asarray(measure['objects']['mini_dust_pan']['median_m'])
    profile = json.loads((ROOT / queue['attributes']).read_text())['objects']['mini_dust_pan']['profile']
    from src.features.mesh_extents._shapes import thin_tool_profile
    first_rotation = matrix(mocap[0, 3:])
    results, all_grasps = [], []
    source_identity = input_inventory([*FEATURE.glob('*.py'), FEATURE / 'queue.json', mocap_path, measure_path, template_path, ROOT / queue['attributes'], ROOT / queue['protocol']])
    write(root_output / 'queue.json', queue)
    # Reuse only this slice's own-reconstruction hull caches, never scan assets.
    for cache in (ROOT / 'data/outputs/moving_policy/ep39/cache/flash-chord/convex').glob('*.npz'):
        target = root_output / 'cache/flash-chord/convex' / cache.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(cache, target)
    for support_name in queue['support_queue']:
        candidate_dir = root_output / support_name
        candidate_dir.mkdir(exist_ok=True)
        trial_profile = dict(profile)
        if support_name == 'S3':
            # Closed prisms from the same constructor. Account for the side-wall
            # slope and clipping so its normal thickness is at least 5 mm.
            trial_profile['floor_thickness_fraction'] = max(profile['floor_thickness_fraction'], .005 / dims[2])
            trial_profile['grip_thickness_fraction'] = max(profile['grip_thickness_fraction'], .005 / dims[2])
            slope = (1 - profile['neck_width_fraction']) / (2 * profile['head_fraction']) * dims[1] / dims[0]
            trial_profile['wall_width_fraction'] = max(profile['wall_width_fraction'], .005 * np.sqrt(1 + slope*slope) / min(dims[:2]))
        canonical = thin_tool_profile(dims, trial_profile)
        mass = .1 if support_name == 'S1' else .3
        inertia = (first_rotation.T @ np.diag(.1 / 12 * (np.sum(dims**2) - dims**2)) @ first_rotation
                   if support_name == 'S1' else np.eye(3) * .001)
        grasps = []
        for pair in ('longitudinal', 'transverse'):
            OUTPUT = candidate_dir / pair
            TASK = OUTPUT / 'motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet'
            OUTPUT.mkdir(exist_ok=True)
            identity = {'sources': source_identity, 'support': support_name, 'pair': pair, 'profile': trial_profile}
            previous = OUTPUT / 'candidate.json'
            if previous.exists():
                saved = json.loads(previous.read_text())
                if saved['identity'] != identity:
                    raise ValueError(f'Cannot resume changed candidate {OUTPUT}')
                grasps.append(saved)
                continue
            mesh = canonical.copy(); mesh.vertices = mesh.vertices @ first_rotation
            mesh.export(OUTPUT / 'mini_dust_pan.obj')
            (OUTPUT / 'mini_dust_pan.urdf').write_text(f'''<robot name="mini_dust_pan"><link name="mini_dust_pan">
<inertial><mass value="{mass}"/><inertia ixx="{inertia[0,0]}" iyy="{inertia[1,1]}" izz="{inertia[2,2]}" ixy="{inertia[0,1]}" ixz="{inertia[0,2]}" iyz="{inertia[1,2]}"/></inertial>
<visual><geometry><mesh filename="mini_dust_pan.obj"/></geometry></visual>
<collision><geometry><mesh filename="mini_dust_pan.obj"/></geometry></collision>
</link></robot>\n''')
            with np.load(root_output / 'template-fk.npz') as fk:
                table, grasp, acquisition, fps = construct(template, parquet.schema_arrow, mocap, times, canonical, fk, True, pair, profile['head_fraction'])
            TASK.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, TASK)
            build = dict(episode=39, frames=len(times), fps=fps, acquisition_end_frame=acquisition, grasp=grasp,
                         task=str(TASK.relative_to(ROOT)), bounds=canonical.bounds.tolist(), dimensions_m=dims.tolist(),
                         motion=motion_summary(mocap[:, :3], times), support_candidate=support_name,
                         mass_kg=mass, inertia_kg_m2=inertia.tolist(), profile=trial_profile,
                         inertia_rule='Original 0.1 kg and rotated bbox tensor' if support_name == 'S1' else 'Shipped URDF constants: 0.3 kg, diag(0.001), no bbox scaling',
                         body_frame_disclosure='Own reconstruction leveled at mocap frame zero; no scan or camera registration',
                         approach_disclosure='0.5 s approach' if acquisition else 'Hands start at grasp: reference fails 1 mm / 1 degree stationarity; no rest prepended')
            write(OUTPUT / 'build.json', build)
            cpu('inspect', TASK)
            info = json.loads((OUTPUT / 'task-fk.json').read_text())
            with np.load(OUTPUT / 'task-fk.npz') as fk:
                pieces = [fk[f'object_piece_{i}'].astype(float) for i in range(info['object_piece_count'])]
            center, radius, top = cylinder_support(pieces, mocap[0, :3], first_rotation)
            support = OUTPUT / 'reconstructed_stage/episode_000039_sharpa_wave_support.usda'
            write_support(support, center, radius, top)
            build.update(support_top_m=top, support_radius_m=radius, support_center_xy_m=center,
                         collision_pieces=info['object_piece_count'], initial_hull_gap_m=.001)
            write(OUTPUT / 'build.json', build)
            checks = geometry_checks(template, table, mocap, times, build)
            eligible = checks['2_closure']['passed'] and checks['3_penetration']['passed']
            speed = max(checks['4a_finger_limits']['maximum_wrist_speed_m_s'].values())
            saved = dict(identity=identity, pair=pair, directory=str(OUTPUT.relative_to(ROOT)), checks=checks,
                         eligible=eligible, peak_wrist_speed_m_s=speed, build=build)
            write(previous, saved)
            grasps.append(saved)
        all_grasps.extend(grasps)
        eligible = [g for g in grasps if g['eligible']]
        # An ineligible grasp never authorizes training; retain a deterministic
        # task solely to finish the independent support measurement if none pass.
        selected = min(eligible or grasps, key=lambda g: (g['peak_wrist_speed_m_s'], g['pair']))
        OUTPUT = ROOT / selected['directory']
        TASK = OUTPUT / 'motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet'
        result = amended_worker('support')
        result.update(candidate=support_name, grasp=selected['pair'], directory=str(OUTPUT.relative_to(ROOT)),
                      mass_kg=mass, collision_pieces=selected['build']['collision_pieces'],
                      inertia_rule=selected['build']['inertia_rule'])
        write(candidate_dir / 'steps.json', result)
        results.append(result)
        with (root_output / 'ledger.jsonl').open('a') as ledger:
            ledger.write(json.dumps({k:v for k,v in result.items() if k != 'measurements'}) + '\n')
        if result['passed']:
            break
    source = OUTPUT
    OUTPUT, TASK = root_output, root_task
    for name in ('mini_dust_pan.obj', 'mini_dust_pan.urdf', 'task-fk.npz', 'task-fk.json'):
        shutil.copyfile(source / name, OUTPUT / name)
    support = OUTPUT / 'reconstructed_stage/episode_000039_sharpa_wave_support.usda'
    support.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / 'reconstructed_stage/episode_000039_sharpa_wave_support.usda', support)
    table = pq.ParquetFile(source / 'motion/sequence_id=episode_000039/robot_name=sharpa_wave/task.parquet').read()
    row = table.to_pylist()[0]
    row['object_mesh_paths'] = [str(OUTPUT / 'mini_dust_pan.obj')]
    row['object_urdf_paths'] = [str(OUTPUT / 'mini_dust_pan.urdf')]
    table = pa.Table.from_pylist([row], schema=table.schema)
    TASK.parent.mkdir(parents=True, exist_ok=True); pq.write_table(table, TASK)
    build = selected['build']; build['task'] = str(TASK.relative_to(ROOT))
    write(OUTPUT / 'build.json', build)
    checks = selected['checks']
    checks['5_support'] = {'status': 'measured', 'passed': any(r['passed'] for r in results), 'candidates': results}
    checks['4b_reachability'] = amended_worker('reachability')
    render(row, build)
    commands = prepare_commands(build)
    artifacts = [TASK, OUTPUT / 'mini_dust_pan.obj', OUTPUT / 'mini_dust_pan.urdf', support,
                 OUTPUT / 'build.json', OUTPUT / 'task-fk.npz', OUTPUT / 'commands.json', OUTPUT / 'reachability.json']
    stage = dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'STOP', checks=checks,
                 build=build, grasp_candidates=all_grasps, selected_grasp=selected['pair'], support_candidates=results,
                 gpu_used=False, training_run=False, protocol='Amendment 1 plus brief 60-step stability requirement',
                 inputs_and_sources=input_inventory([*FEATURE.glob('*.py'), FEATURE / 'queue.json', ROOT / queue['protocol'],
                    ROOT / queue['selection'], ROOT / queue['attributes'], measure_path, mocap_path, template_path,
                    ROOT / 'src/features/dynamic_feasibility/partial_eval.py', ROOT / 'src/features/dynamic_feasibility/handler.py',
                    ROOT / 'src/features/dynamic_feasibility/_worker.py', ROOT / 'src/gpu.py',
                    *list((FC / 'src/flash_chord').rglob('*.py'))]),
                 artifacts=input_inventory(artifacts), commands=commands,
                 runtime_flags=json.loads((FEATURE / 'runtime.json').read_text()))
    # Exclusive creation also closes the race between the initial check and write.
    with (OUTPUT / 'stage0.json').open('x') as stream:
        json.dump(stage, stream, indent=2, allow_nan=False); stream.write('\n')
    summarize_amendment(stage)
    print(json.dumps({'status': stage['status'], 'support': checks['5_support']['passed'],
                      'reachability': checks['4b_reachability']['passed'], 'selected_grasp': selected['pair']}))


def summarize_amendment(stage):
    """Write the one-page amended result from measured records."""
    lines = ['# M2 Amendment 1 Stage 0: ' + stage['status'], '',
             '| Support | Pieces | Hands: acquisition translation m / rotation deg | Through 49 translation m / rotation deg | Cleared 60 translation m / rotation deg | Peak contact N (hands / cleared) | Pass |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in stage['support_candidates']:
        a, b = (row['measurements'][key] for key in ('trajectories', 'hands-cleared'))
        lines.append(f"| {row['candidate']} | {row['collision_pieces']} | {a['gated_maximum_translation_m']:.6g} / {a['gated_maximum_rotation_deg']:.6g} | {a['maximum_translation_m']:.6g} / {a['maximum_rotation_deg']:.6g} | {b['maximum_translation_m']:.6g} / {b['maximum_rotation_deg']:.6g} | {a['peak_contact_force_N']:.6g} / {b['peak_contact_force_N']:.6g} | {row['passed']} |")
    lines += ['', '| Asset / grasp | Closure fraction | Penetration m | Peak wrist m/s | Eligible |', '| --- | ---: | ---: | ---: | --- |']
    for row in stage['grasp_candidates']:
        c = row['checks']
        lines.append(f"| {row['build']['support_candidate']} / {row['pair']} | {c['2_closure']['fraction']:.6g} | {c['3_penetration']['maximum_m']:.6g} | {row['peak_wrist_speed_m_s']:.6g} | {row['eligible']} |")
    lines += ['', '| Wrist diagnostic | Hand | Mean / maximum position m | Mean orientation deg | Maximum allowance m | Pass |', '| --- | --- | ---: | ---: | ---: | --- |']
    for mode, row in stage['checks']['4b_reachability']['measurements'].items():
        for side, h in row['hands'].items():
            lines.append(f"| {mode} | {side} | {h['mean_position_error_m']:.6g} / {h['maximum_position_error_m']:.6g} | {h['mean_orientation_error_deg']:.6g} | {h['allowance_maximum_m']:.6g} | {row['passed']} |")
    lines += ['', f"Selected: **{stage['selected_grasp']}**. {stage['build']['approach_disclosure']}.",
              'S1: original mass/inertia. S2/S3: shipped constants 0.3 kg and diagonal 0.001 kg m². '
              'S3 alone thickens the same closed-prism reconstruction through unedited evaluator CoACD. '
              'Cylinder top is 1 mm below imported collision hulls, height 1 cm; no scan is used.',
              'Support gates: ≤1 cm / ≤2° over acquisition with hands, AND all 60 cleared-hand steps. '
              'Through-frame-49 maxima are descriptive. Forces are raw solver estimates, not verified impulses. '
              'Wrist gate: mean position ≤3 cm, mean orientation ≤15°, maximum position ≤15 cm for each hand. '
              'Zero-residual allowance and speed are descriptive. Robot-only diagnostic uses the evaluator physics/control and is never exported.', '',
              '| Check | Passed |', '| --- | --- |']
    lines += [f"| {name} | {check['passed']} |" for name, check in stage['checks'].items()]
    lines += ['', 'No training, GPU, scoring, install, upload, or excluded-episode read. Dry commands: `commands.json` and `*.argv.txt`. '
              'Raw support poses/forces: `S*/steps.json`; wrist errors: `*-wrists.npz`. '
              'Stage 0 is immutable; any failed check blocks training.']
    (OUTPUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def geometry_checks(template, table, mocap, times, build):
    from src.features.dynamic_feasibility.hand_clearance import nearest_surface, opposing_contacts
    info = json.loads((OUTPUT / 'task-fk.json').read_text())
    first_rotation = matrix(mocap[0, 3:])
    acquisition = build['acquisition_end_frame']
    rotations = matrix(mocap[:, 3:])
    task = table.to_pylist()[0]
    with np.load(OUTPUT / 'task-fk.npz') as fk:
        pieces = [fk[f'object_piece_{i}'].astype(float) @ first_rotation.T for i in range(info['object_piece_count'])]
        triangles = np.concatenate([p[fk[f'object_faces_{i}']] for i, p in enumerate(pieces)])
        groups, hands = [], []
        for side in ('left', 'right'):
            world = fk[f'{side}_groups'][acquisition:]
            local = np.einsum('tji,tkj->tki', rotations[acquisition:], world - mocap[acquisition:, None, :3])
            groups.append(local @ first_rotation.T)
            # All post-acquisition hand/object motion is the same rigid transform;
            # one FK collision pose therefore covers the entire carry interval.
            for i in range(info['hand_piece_counts'][side]):
                p = fk[f'{side}_piece_{i}'] - mocap[acquisition, :3]
                hands.append(p @ rotations[acquisition] @ first_rotation.T)
        points = np.concatenate(groups, axis=1)
        close, distance, normals = nearest_surface(triangles, points.reshape(-1, 3))
        close = close.reshape(points.shape); normals = normals.reshape(points.shape)
        distance = distance.reshape(points.shape[:2])
        bounds = np.array(build['bounds'])
        closure, pairs = opposing_contacts(close, distance, normals, bounds)
        penetration = deepest_piece_penetration(hands, pieces)
        translation_error = float(np.max(np.abs(fk['positions'][:, 0] - mocap[:, :3])))
        loaded = np.concatenate((fk['positions'][:, 0], fk['quaternions'][:, 0, [1, 2, 3, 0]]), axis=1)[:, None, None]
        truth = np.concatenate((mocap[:, :3], mocap[:, [4, 5, 6, 3]]), axis=1)[:, None, None]
    spec = importlib.util.spec_from_file_location('m2_static_score', ROOT / 'scripts/track3/static_pose_baseline.py')
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    # Reference identity is independent of the chosen point set; use OUR mesh,
    # never load the scan just to demonstrate an identity score.
    asset = trimesh.load(OUTPUT / 'mini_dust_pan.obj', force='mesh')
    vertices, _ = trimesh.sample.sample_surface(asset, 1000, seed=0)
    identity = baseline.score(baseline.load_kit(), loaded.copy(), truth, vertices[None])
    speeds = {side: float(np.max(np.linalg.norm(np.diff(task[f'robot_{side}_wrist_position'], axis=0), axis=1) / np.diff(times)))
              for side in ('left', 'right')}
    template_speed = max(float(np.max(np.linalg.norm(np.diff(template[f'robot_{side}_wrist_position'], axis=0), axis=1))) * template['fps']
                         for side in ('left', 'right'))
    ratio = max(speeds.values()) / template_speed
    fraction = float(closure.mean())
    return {
        '1_schema': {'passed': info['frames'] == len(times) and table.schema == pq.ParquetFile(next(TEMPLATE.glob('*.parquet'))).schema_arrow,
                     'frames': info['frames'], 'fields': len(table.schema), 'hands': 2, 'joints_per_hand': 22, 'frames_per_hand': 67, 'contacts_per_hand': 16},
        '2_closure': {'passed': fraction >= .9, 'fraction': fraction, 'required_fraction': .9,
                      'post_acquisition_frames': len(closure), 'frames_passing': int(closure.sum()),
                      'first_frame_pairs': pairs[0], 'nearest_group_distance_m': distance[0].tolist(),
                      'method': 'Installed hand_clearance.opposing_contacts, FK wrist/fingertips vs actual CoACD triangles, canonical asset bounds'},
        '3_penetration': {'passed': penetration['maximum_m'] <= .01, **penetration, 'limit_m': .01,
                          'post_acquisition_rigid_group_span_m': float(np.max(np.ptp(points, axis=0)))},
        '4a_finger_limits': {'passed': info['joint_limit_excess_rad'] <= 1e-6,
                         'joint_limit_excess_rad': info['joint_limit_excess_rad'], 'maximum_wrist_speed_m_s': speeds,
                         'template_maximum_wrist_speed_m_s': template_speed, 'speed_ratio': ratio,
                         'exceeds_example': ratio > 1., 'stop_ratio': 2.},
        '6_truth': {'passed': translation_error <= 1e-6 and identity['AUC'] > .99,
                    'maximum_translation_error_m': translation_error, 'score': identity,
                    'surface_points': '1000 own-reconstruction points, seed 0; identity holds for any point set, scan not read'}
    }


def render(task, build):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = trimesh.load(OUTPUT / 'mini_dust_pan.obj', force='mesh')
    positions = np.asarray(task['object_body_position'])[:, 0]
    rotations = matrix(np.asarray(task['object_body_wxyz'])[:, 0])
    frames = [0, build['acquisition_end_frame'], build['motion']['peak_frame'], build['frames'] - 1]
    fig = plt.figure(figsize=(16, 5))
    for column, frame in enumerate(frames):
        ax = fig.add_subplot(1, 4, column + 1, projection='3d')
        vertices = mesh.vertices @ rotations[frame].T + positions[frame]
        ax.add_collection3d(Poly3DCollection(vertices[mesh.faces], alpha=.4, facecolor='tab:blue', linewidth=.2, edgecolor='navy'))
        for side, color in (('left', 'tab:orange'), ('right', 'tab:green')):
            p = np.asarray(task[f'robot_{side}_frames'])[frame, :, :3]
            ax.scatter(*p.T, s=3, color=color)
            wrist = np.asarray(task[f'robot_{side}_wrist_position'])[frame]
            ax.scatter(*wrist, s=25, marker='x', color=color, label=side)
        center = positions[frame]
        for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=True):
            setter(value - .30, value + .30)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=28, azim=-55)
        ax.set_title(f'frame {frame}, {frame / build["fps"]:.2f} s')
        ax.set_xlabel('world x (m)'); ax.set_ylabel('world y (m)')
    fig.suptitle('M2 tuning 39: own reconstructed pan and two contact-driven Sharpa references; mocap upper bound')
    fig.tight_layout(); fig.savefig(OUTPUT / 'strip.png', dpi=150); plt.close(fig)


def prepare_commands(build):
    from src.features.dynamic_feasibility.handler import training_command, EVALUATOR
    from src.features.dynamic_feasibility.request import Request as TrainingRequest
    queue = json.loads((FEATURE / 'queue.json').read_text())
    thresholds = json.loads((ROOT / queue['selection']).read_text())['chosen']['budget']['curriculum_thresholds']
    budget = dict(world_count=1024, total_environment_steps=66763776, updates_per_collection=2.,
                  capacity=1048576, minimum_size=6400, checkpoint_interval=504832,
                  observation_storage_dtype='float16', save_replay=False, curriculum_thresholds=thresholds)
    argv = training_command(TrainingRequest('D1', 0, budget, 4500), {'path': TASK, 'fps': build['fps']}, OUTPUT)
    argv += ['training.resume=false', 'training.optimization.learning_rate_schedule_updates=130398']
    scores = {}
    for steps in SCORE_STEPS:
        directory = OUTPUT / f'score-{steps}'
        if (directory / 'mocap_metrics.json').exists():
            continue
        scores[str(steps)] = [str(PYTHON), '-B', '-m', 'src.features.dynamic_feasibility._worker', str(EVALUATOR),
                            f'evaluation.checkpoint={OUTPUT / "training" / f"policy_{steps}.safetensors"}',
                            'evaluation.world_count=1', 'evaluation.reset_mode=explicit', 'evaluation.start_frame=0',
                            f'evaluation.object_trajectories_output={directory / "trajectory.parquet"}',
                            f'evaluation.metrics_output={directory / "official_metrics.json"}',
                            f'hydra.run.dir={directory / "hydra"}', 'logging.mode=disabled']
    result = {'train': argv, 'score': scores, 'maximum_seconds': 4500, 'gpu_lease_required': True}
    write(OUTPUT / 'commands.json', result)
    for name, command in [('train', argv), *[(f'score-{step}', value) for step, value in scores.items()]]:
        (OUTPUT / f'{name}.argv.txt').write_text(shlex.join(command) + '\n')
    return result


def verify_scientific_inputs(stage):
    for entry in stage['inputs_and_sources'] + stage['artifacts']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f'Stage 0 input changed: {entry["path"]}')


def execute_gpu(operation):
    """Only this explicit gated boundary acquires the GPU or starts an upstream run."""
    from src.gpu import GpuLease
    flags = json.loads((FEATURE / 'runtime.json').read_text())
    flag = {'train': 'train_authorized', 'score': 'score_authorized', 'support-gpu': 'support_gpu_authorized'}[operation]
    if flags[flag] is not True:
        raise PermissionError(f'{flag} is false; the lead must enable it before GPU execution')
    stage = json.loads((OUTPUT / 'stage0.json').read_text())
    verify_scientific_inputs(stage)
    if operation == 'support-gpu' and stage['checks']['5_support']['status'] == 'measured':
        raise PermissionError('CPU support was measured; the current protocol does not admit a GPU retry of a failed check')
    if operation != 'support-gpu' and not all(c['passed'] for c in stage['checks'].values()):
        # A later GPU support proof is an additional immutable artifact, never a
        # rewrite of the original CPU result.
        later = OUTPUT / 'support-cuda:0.json'
        resolved_support = (stage['checks']['5_support']['status'] == 'unproved'
                            and later.exists() and json.loads(later.read_text())['passed'])
        others = all(c['passed'] for name, c in stage['checks'].items() if name != '5_support')
        if not (resolved_support and others):
            raise PermissionError('M2 stopped at Stage 0; no training/scoring is admitted')
    commands = json.loads((OUTPUT / 'commands.json').read_text())
    if operation == 'train':
        jobs = [('train', commands['train'])]
    elif operation == 'score':
        jobs = [(f'score-{steps}', command) for steps, command in commands['score'].items()]
    else:
        jobs = [('support-gpu', [str(PYTHON), '-B', '-m', 'src.features.moving_policy.cpu', 'support-gpu',
                                 '--task', str(TASK), '--output', str(OUTPUT)])]
    deadline = time.time() + 4500
    with GpuLease(ROOT, 1024, 6000, stage='moving-policy-' + operation) as lease:
        for name, command in jobs:
            record_path = OUTPUT / f'{name}.run.json'
            if record_path.exists():
                saved = json.loads(record_path.read_text())
                if saved.get('status') == 'completed':
                    continue
                raise FileExistsError(f'An unfinished operation exists: {record_path}; no automatic relaunch')
            if name.startswith('score-'):
                checkpoint = OUTPUT / 'training' / f'policy_{name.removeprefix("score-")}.safetensors'
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                (OUTPUT / name).mkdir(exist_ok=True)
            record = {'status': 'running', 'operation': name, 'argv': command,
                      'stage0_sha256': digest(OUTPUT / 'stage0.json'), 'started_unix': time.time()}
            write(record_path, record)
            env = {**environment(cpu=False), 'DYNAMIC_FEASIBILITY_DEADLINE': str(deadline)}
            with (OUTPUT / f'{name}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = child.pid; write(record_path, record)
                code = child.wait()
            record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                          finished_unix=time.time(), gpu=lease.report())
            write(record_path, record)
            if code:
                raise RuntimeError(f'{name} exited {code}; see its log; no automatic retry')
            lease.require_headroom()
    if operation == 'score':
        score_exports()


def score_exports(steps_list=SCORE_STEPS):
    """Score achieved evaluator exports against tuning mocap, never against scored labels."""
    if any(step not in SCORE_STEPS for step in steps_list):
        raise ValueError('Only frozen M2 Amendment 1 checkpoints are admitted')
    from src.features.dynamic_feasibility.partial_eval import motion_metrics
    from .geometry import amended_verdict, orientation_errors
    _, mocap, times = load_mocap()
    truth = np.concatenate((mocap[:, :3], mocap[:, [4, 5, 6, 3]]), axis=1)[:, None, None]
    spec = importlib.util.spec_from_file_location('m2_static_score', ROOT / 'scripts/track3/static_pose_baseline.py')
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    # Stage 1 is measurement only. Public scan points match the mocap body frame;
    # they are never used in task construction or written to a simulated asset.
    points = baseline.vertices('mini_dust_pan')
    e_static = float(np.linalg.norm(mocap[:, :3] - mocap[:1, :3], axis=1).mean())
    for steps in steps_list:
        directory = OUTPUT / f'score-{steps}'
        frame = pq.ParquetFile(directory / 'trajectory.parquet').read().to_pandas().sort_values('reference_frame')
        if (len(frame) != len(times) or not frame['environment_valid'].all()
                or not np.array_equal(frame['reference_frame'], np.arange(len(times)))
                or not (frame['object_body_id'] == 0).all() or not (frame['environment_id'] == 0).all()):
            raise ValueError('Evaluator export has an invalid episode-39 frame/world/body grid')
        metadata = pq.ParquetFile(directory / 'trajectory.parquet').schema_arrow.metadata
        checkpoint_hash = digest(OUTPUT / 'training' / f'policy_{steps}.safetensors')
        if metadata[b'flash_chord.checkpoint_sha256'].decode() != checkpoint_hash:
            raise ValueError('Export/checkpoint hash mismatch')
        official = json.loads((directory / 'official_metrics.json').read_text())
        if official['reference_tracking']['reference_end_fraction'] != 1.:
            raise ValueError('Evaluator did not reach the end of the reference')
        columns = [f'achieved_position_{axis}' for axis in 'xyz'] + [f'achieved_quaternion_{axis}' for axis in 'xyzw']
        achieved = frame[columns].to_numpy(float)[:, None, None]
        if (not np.isfinite(achieved).all()
                or not np.allclose(np.linalg.norm(achieved[..., 3:], axis=-1), 1., atol=1e-4)
                or not np.allclose(frame.time_s, np.arange(len(times)) / ((len(times) - 1) / times[-1]), atol=1e-6, rtol=0.)):
            raise ValueError('Evaluator export has invalid poses or a mismatched source clock')
        motion = motion_metrics(achieved[:, 0, 0, :3], mocap[:, :3])
        error = float(np.linalg.norm(achieved[:, 0, 0, :3] - mocap[:, :3], axis=1).mean())
        score = baseline.score(baseline.load_kit(), achieved.copy(), truth, points[None])
        theta = float(orientation_errors(achieved[:, 0, 0, 3:], mocap[:, 3:]).mean())
        theta_static = float(orientation_errors(np.repeat(mocap[:1, [4, 5, 6, 3]], len(mocap), axis=0), mocap[:, 3:]).mean())
        np.savez_compressed(directory / 'achieved-minus-reference.npz', position_m=achieved[:, 0, 0, :3] - mocap[:, :3], orientation_deg=orientation_errors(achieved[:, 0, 0, 3:], mocap[:, 3:]))
        verdict = amended_verdict(motion['T'], motion['Q'], error, theta, e_static, theta_static, score['AUC'])
        write(directory / 'mocap_metrics.json', {'motion': motion, 'E_m': error, 'E_S_m': e_static, 'Theta_deg': theta, 'Theta_S_deg': theta_static,
              'kit': score, 'verdict': verdict, 'descriptive': steps != 66763776,
              'scoring_geometry': 'Public scan surface points, measurement only, seed 0; no scan geometry enters the simulation',
              'inputs': input_inventory([directory / 'trajectory.parquet', OUTPUT / 'mini_dust_pan.obj',
                        Path(baseline.ROOT) / 'mesh/mini_dust_pan/mini_dust_pan.glb'])})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['stage0', 'train', 'score', 'support-gpu'])
    parser.add_argument('--run', action='store_true', help='Execute the separately authorized GPU operation; default prints argv only')
    args = parser.parse_args()
    if args.operation == 'stage0':
        stage0()
    elif args.run:
        execute_gpu(args.operation)
    elif args.operation == 'support-gpu':
        print('PYTHONPATH=. data/.runtime/linux-cuda/bin/python -B -m src.features.moving_policy support-gpu --run')
    else:
        commands = json.loads((OUTPUT / 'commands.json').read_text())
        value = commands[args.operation]
        for command in (value.values() if isinstance(value, dict) else [value]):
            print(shlex.join(command))


if __name__ == '__main__':
    main()
