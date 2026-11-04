"""CPU clearance of the shipped tissue-box reference against its collision hulls."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def nearest_surface(triangles: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact nearest triangle points, distances and outward normals without rtree."""
    centers = triangles.mean(axis=1)
    radius = np.linalg.norm(triangles - centers[:, None], axis=-1).max()
    tree = cKDTree(centers)
    upper = cKDTree(triangles.reshape(-1, 3)).query(points)[0]
    lower, higher = triangles.min(axis=1), triangles.max(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-30)
    closest, distances, selected_normals = [], [], []
    for point, bound in zip(points, upper, strict=True):
        indices = np.asarray(tree.query_ball_point(point, bound + radius + 1e-12))
        separation = np.maximum(np.maximum(lower[indices] - point, point - higher[indices]), 0.)
        indices = indices[np.einsum('ij,ij->i', separation, separation) <= bound**2 + 1e-14]
        candidates = trimesh.triangles.closest_point(triangles[indices], np.broadcast_to(point, (len(indices), 3)))
        squared = np.sum((candidates - point)**2, axis=-1)
        winner = int(np.argmin(squared))
        closest.append(candidates[winner])
        distances.append(np.sqrt(squared[winner]))
        selected_normals.append(normals[indices[winner]])
    return np.asarray(closest), np.asarray(distances), np.asarray(selected_normals)


def local_points(points: np.ndarray, positions: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    """Transform world points into the per-frame object coordinate system."""
    matrices = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix()
    return np.einsum('tji,tkj->tki', matrices, points - positions[:, None])


def opposing_contacts(closest: np.ndarray, distances: np.ndarray, normals: np.ndarray,
                      bounds: np.ndarray) -> tuple[np.ndarray, list]:
    """Find distinct groups within 1 cm of opposite exterior faces in the fitted box frame."""
    axis = np.argmax(np.abs(normals), axis=-1)
    sign = np.take_along_axis(normals, axis[..., None], axis=-1)[..., 0] > 0
    coordinate = np.take_along_axis(closest, axis[..., None], axis=-1)[..., 0]
    extreme = bounds[sign.astype(int), axis]
    exterior = (distances <= .01) & (np.abs(coordinate - extreme) <= .01)
    pairs = []
    for frame in range(len(closest)):
        found = []
        for a in np.flatnonzero(exterior[frame]):
            for b in np.flatnonzero(exterior[frame]):
                if (b > a and axis[frame, a] == axis[frame, b] and sign[frame, a] != sign[frame, b]
                        and np.dot(normals[frame, a], normals[frame, b]) <= -.8):
                    found.append([int(a), int(b)])
        pairs.append(found)
    return np.asarray([bool(row) for row in pairs]), pairs


def motion_events(positions: np.ndarray) -> dict:
    """Measure onset sensitivity, maximum and final displacement in metres."""
    displacement = positions - positions[0]
    distance = np.linalg.norm(displacement, axis=-1)
    return {'onsets_m': {str(epsilon): int(hits[0]) if len(hits := np.flatnonzero(distance > epsilon)) else None
                         for epsilon in (1e-6, 1e-4, .001, .005, .01)},
            'peak_frame': int(np.argmax(distance)), 'maximum_displacement_m': float(distance.max()),
            'end_displacement_m': float(distance[-1]), 'end_vector_m': displacement[-1].tolist(),
            'peak_height_frame': int(np.argmax(displacement[:, 2])),
            'maximum_height_m': float(displacement[:, 2].max())}


def direction_comparison(achieved: np.ndarray, reference: np.ndarray) -> dict:
    """Compare initial policy motion to simultaneous and cumulative reference motion."""
    events = motion_events(achieved)
    rows = {}
    for threshold, frame in events['onsets_m'].items():
        if frame is None:
            rows[threshold] = None
            continue
        a = achieved[frame] - achieved[0]
        r = reference[frame] - reference[0]
        da = achieved[frame] - achieved[max(0, frame - 1)]
        dr = reference[frame] - reference[max(0, frame - 1)]
        def cosine(x, y):
            norm = np.linalg.norm(x) * np.linalg.norm(y)
            return float(np.dot(x, y) / norm) if norm > 0 else None
        rows[threshold] = {'frame': frame, 'time_s': frame / 30., 'policy_vector_m': a.tolist(),
                           'reference_vector_m': r.tolist(), 'reference_height_m': float(r[2]),
                           'cumulative_cosine': cosine(a, r), 'policy_frame_delta_m': da.tolist(),
                           'reference_frame_delta_m': dr.tolist(), 'instantaneous_cosine': cosine(da, dr)}
    return {**events, 'direction_at_onsets': rows}


def analyze_reference(task: dict, triangles: np.ndarray, hull_vertices: list[np.ndarray],
                      fk_points: np.ndarray) -> dict:
    """Measure CPU robot FK and stored MANO joints against the actual collision pieces."""
    positions = np.asarray(task['object_body_position'])[:, 0]
    quaternions = np.asarray(task['object_body_wxyz'])[:, 0]
    box_transform, box_extents = trimesh.bounds.oriented_bounds(triangles.reshape(-1, 3))
    bounds = np.array([-box_extents / 2, box_extents / 2])
    result = {'object': {'position_m': positions.tolist(), 'quaternion_wxyz': quaternions.tolist(),
                         'height_above_start_m': (positions[:, 2] - positions[0, 2]).tolist(),
                         **motion_events(positions)}, 'representations': {}, 'box_face_frame': {'object_to_box': box_transform.tolist(), 'extents_m': box_extents.tolist()}}
    for representation in ('robot', 'mano'):
        names, arrays = [], []
        for side in ('left', 'right'):
            if representation == 'robot':
                selected = [f'{side}_hand_C_MC', *[f'{side}_{digit}_tip_site'
                            for digit in ('thumb', 'index', 'middle', 'ring', 'pinky')]]
                points = fk_points[:, :6] if side == 'left' else fk_points[:, 6:]
            else:
                selected = [f'{side}_wrist_joint0', *[f'{side}_{digit}_tip'
                            for digit in ('thumb', 'index', 'middle', 'ring', 'pinky')]]
                points = np.asarray(task[f'mano_{side}_joints'])[:, [0, 4, 8, 12, 16, 20]]
            names.extend(selected)
            arrays.append(points)
        world = np.concatenate(arrays, axis=1)
        local = local_points(world, positions, quaternions)
        closest, distance, normal = nearest_surface(triangles, local.reshape(-1, 3))
        closest, distance, normal = (a.reshape((390, 12) + a.shape[1:]) for a in (closest, distance, normal))
        inside = np.zeros((390, 12), dtype=bool)
        for vertices in hull_vertices:
            equations = ConvexHull(vertices).equations
            inside |= np.all(local.reshape(-1, 3) @ equations[:, :3].T + equations[:, 3] < -1e-8,
                             axis=1).reshape(390, 12)
        box_closest = closest @ box_transform[:3, :3].T + box_transform[:3, 3]
        box_normals = normal @ box_transform[:3, :3].T
        mask, pairs = opposing_contacts(box_closest, distance, box_normals, bounds)
        carry = np.asarray(result['object']['height_above_start_m']) > .01
        following = {}
        for side, indices in (('left', slice(0, 6)), ('right', slice(6, 12))):
            mean_z = world[:, indices, 2].mean(axis=1)
            dz = mean_z - mean_z[0]
            object_dz = positions[:, 2] - positions[0, 2]
            high_carry = object_dz > .05
            lift_start = int(np.flatnonzero(high_carry)[0])
            centered_hand = dz[high_carry] - dz[high_carry].mean()
            centered_object = object_dz[high_carry] - object_dz[high_carry].mean()
            following[side] = {'high_carry_frames': int(high_carry.sum()),
                               'high_carry_height_slope': float(np.dot(centered_hand, centered_object) / np.dot(centered_object, centered_object)),
                               'high_carry_height_correlation': float(np.corrcoef(dz[high_carry], object_dz[high_carry])[0, 1]),
                               'high_carry_relative_height_span_m': float(np.ptp((dz - object_dz)[high_carry])),
                               'lift_start_frame_above_5cm': lift_start,
                               'hand_rise_from_5cm_onset_to_peak_m': float(dz[result['object']['peak_height_frame']] - dz[lift_start]),
                               'object_rise_from_5cm_onset_to_peak_m': float(object_dz[result['object']['peak_height_frame']] - object_dz[lift_start]),
                               'carry_frames': int(carry.sum()),
                               'height_correlation_during_carry': float(np.corrcoef(dz[carry], object_dz[carry])[0, 1]),
                               'height_gain_during_carry': float(np.dot(dz[carry], object_dz[carry]) / np.dot(object_dz[carry], object_dz[carry])),
                               'hand_rise_at_object_peak_m': float(dz[result['object']['peak_height_frame']]),
                               'relative_height_change_p95_m': float(np.quantile(np.abs(dz[carry] - object_dz[carry]), .95))}
        minima = []
        for i, name in enumerate(names):
            frame = int(np.argmin(distance[:, i]))
            minima.append({'group': name, 'distance_m': float(distance[frame, i]), 'frame': frame,
                           'inside_any_hull_frames': int(inside[:, i].sum())})
        high_carry = positions[:, 2] - positions[0, 2] > .05
        runs = ''.join('1' if value else '0' for value in mask).split('0')
        result['representations'][representation] = {'groups': names, 'position_world_m': world.tolist(),
            'position_object_m': local.tolist(), 'distance_to_collision_piece_surface_m': distance.tolist(),
            'closest_point_object_m': closest.tolist(), 'surface_normal_object': normal.tolist(),
            'inside_any_hull': inside.tolist(), 'minimum_per_group': minima, 'opposing_pairs': pairs,
            'plausible_grasp_frames': int(mask.sum()), 'first_plausible_grasp_frame': int(np.flatnonzero(mask)[0]) if mask.any() else None,
            'plausible_grasp_during_carry_frames': int((mask & carry).sum()),
            'plausible_grasp_during_high_carry_frames': int((mask & high_carry).sum()),
            'longest_plausible_grasp_run_frames': max(map(len, runs)), 'hand_object_following': following}
    return result


def export_cpu_fk() -> None:
    """Export the training binding's own CPU forward kinematics using the installed runtime."""
    from pathlib import Path
    import warp as wp
    root = Path(__file__).resolve().parents[3]
    output = root / 'data/outputs/dynamic_feasibility/m0b/clearance'
    wp.config.kernel_cache_dir = str(output / 'warp-cache')
    import newton
    from flash_chord.data.loader import load_reference
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    task = next((root / 'data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave').glob('*.parquet'))
    with wp.ScopedDevice('cpu'):
        reference = load_reference(task, control_fps=30., motion_speed=1.)
        builder = newton.ModelBuilder()
        embodiment = SharpaHands()
        layout = embodiment.build(builder)
        bound = embodiment.bind_reference(builder, layout, reference)
        points = np.concatenate([np.concatenate([hand.wrist_pos_w[:, None], hand.fingertip_pos_w], axis=1)
                                 for hand in bound.hands], axis=1)
    np.savez_compressed(output / 'fk-reference.npz', points=points)


def run() -> None:
    """Read the fixed example and M0 exports, then write CPU measurements and input hashes."""
    import hashlib
    import json
    import os
    import subprocess
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import pyarrow.parquet as pq

    root = Path(__file__).resolve().parents[3]
    output = root / 'data/outputs/dynamic_feasibility/m0b/clearance'
    output.mkdir(parents=True, exist_ok=True)
    inputs = []
    def read_path(relative):
        path = root / relative
        inputs.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return path
    base = 'data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed'
    task_path = next((root / base / 'sequence_id=tissue_box_simple/robot_name=sharpa_wave').glob('*.parquet'))
    task = pq.ParquetFile(read_path(task_path.relative_to(root))).read().to_pylist()[0]
    urdf = read_path(f'{base}/tissue_box_simple_rigid.urdf')
    collision = ET.parse(urdf).find('.//collision')
    origin, mesh_spec = collision.find('origin'), collision.find('geometry/mesh')
    if origin.attrib != {'rpy': '0 0 0', 'xyz': '0 0 0'} or mesh_spec.attrib['scale'] != '1 1 1':
        raise ValueError('This diagnostic requires the shipped identity collision transform')
    mesh_path = read_path((urdf.parent / mesh_spec.attrib['filename']).relative_to(root))
    mesh = trimesh.load(mesh_path, force='mesh')
    params = dict(threshold=.05, max_convex_hull=-1, preprocess_mode='auto')
    digest = hashlib.sha1(b'v1')
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    digest.update(repr(sorted(params.items())).encode())
    cache = read_path(f'data/.cache/flash-chord/convex/{digest.hexdigest()[:16]}.npz')
    with np.load(cache, allow_pickle=False) as data:
        hull_vertices = [data[f'v{i}'].astype(np.float64) for i in range(int(data['n']))]
        triangles = np.concatenate([v[data[f'f{i}']] for i, v in enumerate(hull_vertices)])
    fc_src = root / 'data/.runtime/flashchord/video_to_data/robotic_grounding/flash_chord/src'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(fc_src),
           'XDG_CACHE_HOME': str(output / 'cache'), 'MPLCONFIGDIR': str(output / 'mpl-cache')}
    with (output / 'cpu-fk.log').open('w') as log:
        subprocess.run([str(root / 'data/.runtime/flashchord/venv/bin/python'), '-B', str(Path(__file__).resolve()), '--fk'],
                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(read_path('data/outputs/dynamic_feasibility/m0b/clearance/fk-reference.npz'), allow_pickle=False) as saved:
        fk_points = saved['points'].astype(np.float64)
    result = analyze_reference(task, triangles, hull_vertices, fk_points)
    stored = []
    for side in ('left', 'right'):
        names = task[f'{side}_robot_frame_names']
        selected = [f'{side}_hand_C_MC', *[f'{side}_{digit}_tip_site' for digit in ('thumb', 'index', 'middle', 'ring', 'pinky')]]
        stored.append(np.asarray(task[f'robot_{side}_frames'])[:, [names.index(n) for n in selected], :3])
    result['stored_vs_cpu_fk_max_abs_m'] = float(np.max(np.abs(np.concatenate(stored, axis=1) - fk_points)))
    result.update(fps=task['fps'], frames=390, units='metres, seconds; wxyz quaternions; zero-based frames',
                  collision={'source_triangles': len(mesh.faces), 'convex_pieces': len(hull_vertices),
                             'convex_triangles': len(triangles), 'cache_key_matched': True,
                             'local_bounds_m': np.array([triangles.min((0, 1)), triangles.max((0, 1))]).tolist()},
                  method={'surface': 'Exact point-to-triangle minimum over cached simulation collision pieces; no center distance.',
                          'inside': 'Unsigned constituent-piece boundary distance; inside-any-hull recorded separately. Internal interfaces do not count as exterior opposing faces.',
                          'grasp': 'Two distinct digit/palm/wrist groups <=0.01 m from surfaces, closest points within 0.01 m of opposite extrema of the minimum-volume oriented box fitted to the collision hulls on the same axis, outward normals dot <=-0.8. Geometric plausibility, not force closure.',
                          'carry': 'Reference height >0.01 m above frame 0; no selected challenge episode.',
                          'mano_palm': 'Stored wrist joint0 proxy; robot palm is hand_C_MC frame origin, not skin or collision pad.'})
    reference = np.asarray(result['object']['position_m'])
    result['policies'] = {}
    exports = {'Z': 'data/runs/Z-seed0', 'L1-seed0': 'data/runs/L1-seed0', 'L1-seed1': 'data/runs/L1-seed1',
               'D1p-seed0': 'm0/runs/D1p-seed0/attempt0', 'D1p-seed1': 'm0/runs/D1p-seed1/attempt0'}
    for name, directory in exports.items():
        table = pq.ParquetFile(read_path(f'data/outputs/dynamic_feasibility/{directory}/trajectory.parquet')).read().to_pandas().sort_values('reference_frame')
        achieved = table[[f'achieved_position_{axis}' for axis in 'xyz']].to_numpy()
        result['policies'][name] = direction_comparison(achieved, reference)
    result['contact_labels'] = {side: {'active_frames': int(np.any(np.asarray(task[f'mano_{side}_object_contact_part_ids']) > 0, axis=1).sum())}
                                for side in ('left', 'right')}
    source_files = [root / name for name in (
        'AGENTS.md', 'CLAUDE.md', 'docs/chain/PROGRAM.md', 'docs/design/simplification-standard.md',
        'docs/chain/moving-m0-outcome.md', 'docs/chain/moving-m0-protocol.md',
        'docs/chain/dynamic-scaling-protocol.md',
        'data/.runtime/flashchord/install.log',
        'upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/training/flash_sac/runner.py',
        'src/gpu.py', 'src/features/dynamic_feasibility/README.md',
        'src/features/dynamic_feasibility/queue.json', 'src/features/dynamic_feasibility/scaling_queue.json',
        'src/features/dynamic_feasibility/runtime.json',
        'data/outputs/dynamic_feasibility/d1/sizing-fixed/selection.json',
        'data/outputs/dynamic_feasibility/d1/sizing-fixed/partial_checkpoints.json',
        'data/outputs/dynamic_feasibility/d1/sizing-fixed/runs/seed0/run.json',
        'data/outputs/dynamic_feasibility/d1/sizing-fixed/runs/seed1/run.json',
        'data/outputs/dynamic_feasibility/d1/sizing-fixed/runs/seed0/training.log',
        'data/.runtime/tmp/dynamic-scaling-seed0-k62pzjmq/training/state_21575680.safetensors')]
    source_files.extend(Path(__file__).parent.glob('*.py'))
    fc = fc_src.parent
    for pattern in ('src/flash_chord/training/**/*.py', 'src/flash_chord/data/*.py',
                    'src/flash_chord/embodiments/*.py', 'src/flash_chord/scene/*.py',
                    'src/flash_chord/configuration.py', 'src/flash_chord/lifecycle/curriculum.py',
                    'src/flash_chord/evaluation/harness.py',
                    'src/flash_chord/assets/mesh.py', 'src/flash_chord/assets/__init__.py',
                    'src/flash_chord/assets/urdfs/sharpawave/**/*', 'src/flash_chord/assets/meshes/sharpawave/**/*',
                    'src/flash_chord/configs/**/*.yaml', 'workflow/train_flash_sac.yaml', 'scripts/train_flash_sac.py'):
        source_files.extend(path for path in fc.glob(pattern) if path.is_file())
    source_files.extend([root / 'data/.runtime/flashchord/venv/lib/python3.11/site-packages/newton/_src/utils' / name
                         for name in ('mesh.py', 'import_urdf.py')])
    source_files.extend([root / 'data/.runtime/flashchord/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/retarget' / name
                         for name in ('params.py', 'retarget_utils.py', 'data_logger.py')])
    already_read = {entry['path'] for entry in inputs}
    for path in sorted(set(source_files)):
        if str(path.relative_to(root)) not in already_read:
            read_path(path.relative_to(root))
    installed_runner = fc / 'src/flash_chord/training/flash_sac/runner.py'
    pinned_runner = root / 'upstream/video_to_data-flashchord-aee95f2/robotic_grounding/flash_chord/src/flash_chord/training/flash_sac/runner.py'
    result['runner_matches_pinned_aee95f25_normalized_newlines'] = installed_runner.read_text() == pinned_runner.read_text()
    result['inputs'] = inputs
    robot = result['representations']['robot']
    supported = robot['plausible_grasp_during_high_carry_frames'] > 0
    result['verdict'] = ('reference carries the box with a geometrically plausible grasp' if supported else 'cannot tell whether the reference supports a sustained grasp') + '; contact forces and actual policy hand trajectories are not exported.'
    (output / 'clearance.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['| Frame (30 Hz) | Box x / y / z (m) | Rise (cm) | Left tips min / palm (cm) | Right tips min / palm (cm) | Opposing grasp |',
             '| ---: | --- | ---: | --- | --- | --- |']
    for frame in [*range(0, 390, 10), 389]:
        d = np.asarray(robot['distance_to_collision_piece_surface_m'])[frame] * 100
        lines.append(f"| {frame} | {' / '.join(f'{x:.5f}' for x in reference[frame])} | {(reference[frame, 2] - reference[0, 2])*100:.3f} | {d[1:6].min():.3f} / {d[0]:.3f} | {d[7:12].min():.3f} / {d[6]:.3f} | {'yes' if robot['opposing_pairs'][frame] else 'no'} |")
    onsets = ', '.join(f"{name}: {row['onsets_m']['0.001']}" for name, row in result['policies'].items())
    left, right = (robot['hand_object_following'][side] for side in ('left', 'right'))
    obj = result['object']
    lines += ['', f"Verdict: {result['verdict']} Robot opposing groups occur in {robot['plausible_grasp_frames']}/390 frames "
              f"(first {robot['first_plausible_grasp_frame']}; longest run {robot['longest_plausible_grasp_run_frames']}), "
              f"including {robot['plausible_grasp_during_high_carry_frames']}/{left['high_carry_frames']} frames above 5 cm. "
              f"The hands rise {left['hand_rise_from_5cm_onset_to_peak_m']*100:.2f}/{right['hand_rise_from_5cm_onset_to_peak_m']*100:.2f} cm "
              f"while the box rises {left['object_rise_from_5cm_onset_to_peak_m']*100:.2f} cm from frame {left['lift_start_frame_above_5cm']} "
              f"to peak {obj['peak_height_frame']}; high-carry height correlations are {left['high_carry_height_correlation']:.4f}/{right['high_carry_height_correlation']:.4f}, "
              f"with relative height spans {left['high_carry_relative_height_span_m']*100:.2f}/{right['high_carry_relative_height_span_m']*100:.2f} cm. "
              f"The reference first exceeds 1 mm at frame {obj['onsets_m']['0.001']}, peaks at {obj['maximum_displacement_m']*100:.3f} cm displacement "
              f"({obj['maximum_height_m']*100:.3f} cm rise) at {obj['peak_frame']}, and ends {obj['end_displacement_m']*100:.3f} cm away "
              f"({obj['height_above_start_m'][-1]*100:.3f} cm higher). "
              f"Policy first 1 mm displacements (zero-based frames) are {onsets}; all have smaller shifts at frame 1. "
              'Opposing faces use oriented bounds; distances use the actual 52 convex collision pieces. '
              'Unsigned constituent-piece surface distances and penetration flags do not establish grip force; '
              f"palm origins are not collision pads. CPU FK agrees with stored robot frames within {result['stored_vs_cpu_fk_max_abs_m']*1e6:.3f} micrometres. "
              'Every frame/quaternion/group, MANO evidence, policy direction vectors/cosines, onset sensitivity '
              'and input SHA-256 is in clearance.json; source-read resume/ratio details are in resume-facts.md.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'verdict': result['verdict'], 'object': {k: v for k, v in result['object'].items() if k not in ('position_m', 'quaternion_wxyz', 'height_above_start_m')},
                      'robot': {k: robot[k] for k in ('plausible_grasp_frames', 'first_plausible_grasp_frame', 'plausible_grasp_during_carry_frames', 'hand_object_following')},
                      'policy_onsets_1mm': {k: v['direction_at_onsets']['0.001'] for k, v in result['policies'].items()}}, indent=2))


def main() -> None:
    """Dispatch the CPU diagnostic or its installed-runtime FK subprocess."""
    import sys
    export_cpu_fk() if sys.argv[1:] == ['--fk'] else run()


if __name__ == '__main__':
    main()
