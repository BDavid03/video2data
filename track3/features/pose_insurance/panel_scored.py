"""Run the frozen episode-30-only P route, saving each stage before the next."""
import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from track3.gpu import GpuLease
from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, checked_entry,
                         heavy_execution, producer, read_json, recorded, runtime_flag)
from .request import Request
from .scored_stereo_config import BUDGET, FC, FEATURE, KIT

OUTPUT = 'outputs/pose_insurance/frame0-panel/scored-30'
PROTOCOL = 'docs/chain/frame0-panel-protocol.md'
PLAN = 'outputs/pose_insurance/stereo-placement-scored/plans/episode_000030.json'
MANIFEST = 'submissions/20261103T2024Z-d2-v5/track3_poses_v5_final.manifest.json'
NAMES = ['mini_sweeper', 'plastic_dish_rack']


def environment() -> dict:
    """Admit episode 30 and keep worker caches in permitted scratch."""
    runtime = runtime_flag(FEATURE, 'scored_stereo_authorized', 'Panel scored P')
    authorize_episode(runtime, Request(episode_index=30, budget=BUDGET), 'Panel scored P')
    selected_python = runtime['python'] if Path(sys.executable).resolve() == (ROOT/runtime['python']).resolve() else runtime['diagnostic_python']
    heavy_execution(dict(runtime, python=selected_python))
    scratch = output_path(ROOT, '.runtime/tmp/panel-scored-30')
    for key in (*CACHE_FOLDERS, 'WARP_CACHE_PATH', 'JAX_COMPILATION_CACHE_DIR'):
        path = scratch / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(FC/'src')]), CUDA_VISIBLE_DEVICES='0',
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                      JAX_ENABLE_COMPILATION_CACHE='true', JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
                      WANDB_MODE='disabled', HYDRA_FULL_ERROR='1', PYTHONUNBUFFERED='1')
    return os.environ.copy()


def lease(stage: str, allowance: int) -> GpuLease:
    """Refuse either checkout's existing owner before acquiring the slice lease."""
    main = Path('/home/ceolwolf/repos/video2data/.runtime/gpu-stage.lock')
    for path in {main, ROOT/'.runtime/gpu-stage.lock'}:
        if path.exists():
            raise RuntimeError(f'GPU lease already exists: {path}')
    return GpuLease(ROOT, 2048, allowance, stage=stage)


def record(operation: str, parents=(), inputs=()) -> dict:
    """Record the fixed route and its consumed stage evidence."""
    return run_fields(ROOT, 'pose_insurance', operation, dict(episode=30, budget=BUDGET, candidate='P'),
                      parents=parents, inputs=[PROTOCOL, PLAN, MANIFEST, FEATURE/'runtime.json', *inputs])


def incumbent(rec: dict) -> tuple:
    """Bind the shipped row to the exact inherited T1 build and prior physical run."""
    plan = read_json(PLAN)
    manifest = read_json(MANIFEST)
    export = next(e for e in manifest['exports'] if Path(e['path']).stem == 'episode_000030')
    policy = producer(export['run'], 'pose_insurance.run', Request(episode_index=30, budget=BUDGET))
    build = producer(plan['base_build'], 'pose_insurance.build', Request(episode_index=30, budget=BUDGET))
    trajectory = artifact(policy, 'trajectory.parquet', rec)
    assert sha256_file(trajectory) == export['sha256'] == sha256_file(ROOT/export['path'])
    assert policy[1]['scene'] == plan['scene'] == build[1]['scene']
    assert policy[1]['request']['budget'] == BUDGET and policy[1]['checks']['accepted']
    assert build[1]['object_names'] == NAMES and not plan['changed']
    rec['inputs'].extend(input_file(ROOT, p[0]) for p in (policy, build))
    return plan, build, policy


def prepare(rec: dict, directory: Path) -> dict:
    """Read only episode 30's exact frame-zero remap, stereo and T1 table basis."""
    import cv2
    import numpy as np
    from ._scene import table_basis
    from .length_diagnostic import frame_arrays
    request = Request(episode_index=30, budget=BUDGET)
    plan, build, policy = incumbent(rec)
    original = read_json('outputs/pose_insurance/v3/plans/episode_000030.json')
    old_build = producer(original['base_build'], 'pose_insurance.build', request)
    entry = next(e for e in old_build[1]['inputs'] if '/mesh_extents/evaluation/episode_000030/' in e['path'] and e['path'].endswith('/run.json'))
    measure = producer(checked_entry(entry, rec), 'mesh_extents.measure', request)
    depth = producer(original['depth_run'], 'depth.generate', request)
    scale = read_json('outputs/pose_insurance/v3/stereo_000030.json')
    rec['inputs'].append(input_file(ROOT, 'outputs/pose_insurance/v3/stereo_000030.json'))
    for anchor_entry in scale['anchors']:
        anchor = producer(checked_entry(anchor_entry, rec), 'stereo_depth.anchor', request)
        if any(f['frame_index'] == 0 for f in anchor[1]['frames']):
            break
    else:
        raise ValueError('No frame-zero stereo anchor')
    stereo_run = producer(anchor[1]['request']['stereo_run'], 'stereo_depth.generate', request)
    assert measure[1]['request']['depth_run'] == anchor[1]['request']['depth_run'] == original['depth_run']
    assert stereo_run[1]['grid'] == depth[1]['preprocessing_artifact_sha256']
    stereo = frame_arrays(stereo_run, rec)
    with np.load(artifact(depth, 'preprocessing.npz', rec, depth[1]['preprocessing_artifact_sha256'])) as grid:
        intrinsics = grid['input_intrinsics_pixels'].copy()
        source_x, source_y = grid['source_x'].copy(), grid['source_y'].copy()
    overlay = cv2.imread(str(artifact(measure, 'overlay_000000.png', rec)))
    rgb = overlay[:, :overlay.shape[1]//2]
    video = next(e for e in depth[1]['inputs'] if '/observation.images.ego_cam_a/' in e['path'])
    capture = cv2.VideoCapture(str(checked_entry(video, rec)))
    ok, raw = capture.read()
    capture.release()
    assert ok and np.array_equal(rgb, cv2.remap(raw, source_x, source_y, cv2.INTER_LINEAR))
    assert int(stereo['frame_index']) == int(stereo['pts']) == measure[1]['frames'][0]['frame_index'] == 0
    assert stereo['rgb_depth_m'].shape == rgb.shape[:2]
    assert np.allclose(stereo['rgb_intrinsics_pixels'], intrinsics, atol=1e-5, rtol=1e-5)
    basis = table_basis(np.array(measure[1]['frames'][0]['plane']['normal']))
    rec['inputs'].extend(input_file(ROOT, p[0]) for p in (old_build, measure, depth, stereo_run))
    cv2.imwrite(str(directory/'rgb.png'), rgb)
    np.savez_compressed(directory/'frame.npz', **stereo, basis=basis)
    write_json(directory/'inputs.json', dict(frame_index=0, pts=0, camera='ego_cam_a',
               rgb_kind='recorded_camera_a_remap', remap_pixel_equivalence=True,
               grid_sha256=stereo_run[1]['grid'], intrinsics_pixels=intrinsics.tolist(), basis=basis.tolist(),
               t1_build=input_file(ROOT, build[0]), v5_policy=input_file(ROOT, policy[0]), inputs=rec['inputs']))
    return dict(stereo=stereo, rgb=rgb, basis=basis, scene=plan['scene'])


def stage_a() -> Path:
    """Apply the unchanged detector, segmenter and overlap rules once to the pair."""
    import cv2
    import numpy as np
    from .panel_masks import infer, contact_sheet
    from ._panel_masks import resolve_overlap, box_iou, mask_reading, centroid_offset
    from ._scene import camera_points
    from ._shrink import footprint_clearance
    from track3.features.mesh_generative.g3_math import largest_component
    queue = read_json('track3/features/pose_insurance/panel_queue.json')
    assert queue['detector'] == dict(revision='a2bb814dd30d776dcf7e30523b00659f4f141c71',
                                      box_threshold=.4, text_threshold=.3)
    assert queue['segmenter'] == dict(source_revision='2b90b9f5ceec907a1c18123530e92e794ad901a4',
        config='configs/sam2.1/sam2.1_hiera_l.yaml', multimask_output=True,
        postprocessing='largest connected component only')
    rec = record('panel_scored_masks', inputs=['track3/features/pose_insurance/panel_queue.json',
                 'track3/features/object_masks/prompts.json',
                 'track3/features/mesh_generative/g3_detect.py',
                 'track3/features/mesh_generative/g3_segment.py',
                 'track3/features/mesh_generative/g3_math.py'])
    rec['queue_protocol_note'] = 'Train queue pins the pre-amendment protocol; this scored adapter binds the current amended protocol directly'
    for model, expected in [('grounding-dino', queue['detector']['revision']), ('sam2', queue['segmenter']['source_revision'])]:
        selected = read_json(f'.runtime/{model}/weights-selected.json')
        assert selected['revision'] == expected
        rec['inputs'].append(input_file(ROOT, f'.runtime/{model}/weights-selected.json'))
        for entry in selected['files']:
            checked_entry({k: entry[k] for k in ('path', 'sha256')}, rec)
    directory = output_path(ROOT, f'{OUTPUT}/stage-a')
    started = time.perf_counter()
    with recorded(directory, rec):
        data = prepare(rec, directory)
        prompts = read_json('track3/features/object_masks/prompts.json')['objects']
        masks, detections, segments = [], [], []
        with lease('mesh_generative', 10000) as gpu:
            gpu_started = time.perf_counter()
            try:
                for slot, name in enumerate(NAMES):
                    work = directory/f'slot_{slot}'
                    detection = infer('detect', dict(image=f'{OUTPUT}/stage-a/rgb.png',
                        rule=dict(queue['detector'], prompt=prompts[name]+'.'),
                        output=f'{OUTPUT}/stage-a/slot_{slot}/detect.json'), work)
                    mask, segment = np.zeros(data['rgb'].shape[:2], bool), None
                    if detection['box'] is not None:
                        segment = infer('segment', dict(image=f'{OUTPUT}/stage-a/rgb.png', box=detection['box'],
                            rule=queue['segmenter'], output=f'{OUTPUT}/stage-a/slot_{slot}/sam.npz'), work)
                        with np.load(work/'sam.npz', allow_pickle=False) as arrays:
                            mask = largest_component(arrays['mask'])
                        assert mask.shape == data['rgb'].shape[:2]
                    cv2.imwrite(str(directory/f'component_slot_{slot}.png'), mask.astype(np.uint8)*255)
                    masks.append(mask)
                    detections.append(detection)
                    segments.append(segment)
                gpu.require_headroom()
            finally:
                rec.update(gpu=gpu.report(), gpu_seconds=time.perf_counter()-gpu_started)
        masks, overlap = resolve_overlap(masks)
        boxes = [d['box'] for d in detections]
        overlap['box_iou'] = box_iou(boxes)
        stereo = data['stereo']
        valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
        objects = []
        for slot, mask in enumerate(masks):
            row = dict(slot=slot, name=NAMES[slot], box=boxes[slot], score=detections[slot]['box_score'],
                       predicted_iou=None if segments[slot] is None else segments[slot]['selected_predicted_iou'],
                       **mask_reading(mask, valid, boxes[slot] is not None, None))
            cv2.imwrite(str(directory/f'mask_slot_{slot}.png'), mask.astype(np.uint8)*255)
            write_json(directory/f'object_slot_{slot}.json', row)
            objects.append(row)
        reasons = [f'slot_{r["slot"]}: {reason}' for r in objects for reason in r['reasons']]
        if overlap['box_iou'] is not None and overlap['box_iou'] > .8:
            reasons.append('box_iou_above_0.8')
        offset = centroid_offset(camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels']),
                                 valid, masks, data['basis'])
        proposed = np.array([r['position'] for r in data['scene']])
        gap = None
        if offset['d_table_m'] is not None:
            proposed[1] = proposed[0] + offset['d_table_m']
            gap = footprint_clearance(proposed, np.array([r['dimensions_m'] for r in data['scene']]),
                                      np.array([r['quaternion_wxyz'] for r in data['scene']]))
        row = dict(episode=30, label_free=True, admitted=not reasons, reasons=reasons, objects=objects,
                   overlap=overlap, **offset, clearance_before_m=gap, wall_seconds=time.perf_counter()-started,
                   gpu_seconds=rec['gpu_seconds'])
        contact_sheet(data['rgb'], masks, boxes, NAMES, directory/'contact-sheet.png')
        write_json(directory/'episode.json', row)
    return directory/'run.json'


def construct() -> Path:
    """Freeze the maximal common footprint factor before any policy execution."""
    import numpy as np
    import pyarrow.parquet as pq
    from ._panel_scored import common_xy_shrink
    from .scored_heading_assets import origin
    masks = producer(f'{OUTPUT}/stage-a/run.json', 'pose_insurance.panel_scored_masks')
    rec = record('panel_scored_construction', parents=[masks[0]])
    reading = read_json(artifact(masks, 'episode.json', rec))
    if not reading['admitted']:
        raise ValueError('Stage A failed; no construction allowed')
    plan, parent, policy = incumbent(rec)
    task = pq.read_table(artifact(parent, 'task.parquet', rec)).to_pylist()[0]
    masses = []
    for slot, path in enumerate(task['object_urdf_paths']):
        link = ET.parse(project_path(ROOT, path)).getroot().find('link')
        rec['inputs'].append(input_file(ROOT, project_path(ROOT, path)))
        for key in ('collision', 'inertial', 'visual'):
            centre, rotation = origin(link.find(key))
            assert np.array_equal(centre, np.zeros(3)) and np.array_equal(rotation, np.eye(3))
        assert np.array_equal(np.fromstring(link.find('collision/geometry/box').get('size'), sep=' '), plan['scene'][slot]['dimensions_m'])
        masses.append(float(link.find('inertial/mass').get('value')))
    result = common_xy_shrink(plan['scene'], reading['d_table_m'], masses)
    result.update(episode=30, label_free=True, d_table_m=reading['d_table_m'], budget=BUDGET,
                  base_build=str(parent[0].relative_to(ROOT)), base_scene=plan['scene'],
                  v5_reproduction=dict(skipped=True, reason='Shipped row hash equals completed physical run',
                                       run=str(policy[0].relative_to(ROOT)), checks=policy[1]['checks']),
                  stage_a=input_file(ROOT, masks[0]), inputs=rec['inputs'])
    directory = output_path(ROOT, f'{OUTPUT}/construction')
    with recorded(directory, rec):
        write_json(directory/'construction.json', result)
    return directory/'run.json'


def build(construction: tuple) -> Path:
    """Write fresh solid-box assets and amend only the episode's task fields."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    con = read_json(construction[0].parent/'construction.json')
    parent = producer(con['base_build'], 'pose_insurance.build', Request(episode_index=30, budget=BUDGET))
    rec = record('build', parents=[parent[0], construction[0]])
    directory = output_path(ROOT, f'{OUTPUT}/build')
    with recorded(directory, rec):
        original = pq.read_table(artifact(parent, 'task.parquet', rec))
        task = original.to_pylist()[0]
        previous = deepcopy(task)
        evidence = []
        for slot, (row, props) in enumerate(zip(con['scene'], con['properties'])):
            assert task['object_body_wxyz'][0][slot] == row['quaternion_wxyz']
            source = project_path(ROOT, task['object_urdf_paths'][slot])
            tree = ET.parse(source)
            link = tree.getroot().find('link')
            assert float(link.find('inertial/mass').get('value')) == props['mass_kg']
            mesh = trimesh.creation.box(extents=row['dimensions_m'])
            target_mesh, target_urdf = directory/f'{row["name"]}.obj', directory/f'{row["name"]}.urdf'
            target_mesh.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17, include_texture=False))
            link.find('visual/geometry/mesh').set('filename', str(target_mesh))
            link.find('collision/geometry/box').set('size', ' '.join(map(str, row['dimensions_m'])))
            inertia = link.find('inertial/inertia')
            for key, value in zip(('ixx','iyy','izz'), props['inertia_diagonal_kg_m2']):
                inertia.set(key, str(value))
            for key in ('ixy','ixz','iyz'):
                inertia.set(key, '0')
            ET.indent(tree, space='  ')
            tree.write(target_urdf, encoding='unicode', xml_declaration=True)
            restored = ET.parse(target_urdf).getroot().find('link')
            assert float(restored.find('inertial/mass').get('value')) == props['mass_kg']
            assert np.array_equal(np.fromstring(restored.find('collision/geometry/box').get('size'), sep=' '), row['dimensions_m'])
            task['object_mesh_paths'][slot], task['object_urdf_paths'][slot] = str(target_mesh), str(target_urdf)
            task['object_mesh_radius'][slot] = props['radius_m']
            evidence.append(dict(slot=slot, **props, collision_dimensions_m=row['dimensions_m']))
        task['object_body_position'] = np.repeat([[r['position'] for r in con['scene']]], len(task['object_body_position']), axis=0).tolist()
        allowed = ('object_body_position', 'object_mesh_paths', 'object_urdf_paths', 'object_mesh_radius')
        assert all(task[k] == previous[k] for k in task if k not in allowed)
        pq.write_table(pa.Table.from_pylist([task], schema=original.schema), directory/'task.parquet')
        rec.update(split='evaluation', episode_index=30, scene=con['scene'], object_names=NAMES,
                   roster_frames=parent[1]['roster_frames'], task_frames=parent[1]['task_frames'],
                   dataset_revision=parent[1]['dataset_revision'], factor=con['factor'])
        write_json(directory/'assets.json', evidence)
    return directory/'run.json'


def policy() -> Path:
    """Run exactly the shipped 512-step budget and the real policy evaluator once."""
    from .handler import training_command, execute
    from .scored_scale_handler import optimizer_state, inspect
    construction = producer(f'{OUTPUT}/construction/run.json', 'pose_insurance.panel_scored_construction')
    con = read_json(artifact(construction, 'construction.json', {'inputs': []}))
    assert con['factor'] is not None and con['clearance_m'] >= .01
    parent = producer(build(construction), 'pose_insurance.build', Request(episode_index=30, budget=BUDGET))
    rec = record('run', parents=[parent[0]], inputs=[KIT/'flash_chord/scripts/evaluate_policy.py',
                 f'{OUTPUT}/geometry-check.json'])
    rec.update(split='evaluation', episode_index=30, scene=parent[1]['scene'],
               dataset_revision=parent[1]['dataset_revision'], owner_pid=os.getpid())
    directory = output_path(ROOT, f'{OUTPUT}/policy')
    with recorded(directory, rec):
        task = artifact(parent, 'task.parquet', rec)
        with lease('pose_insurance_panel_p', 1024) as gpu:
            started = time.perf_counter()
            try:
                execute(training_command(Request(episode_index=30, budget=BUDGET), task, directory),
                        directory/'training.log', os.environ.copy(), rec)
                rec.update(optimizer_state(directory))
                command = [str(ROOT/'.runtime/flashchord/venv/bin/python'), str(KIT/'flash_chord/scripts/evaluate_policy.py'),
                           f'evaluation.checkpoint={ROOT/rec["checkpoint"]}', 'evaluation.world_count=1',
                           'evaluation.reset_mode=explicit', 'evaluation.start_frame=0',
                           f'evaluation.object_trajectories_output={directory/"trajectory.parquet"}',
                           f'evaluation.metrics_output={directory/"official_metrics.json"}',
                           f'hydra.run.dir={directory/"evaluate_hydra"}', 'logging.mode=disabled']
                execute(command, directory/'evaluation.log', os.environ.copy(), rec)
                gpu.require_headroom()
            finally:
                rec.update(gpu=gpu.report(), gpu_seconds=time.perf_counter()-started)
        rec['checks'] = inspect(directory/'trajectory.parquet', parent[1])
        write_json(directory/'result.json', {k: rec[k] for k in ('checks','gpu_seconds','optimizer_updates','actor_adam_steps')})
    return directory/'run.json'


def decision() -> Path:
    """Choose adoption solely from input admission, proxy feasibility and physical checks."""
    root = output_path(ROOT, OUTPUT)
    masks = producer(root/'stage-a/run.json', 'pose_insurance.panel_scored_masks')
    stage = read_json(artifact(masks, 'episode.json', {'inputs': []}))
    adopt, rule, checks = False, 'adoption rule 3: Stage A input admission', None
    con, run = None, None
    if stage['admitted']:
        construction = producer(root/'construction/run.json', 'pose_insurance.panel_scored_construction')
        con = read_json(artifact(construction, 'construction.json', {'inputs': []}))
        rule = 'Amendment 1 item 10: no feasible positive common footprint factor'
        if con['factor'] is not None:
            run = producer(root/'policy/run.json', 'pose_insurance.run')
            checks = run[1]['checks']
            adopt = checks['accepted']
            rule = 'adoption rule 5: physical and export checks' + (' passed' if adopt else ' failed')
    manifest = read_json(MANIFEST)
    unchanged = [dict(path=e['path'], sha256=sha256_file(ROOT/e['path']), byte_identical=sha256_file(ROOT/e['path']) == e['sha256'])
                 for e in manifest['exports']]
    assert all(e['byte_identical'] for e in unchanged) and len(unchanged) == 20
    records = [read_json(p) for p in (root/'stage-a/run.json', root/'construction/run.json', root/'build/run.json', root/'policy/run.json') if p.exists()]
    result = dict(episode=30, verdict='PASS' if adopt else 'FAIL', decision='adopt' if adopt else 'keep v5',
                  rule=rule, scored_label_read=False, scored_score_read=False, stage_a_admitted=stage['admitted'],
                  d_table_m=stage['d_table_m'], factor=None if con is None else con['factor'], physical_checks=checks,
                  evaluator_output=None if run is None else input_file(ROOT, run[0].parent/'trajectory.parquet'),
                  adopted_export=None if not adopt else input_file(ROOT, run[0].parent/'trajectory.parquet'),
                  wall_seconds=(datetime.fromisoformat(records[-1]['finished_utc']) - datetime.fromisoformat(records[0]['started_utc'])).total_seconds(),
                  stage_wall_seconds=sum(r['timing']['wall_seconds'] for r in records),
                  gpu_seconds=sum(r.get('gpu_seconds', 0.) for r in records), incumbent_exports=unchanged,
                  inputs=[input_file(ROOT, p) for p in (PROTOCOL, MANIFEST, masks[0])],
                  limitation='Simulation consistency only; proxy shrink proves neither realistic geometry nor scored accuracy')
    write_json(root/'decision.json', result)
    return root/'decision.json'


def main() -> None:
    """Dispatch one explicitly saved stage without automatic retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('stage-a', 'construct', 'policy', 'decision'))
    args = parser.parse_args()
    environment()
    gates = read_json('outputs/pose_insurance/frame0-panel/stage-c/summary.json')['gates']['0']
    assert gates['P']['passed'] and not gates['TW4']['passed']
    actions = {'stage-a': stage_a, 'construct': construct, 'policy': policy, 'decision': decision}
    print(actions[args.stage](), flush=True)


if __name__ == '__main__':
    main()
