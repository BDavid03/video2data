"""Run and resume the protocol's label-free Stage A on its seven fixed pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from track3.gpu import GpuLease
from track3.paths import output_path, project_path
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, checked_entry,
                         heavy_execution, now_utc, producer, read_json, recorded, runtime_flag)
from .registration_h3_loop import append, identity

FEATURE = Path(__file__).parent
OUTPUT = 'outputs/pose_insurance/frame0-panel'
QUEUE = 'track3/features/pose_insurance/panel_queue.json'
PROTOCOL = 'docs/chain/frame0-panel-protocol.md'
METHOD = 'docs/chain/pose-heading-method-hr2.json'
METHOD_HASH = 'c1fd86776f0a5e39eaf0f79273cbb8c6098ef585b59783b00a30ef1d65772e25'
ROSTER = {0: ['white_pot', 'white_pot_lid'], 26: ['plastic_dish_rack', 'white_pot'],
          31: ['mini_sweeper', 'plastic_dish_rack'], 11: ['blue_cup', 'beige_cup'],
          12: ['blue_cup', 'plastic_dish_rack'], 18: ['mini_sweeper', 'mini_dust_pan'],
          21: ['blue_cup', 'water_pitcher']}


def environment(flag: str) -> dict:
    """Admit the existing runtime and put worker caches inside the panel output."""
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, flag, 'Frame-zero panel')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    for key in CACHE_FOLDERS:
        path = output_path(ROOT, f'{OUTPUT}/cache/{key.lower()}')
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    return runtime


def contract(queue_path: str) -> dict:
    """Bind the exact episode roster, model settings and committed protocol."""
    queue = read_json(queue_path)
    if queue['episodes'] != list(ROSTER) or queue['registration_episodes'] != [0, 26, 31]:
        raise PermissionError('Panel roster differs from the fixed protocol')
    for entry in (queue['protocol'], queue['method']):
        checked_entry(entry, {'inputs': []})
    if queue['protocol']['path'] != PROTOCOL or queue['method'] != dict(path=METHOD, sha256=METHOD_HASH):
        raise PermissionError('Pinned frame-zero protocol and HR2 method required')
    if queue['detector'] != dict(revision='a2bb814dd30d776dcf7e30523b00659f4f141c71',
                                box_threshold=.4, text_threshold=.3):
        raise PermissionError('Frozen detector thresholds required')
    if queue['segmenter'] != dict(source_revision='2b90b9f5ceec907a1c18123530e92e794ad901a4',
                                 config='configs/sam2.1/sam2.1_hiera_l.yaml', multimask_output=True,
                                 postprocessing='largest connected component only'):
        raise PermissionError('Frozen SAM2.1 rule required')
    if queue['max_trials'] != 7 or not 1 <= queue['max_seconds'] <= 7200 or not 1 <= queue['cpu_workers'] <= 3:
        raise ValueError('Panel requires bounded trial/time/process budgets')
    return queue


def base_record(operation: str, request: dict, paths: list) -> dict:
    """Record the adapter without importing or calling any label-dependent handler."""
    return run_fields(ROOT, 'pose_insurance', operation, request,
                      inputs=[PROTOCOL, METHOD, FEATURE/'runtime.json', *paths],
                      provenance='public_asset_diagnostic')


def resume(path: Path) -> dict:
    """Re-admit completed output bytes and all explicit consumed inputs."""
    parent = producer(path, 'pose_insurance.panel_masks', allowed=('public_asset_diagnostic',))
    for entry in (*parent[1]['inputs'], *parent[1]['sources']):
        checked_entry(entry, {'inputs': []})
    for name in parent[1]['artifacts']:
        artifact(parent, name, {'inputs': []})
    return read_json(path.parent/'episode.json')


def prepare(episode: int, queue: dict, record: dict, directory: Path) -> dict:
    """Load the T1 build's exact video grid, stereo and depth-cluster background."""
    import cv2
    import numpy as np
    from ._scene import table_basis
    from .length_diagnostic import frame_arrays, load_stereo
    request = SimpleNamespace(source_split='public', episode_index=episode)
    build = producer(queue['builds'][str(episode)], 'pose_insurance.stereo_placement_build', request)
    record['inputs'].append(input_file(ROOT, build[0]))
    construction = read_json(artifact(build, 'construction.json', record))
    if construction['names'] != ROSTER[episode]:
        raise ValueError('T1 object order differs from panel roster')
    measure_entry = next(e for e in build[1]['inputs'] if e['path'].startswith(
        f'outputs/mesh_extents/public/episode_{episode:06d}/') and e['path'].endswith('/run.json'))
    measure = producer(checked_entry(measure_entry, record), 'mesh_extents.measure', request)
    depth_entry = next(e for e in build[1]['inputs'] if e['path'] == measure[1]['request']['depth_run'])
    depth = producer(checked_entry(depth_entry, record), 'depth.generate', request)
    scale_entry = next(e for e in build[1]['inputs'] if e['path'] == f'outputs/pose_insurance/scale/stereo_{episode:06d}.json')
    scale = read_json(checked_entry(scale_entry, record))
    for anchor_path in scale['anchor_runs']:
        checked_entry(next(e for e in build[1]['inputs'] if e['path'] == anchor_path), record)
    stereo = load_stereo(episode, scale, depth, record, request)
    with np.load(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256'])) as grid:
        intrinsics = grid['input_intrinsics_pixels'].copy()
        source_x, source_y = grid['source_x'].copy(), grid['source_y'].copy()
    with np.load(artifact(measure, 'labels_000000.npz', record)) as arrays:
        clusters = arrays['label'].copy()
    if (measure[1]['frames'][0]['frame_index'] != 0 or stereo['rgb_depth_m'].shape != clusters.shape
            or not np.allclose(stereo['rgb_intrinsics_pixels'], intrinsics, atol=1e-5, rtol=1e-5)):
        raise ValueError('Frame-zero mask/stereo timestamps, grid or intrinsics differ')
    overlay = cv2.imread(str(artifact(measure, 'overlay_000000.png', record)))
    rgb = overlay[:, :overlay.shape[1]//2]
    video = next(e for e in depth[1]['inputs'] if '/observation.images.ego_cam_a/' in e['path'])
    capture = cv2.VideoCapture(str(checked_entry(video, record)))
    ok, raw = capture.read()
    capture.release()
    if not ok or not np.array_equal(rgb, cv2.remap(raw, source_x, source_y, cv2.INTER_LINEAR)):
        raise ValueError('Overlay RGB differs from the recorded frame-zero camera-A remap')
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory/'rgb.png'), rgb)
    normal = np.array(measure[1]['frames'][0]['plane']['normal'])
    basis = table_basis(normal)
    evidence = dict(frame_index=0, pts=0, camera='ego_cam_a', rgb_kind='recorded_camera_a_remap',
                    remap_pixel_equivalence=True, grid_sha256=depth[1]['preprocessing_artifact_sha256'],
                    intrinsics_pixels=intrinsics.tolist(), normal_hint=normal.tolist(), basis=basis.tolist(),
                    t1_build=input_file(ROOT, build[0]), inputs=record['inputs'].copy())
    write_json(directory/'inputs.json', evidence)
    np.savez_compressed(directory/'frame.npz', rgb_depth_m=stereo['rgb_depth_m'], rgb_valid=stereo['rgb_valid'],
                        rgb_intrinsics_pixels=intrinsics, cluster_labels=clusters, normal_hint=normal, basis=basis)
    return dict(stereo=stereo, clusters=clusters, basis=basis, construction=construction, evidence=evidence)


def infer(operation: str, job: dict, directory: Path) -> dict:
    """Run the unchanged isolated G3 worker under the caller's single GPU lease."""
    directory.mkdir(parents=True, exist_ok=True)
    target = project_path(ROOT, job['output'])
    report_path = target if operation == 'detect' else target.with_suffix('.json')
    request_path = directory/f'{operation}-request.json'
    job_hash = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if report_path.exists():
        report = read_json(report_path)
        if report.get('job_sha256') != job_hash:
            raise ValueError('Unsealed or changed inference; preserve for review')
        if operation == 'segment' and sha256_file(target) != report['mask_sha256']:
            raise ValueError('Raw SAM mask changed')
        return report
    marker = directory/f'{operation}.started.json'
    if marker.exists() or target.exists():
        raise RuntimeError('Unfinished inference: preserve for review, never automatically retry')
    write_json(request_path, job)
    write_json(marker, dict(started_utc=now_utc(), job_sha256=job_hash))
    name = 'grounding-dino' if operation == 'detect' else 'sam2'
    script = project_path(ROOT, f'track3/features/mesh_generative/g3_{operation}.py')
    with (directory/f'{operation}.log').open('w') as log:
        subprocess.run([str(ROOT/f'.runtime/{name}/venv/bin/python'), '-B', str(script),
                        '--request', str(request_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read_json(report_path)
    report['job_sha256'] = job_hash
    if operation == 'segment':
        report['mask_sha256'] = sha256_file(target)
    write_json(report_path, report)
    return report


def contact_sheet(rgb, masks: list, boxes: list, names: list, path: Path) -> None:
    """Write original RGB and both post-overlap silhouettes with detector boxes."""
    import cv2
    import numpy as np
    panels = [rgb.copy()]
    for slot, mask in enumerate(masks):
        panel = rgb.copy()
        colour = np.array((40, 210, 50) if slot == 0 else (230, 130, 40))
        panel[mask] = (.55*panel[mask] + .45*colour).astype(np.uint8)
        if boxes[slot] is not None:
            x0, y0, x1, y1 = np.rint(boxes[slot]).astype(int)
            cv2.rectangle(panel, (x0,y0), (x1,y1), colour.tolist(), 2)
        panels.append(panel)
    labelled = []
    for panel, title in zip(panels, ['Frame 0: camera-A remap', *[f'Slot {i}: {name}' for i,name in enumerate(names)]]):
        panel = cv2.copyMakeBorder(panel, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))
        cv2.putText(panel, title, (8,26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
        labelled.append(panel)
    cv2.imwrite(str(path), np.concatenate(labelled, axis=1))


def episode_masks(episode: int, queue: dict, prepared: Path, data: dict, directory: Path) -> dict:
    """Detect each object once and freeze overlap-resolved masks and measurements."""
    import cv2
    import numpy as np
    from track3.features.mesh_generative.g3_math import largest_component
    from ._panel_masks import box_iou, resolve_overlap, mask_reading, centroid_offset, placement_reading
    from ._scene import camera_points
    names = ROSTER[episode]
    prompts = read_json('track3/features/object_masks/prompts.json')['objects']
    raw_masks, detections, segmentations = [], [], []
    image = str((prepared/'rgb.png').relative_to(ROOT))
    shape = data['clusters'].shape
    for slot, name in enumerate(names):
        work = prepared/f'slot_{slot}'
        detection = infer('detect', dict(image=image, rule=dict(queue['detector'], prompt=prompts[name]+'.'),
                          output=str((work/'detect.json').relative_to(ROOT))), work)
        segmentation, mask = None, np.zeros(shape, bool)
        if detection['box'] is not None:
            segmentation = infer('segment', dict(image=image, box=detection['box'], rule=queue['segmenter'],
                                 output=str((work/'sam.npz').relative_to(ROOT))), work)
            with np.load(work/'sam.npz', allow_pickle=False) as arrays:
                mask = largest_component(arrays['mask'])
            if mask.shape != shape:
                raise ValueError('Segmenter mask differs from the admitted stereo grid')
        raw_masks.append(mask)
        detections.append(detection)
        segmentations.append(segmentation)
    masks, overlap = resolve_overlap(raw_masks)
    boxes = [d['box'] for d in detections]
    overlap['box_iou'] = box_iou(boxes)
    stereo = data['stereo']
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    objects = []
    for slot, (name, mask) in enumerate(zip(names, masks)):
        ids = data['construction']['observations']['cluster_ids'][slot]['observed']
        reference = np.isin(data['clusters'], np.array(ids)+1) if episode in (11,12,18,21) else None
        row = dict(slot=slot, name=name, box=boxes[slot], score=detections[slot]['box_score'],
                   predicted_iou=None if segmentations[slot] is None else segmentations[slot]['selected_predicted_iou'],
                   **mask_reading(mask, valid, boxes[slot] is not None, reference))
        cv2.imwrite(str(directory/f'mask_slot_{slot}.png'), mask.astype(np.uint8)*255)
        cv2.imwrite(str(directory/f'component_slot_{slot}.png'), raw_masks[slot].astype(np.uint8)*255)
        write_json(directory/f'object_slot_{slot}.json', row)
        objects.append(row)
    reasons = [f'slot_{r["slot"]}: {reason}' for r in objects for reason in r['reasons']]
    if overlap['box_iou'] is not None and overlap['box_iou'] > .8:
        reasons.append('box_iou_above_0.8')
    offset = centroid_offset(camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels']),
                             valid, masks, data['basis'])
    placement = placement_reading(data['construction']['scenes']['T1'], offset['d_table_m'], not reasons)
    row = dict(episode=episode, names=names, label_free=True, admitted=not reasons, reasons=reasons,
               objects=objects, overlap=overlap, **offset, placement=placement, grid=data['evidence'])
    write_json(directory/'episode.json', row)
    contact_sheet(cv2.imread(str(prepared/'rgb.png')), masks, boxes, names, directory/'contact-sheet.png')
    return row


def run(queue_path: str = QUEUE) -> Path:
    """Complete or resume all seven pairs under one sequential detector/segmenter lease."""
    started = time.monotonic()
    runtime = environment('registration_authorized')
    queue = contract(queue_path)
    root = output_path(ROOT, OUTPUT)
    rows, executed, resumed = [], 0, 0
    prior = [json.loads(line) for line in (root/'stage-a-ledger.jsonl').read_text().splitlines()] if (root/'stage-a-ledger.jsonl').exists() else []
    lease = None
    try:
        for episode in queue['episodes']:
            authorize_episode(runtime, SimpleNamespace(source_split='public', episode_index=episode), 'Frame-zero panel')
            record = base_record('panel_masks', dict(episode=episode, queue=queue), [queue_path,
                'track3/features/object_masks/prompts.json', 'track3/features/mesh_generative/g3_detect.py',
                'track3/features/mesh_generative/g3_segment.py', 'track3/features/mesh_generative/g3_math.py'])
            for model in ('grounding-dino', 'sam2'):
                selected = read_json(f'.runtime/{model}/weights-selected.json')
                expected = queue['detector']['revision'] if model == 'grounding-dino' else queue['segmenter']['source_revision']
                if selected['revision'] != expected:
                    raise ValueError('Installed model revision differs from protocol')
                record['inputs'].append(input_file(ROOT, ROOT/f'.runtime/{model}/weights-selected.json'))
                for entry in selected['files']:
                    checked_entry({k:entry[k] for k in ('path','sha256')}, record)
            prepared = root/f'prepared/episode_{episode:06d}'
            data = prepare(episode, queue, record, prepared)
            record['inputs'].extend(input_file(ROOT, prepared/name) for name in ('rgb.png','frame.npz','inputs.json'))
            key = identity(record)
            previous = next((r for r in reversed(prior) if r['identity'] == key), None)
            target = root/f'stage-a/episode_{episode:06d}'
            if previous:
                row = resume(ROOT/previous['run'])
                resumed += 1
            else:
                if time.monotonic() - started > queue['max_seconds'] or executed >= queue['max_trials']:
                    raise RuntimeError('Stage A budget reached; resume unchanged queue')
                if lease is None:
                    lease = GpuLease(ROOT, 1024, 10000, stage='mesh_generative')
                    lease.__enter__()
                record.update(identity=key, split='public', episode_index=episode)
                with recorded(target, record):
                    row = episode_masks(episode, queue, prepared, data, target)
                append(root/'stage-a-ledger.jsonl', dict(identity=key, episode=episode, status='completed',
                                                       run=str((target/'run.json').relative_to(ROOT))))
                executed += 1
            rows.append(row)
            print(json.dumps(dict(stage='A', episode=episode, admitted=row['admitted'], reasons=row['reasons'],
                                  d_table_m=row['d_table_m'])), flush=True)
    finally:
        if lease is not None:
            write_json(root/f'gpu-lease-{time.time_ns()}.json', lease.report())
            lease.__exit__(None, None, None)
    write_json(root/'stage-a.json', dict(label_free=True, episodes=rows, executed=executed, resumed=resumed,
                                        wall_seconds=time.monotonic()-started))
    return root/'stage-a.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default=QUEUE)
    run(parser.parse_args().queue)
