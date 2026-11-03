"""Recorded frame inputs, isolated mask workers and immutable mask producer records."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from track3.gpu import GpuLease
from track3.paths import PROJECT as ROOT
from track3.provenance import input_file, run_fields, sha256_file, write_json
from track3.runs import now_utc
from .g2 import key

FEATURE = Path(__file__).parent


def prepare(pair, directory, control=False):
    """Read the pinned overlay/depth and preserve their validity masks without inference."""
    import cv2
    import numpy as np
    from track3.features.object_masks.opening_clusters import _object_names

    parent = ROOT/pair['measurement']
    measure = json.loads(parent.read_text())
    if (measure['status'] != 'completed' or measure['provenance'] != 'video_only'
        or measure['split'] != pair['source_split'] or measure['episode_index'] != pair['episode']
        or measure['frames'][0]['frame_index'] != 0):
        raise ValueError('Not a matching completed video-only frame-zero measurement')
    depth_entry = next(e for e in measure['inputs'] if '/depth/' in e['path'] and e['path'].endswith('run.json'))
    depth_path = ROOT/depth_entry['path']
    if sha256_file(depth_path) != depth_entry['sha256']:
        raise ValueError('Depth producer changed')
    depth = json.loads(depth_path.read_text())
    frame = next(f for f in depth['frames'] if f['frame_index'] == 0 and f['status'] == 'completed')
    files = [(parent, None), (parent.parent/'overlay_000000.png', measure['artifacts']['overlay_000000.png']),
             (depth_path, None), (depth_path.parent/'preprocessing.npz', depth['preprocessing_artifact_sha256']),
             (depth_path.parent/frame['artifact'], frame['sha256'])]
    if control:
        files.append((parent.parent/'labels_000000.npz', measure['artifacts']['labels_000000.npz']))
    inputs = []
    for path, digest in files:
        entry = input_file(ROOT, path)
        if digest and entry['sha256'] != digest:
            raise ValueError(f'Producer artifact changed: {path}')
        inputs.append(entry)
    video = next(e for e in measure['inputs'] + depth['inputs'] if '/observation.images.ego_cam_a/' in e['path'])
    video_input = input_file(ROOT, ROOT/video['path'])
    if video_input['sha256'] != video['sha256']:
        raise ValueError('Recorded source video changed')
    inputs.append(video_input)
    image = cv2.imread(str(files[1][0]))
    rgb = image[:, :image.shape[1]//2]
    if rgb.shape != (576, 768, 3):
        raise ValueError('G3 requires the recorded 576 x 768 RGB')
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory/'rgb.png').exists():
        cv2.imwrite(str(directory/'rgb.png'), rgb)
    elif not np.array_equal(cv2.imread(str(directory/'rgb.png')), rgb):
        raise ValueError('Prepared RGB differs from the frozen overlay')
    with np.load(files[3][0], allow_pickle=False) as prep, np.load(files[4][0], allow_pickle=False) as frame:
        z = frame['depth_m'].astype(float)
        valid = prep['input_valid'] & frame['valid'] & frame['input_valid'] & np.isfinite(z) & (z > 0)
        if not (directory/'frame.npz').exists():
            np.savez_compressed(directory/'frame.npz', depth_m=z, observed_valid=valid,
                                remap_valid=prep['input_valid'], intrinsics=prep['input_intrinsics_pixels'])
    names = _object_names(SimpleNamespace(source_split=pair['source_split'], episode_index=pair['episode']),
                          measure['dataset_revision'])
    if control:
        assignments = json.loads((ROOT/'track3/features/mesh_extents/assignments.json').read_text())
        ids = assignments['episodes'][f"{pair['source_split']}:{pair['episode']}"]['objects'][pair['object']]
        if measure['request']['objects'][pair['object']] != ids:
            raise ValueError('Recorded control assignment changed')
        with np.load(files[-1][0], allow_pickle=False) as labels:
            if not (directory/'reference_mask.npz').exists():
                np.savez_compressed(directory/'reference_mask.npz', mask=np.isin(labels['label'], np.array(ids)+1))
    return dict(inputs=inputs, object_names=names, lid='white_pot_lid' in names, source_video=video)


def infer(operation, pair, job, directory):
    """Run one mask model under the parent lease, preserving completed predictions on resume."""
    target = ROOT/job['output']
    report_path = target if operation == 'detect' else target.with_suffix('.json')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if 'gpu_lease' not in report or report['rule'] != job['rule']:
            raise ValueError('Incomplete or changed mask prediction; no automatic retry')
        return report
    marker = directory/f'{operation}.started.json'
    if marker.exists() or target.exists():
        raise RuntimeError(f'Unfinished mask inference; preserve evidence: {marker}')
    environment = 'grounding-dino' if operation == 'detect' else 'sam2'
    request = directory/f'{operation}-request.json'
    write_json(request, job)
    log_path = directory.parent.parent/f'g3-{key(pair)}-{operation}.log'
    with GpuLease(ROOT, 1024, 10000, stage='mesh_generative') as lease:
        write_json(marker, dict(started_utc=now_utc(), request=input_file(ROOT, request)))
        with log_path.open('w') as log:
            subprocess.run([str(ROOT/f'.runtime/{environment}/venv/bin/python'), '-B',
                            str(FEATURE/f'g3_{operation}.py'), '--request', str(request)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        telemetry = lease.report()
    report = json.loads(report_path.read_text())
    report['gpu_lease'] = telemetry
    write_json(report_path, report)
    return report


def make_mask(pair, directory, queue, prepared, control=False):
    """Detect once, segment once and seal the largest component plus rejection measurements."""
    import cv2
    import numpy as np
    from .g3_math import largest_component, mask_checks, iou
    from ._geometry import masked_square
    from track3.features.mesh_extents._geometry import camera_points

    path = directory/'mask.json'
    if path.exists():
        result = json.loads(path.read_text())
        for name, digest in result['artifacts'].items():
            if sha256_file(directory/name) != digest:
                raise ValueError('Sealed mask artifact changed')
        return result
    image = str((directory/'rgb.png').relative_to(ROOT))
    detection = infer('detect', pair, dict(image=image, rule=queue['detector'],
                      output=str((directory/'detect.json').relative_to(ROOT))), directory)
    result = run_fields(ROOT, 'mesh_generative', 'g3_mask', pair,
                        inputs=[ROOT/e['path'] for e in prepared['inputs']])
    result.update(detection=detection, segmentation=None, box_score=detection['box_score'],
                  accepted=False, reason='no detector box', control_iou=0.0 if control else None)
    if detection['box'] is not None:
        segmentation = infer('segment', pair, dict(image=image, box=detection['box'], rule=queue['segmenter'],
                             output=str((directory/'sam.npz').relative_to(ROOT))), directory)
        with np.load(directory/'sam.npz', allow_pickle=False) as raw:
            mask = largest_component(raw['mask'])
        with np.load(directory/'frame.npz', allow_pickle=False) as frame:
            checks = mask_checks(mask, frame['remap_valid'], frame['observed_valid'], queue['mask'])
            observed = camera_points(frame['depth_m'], frame['intrinsics'])[mask & frame['observed_valid']]
        np.savez_compressed(directory/'mask.npz', mask=mask)
        np.savez_compressed(directory/'observed.npz', points=observed)
        if mask.any():
            rgb = cv2.cvtColor(cv2.imread(str(directory/'rgb.png')), cv2.COLOR_BGR2RGB)
            crop, bounds = masked_square(rgb, mask)
            cv2.imwrite(str(directory/'crop.png'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            result['crop_bounds'] = bounds
        result.update(segmentation=segmentation, **checks)
        if control:
            with np.load(directory/'reference_mask.npz', allow_pickle=False) as reference:
                result['control_iou'] = iou(mask, reference['mask'])
    result.update(status='completed', completed_utc=now_utc(), artifacts={
        p.name:sha256_file(p) for p in directory.iterdir() if p.is_file() and p.name != 'mask.json'})
    write_json(path, result)
    return result
