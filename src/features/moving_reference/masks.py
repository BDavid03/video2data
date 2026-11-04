"""Authorize, prepare, run and summarize the fixed two-episode M1 mask queue."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.gpu import GpuLease
from src.paths import project_path
from src.provenance import input_file, sha256_file, write_json
from src.runs import now_utc

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parent


def authorize(rule):
    """Admit only this task's fixed, video-only episode pairs and isolated environments."""
    if rule['execution']['masks_authorized'] is not True:
        raise PermissionError('M1 masks disabled')
    if rule['queue'] != [{'split': 'public', 'episode': 39}, {'split': 'evaluation', 'episode': 38}]:
        raise PermissionError('Only the ordered M1 39/38 queue is authorized')
    if Path(sys.executable).resolve() != project_path(ROOT, rule['python']):
        raise PermissionError('Use the project Linux interpreter')
    if rule['stream'] != 'ego_cam_c' or rule['maximum_hands'] != 2:
        raise ValueError('M1 expects the stereo-left stream and two hand slots')


def environment(directory):
    """Keep worker caches and temporary files in this episode's authorized output tree."""
    cache = directory / '.cache'
    cache.mkdir(exist_ok=True)
    temp = directory / '.tmp'
    temp.mkdir(exist_ok=True)
    result = os.environ.copy()
    for key in ('TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                'CUDA_CACHE_PATH', 'HF_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
        result[key] = str(cache / key.lower())
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        result[key] = str(temp)
    result.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', CUDA_VISIBLE_DEVICES='0')
    return result


def prepare(item, rule, directory):
    """Decode only the selected C video, preserving its recorded frame/PTS identities."""
    import cv2
    source = json.loads((ROOT / 'data/configs/sources.lock.json').read_text())['dataset']
    relative = f"track_3/{item['split']}/videos/chunk-000/observation.images.ego_cam_c/episode_{item['episode']:06d}.mp4"
    video = project_path(ROOT, f"{source['cache_path']}/snapshots/{source['revision']}/{relative}")
    audit = ROOT / 'data/reports/data-audit/videos.json'
    clocks = json.loads(audit.read_text())
    clock, = [row for row in clocks['videos'] if row['path'] == relative]
    if clock['status'] != 'ok' or clock.get('rotation_metadata_lines') or clocks['revision'] != source['revision']:
        raise ValueError('Video inventory is incomplete or differs from the pinned revision')
    count = clock['decoded_frame_count']
    if len(clock['pts_integer']) != count or len(clock['pts_seconds']) != count:
        raise ValueError('Video timestamps are incomplete')
    sources = [*sorted(FEATURE.glob('*.py')), FEATURE / 'runtime.json', ROOT / 'src/gpu.py']
    inputs = [video, audit, ROOT / 'data/configs/sources.lock.json', ROOT / 'data/configs/calibration/ego.json']
    weights = [*sorted((ROOT / 'data/.runtime/grounding-dino/weights').glob('*')),
               ROOT / 'data/.runtime/sam2/weights/sam2.1_hiera_large.pt',
               ROOT / 'data/.runtime/sam2/source/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_l.yaml']
    identity = dict(sources=[input_file(ROOT, path) for path in sources],
                    inputs=[input_file(ROOT, path) for path in inputs],
                    weights=[input_file(ROOT, path) for path in weights if path.is_file()])
    job_path = directory / 'request.json'
    if job_path.exists():
        old = json.loads(job_path.read_text())
        if old['identity'] != identity or old['rule'] != rule or old['episode'] != item:
            raise ValueError('Existing mask run has a different input/source identity; preserve it')
        return old
    frames = directory / 'input_frames'
    frames.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError('Cannot open the selected video')
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    hashes = {}
    try:
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Video ended at frame {index}, expected {count}')
            height, width = frame.shape[:2]
            scale = min(1.0, rule['maximum_input_side'] / max(width, height))
            size = (round(width * scale), round(height * scale))
            if scale < 1:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            path = frames / f'{index:06d}.jpg'
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError('Could not persist model input')
            hashes[path.name] = sha256_file(path)
        if capture.read()[0]:
            raise ValueError('Video contains more frames than the inventory')
    finally:
        capture.release()
    job = dict(episode=item, rule=rule, identity=identity, frame_count=count,
               pan_box_selection='smallest_area_above_g3_threshold',
               source_resolution_width_height=[width, height], mask_shape=[size[1], size[0]],
               input_resize_factor=scale, sam_internal_square_side=1024,
               grid='raw distorted ego_cam_c, no rectification or rotation; resize only',
               timestamps=clock['pts_seconds'], pts_integer=clock['pts_integer'],
               time_base_and_frame_rate=clock['time_base_and_frame_rate'],
               prepared_jpeg_hashes=hashes, dataset_revision=source['revision'])
    write_json(job_path, job)
    return job


def summarize(directory, job, record):
    """Write coverage measurements and a 12-frame contact sheet after releasing the GPU."""
    import cv2
    import numpy as np
    from ._masks import track_statistics
    count = job['frame_count']
    areas = np.zeros((count, 3), dtype=np.int64)
    forward_frames = reverse_frames = 0
    agreements = [[], [], []]
    sample_indices = np.rint(np.linspace(0, count - 1, 12)).astype(int).tolist()
    tiles = []
    colors = np.asarray([[40, 230, 40], [230, 70, 220], [255, 180, 30]], dtype=np.uint8)
    for index in range(count):
        masks = np.zeros((3, *job['mask_shape']), dtype=bool)
        path = directory / 'frames' / f'frame_{index:06d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as arrays:
                if int(arrays['frame_index']) != index or float(arrays['timestamp_seconds']) != job['timestamps'][index]:
                    raise ValueError('Saved mask identities differ from the source clock')
                masks = arrays['masks'].copy()
                reverse_frames += int(arrays['reverse_processed'])
                for slot, iou in enumerate(arrays['forward_reverse_iou']):
                    if iou >= 0:
                        agreements[slot].append(float(iou))
            forward_frames += 1
        areas[index] = masks.reshape(3, -1).sum(axis=1)
        if index in sample_indices:
            tile = cv2.imread(str(directory / 'input_frames' / f'{index:06d}.jpg'))
            for slot, mask in enumerate(masks):
                tile[mask] = (0.6 * tile[mask] + 0.4 * colors[slot]).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, contours, -1, tuple(int(x) for x in colors[slot]), 1)
            tile = cv2.resize(tile, (400, round(tile.shape[0] * 400 / tile.shape[1])))
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(tile, f"ep{job['episode']['episode']} f{index} t={job['timestamps'][index]:.2f}s", (8, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[start:start + 4]) for start in range(0, 12, 4)])
    cv2.imwrite(str(directory / 'contact-sheet.png'), sheet)
    names = ['pan', 'hand_0', 'hand_1']
    tracks = {name:track_statistics(areas[:, slot], job['timestamps']) for slot, name in enumerate(names)}
    for slot, name in enumerate(names):
        tracks[name]['forward_reverse_iou_median'] = float(np.median(agreements[slot])) if agreements[slot] else None
    detection = json.loads((directory / 'detections.json').read_text()) if (directory / 'detections.json').exists() else None
    summary = dict(status=record['status'], episode=job['episode'], stream='ego_cam_c',
                   pan_box_selection=job['pan_box_selection'],
                   frame_count=count, saved_frames=forward_frames, reverse_processed_frames=reverse_frames,
                   source_resolution_width_height=job['source_resolution_width_height'],
                   mask_resolution_width_height=job['mask_shape'][::-1], input_resize_factor=job['input_resize_factor'],
                   sam_internal_square_side=1024, grid=job['grid'], tracks=tracks,
                   coverage_definition='nonempty mask, not a correctness or usability measurement',
                   hand_identity='anonymous slots matched by box centers; no anatomical identity claim',
                   pan_detection=None if detection is None else detection['pan'],
                   seconds=record['wall_seconds'], stages=record.get('stages', {}), gpu=record.get('gpu'),
                   identity=job['identity'], contact_sheet_indices=sample_indices,
                   error=record.get('error'), completed_utc=now_utc())
    summary['peak_gpu_reserved_mib'] = max((x.get('peak_reserved_mib', 0) for x in summary['stages'].values()), default=0)
    summary['peak_sampled_device_mib'] = (record.get('gpu') or {}).get('peak_sampled_total_device_used_mib')
    write_json(directory / 'summary.json', summary)
    return summary


def select_pan_box(directory, rule):
    """Apply the frozen smallest-box rule to the detector's unchanged prompt results."""
    from ._masks import smallest_box_index
    path = directory / 'detections.json'
    detection = json.loads(path.read_text())
    original = directory / 'detections-highest-score.json'
    if not original.exists():
        write_json(original, detection)
    detection['pan'] = None
    detection['anchors']['0'] = []
    for attempt in detection['pan_attempts']:
        selected = smallest_box_index(attempt['boxes'], attempt['scores'], rule['box_threshold'])
        if selected is not None:
            detection['pan'] = dict(frame_index=0, box=attempt['boxes'][selected],
                                    score=attempt['scores'][selected], prompt=attempt['prompt'],
                                    selected_index=selected)
            detection['anchors']['0'] = [0]
            break
    detection['pan_box_selection'] = 'smallest_area_above_g3_threshold'
    write_json(path, detection)
    return detection['pan']


def run(episode=None, output_name='masks'):
    """Run the frozen episode queue sequentially, skipping completed stages on resume."""
    rule = json.loads((FEATURE / 'runtime.json').read_text())
    authorize(rule)
    if output_name not in ('masks', 'masks-smallest') or (output_name == 'masks-smallest' and episode != 38):
        raise PermissionError('The smallest-box output is authorized only for episode 38')
    deadline = datetime.fromisoformat(rule['stop_utc']) if rule.get('stop_utc') else None
    queue = [item for item in rule['queue'] if episode is None or item['episode'] == episode]
    for item in queue:
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            print('GPU deadline reached; remaining queue not started', flush=True)
            break
        directory = ROOT / f"data/outputs/moving_reference/ep{item['episode']:02d}/{output_name}"
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        job = prepare(item, rule, directory)
        run_path = directory / 'run.json'
        if run_path.exists() and json.loads(run_path.read_text())['status'] == 'completed':
            print(f"Already completed: {run_path}", flush=True)
            continue
        record = dict(kind='moving_reference.masks', status='running', provenance='video_only',
                      episode=item, started_utc=now_utc(), request='request.json', stages={})
        write_json(run_path, record)
        lease = None
        try:
            with GpuLease(ROOT, 1024, 10000, stage='moving_reference.masks') as lease:
                for operation, environment_name in [('detect', 'grounding-dino'), ('segment', 'sam2')]:
                    lease.require_headroom()
                    report_path = directory / f'{operation}-report.json'
                    existing = json.loads(report_path.read_text()) if report_path.exists() else None
                    if existing and existing['status'] == 'completed':
                        if operation == 'detect':
                            select_pan_box(directory, rule)
                        record['stages'][operation] = existing
                        continue
                    if deadline is not None and datetime.now(timezone.utc) >= deadline:
                        raise TimeoutError('GPU deadline reached')
                    command = [str(ROOT / f'data/.runtime/{environment_name}/venv/bin/python'), '-B',
                               '-m', 'src.features.moving_reference._worker', operation, str(directory / 'request.json')]
                    print(f"ep{item['episode']} {operation} started {now_utc()}", flush=True)
                    with (directory / f'{operation}.log').open('a') as log:
                        result = subprocess.run(command, cwd=ROOT, env=environment(directory),
                                                stdout=log, stderr=subprocess.STDOUT)
                    if report_path.exists():
                        record['stages'][operation] = json.loads(report_path.read_text())
                    if result.returncode:
                        raise RuntimeError(f'{operation} exited {result.returncode}; inspect {operation}.log')
                    if operation == 'detect':
                        pan = select_pan_box(directory, rule)
                        report = record['stages'][operation]
                        report.update(pan_detected=pan is not None,
                                      pan_box_selection='smallest_area_above_g3_threshold', selected_pan=pan)
                        write_json(report_path, report)
                    write_json(run_path, record)
                record['status'] = 'completed'
        except Exception as error:
            record.update(status='interrupted' if isinstance(error, TimeoutError) else 'failed',
                          error=f'{type(error).__name__}: {error}')
        finally:
            record['wall_seconds'] = time.perf_counter() - started
            record['completed_utc'] = now_utc()
            record['gpu'] = lease.report() if lease else None
            write_json(run_path, record)
            summary = summarize(directory, job, record)
            with (directory / 'ledger.jsonl').open('a') as ledger:
                ledger.write(json.dumps(dict(status=record['status'], episode=item, wall_seconds=record['wall_seconds'],
                                             saved_frames=summary['saved_frames'], completed_utc=record['completed_utc'])) + '\n')
            record['artifacts'] = {str(path.relative_to(directory)):sha256_file(path)
                                   for path in sorted(directory.rglob('*'))
                                   if path.is_file() and '.cache' not in path.parts and '.tmp' not in path.parts and path != run_path}
            write_json(run_path, record)
        print(json.dumps(dict(episode=item, status=record['status'], saved_frames=summary['saved_frames'],
                              tracks=summary['tracks'], summary=str(directory / 'summary.json'))), flush=True)
        if record['status'] != 'completed':
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the fixed smallest-box M1 mask correction')
    parser.add_argument('--episode', type=int, choices=[38], required=True)
    parser.add_argument('--output-name', choices=['masks-smallest'], required=True)
    options = parser.parse_args()
    run(options.episode, options.output_name)
