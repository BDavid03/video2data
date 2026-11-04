"""Isolated DINO and SAM workers under the orchestration process's GPU lease."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[3]


def check_time(rule):
    """Leave time for interpreter teardown before the owner's GPU deadline, when one is set."""
    if rule.get('stop_utc') and datetime.now(timezone.utc) >= datetime.fromisoformat(rule['stop_utc']):
        raise TimeoutError('M1 GPU deadline reached')


def save_json(path, value):
    """Replace a worker record atomically next to its destination."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def detect(directory, job, torch):
    """Detect the pan at frame zero and hands at the predeclared fixed cadence."""
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from src.features.moving_reference._masks import hand_boxes
    rule = job['rule']
    weights = ROOT / 'data/.runtime/grounding-dino/weights'
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(weights, local_files_only=True).to('cuda').eval()

    def prediction(index, text):
        check_time(rule)
        image = Image.open(directory / 'input_frames' / f'{index:06d}.jpg').convert('RGB')
        inputs = processor(images=image, text=text.rstrip('.') + '.', return_tensors='pt').to('cuda')
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            raw, inputs.input_ids, box_threshold=rule['box_threshold'],
            text_threshold=rule['text_threshold'], target_sizes=[image.size[::-1]])[0]
        return dict(frame_index=index, prompt=text, boxes=result['boxes'].detach().cpu().tolist(),
                    scores=result['scores'].detach().cpu().tolist())

    report = dict(pan_attempts=[], pan=None, hands=[], anchors={'0': [], '1': [], '2': []})
    for prompt in rule['pan_prompts']:
        found = prediction(0, prompt)
        report['pan_attempts'].append(found)
        if found['scores']:
            best = max(range(len(found['scores'])), key=found['scores'].__getitem__)
            report['pan'] = dict(frame_index=0, box=found['boxes'][best], score=found['scores'][best], prompt=prompt)
            report['anchors']['0'] = [0]
            break
    previous = {}
    for index in range(0, job['frame_count'], rule['hand_cadence_frames']):
        found = prediction(index, rule['hand_prompt'])
        assigned = hand_boxes(found['boxes'], found['scores'], previous,
                              rule['maximum_hands'], rule['hand_nms_iou'])
        found['assigned'] = {str(slot): box for slot, box in assigned.items()}
        for slot, box in assigned.items():
            previous[slot] = box
            report['anchors'][str(slot)].append(index)
        report['hands'].append(found)
        save_json(directory / 'detections.partial.json', report)
    save_json(directory / 'detections.json', report)
    return dict(pan_detected=report['pan'] is not None, hand_seed_counts={k:len(v) for k,v in report['anchors'].items() if k != '0'})


def segment(directory, job, torch):
    """Propagate all tracks, keeping the pan forward and hands nearest-seed bidirectional."""
    import numpy as np
    from sam2.build_sam import build_sam2_video_predictor
    from src.features.moving_reference._masks import choose_directions
    rule = job['rule']
    found = json.loads((directory / 'detections.json').read_text())
    seeds = []
    if found['pan']:
        seeds.append((0, 0, found['pan']['box']))
    for row in found['hands']:
        seeds.extend((row['frame_index'], int(slot), box) for slot, box in row['assigned'].items())
    if not seeds:
        return dict(status='no_detections', forward_frames=0, reverse_frames=0)
    predictor = build_sam2_video_predictor(rule['sam_config'],
        str(ROOT / 'data/.runtime/sam2/weights/sam2.1_hiera_large.pt'),
        device='cuda', mode='eval', apply_postprocessing=False)
    predictor.fill_hole_area = 0
    counts = dict(forward_frames=0, reverse_frames=0)
    (directory / 'frames').mkdir(exist_ok=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(str(directory / 'input_frames'),
                                     offload_video_to_cpu=True, offload_state_to_cpu=True)
        try:
            for index, slot, box in seeds:
                check_time(rule)
                predictor.add_new_points_or_box(state, frame_idx=index, obj_id=slot,
                                                 box=np.asarray(box, dtype=np.float32))
            for reverse in (False, True) if rule['bidirectional'] else (False,):
                for index, ids, logits in predictor.propagate_in_video(
                        state, start_frame_idx=job['frame_count'] - 1 if reverse else 0,
                        reverse=reverse):
                    check_time(rule)
                    masks = np.zeros((3, *job['mask_shape']), dtype=bool)
                    values = (logits[:, 0] > rule['logit_threshold']).cpu().numpy()
                    for slot, mask in zip(ids, values):
                        masks[int(slot)] = mask
                    path = directory / 'frames' / f'frame_{index:06d}.npz'
                    direction = np.zeros(3, dtype=np.uint8)
                    agreement = np.full(3, -1, dtype=np.float32)
                    if reverse:
                        with np.load(path, allow_pickle=False) as old:
                            forward = old['masks'].copy()
                        masks, direction, agreement = choose_directions(forward, masks, index, found['anchors'])
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, masks=masks, frame_index=np.int64(index),
                            timestamp_seconds=np.float64(job['timestamps'][index]),
                            pts_integer=np.int64(job['pts_integer'][index]),
                            track_ids=np.arange(3, dtype=np.int32), selected_direction=direction,
                            forward_reverse_iou=agreement, reverse_processed=np.bool_(reverse))
                    temporary.replace(path)
                    counts['reverse_frames' if reverse else 'forward_frames'] += 1
                    if index % 20 == 0:
                        print(json.dumps(dict(frame=index, reverse=reverse, **counts)), flush=True)
        finally:
            predictor.reset_state(state)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['detect', 'segment'])
    parser.add_argument('job')
    args = parser.parse_args()
    job_path = Path(args.job)
    directory = job_path.parent
    job = json.loads(job_path.read_text())
    rule = job['rule']
    lock = json.loads((ROOT / 'data/.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'moving_reference.masks':
        raise PermissionError('Worker requires its parent M1 GPU lease')
    check_time(rule)
    def deadline(signum, frame):
        raise TimeoutError('M1 GPU deadline reached')
    if rule.get('stop_utc'):
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, max(0.1, (datetime.fromisoformat(rule['stop_utc']) - datetime.now(timezone.utc)).total_seconds()))
    import torch
    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('Use the pinned CUDA build')
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 15000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = dict(status='running', operation=args.operation)
    try:
        report.update((detect if args.operation == 'detect' else segment)(directory, job, torch))
        report.setdefault('status', 'completed')
        if report['status'] == 'running':
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, TimeoutError) else 'failed',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(wall_seconds=time.perf_counter() - started,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        save_json(directory / f'{args.operation}-report.json', report)


if __name__ == '__main__':
    main()
