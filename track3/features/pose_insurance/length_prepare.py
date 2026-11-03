"""Prepare length-confirmation inputs and freeze reviewed RGB assignments through one driver."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from track3.paths import output_path
from track3.provenance import input_file, run_fields, write_json
from track3.runs import (ROOT, artifact, authorize_episode, heavy_execution,
                         producer, read_json, recorded, runtime_flag, source_hashes)
from .length_diagnostic import FEATURE, LengthRequest

REQUEST = 'track3/features/pose_insurance/length_confirm_request.json'
OUTPUT = 'outputs/pose_insurance/length-confirm/preparation'


def append(path, row):
    """Persist a completed stage before moving to the next queue entry."""
    with path.open('a') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def existing(feature, operation, request):
    """Find reusable completed video producers with matching normalized requests."""
    wanted = asdict(request)
    wanted.pop('assignment_note', None)
    directory = ROOT/f'outputs/{feature}/public/episode_{request.episode_index:06d}'
    for path in sorted(directory.glob('*/run.json')):
        row = read_json(path)
        if row.get('kind') != f'{feature}.{operation}' or row.get('status') != 'completed':
            continue
        values = asdict(type(request)(**row['request']))
        values.pop('assignment_note', None)
        if values == wanted:
            return producer(path, f'{feature}.{operation}', request)
    return None


def prepare(request, state, directory, reuse_only=False):
    """Reuse or execute each depth/measure pair sequentially through its owning handler."""
    from ..depth.generate import GenerateDepthRequest, generate_depth
    from ..mesh_extents.measure import MeasureRequest, measure
    for item in request.preparation:
        selected = state.setdefault(str(item['episode']), {})
        for feature, operation, cls, handler, key in (
            ('depth', 'generate', GenerateDepthRequest, generate_depth, 'depth'),
            ('mesh_extents', 'measure', MeasureRequest, measure, 'measure'),
        ):
            job = cls.from_file(item[f'{key}_request'])
            if job.episode_index != item['episode'] or job.source_split != request.source_split:
                raise ValueError('Preparation request differs from its queued episode')
            if key == 'measure':
                job = replace(job, depth_run=selected['depth'])
            prior = existing(feature, operation, job)
            tick = time.perf_counter()
            if prior is None:
                if reuse_only:
                    raise FileNotFoundError(f'No reusable {feature}.{operation} for episode {job.episode_index}')
                runtime = runtime_flag(FEATURE.parent/feature, f'{operation}_authorized', 'Length preparation')
                authorize_episode(runtime, job, 'length preparation')
                prior = producer(handler(job), f'{feature}.{operation}', job)
                reused = False
            else:
                reused = True
            parent = prior
            if key == 'measure':
                artifact(parent, 'overlay_000000.png', {'inputs': []})
            else:
                frame = next(row for row in parent[1]['frames'] if row['frame_index'] == 0)
                artifact(parent, frame['artifact'], {'inputs': []}, frame['sha256'])
            selected[key] = str(parent[0].relative_to(ROOT))
            identity = hashlib.sha256(json.dumps(dict(request=asdict(job), sources=source_hashes(FEATURE.parent/feature)), sort_keys=True).encode()).hexdigest()
            write_json(directory/'state.json', state)
            append(directory/'ledger.jsonl', dict(identity=identity, stage=key, episode=item['episode'],
                                                   run=selected[key], resumed=reused,
                                                   seconds=time.perf_counter()-tick))
    return state


def stereo(request, state, directory, request_path):
    """Reuse stereo handlers and record the unchanged scale estimator on frozen RGB unions."""
    from .scale_stereo import matching_run
    from ._scale import estimate_scale
    from ..stereo_depth._common import StereoDepthRequest
    assignments = read_json(request.assignments)['episodes']
    for episode in request.episodes:
        entry = assignments[str(episode)]
        if entry['merged'] and episode not in request.counted_episodes:
            continue
        depth = producer(state[str(episode)]['depth'], 'depth.generate')
        rect = matching_run('rectify', StereoDepthRequest(episode_index=episode))
        frames, parents = [], {}
        for frame in depth[1]['frames']:
            if frame['status'] != 'completed':
                raise ValueError('Incomplete teacher frame')
            index = frame['frame_index']
            common = dict(episode_index=episode, start_frame=index, stop_frame=index+1, depth_run=state[str(episode)]['depth'])
            generated = matching_run('generate', StereoDepthRequest(**common, rectification_run=str(rect[0].relative_to(ROOT))))
            anchored = matching_run('anchor', StereoDepthRequest(**common, stereo_run=str(generated[0].relative_to(ROOT))))
            document = read_json(artifact(anchored, 'anchor.json', {'inputs': []}))
            row = next(row for row in document['frames'] if row['frame_index'] == index)
            if row['video_pts'] != frame['video_pts']:
                raise ValueError('Mismatched timestamps')
            frames.append(row)
            parents[str(anchored[0].relative_to(ROOT))] = input_file(ROOT, anchored[0])
        result = dict(episode=episode, depth_run=state[str(episode)]['depth'], frame_indices=[row['frame_index'] for row in frames],
                      anchor_runs=list(parents), anchor_inputs=list(parents.values()), frames=frames, **estimate_scale(frames))
        identity = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        folder = directory/f'scale-{episode:06d}-{identity[:12]}'
        if not folder.exists():
            record = run_fields(ROOT, 'pose_insurance', 'length_confirmation_scale',
                                dict(episode_index=episode, source_split='public'), parents=list(parents),
                                inputs=[request.assignments, request_path], provenance='video_only')
            with recorded(folder, record):
                write_json(folder/'scale.json', result)
        state[str(episode)]['scale'] = str((folder/'run.json').relative_to(ROOT))
        write_json(directory/'state.json', state)
        append(directory/'ledger.jsonl', dict(identity=identity, stage='stereo_scale', episode=episode,
                                               run=state[str(episode)]['scale'], s=result['s'], valid=result['valid']))
    return state


def setup(request, directory):
    """Derive request drafts from the recorded templates without changing runtime flags."""
    placements = read_json(FEATURE/'placements_train.json')['episodes']
    queue = asdict(request)
    for item in queue['preparation']:
        measure = read_json(placements[str(item['template_episode'])]['measure_run'])['request'].copy()
        depth = read_json(measure['depth_run'])['request'].copy()
        depth['episode_index'] = measure['episode_index'] = item['episode']
        measure['depth_run'] = None
        for key, values in (('depth', depth), ('measure', measure)):
            path = directory/f"{key}_{item['episode']}.json"
            write_json(path, values)
            item[f'{key}_request'] = str(path.relative_to(ROOT))
    write_json(directory/'request.json', queue)


def freeze(request, state, observations_path, directory):
    """Hash reviewed RGB overlays into a new assignment file, never rewriting the frozen input."""
    from datetime import datetime, timezone
    observations = read_json(observations_path)
    if sorted(map(int, observations['episodes'])) != list(request.episodes):
        raise ValueError('RGB observations do not match the requested roster')
    for episode, row in observations['episodes'].items():
        row.update(frame_index=0, depth_run=state[episode]['depth'], measure_run=state[episode]['measure'],
                   overlay=input_file(ROOT, Path(state[episode]['measure']).parent/'overlay_000000.png'),
                   counted=int(episode) in request.counted_episodes)
    observations.update(created_utc=datetime.now(timezone.utc).isoformat(), frame_index=0,
                        basis='RGB overlays only; this operation loads no mocap')
    write_json(directory/'assignments.json', observations)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['setup', 'prepare', 'freeze', 'stereo'])
    parser.add_argument('--request', default=REQUEST)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--reuse-only', action='store_true', help='Prepare from existing completed video producers; never execute a handler')
    parser.add_argument('--observations', help='Reviewed RGB observations for freeze; no mocap is loaded')
    args = parser.parse_args()
    request = LengthRequest.from_file(args.request)
    runtime = runtime_flag(FEATURE, 'length_diagnostic_authorized', 'Length preparation')
    for episode in request.episodes:
        authorize_episode(runtime, SimpleNamespace(episode_index=episode, source_split=request.source_split), 'length preparation')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    directory = output_path(ROOT, args.output)
    directory.mkdir(parents=True, exist_ok=True)
    state = read_json(directory/'state.json') if (directory/'state.json').exists() else read_json(request.preparation_state)
    if args.stage == 'setup':
        setup(request, directory)
    elif args.stage == 'prepare':
        prepare(request, state, directory, args.reuse_only)
    elif args.stage == 'stereo':
        stereo(request, state, directory, args.request)
    elif args.observations:
        freeze(request, state, args.observations, directory)
    else:
        parser.error('freeze requires --observations')
    print(directory)


if __name__ == '__main__':
    main()
