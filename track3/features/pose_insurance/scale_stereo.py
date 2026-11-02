"""Run and resume unchanged stereo handlers on the exact scene depth frames."""
from dataclasses import asdict
from track3.provenance import input_file, write_json
from track3.runs import ROOT, producer, artifact, read_json
from ._scale import estimate_scale
from ..stereo_depth._common import StereoDepthRequest
from ..stereo_depth.rectify import rectify
from ..stereo_depth.generate import generate
from ..stereo_depth.anchor import anchor


def matching_run(operation, request):
    """Reuse a completed exact request, including containing default anchor intervals."""
    folder = ROOT / f'outputs/stereo_depth/public/episode_{request.episode_index:06d}'
    wanted = asdict(request)
    for path in sorted(folder.glob('*/run.json')):
        row = read_json(path)
        if row['kind'] != f'stereo_depth.{operation}' or row['status'] != 'completed':
            continue
        recorded = row['request']
        if operation == 'anchor':
            shared = [k for k in wanted if k not in ('start_frame', 'stop_frame')]
            matches = (all(recorded.get(k) == wanted[k] for k in shared)
                       and recorded['start_frame'] <= request.start_frame
                       and recorded['stop_frame'] >= request.stop_frame)
        else:
            matches = recorded == wanted
        if matches:
            return producer(path, f'stereo_depth.{operation}', request)
    path = dict(rectify=rectify, generate=generate, anchor=anchor)[operation](request)
    return producer(path, f'stereo_depth.{operation}', request)


def episode_stereo(episode, depth_run):
    """Aggregate per-frame anchors without enabling the sparse research gate."""
    depth = producer(depth_run, 'depth.generate')
    indices = [f['frame_index'] for f in depth[1]['frames'] if f['status'] == 'completed']
    if len(indices) != len(depth[1]['frames']):
        raise ValueError('Scene depth run contains incomplete frames')
    frames, parents = [], {}
    rect = None
    folder = ROOT / f'outputs/stereo_depth/public/episode_{episode:06d}'
    for index in indices:
        existing = None
        for path in sorted(folder.glob('*/run.json')):
            row = read_json(path)
            req = row['request']
            if (row['kind'] == 'stereo_depth.anchor' and row['status'] == 'completed'
                    and req.get('depth_run') == depth_run and req.get('masks_run') is None
                    and req.get('holdback_modulus', 0) == 0 and req.get('episode_aggregation', 'median') == 'median'
                    and req == asdict(StereoDepthRequest(episode_index=episode, start_frame=req['start_frame'],
                        stop_frame=req['stop_frame'], depth_run=depth_run, stereo_run=req['stereo_run']))
                    and any(f['frame_index'] == index for f in row['frames'])):
                existing = producer(path, 'stereo_depth.anchor')
                break
        if existing is None:
            if rect is None:
                rect = matching_run('rectify', StereoDepthRequest(episode_index=episode))
            common = dict(episode_index=episode, start_frame=index, stop_frame=index + 1, depth_run=depth_run)
            stereo = matching_run('generate', StereoDepthRequest(**common, rectification_run=str(rect[0].relative_to(ROOT))))
            existing = matching_run('anchor', StereoDepthRequest(**common, stereo_run=str(stereo[0].relative_to(ROOT))))
        doc = read_json(artifact(existing, 'anchor.json', {'inputs': []}))
        frame = next(f for f in doc['frames'] if f['frame_index'] == index)
        depth_frame = next(f for f in depth[1]['frames'] if f['frame_index'] == index)
        if frame['video_pts'] != depth_frame['video_pts']:
            raise ValueError('Anchor and scene depth timestamps differ')
        frames.append(frame)
        parents[str(existing[0].relative_to(ROOT))] = input_file(ROOT, existing[0])
    result = dict(episode=episode, depth_run=depth_run, frame_indices=indices,
                  anchor_runs=list(parents), anchor_inputs=list(parents.values()),
                  frames=frames, **estimate_scale(frames))
    write_json(ROOT / f'outputs/pose_insurance/scale/stereo_{episode:06d}.json', result)
    return result
