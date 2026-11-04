"""Recorded evaluation stereo preparation with the unchanged slice handlers."""
from dataclasses import asdict
import numpy as np
from src.provenance import input_file, write_json
from src.runs import ROOT, producer, artifact, read_json
from ._scale import estimate_scale
from ..stereo_depth._common import StereoDepthRequest
from ..stereo_depth.rectify import rectify
from ..stereo_depth.generate import generate
from ..stereo_depth.anchor import anchor


def operation(operation_name: str, request: StereoDepthRequest) -> tuple:
    """Resume only an exact completed handler request with admitted producer artifacts."""
    folder = ROOT / f'data/outputs/stereo_depth/evaluation/episode_{request.episode_index:06d}'
    for path in sorted(folder.glob('*/run.json')):
        row = read_json(path)
        if (row['status'] == 'completed' and row['kind'] == f'stereo_depth.{operation_name}'
                and row['request'] == asdict(request)):
            return producer(path, f'stereo_depth.{operation_name}', request)
    handler = {'rectify': rectify, 'generate': generate, 'anchor': anchor}[operation_name]
    return producer(handler(request), f'stereo_depth.{operation_name}', request)


def prepare(episode: int, depth_run: str) -> dict:
    """Anchor exactly the v2 depth frames using public calibration for the evaluation rig."""
    depth = producer(depth_run, 'depth.generate')
    indices = [r['frame_index'] for r in depth[1]['frames'] if r['status'] == 'completed']
    if len(indices) != len(depth[1]['frames']) or indices != sorted(set(indices)):
        raise ValueError('The v2 depth run must contain unique completed frames')
    rect = operation('rectify', StereoDepthRequest(source_split='evaluation', episode_index=episode))
    groups = np.split(np.array(indices), np.flatnonzero(np.diff(indices) != 1) + 1)
    frames, anchors = [], []
    for group in groups:
        common = dict(source_split='evaluation', episode_index=episode,
                      start_frame=int(group[0]), stop_frame=int(group[-1]) + 1, depth_run=depth_run)
        stereo = operation('generate', StereoDepthRequest(**common, rectification_run=str(rect[0].relative_to(ROOT))))
        parent = operation('anchor', StereoDepthRequest(**common, stereo_run=str(stereo[0].relative_to(ROOT))))
        doc = read_json(artifact(parent, 'anchor.json', {'inputs': []}))
        frames.extend(doc['frames'])
        anchors.append(input_file(ROOT, parent[0]))
    if [f['frame_index'] for f in frames] != indices:
        raise ValueError('Stereo anchors do not match the scene depth frame identities')
    for frame, original in zip(frames, depth[1]['frames']):
        if frame['video_pts'] != original['video_pts']:
            raise ValueError('Stereo and v2 depth timestamps differ')
    result = dict(episode=episode, depth_run=depth_run, depth_input=input_file(ROOT, depth[0]),
                  frame_indices=indices, frames=frames, anchors=anchors,
                  calibration_assumption='Public C-left/B-right calibration is assumed for the evaluation rig',
                  **estimate_scale(frames))
    write_json(ROOT / f'data/outputs/pose_insurance/v3/stereo_{episode:06d}.json', result)
    return result
