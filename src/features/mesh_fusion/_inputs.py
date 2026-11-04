"""Producer admission and CPU stereo integration for the F1 handler."""

from dataclasses import replace

from src.paths import output_path, project_path
from src.provenance import input_file, read_run, write_json
from src.runs import (ROOT, artifact, checked_entry, dataset_revision, producer, read_json,
                         record_input, same_revision)


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def load_opening(request, record):
    depth = producer(request.depth_run, 'depth.generate', request)
    measured = producer(request.measure_run, 'mesh_extents.measure', request)
    stereo = producer(request.stereo_reference_run, 'stereo_depth.generate', request)
    same_revision([depth, measured, stereo], dataset_revision())
    grid = arrays(artifact(depth, 'preprocessing.npz', record, depth[1]['preprocessing_artifact_sha256']))
    labels = arrays(artifact(measured, 'labels_000000.npz', record))['label']
    entry = next(row for row in depth[1]['inputs'] if row['path'] == depth[1]['source_video'])
    video = checked_entry(entry, record)
    return dict(depth=depth, measured=measured, stereo=stereo, grid=grid, labels=labels, video=video)


def stereo_frame(request, opening, index, record):
    """Reuse cached single-frame producers, or call the unchanged CPU stereo handler."""
    from src.features.stereo_depth._common import StereoDepthRequest
    from src.features.stereo_depth.generate import generate

    base_path, base = opening['stereo']
    if index == 0:
        path = base_path
    else:
        selection = replace(StereoDepthRequest(**base['request']), start_frame=index, stop_frame=index + 1)
        cache = output_path(ROOT, f'data/outputs/mesh_fusion/stereo-cache/public{request.episode_index}/frame_{index:06d}.json')
        if cache.is_file() and read_json(cache)['request'] == selection.__dict__:
            path = project_path(ROOT, read_json(cache)['run'])
        else:
            path = generate(selection)
            write_json(cache, dict(request=selection.__dict__, run=path.relative_to(ROOT).as_posix()))
    source = read_run(ROOT, path, allowed=('video_only',), kind='stereo_depth.generate')
    if source['episode_index'] != request.episode_index or source['split'] != request.source_split:
        raise ValueError('Stereo producer episode does not match the F1 request')
    if source['grid'] != opening['depth'][1]['preprocessing_artifact_sha256']:
        raise ValueError('Stereo and label grids differ')
    row = next(r for r in source['frames'] if r['frame_index'] == index and r['status'] == 'completed')
    record_input(input_file(ROOT, path), record)
    path_to_frame = artifact((path, source), row['artifact'], record, row['sha256'])
    result = arrays(path_to_frame)
    return result, path.relative_to(ROOT).as_posix(), row['video_pts']


def rgb_frames(video, indices, grid=None):
    import cv2
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    frames = {}
    try:
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'RGB decode failed at frame {index}')
            if grid is not None:
                image = cv2.remap(image, grid['source_x'], grid['source_y'], cv2.INTER_LINEAR)
            frames[index] = image
    finally:
        capture.release()
    return frames
