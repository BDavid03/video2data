"""CPU-only orchestration for one F1 Stage A object."""

from dataclasses import asdict
import os
from pathlib import Path
import sys

from track3.paths import interpreter, output_path
from track3.provenance import run_fields, write_json
from track3.runs import (ROOT, authorize_episode, dataset_revision, new_run_id, recorded,
                         runtime_flag, source_hashes)

FEATURE = Path(__file__).parent
PROTOCOL = 'docs/chain/mesh-fusion-protocol.md'
REUSED = ('track3/features/camera_motion/estimate.py', 'track3/features/mesh_extents/_geometry.py',
          'track3/features/object_masks/opening_clusters.py', 'track3/features/pose_insurance/_registration.py',
          'track3/features/pose_insurance/_scene.py', 'track3/features/stereo_depth/generate.py',
          'track3/features/stereo_depth/_common.py')


def cpu_runtime():
    """Keep the narrower job's caches under its permitted scratch directory."""
    runtime = runtime_flag(FEATURE, 'stage_a_authorized', 'F1 Stage A')
    if Path(sys.executable).resolve() != interpreter(ROOT, runtime['python']):
        raise RuntimeError('Use .runtime/linux-cuda/bin/python')
    scratch = output_path(ROOT, '.runtime/tmp/ctx20/fusion-cache')
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TEMP', 'TMP', 'TMPDIR'):
        os.environ[name] = str(scratch)
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    return runtime


def reference_geometry(opening, stereo, grid):
    import numpy as np
    from ._coverage import object_region
    from ._registration import background_mask, features, region_box
    from track3.features.mesh_extents._geometry import camera_points
    from track3.features.pose_insurance._registration import fit_table

    labels, ids = opening['labels'], opening['request'].cluster_ids
    k = grid['input_intrinsics_pixels']
    points = camera_points(stereo['rgb_depth_m'], k)
    valid = stereo['rgb_valid'] & grid['input_valid'] & np.isfinite(points).all(-1)
    target = np.isin(labels, np.array(ids) + 1) & valid
    seed = points[target]
    hint = opening['measured'][1]['frames'][0]['plane']['normal']
    parameters = {'table': dict(radius_m=0.65, iterations=300, inlier_m=0.003,
                               normal_hint_max_deg=25, seed=31, degenerate_cross_norm=1e-10)}
    normal, offset, table = fit_table(points, valid, labels, seed, np.array(hint), parameters)
    seed = seed[seed @ normal - offset > 0.006]
    if not len(seed):
        raise ValueError('No opening object stereo points remain above its support plane')
    boxes = [region_box(points[valid & (labels == value)]) for value in np.unique(labels[valid])
             if value > 0 and (valid & (labels == value)).sum() >= 30]
    static = background_mask(valid.shape, k, np.eye(4), boxes, grid['input_valid'])
    reference = dict(stereo=stereo, static=static,
                     features=features(opening['rgb'][0], stereo, k, static))
    return seed, object_region(seed, normal, offset), boxes, reference, dict(
        normal=normal.tolist(), offset_m=offset, **table)


def measure_frames(request, opening, directory, record):
    import numpy as np
    from ._coverage import select_object
    from ._inputs import stereo_frame
    from ._registration import camera_points, register

    grid = opening['grid']
    k = grid['input_intrinsics_pixels']
    first, path, _ = stereo_frame(request, opening, 0, record)
    seed, region, boxes, reference, table = reference_geometry(opening, first, grid)
    opening_points = camera_points(first['rgb_depth_m'], k)
    opening_cloud = select_object(opening_points, first['rgb_valid'] & grid['input_valid'], np.eye(4), region)
    record.update(table=table, opening_object_points=len(seed),
                  object_centre_m=np.median(seed, axis=0).tolist(), stereo_reference=path)
    clouds, centres = [], []
    for index in request.candidate_frames:
        stereo, parent, pts = stereo_frame(request, opening, index, record)
        pose, row = register(reference, opening['rgb'][index], stereo, k, boxes, grid['input_valid'])
        row.update(frame_index=index, pts=pts, status='completed', stereo_run=parent,
                   reference_identity=index == 0)
        if pose is not None:
            if index == 0:
                pose = np.eye(4)
                row['camera_centre_m'] = [0., 0., 0.]
            points = camera_points(stereo['rgb_depth_m'], k)
            cloud = select_object(points, stereo['rgb_valid'] & grid['input_valid'], pose, region)
            row['object_points'] = len(cloud)
            if row['accepted'] and not len(cloud):
                row['accepted'] = False
                row['reasons'].append('no object stereo points in the fixed opening region')
            np.savez_compressed(directory / f'frame_{index:06d}.npz', pose=pose, object_points=cloud)
            if row['accepted']:
                clouds.append(cloud)
                centres.append(pose[:3, 3])
        record['frames'].append(row)
        write_json(directory / 'run.json', record)
        print(f"public{request.episode_index} frame {index}: {'accept' if row['accepted'] else 'reject'} {row['reasons']}", flush=True)
    return opening_cloud, clouds, centres


def stage_a(request, trial_identity=None):
    runtime = cpu_runtime()
    authorize_episode(runtime, request, 'F1 Stage A')
    from ._inputs import load_opening, rgb_frames
    from ._render import camera_figure, contact_sheet
    from ._coverage import angular_span, supported_surface, verdict
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    record = run_fields(ROOT, 'mesh_fusion', 'stage_a', asdict(request),
                        parents=[request.depth_run, request.measure_run, request.stereo_reference_run],
                        allowed=('video_only',), inputs=[PROTOCOL, FEATURE / 'runtime.json'])
    record.update(run_id=new_run_id(), split='public', episode_index=request.episode_index,
                  dataset_revision=dataset_revision(), gpu=None, frames=[], artifacts={},
                  trial_identity=trial_identity, source_hashes=source_hashes(FEATURE, REUSED),
                  interval=dict(start_frame=0, stop_frame_exclusive=request.first_contact_frame,
                                candidates=request.candidate_frames, annotation=request.contact_annotation))
    directory = output_path(ROOT, f"outputs/mesh_fusion/stage_a/public{request.episode_index}/{record['run_id']}")
    with recorded(directory, record):
        opening = load_opening(request, record)
        opening['request'] = request
        opening['rgb'] = rgb_frames(opening['video'], request.candidate_frames, opening['grid'])
        coarse = rgb_frames(opening['video'], request.contact_annotation['coarse_frames'])
        refined = rgb_frames(opening['video'], request.contact_annotation['refined_frames'])
        cv2.imwrite(str(directory / 'contact-coarse.png'), contact_sheet(coarse, request.first_contact_frame))
        cv2.imwrite(str(directory / 'contact-refined.png'), contact_sheet(refined, request.first_contact_frame, crop=True))
        cv2.imwrite(str(directory / 'pre-contact.png'), contact_sheet(opening['rgb'], request.first_contact_frame))
        seed, clouds, centres = measure_frames(request, opening, directory, record)
        centre = np.asarray(record['object_centre_m'])
        coverage = supported_surface(seed, clouds)
        angle = angular_span(centres, centre)
        record['result'] = dict(object=request.object_name, accepted_frames=[r['frame_index'] for r in record['frames'] if r['accepted']],
                                angular_span_deg=angle, **coverage, **verdict(len(centres), angle, coverage['additional_supported_surface_percent']))
        figure = camera_figure(record['frames'], centre, seed)
        figure.savefig(directory / 'camera-centres.png', dpi=150)
        plt.close(figure)
        write_json(directory / 'result.json', record['result'])
        (directory / 'summary.md').write_text(object_summary(record), encoding='utf-8')
    return directory / 'run.json'


def object_summary(record):
    result, interval = record['result'], record['interval']
    lines = [f"# F1 Stage A: {result['object']} — {result['verdict']}", '',
             f"Pre-contact interval: [0, {interval['stop_frame_exclusive']}); candidates {interval['candidates']}.",
             interval['annotation']['method'], interval['annotation']['interpretation'], '',
             f"Accepted: {result['accepted_frames']}; angular span {result['angular_span_deg']:.6f} deg; "
             f"additional supported surface {result['additional_supported_surface_percent']:.6f}% "
             f"({result['additional_voxels']}/{result['fused_voxels']} fused 3 mm voxels).", '',
             f"Frozen rule: {result['frozen_rule']}. Reasons: {result['reasons'] or 'none'}.", '',
             '| Frame | PnP inliers | Reprojection px | Independent matches | Disagreement mm | Accepted | Reasons |',
             '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for row in record['frames']:
        error = row.get('independent_disagreement_m')
        lines.append(f"| {row['frame_index']} | {row.get('pnp_inliers', 0)} | {row.get('reprojection_px')} | "
                     f"{row.get('independent_correspondences', 0)} | {None if error is None else 1000 * error} | "
                     f"{row['accepted']} | {', '.join(row['reasons'])} |")
    lines += ['', 'Frame 0 uses identity correspondences to the opening reference; it is not a second view.',
              'Independent = descriptor correspondences at reference landmark index modulo five == zero; excluded from PnP and 8 px ICP patches.',
              'Disagreement is median Euclidean 3D distance on those unused matches, without residual trimming; >=10 matches required.',
              'Reprojection is measured after ICP on the original PnP-consistent training matches (<=3 px before ICP).',
              'Exact voxel novelty is sampling/noise sensitive and is not evidence that an unseen backside was reconstructed.',
              'Only accepted poses contribute to angular span and fused support. The opening cloud is the fixed coverage reference.',
              'No scan or CD-O used. No Stage B geometry or completion was built.', '',
              '![Selected pre-contact frames](pre-contact.png)', '![Contact boundary](contact-refined.png)',
              '![Camera centres](camera-centres.png)', '']
    return '\n'.join(lines)
