"""Independently recompute recorded H3 geometry from raw arrays and train mocap."""
import json
import os
from pathlib import Path
import sys
import tempfile

from track3.paths import output_path
from track3.provenance import input_file, write_json
from track3.runs import (ROOT, CACHE_FOLDERS, artifact, authorize_episode, checked_entry,
                         heavy_execution, producer, read_json, runtime_flag)
from .request import Request

FEATURE = Path(__file__).parent
QUEUE = 'track3/features/pose_insurance/heading_h3_queue.json'


def recompute(queue: dict, summary: dict) -> dict:
    """Recover b/C directly without importing the H3 construction or scoring modules."""
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation
    normal = np.array(read_json(checked_entry(queue['measurement'], {'inputs': []}))['frames'][0]['plane']['normal'])
    x = np.array([1., 0., 0.]) - normal[0] * normal
    x /= np.linalg.norm(x)
    basis = np.stack((x, np.cross(normal, x), normal))
    rotations = []
    for entry in queue['registrations']:
        parent = producer(checked_entry(entry, {'inputs': []}), 'pose_insurance.registration_trial',
                          allowed=('public_asset_diagnostic',))
        prediction = read_json(artifact(parent, 'prediction.json', {'inputs': []}))
        with np.load(artifact(parent, 'registrations.npz', {'inputs': []}), allow_pickle=False) as data:
            index = int(np.argmin(data['search_0_losses_m']))
            assert index == prediction['best_index']
            rotations.append(data['search_0_rotations'][index])
    rotations = basis @ np.array(rotations)
    build = producer(checked_entry(queue['build'], {'inputs': []}), 'pose_insurance.stereo_placement_build')
    construction = read_json(artifact(build, 'construction.json', {'inputs': []}))
    positions = np.array(construction['scenes']['T1']['positions'])
    b = rotations[0].T @ (positions[1] - positions[0])
    c = rotations[0].T @ rotations[1]
    dataset = next(ROOT.glob('datasets--nvidia--video_to_data_challenge/snapshots/*/track_3/public'))
    objects = pq.read_table(dataset / f"data/chunk-000/episode_{queue['episode']:06d}.parquet",
                            columns=['observation.objects'])['observation.objects'][0].as_py()
    objects = {row['name']: row for row in objects}
    names = construction['names']
    assert all(objects[name]['visible'] for name in names)
    reference = np.array([objects[name]['pose'] for name in names])
    ref_r = Rotation.from_quat(reference[:, [4, 5, 6, 3]]).as_matrix()
    ref_b = ref_r[0].T @ (reference[1, :3] - reference[0, :3])
    ref_c = ref_r[0].T @ ref_r[1]
    b_error = float(np.degrees(np.arccos(np.clip(b @ ref_b / np.linalg.norm(b) / np.linalg.norm(ref_b), -1., 1.))))
    c_error = float(np.degrees(Rotation.from_matrix(ref_c.T @ c).magnitude()))
    expected = summary['results']['H3-sel']
    assert np.max(np.abs(Rotation.from_quat(np.array(expected['start_poses'])[:, 3:]).as_matrix() - rotations)) < 1e-12
    assert np.array_equal(np.array(expected['start_poses'])[:, :3], positions)
    assert abs(b_error - expected['frame0']['b_error_deg']) < 1e-10
    assert abs(c_error - expected['frame0']['C_error_deg']) < 1e-10
    assert np.max(np.abs(b - expected['frame0']['b_m'])) < 1e-12
    assert np.max(np.abs(c - expected['frame0']['C'])) < 1e-12
    synthetic_cam = basis.T @ ref_r
    recovered_r = basis @ synthetic_cam
    recovered_b = recovered_r[0].T @ (reference[1, :3] - reference[0, :3])
    recovered_c = recovered_r[0].T @ recovered_r[1]
    assert np.max(np.abs(recovered_b - ref_b)) < 1e-12
    assert np.max(np.abs(recovered_c - ref_c)) < 1e-12
    return dict(b_error_deg=b_error, C_error_deg=c_error, b_m=b.tolist(), C=c.tolist(),
                mocap_frame0_visible=True, O3_known_frame_b_max_error_m=float(np.max(np.abs(recovered_b-ref_b))),
                O3_known_frame_C_max_error=float(np.max(np.abs(recovered_c-ref_c))))


def main() -> None:
    """Admit the CPU check and verify the durable inventory before raw recomputation."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    runtime = runtime_flag(FEATURE, 'heading_authorized', 'Independent H3 CPU check')
    queue = read_json(QUEUE)
    authorize_episode(runtime, Request(source_split=queue['source_split'], episode_index=queue['episode']),
                      'independent H3 CPU check')
    heavy_execution(dict(runtime, python=runtime['diagnostic_python']))
    for key in CACHE_FOLDERS:
        path = output_path(ROOT, f'.runtime/tmp/h3/cache/{key.lower()}')
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    sys.pycache_prefix = os.environ['PYTHONPYCACHEPREFIX']
    tempfile.tempdir = os.environ['TMPDIR']
    inventory = read_json(checked_entry(queue['integrity_snapshot'], {'inputs': []}))['files']
    changed = [p for p, digest in inventory.items() if input_file(ROOT, p)['sha256'] != digest]
    assert not changed, changed
    latest = read_json('outputs/pose_insurance/heading-h3/latest.json')
    parent = producer(latest['report'], 'pose_insurance.heading_h3_report', allowed=('public_label_diagnostic',))
    checked_entry(next(entry for entry in parent[1]['inputs'] if entry['path'] == QUEUE), {'inputs': []})
    if parent[1]['request'] != queue:
        raise ValueError('Independent check requires the exact recorded heading queue')
    summary = read_json(artifact(parent, 'summary.json', {'inputs': []}))
    result = dict(recompute(queue, summary), historical_files_byte_identical=len(inventory), passed=True)
    write_json(output_path(ROOT, 'outputs/pose_insurance/heading-h3/independent-check.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
