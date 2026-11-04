"""Register and render one saved frame with the unchanged FoundationPose adapter (refresh copy)."""

import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    root = Path.cwd()
    lock = json.loads((root / 'data/.runtime/gpu-stage.lock').read_text())
    if lock['pid'] != os.getppid() or lock['stage'] != 'pose_refresh.frame0_fp':
        raise PermissionError('The campaign parent must own the GPU lease')
    from register_one_frame import register_frame, load_textured_glb
    from Utils import set_seed, nvdiffrast_render
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from src.provenance import sha256_file
    from src.features.moving_policy._two_object_rotation import orientation_degrees
    set_seed(0)
    output = root / request['output']
    output.mkdir(parents=True, exist_ok=True)
    frame = root / request['frame']
    glb = root / request['glb']
    if sha256_file(frame) != request['frame_sha256'] or sha256_file(glb) != request['glb_sha256']:
        raise ValueError('Registration input changed')
    with np.load(frame) as saved:
        rgb, depth, mask, k = [saved[key].copy() for key in ('rgb', 'depth', 'mask', 'K')]
    result = register_frame(rgb, depth, mask, k, glb, output / 'debug')
    np.savez_compressed(output / 'predictions.npz', **result)
    mesh = load_textured_glb(glb)
    context = dr.RasterizeCudaContext()
    _, rendered, _ = nvdiffrast_render(K=k, H=depth.shape[0], W=depth.shape[1],
        ob_in_cams=torch.as_tensor(result['pose'], device='cuda', dtype=torch.float32)[None],
        glctx=context, mesh=mesh, use_light=False)
    rendered = rendered[0].detach().cpu().numpy()
    occupied = rendered > 0
    overlap = float((occupied & mask).sum() / max((occupied | mask).sum(), 1))
    supported = mask & np.isfinite(depth) & (depth > 0)
    residual = float(np.median(abs(rendered[supported] - depth[supported]))) if supported.any() else None
    # Same public-surface PCA axis convention as the current kit_score proxy.
    axis = np.array([0., 0., 1.])
    if request['name'] in ('blue_cup', 'beige_cup', 'white_pot_lid'):
        points, _ = trimesh.sample.sample_surface(mesh, 10000, seed=0)
        _, vectors = np.linalg.eigh(np.cov(points.T))
        axis = vectors[:, 0 if request['name'] == 'white_pot_lid' else -1]
    from scipy.spatial.transform import Rotation
    poses = result['hypotheses']
    translations = np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1)
    turns = orientation_degrees(Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
                               Rotation.from_matrix(poses[0, :3, :3]).as_quat(), request['name'], axis)
    distinct = (translations > .01) | (turns > 5.)
    margins = (result['scores'][0] - result['scores']) / max(abs(float(result['scores'][0])), 1e-12)
    rivals = np.flatnonzero(distinct)
    rival = int(rivals[np.argmin(margins[rivals])]) if len(rivals) else None
    margin = float(margins[rival]) if rival is not None else None
    record = dict(episode=request['episode'], slot=request['slot'], name=request['name'],
        pose=result['pose'].tolist(), silhouette_iou=overlap, median_depth_residual_m=residual,
        supported_depth_pixels=int(supported.sum()), rendered_pixels=int(occupied.sum()),
        silhouette_passed=overlap >= .7, depth_passed=residual is not None and residual <= .01,
        distinct_rival_index=rival, distinct_rival_relative_margin=margin,
        ambiguity_passed=margin is None or margin > .01, symmetry_axis_scan=axis.tolist(),
        symmetry_axis_source='Public scan surface PCA, same disclosed local convention as kit_score',
        hypotheses=len(poses), hypothesis_translation_m=translations.tolist(),
        hypothesis_rotation_deg=turns.tolist(), hypothesis_relative_margins=margins.tolist(),
        register_seconds=float(result['register_seconds']), labels_read=False,
        prediction_sha256=sha256_file(output / 'predictions.npz'), request=request)
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    np.savez_compressed(output / 'rendered-depth.npz', depth=rendered, mask=occupied)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
