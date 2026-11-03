"""Pure similarity fitting and exact observed-point surface residuals."""

import numpy as np
import trimesh


def _frame(points):
    center = points.mean(axis=0)
    axes = np.linalg.svd(points-center, full_matrices=False)[2]
    for axis in axes:
        axis *= 1 if axis[np.abs(axis).argmax()] >= 0 else -1
    axes[-1] *= np.sign(np.linalg.det(axes))
    radius = np.sqrt(np.square(points-center).sum(axis=1).mean())
    return center, axes, radius


def fit_residual(vertices, faces, observed, settings):
    """Fit one proper similarity by ICP, then measure every observation against triangles."""
    triangles = vertices[faces]
    keep = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)>1e-12
    mesh = trimesh.Trimesh(vertices, faces[keep], process=False)
    surface = mesh.sample(settings['surface_samples'], seed=settings['seed'])
    fit_points = observed[np.linspace(0,len(observed)-1,min(len(observed),settings['observed_fit_points']),dtype=int)]
    oc, oa, od = _frame(fit_points)
    mc, ma, md = _frame(surface)
    initial = np.eye(4)
    initial[:3,:3] = (md/od) * ma.T @ oa
    initial[:3,3] = mc-initial[:3,:3]@oc
    matrix, _, cost = trimesh.registration.icp(
        fit_points, surface, initial=initial, scale=True, reflection=False,
        threshold=settings['threshold'], max_iterations=settings['max_iterations'])
    scale = np.cbrt(np.linalg.det(matrix[:3,:3]))
    transformed = trimesh.transform_points(observed,matrix)
    distances = []
    for start in range(0,len(transformed),64):
        distances.append(trimesh.proximity.closest_point_naive(mesh,transformed[start:start+64])[1]/scale)
    residuals = np.concatenate(distances)
    return dict(median_residual_m=float(np.median(residuals)), mean_residual_m=float(residuals.mean()),
                observed_points=len(observed), fit_points=len(fit_points), scale_observed_to_model=float(scale),
                observed_to_model=matrix.tolist(), icp_cost_model_units=float(cost))
