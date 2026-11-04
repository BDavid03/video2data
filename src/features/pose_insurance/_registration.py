"""Pure surface registration, rotation ambiguity and frame-zero pose errors."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ._scene import table_basis


def surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Sample triangle area uniformly without loading or copying scan textures."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return trimesh.sample.sample_surface(mesh, count, seed=seed)[0]


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple:
    """Return the proper rigid transform mapping paired rows of a onto b."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    r = vt.T @ np.diag([1., 1., np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return r, cb - r @ ca


def trimmed(distances: np.ndarray, fraction: float) -> float:
    """Return HR1's mean of the rounded retained nearest-surface distances."""
    return float(np.sort(distances)[:max(1, round(fraction * len(distances)))].mean())


def icp(observed: np.ndarray, tree: cKDTree, model: np.ndarray,
        rotation: np.ndarray, translation: np.ndarray, parameters: dict) -> tuple:
    """Refine observed-to-model correspondences with HR1 translation and Kabsch steps."""
    r, t = rotation.copy(), translation.copy()
    for iteration in range(parameters['translation_only_iters'] + parameters['rigid_iters']):
        distances, indices = tree.query((observed - t) @ r)
        keep = distances <= np.quantile(distances, parameters['keep_fraction'])
        m = model[indices[keep]] @ r.T + t
        o = observed[keep]
        if iteration < parameters['translation_only_iters']:
            t += (o - m).mean(0)
        else:
            dr, dt = kabsch(m, o)
            r, t = dr @ r, dr @ t + dt
    distances, _ = tree.query((observed - t) @ r)
    return r, t, trimmed(distances, parameters['keep_fraction'])


def starts(normal: np.ndarray, yaw_step_deg: float) -> tuple:
    """Map each signed scan axis onto table up, followed by table yaw."""
    rotations, axes, yaws = [], [], []
    basis = table_basis(normal)
    for axis in range(3):
        for sign in (1, -1):
            up = np.eye(3)[axis] * sign
            x = np.eye(3)[(axis + 1) % 3]
            alignment = np.stack((x, np.cross(up, x), up))
            for yaw in np.arange(0., 360., yaw_step_deg):
                rz = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
                rotations.append(basis.T @ rz @ alignment)
                axes.append(up)
                yaws.append(yaw)
    return np.array(rotations), np.array(axes), np.array(yaws)


def fit_table(points: np.ndarray, valid: np.ndarray, labels: np.ndarray,
              object_points: np.ndarray, normal_hint: np.ndarray, parameters: dict) -> tuple:
    """Fit HR1's nearby stereo background plane with a MoGe-normal orientation hint."""
    p = parameters['table']
    normal_hint = normal_hint / np.linalg.norm(normal_hint)
    background = points[valid & (labels == 0)]
    background = background[np.linalg.norm(background - object_points.mean(0), axis=1) < p['radius_m']]
    if len(background) < 3:
        raise ValueError('Fewer than three nearby stereo table points')
    rng = np.random.default_rng(p['seed'])
    best = None
    for _ in range(p['iterations']):
        sample = background[rng.choice(len(background), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(n) < p['degenerate_cross_norm']:
            continue
        n /= np.linalg.norm(n)
        if n @ normal_hint < 0:
            n = -n
        if n @ normal_hint < np.cos(np.radians(p['normal_hint_max_deg'])):
            continue
        inliers = np.abs((background - sample[0]) @ n) < p['inlier_m']
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 3:
        raise ValueError('No stereo table plane consistent with the normal hint')
    selected = background[best]
    centre = selected.mean(0)
    _, _, vt = np.linalg.svd(selected - centre, full_matrices=False)
    n = vt[2] if vt[2] @ normal_hint > 0 else -vt[2]
    return n, float(n @ centre), dict(inliers=int(best.sum()), nearby_points=len(background),
        angle_to_hint_deg=vector_error_deg(n, normal_hint))


def search(model: np.ndarray, observed: np.ndarray, normal: np.ndarray,
           offset: float, parameters: dict, seed: int, sample_fraction: float = 1.) -> dict:
    """Search every signed-axis/yaw start on one fixed observed-point subsample."""
    rng = np.random.default_rng(seed)
    count = max(3, int(min(parameters['max_observed_points'], len(observed)) * sample_fraction))
    obs = observed if count == len(observed) else observed[rng.choice(len(observed), count, replace=False)]
    tree = cKDTree(model)
    initial, axes, yaws = starts(normal, parameters['yaw_step_deg'])
    centre = obs.mean(0)
    back = centre - (centre @ normal) * normal
    back /= np.linalg.norm(back)
    rotations, translations, losses = [], [], []
    for r in initial:
        transformed = model @ r.T
        mean = transformed.mean(0)
        t = centre + parameters['push_back_m'] * back - (mean - (mean @ normal) * normal)
        t = t - (t @ normal) * normal + (offset - (transformed @ normal).min()) * normal
        r, t, loss = icp(obs, tree, model, r, t, parameters)
        rotations.append(r)
        translations.append(t)
        losses.append(loss)
    return dict(rotations=np.array(rotations), translations=np.array(translations),
                losses_m=np.array(losses), start_axes=axes, start_yaws_deg=yaws,
                observed_count=np.array(len(obs)))


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return raw SO(3) distance in degrees, allowing a batch of rotations."""
    return np.degrees(Rotation.from_matrix(a @ np.swapaxes(b, -1, -2)).magnitude())


def basins(rotations: np.ndarray, losses: np.ndarray, separation_deg: float) -> list:
    """Greedily group finals around loss-ranked representatives on SO(3)."""
    remaining = np.argsort(losses, kind='stable')
    rows = []
    while len(remaining):
        best = int(remaining[0])
        belongs = geodesic_deg(rotations[remaining], rotations[best]) <= separation_deg
        rows.append(dict(start_index=best, loss_mm=float(losses[best] * 1000),
                         starts=int(belongs.sum()), rotation=rotations[best].tolist()))
        remaining = remaining[~belongs]
    return rows


def decision(primary: dict, repeats: list, parameters: dict) -> dict:
    """Decline any distant near-best final or an unstable subsampling winner."""
    r, loss = primary['rotations'], primary['losses_m']
    best = int(np.argmin(loss))
    angles = geodesic_deg(r, r[best])
    distant = np.flatnonzero(angles > parameters['basin_separation_deg'])
    rival = int(distant[np.argmin(loss[distant])]) if len(distant) else None
    ambiguous = rival is not None and loss[rival] <= loss[best] * (1 + parameters['relative_loss_margin'])
    stability = [float(geodesic_deg(row['rotations'][np.argmin(row['losses_m'])], r[best])) for row in repeats]
    unstable = max(stability, default=0.) > parameters['stability_max_deg']
    return dict(best_index=best, best_loss_mm=float(loss[best] * 1000),
                rival_index=rival, rival_loss_mm=None if rival is None else float(loss[rival] * 1000),
                rival_separation_deg=None if rival is None else float(angles[rival]),
                rival_relative_margin=None if rival is None else float(loss[rival] / loss[best] - 1),
                stability_deg=stability, accepted=not (ambiguous or unstable),
                decline_reasons=[name for name, yes in [('competing_rotation', ambiguous),
                                  ('subsampling_instability', unstable)] if yes],
                basins=basins(r, loss, parameters['basin_separation_deg']))


def vector_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return the full three-dimensional angle between two nonzero vectors."""
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.))))


def up_error_deg(estimated: np.ndarray, normal: np.ndarray, reference: np.ndarray) -> float:
    """Compare estimated table up and mocap world up in the scan body frame."""
    return vector_error_deg(estimated.T @ normal, reference.T @ np.array([0., 0., 1.]))


def pair_errors(rotations: np.ndarray, translations: np.ndarray, reference: np.ndarray) -> dict:
    """Compare anchor-frame b direction and raw relative C without symmetry quotients."""
    ref_r = Rotation.from_quat(reference[:, 3:]).as_matrix()
    b = rotations[0].T @ (translations[1] - translations[0])
    ref_b = ref_r[0].T @ (reference[1, :3] - reference[0, :3])
    c, ref_c = rotations[0].T @ rotations[1], ref_r[0].T @ ref_r[1]
    return dict(b_error_deg=vector_error_deg(b, ref_b), C_error_deg=float(geodesic_deg(c, ref_c)))
