"""Pure horizontal lengths, footprint centres and diagnostic error decompositions."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._scene import table_basis

VIDEO = ('V_inc', 'V_moge', 'V_stereo', 'V_hull', 'V_stereo_hull')
CLEAN = (11, 12, 18, 21, 23)
SPECIAL = (0, 26, 31)


def footprint_centres(clusters: list, normal: np.ndarray, bounds: bool = False) -> list:
    """Project point means or axis-aligned footprint-bounds midpoints into the table plane."""
    basis = table_basis(normal)[:2]
    result = []
    for points in clusters:
        if points is None or not len(points):
            result.append(None)
        else:
            xy = points @ basis.T
            result.append((xy.min(0) + xy.max(0)) / 2 if bounds else xy.mean(0))
    return result


def offset(centres: list) -> np.ndarray | None:
    """Return the slot-one displacement only when both centres exist."""
    return None if any(c is None for c in centres) else np.asarray(centres[1]) - centres[0]


def reference_offsets(poses: np.ndarray, scan_centres: np.ndarray) -> dict:
    """Transform exact scan AABB centres with mocap poses, retaining world-horizontal offsets."""
    rotations = Rotation.from_quat(poses[:, 3:])
    centres = poses[:, :3] + rotations.apply(scan_centres)
    return dict(M_o=poses[1, :3] - poses[0, :3], M_c=centres[1] - centres[0])


def hull_offset(geometry: dict, clusters: list, normal: np.ndarray, factor: float) -> np.ndarray:
    """Replace incumbent point means with footprint centres, keeping defaults and recorded shifts."""
    positions = geometry['positions'][:, :2].copy()
    means = footprint_centres(clusters, normal)
    bounds = footprint_centres(clusters, normal, bounds=True)
    for slot, centre in enumerate(bounds):
        if centre is not None:
            positions[slot] += centre - means[slot]
    return (positions[1] - positions[0]) * factor


def measurements(vectors: dict) -> dict:
    """Convert horizontal displacements to centimetres and ratios to the origin reference."""
    lengths = {k: float(np.linalg.norm(v[:2]) * 100) if v is not None else None
               for k, v in vectors.items()}
    return dict(lengths_cm=lengths,
                decomposition_cm={name: None if lengths[name] is None else
                    dict(origin=lengths['M_c'] - lengths['M_o'],
                         video_vs_centre=lengths[name] - lengths['M_c'],
                         total=lengths[name] - lengths['M_o'],
                         absolute_origin=abs(lengths['M_c'] - lengths['M_o']),
                         absolute_video_vs_centre=abs(lengths[name] - lengths['M_c']))
                    for name in VIDEO},
                ratios_to_M_o={k: v / lengths['M_o'] if v is not None and lengths['M_o'] else None
                               for k, v in lengths.items()})


def ray_errors(vectors: dict, references: dict, poses: np.ndarray,
               anchor_wxyz: np.ndarray, normal: np.ndarray) -> dict:
    """Split offset error along/cross camera-forward after the kit's slot-zero rigid alignment."""
    ray = (table_basis(normal) @ np.array([0., 0., 1.]))[:2]
    ray /= np.linalg.norm(ray)
    cross = np.array([-ray[1], ray[0]])
    alignment = (Rotation.from_quat(anchor_wxyz[[1, 2, 3, 0]]) *
                 Rotation.from_quat(poses[0, 3:]).inv())
    aligned = {k: alignment.apply(v)[:2] for k, v in references.items()}
    errors = {}
    for name in VIDEO:
        v = vectors[name]
        errors[name] = None if v is None else {
            ref: dict(along_cm=float((v[:2] - target) @ ray * 100),
                      cross_cm=float((v[:2] - target) @ cross * 100))
            for ref, target in aligned.items()}
    return dict(camera_forward_table_xy=ray.tolist(),
                aligned_reference_xy_m={k: v.tolist() for k, v in aligned.items()}, errors=errors)


def summarize(rows: list) -> dict:
    """Compute fixed-subset MAEs and apply the pre-registered five-episode authorization test."""
    groups = dict(clean=CLEAN, merged_or_default=SPECIAL,
                  pooled=(*SPECIAL, *CLEAN), changed_six=(*CLEAN, 31))
    means = {}
    for group, episodes in groups.items():
        selected = [r for r in rows if r['episode'] in episodes]
        means[group] = {}
        for name in VIDEO:
            present = [r for r in selected if r['lengths_cm'][name] is not None]
            means[group][name] = dict(count=len(present), episodes=[r['episode'] for r in present])
            for ref in ('M_o', 'M_c'):
                errors = [abs(r['lengths_cm'][name] - r['lengths_cm'][ref]) for r in present]
                means[group][name][f'mae_vs_{ref}_cm'] = float(np.mean(errors)) if errors else None
        means[group]['origin_offset_mae_cm'] = float(np.mean([
            abs(r['lengths_cm']['M_o'] - r['lengths_cm']['M_c']) for r in selected]))
    checks = {}
    for name in ('V_stereo', 'V_hull', 'V_stereo_hull'):
        present = [r for r in rows if r['episode'] in CLEAN and r['lengths_cm'][name] is not None]
        gain = means['clean']['V_inc']['mae_vs_M_o_cm'] - means['clean'][name]['mae_vs_M_o_cm'] if len(present) == 5 else None
        improved = sum(abs(r['lengths_cm'][name] - r['lengths_cm']['M_o']) <
                       abs(r['lengths_cm']['V_inc'] - r['lengths_cm']['M_o']) for r in present)
        checks[name] = dict(count=len(present), gain_cm=gain, lower_episodes=improved,
                            passed=gain is not None and gain >= 5 and improved >= 4)
    warranted = any(row['passed'] for row in checks.values())
    verdict = ('A later, separately pre-registered candidate is warranted; nothing is adopted and v3 stays unchanged.'
               if warranted else 'The fixed test does not warrant a later candidate: length error is not explained by depth source or centroid choice under this protocol, and the pose line stays at v3.')
    return dict(means=means, authorization=dict(checks=checks, warranted=warranted, verdict=verdict))
