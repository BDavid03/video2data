"""Pure horizontal lengths, footprint centres and diagnostic error decompositions."""
import numpy as np
from scipy.spatial.transform import Rotation
from ._scene import table_basis, camera_points

VIDEO = ('V_inc', 'V_moge', 'V_stereo', 'V_hull', 'V_stereo_hull')


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


def measurements(vectors: dict, video: tuple = VIDEO) -> dict:
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
                    for name in video},
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


def summarize(rows: list, groups: dict, video: tuple, candidates: tuple,
              minimum_gain_cm: float, minimum_lower: int) -> dict:
    """Compute fixed-subset MAEs and apply the pre-registered five-episode authorization test."""
    means = {}
    for group, episodes in groups.items():
        selected = [r for r in rows if r['episode'] in episodes]
        means[group] = {}
        for name in video:
            present = [r for r in selected if r['lengths_cm'][name] is not None]
            means[group][name] = dict(count=len(present), episodes=[r['episode'] for r in present])
            for ref in ('M_o', 'M_c'):
                errors = [abs(r['lengths_cm'][name] - r['lengths_cm'][ref]) for r in present]
                means[group][name][f'mae_vs_{ref}_cm'] = float(np.mean(errors)) if errors else None
        means[group]['origin_offset_mae_cm'] = float(np.mean([
            abs(r['lengths_cm']['M_o'] - r['lengths_cm']['M_c']) for r in selected]))
    checks = {}
    for name in candidates:
        present = [r for r in rows if r['episode'] in groups['clean'] and r['lengths_cm'][name] is not None]
        gain = means['clean']['V_inc']['mae_vs_M_o_cm'] - means['clean'][name]['mae_vs_M_o_cm'] if len(present) == len(groups['clean']) else None
        improved = sum(abs(r['lengths_cm'][name] - r['lengths_cm']['M_o']) <
                       abs(r['lengths_cm']['V_inc'] - r['lengths_cm']['M_o']) for r in present)
        checks[name] = dict(count=len(present), gain_cm=gain, lower_episodes=improved,
                            passed=gain is not None and gain >= minimum_gain_cm and improved >= minimum_lower)
    warranted = any(row['passed'] for row in checks.values())
    verdict = ('A later, separately pre-registered candidate is warranted; nothing is adopted and v3 stays unchanged.'
               if warranted else 'The fixed test does not warrant a later candidate: length error is not explained by depth source or centroid choice under this protocol, and the pose line stays at v3.')
    return dict(means=means, authorization=dict(checks=checks, warranted=warranted, verdict=verdict))


def summarize_confirmation(rows: list, counted_episodes: tuple, video: tuple,
                           maximum_mae_cm: float, minimum_gain_cm: float) -> dict:
    """Keep mandatory episodes in the denominator and apply exactly Amendment 1's three checks."""
    counted = [r for r in rows if r['episode'] in counted_episodes]
    complete = all(r['lengths_cm']['V_stereo'] is not None and r['lengths_cm']['V_s'] is not None
                   for r in counted)
    means = {}
    for name in video:
        present = [r for r in counted if r['lengths_cm'][name] is not None]
        means[name] = dict(count=len(present), expected_count=len(counted),
                           episodes=[r['episode'] for r in present])
        for ref in ('M_o', 'M_c'):
            errors = [abs(r['lengths_cm'][name] - r['lengths_cm'][ref]) for r in present]
            value = float(np.mean(errors)) if errors else None
            means[name][f'available_subset_mae_vs_{ref}_cm'] = value
            means[name][f'counted_mae_vs_{ref}_cm'] = value if len(present) == len(counted) else None
    stereo = means['V_stereo']['counted_mae_vs_M_o_cm']
    scaled = means['V_s']['counted_mae_vs_M_o_cm']
    gain = scaled - stereo if complete else None
    lower = [r['episode'] for r in counted if r['lengths_cm']['V_stereo'] is not None
             and r['lengths_cm']['V_s'] is not None
             and abs(r['lengths_cm']['V_stereo']-r['lengths_cm']['M_o']) <
                 abs(r['lengths_cm']['V_s']-r['lengths_cm']['M_o'])]
    checks = dict(stereo_mae_below_3_cm=stereo is not None and stereo < maximum_mae_cm,
                  improvement_at_least_5_cm=gain is not None and gain >= minimum_gain_cm,
                  lower_on_every_counted_episode=complete and len(lower) == len(counted))
    confirmed = complete and all(checks.values())
    verdict = ('Confirmed under Amendment 1; nothing is adopted and a separate candidate protocol is still required.'
               if confirmed else 'Not confirmed under Amendment 1; no candidate follows and v3 remains unchanged.')
    return dict(counted_episodes=[r['episode'] for r in counted], complete=complete, means=means,
                measured_episodes=[r['episode'] for r in counted if r['lengths_cm']['V_stereo'] is not None],
                gain_cm=gain, lower_episodes=lower, checks=checks, confirmed=confirmed, verdict=verdict)


def stereo_clusters(stereo: dict | None, masks: list, labels: np.ndarray,
                    intrinsics: np.ndarray, minimum_pixels: int) -> tuple:
    """Select supported stereo pixels on the admitted monocular grid."""
    if stereo is None:
        return [None, None], [0, 0]
    if stereo['rgb_depth_m'].shape != labels.shape or not np.allclose(
            stereo['rgb_intrinsics_pixels'], intrinsics, atol=1e-5):
        raise ValueError('Stereo labels or intrinsics differ from the incumbent grid')
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & (stereo['rgb_depth_m'] > 0)
    points = camera_points(stereo['rgb_depth_m'].astype(float), stereo['rgb_intrinsics_pixels'])
    counts = [0 if mask is None else int((mask & valid).sum()) for mask in masks]
    clusters = [points[mask & valid] if mask is not None and count >= minimum_pixels else None
                for mask, count in zip(masks, counts)]
    return clusters, counts


def build_row(episode: int, names: list, data: dict, video: tuple,
              minimum_pixels: int, counted: bool | None = None) -> dict:
    """Build either protocol's measurement row from admitted arrays and construction choices."""
    labels, normal = data['labels'], data['normal']
    points = camera_points(data['depth']['depth_m'].astype(float), data['intrinsics'])
    inc_masks, masks, ids, defaults, merged = data['masks']
    clusters = [None if mask is None else points[mask] for mask in masks]
    stereo, counts = stereo_clusters(data['stereo'], masks, labels, data['intrinsics'], minimum_pixels)
    refs = reference_offsets(data['poses'], data['scan_centres'])
    moge = offset(footprint_centres(clusters, normal))
    available = dict(V_inc=None, V_moge=moge,
                     V_stereo=offset(footprint_centres(stereo, normal)))
    scale = data['scale']
    if 'V_s' in video:
        available['V_s'] = moge * scale['s'] if moge is not None and scale and scale['valid'] else None
    if 'V_hull' in video:
        inc_clusters = [None if mask is None else points[mask] for mask in inc_masks]
        positions = data['scaled']['positions']
        available.update(V_inc=positions[1, :2] - positions[0, :2],
                         V_hull=hull_offset(data['geometry'], inc_clusters, normal,
                                            scale['s'] if scale['valid'] else 1.),
                         V_stereo_hull=offset(footprint_centres(stereo, normal, bounds=True)))
    vectors = dict(refs, **{name: available[name] for name in video})
    row = dict(episode=episode, names=names, **measurements(vectors, video),
               vectors_m={key: value.tolist() if value is not None else None for key, value in vectors.items()},
               merged_cluster=merged, stereo_valid_pixels=counts,
               scale=None if scale is None else scale['s'])
    if counted is not None:
        row.update(counted=counted, scale_valid=None if scale is None else scale['valid'],
                   cluster_pixels=[int(np.isin(labels, np.array(data['merged_cluster_ids'])+1).sum())]*2
                                  if merged else [int(mask.sum()) for mask in masks],
                   missing_reason='No independent clusters; no split or fallback authorized' if merged else
                                  f'Stereo needs at least {minimum_pixels} valid pixels per object' if min(counts)<minimum_pixels else None)
        return row
    mismatch = float(np.max(np.abs(positions - np.asarray(data['recorded_scene']['positions']))))
    if mismatch > 1e-6:
        raise ValueError(f'S1 scene reproduction failed: {mismatch} m')
    pixel_counts = {str(int(key)-1): int(value) for key, value in
                    zip(*np.unique(labels[labels > 0], return_counts=True))}
    row.update(ray=ray_errors(vectors, refs, data['poses'], data['geometry']['quaternions'][0], normal),
               cluster_ids=ids, default_slots=defaults, default_half_m_offset=data['default_half_m_offset'],
               cluster_pixels=[sum(pixel_counts[str(i)] for i in item['observed'])
                               if item['observed'] is not None else None for item in ids],
               label_pixel_counts=pixel_counts, stereo_minimum_pixels=minimum_pixels,
               missing={key: ('merged cluster has no independent object pixels' if merged else
                              f'stereo cluster support {counts}; need {minimum_pixels} valid pixels in each')
                        for key in video if vectors[key] is None},
               separation_shifts_cm=data['geometry']['shifts'].tolist(),
               s1_max_position_difference_m=mismatch)
    return row


def cluster_masks(names: list, entries: dict, assignments: dict, labels: np.ndarray,
                  shared_incumbent: bool) -> tuple:
    """Resolve incumbent and observed cluster masks from the recorded assignments."""
    incumbent, observed, ids, defaults = [], [], [], []
    for name in names:
        entry = entries['slots'].get(name)
        chosen = entry['cluster_ids'] if entry else assignments[name]
        seen = chosen if chosen is not None else entry.get('observed_cluster_ids')
        defaults.append(chosen is None)
        ids.append(dict(incumbent=chosen, observed=seen))
        incumbent.append(None if chosen is None else np.isin(labels, np.asarray(chosen) + 1))
        observed.append(None if seen is None else np.isin(labels, np.asarray(seen) + 1))
    merged = entries['merged_cluster_ids'] is not None
    if merged:
        observed = [None, None]
        if shared_incumbent:
            mask = np.isin(labels, np.asarray(entries['merged_cluster_ids']) + 1)
            incumbent = [mask, mask]
    return incumbent, observed, ids, defaults, merged


def confirmation_masks(entry: dict, labels: np.ndarray) -> tuple:
    """Build independent frozen RGB unions while retaining merged missingness."""
    masks = [None if entry['merged'] else np.isin(labels, np.array(entry['objects'][name])+1)
             for name in entry['names']]
    return masks, masks, [], [], entry['merged']
