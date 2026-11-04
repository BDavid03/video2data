"""Pure orchestration of the kit's registered-pose arithmetic on supplied arrays."""
import numpy as np
from scipy.spatial.transform import Rotation

METRICS = dict(AUC='add_auc', SP_SR='spider_sr', MP_SR='maniptrans_sr',
               RPE_cm='rpe_cm', MPPE_cm='mppe_cm')


def registered_metrics(kit, achieved, reference, candidates, geometry, episode, seed_offset=0):
    """Mirror AUC.py::_score_registered_track3 after row/provenance parsing."""
    achieved = kit._normalise_quaternions(achieved, 'submission', ValueError)
    reference = kit._normalise_quaternions(reference, 'solution', ValueError)
    registrations, vertices, axes, orders = [], [], [], []
    for slot, (candidate, item) in enumerate(zip(candidates, geometry)):
        anchor, rotation = kit._register_object_body(candidate, item['mesh'], item['anchor'],
                                                    1_000_003 * episode + 101 * slot + seed_offset)
        achieved[:, :, slot] = kit._convert_object_body(achieved[:, :, slot], anchor, rotation)
        reference[:, :, slot, :3] += np.einsum('...ij,j->...i',
            kit._rotation_matrix(reference[:, :, slot, 3:]), item['anchor'])
        vertices.append(item['points'] - item['anchor'])
        axes.append(item['axis'])
        orders.append(item['order'])
        registrations.append(dict(candidate_anchor=anchor.tolist(), rotation=rotation.tolist()))
    achieved = kit._align_registered_scene(achieved, reference, axes)
    metrics = {name: kit._registered_pose_metric(metric, achieved, reference, vertices, axes, orders)
               for name, metric in METRICS.items()}
    return metrics, registrations


def rotation_spread(rotations):
    """Pairwise geodesic angles; a value near 180 degrees exposes a sign flip."""
    return [float(np.degrees(Rotation.from_matrix(a @ b.T).magnitude()))
            for i, a in enumerate(rotations) for b in rotations[i+1:]]
