"""Pure common-footprint shrink for the amended panel placement."""
from copy import deepcopy
import numpy as np
from ._scene import box_properties
from ._shrink import footprint_clearance


def common_xy_shrink(scene: list, displacement: list, masses: list) -> dict:
    """Keep T1 frames and masses while maximising a shared, clear XY box factor."""
    positions = np.array([r['position'] for r in scene], float)
    dimensions = np.array([r['dimensions_m'] for r in scene], float)
    quaternions = np.array([r['quaternion_wxyz'] for r in scene], float)
    positions[1] = positions[0] + np.asarray(displacement)

    def clearance(factor):
        scaled = dimensions.copy()
        scaled[:, :2] *= factor
        return footprint_clearance(positions, scaled, quaternions)

    original_gap = clearance(1.)
    if original_gap >= .01:
        factor = 1.
    elif clearance(0.) <= .01:
        return dict(factor=None, clearance_before_m=original_gap, scene=None)
    else:
        lower, upper = 0., 1.
        for _ in range(80):
            middle = (lower + upper) / 2
            if clearance(middle) >= .01:
                lower = middle
            else:
                upper = middle
        factor = lower
    result = deepcopy(scene)
    properties = []
    for slot, row in enumerate(result):
        dims = dimensions[slot].copy()
        dims[:2] *= factor
        inertia, radius = box_properties(dims, masses[slot])
        row.update(position=positions[slot].tolist(), dimensions_m=dims.tolist())
        properties.append(dict(mass_kg=masses[slot], inertia_diagonal_kg_m2=inertia.tolist(), radius_m=radius))
    assert np.array_equal(quaternions, [r['quaternion_wxyz'] for r in result])
    assert result[0]['position'] == scene[0]['position']
    return dict(factor=factor, clearance_before_m=original_gap, clearance_m=clearance(factor),
                scene=result, properties=properties, counter_rotation=np.repeat(np.eye(3)[None], 2, axis=0).tolist())
