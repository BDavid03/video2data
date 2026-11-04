"""Pure frame conversion and world-space equivalence for the frozen TW assets."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from ._placement import reexpress_rotations


def body_inertia(rotation, inertia) -> np.ndarray:
    """Express a COM tensor in its body frame with the URDF inertial rotation."""
    return rotation @ inertia @ rotation.T


def transform_vertices(vertices, rotation) -> np.ndarray:
    """Rotate row-vector mesh vertices into another frame."""
    return vertices @ rotation.T


def collision_corners(dimensions, rotation, centre) -> np.ndarray:
    """Express the eight ordered collision-box corners in the body frame."""
    corners = np.array(list(product((-.5, .5), repeat=3))) * dimensions
    return corners @ rotation.T + centre


def asset_frame(old_wxyz, new_wxyz, inertia, centre, collision_rotation, collision_centre) -> dict:
    """Counter-rotate geometry, COM inertia and the collision frame into the new body frame."""
    old = Rotation.from_quat(np.asarray(old_wxyz)[None, [1, 2, 3, 0]])
    new = Rotation.from_quat(np.asarray(new_wxyz)[None, [1, 2, 3, 0]])
    result = reexpress_rotations(old, new, np.asarray(inertia)[None])
    geometry = result['geometry_rotation'][0]
    return dict(geometry=geometry, inertia=result['inertia'][0], centre=geometry @ centre,
                collision_rotation=geometry @ collision_rotation, collision_centre=geometry @ collision_centre)


def equivalence(old_wxyz, new_wxyz, old_vertices, new_vertices, old_inertia, new_inertia,
                old_centre, new_centre, old_collision, new_collision) -> dict:
    """Compare serialized mesh, collision corners, COM and inertia in the common world frame."""
    old = Rotation.from_quat(np.asarray(old_wxyz)[[1, 2, 3, 0]]).as_matrix()
    new = Rotation.from_quat(np.asarray(new_wxyz)[[1, 2, 3, 0]]).as_matrix()
    errors = dict(world_geometry_max_m=float(np.max(np.abs(old_vertices @ old.T - new_vertices @ new.T))),
                  world_collision_max_m=float(np.max(np.abs(old_collision @ old.T - new_collision @ new.T))),
                  world_com_max_m=float(np.max(np.abs(old @ old_centre - new @ new_centre))),
                  world_inertia_max_kg_m2=float(np.max(np.abs(old @ old_inertia @ old.T - new @ new_inertia @ new.T))),
                  inertia_min_eigenvalue_kg_m2=float(np.linalg.eigvalsh(new_inertia).min()))
    errors['passed'] = (max(errors[key] for key in ('world_geometry_max_m', 'world_collision_max_m', 'world_com_max_m')) < 1e-12
                        and errors['world_inertia_max_kg_m2'] < 1e-15 and errors['inertia_min_eigenvalue_kg_m2'] > 0)
    return errors
