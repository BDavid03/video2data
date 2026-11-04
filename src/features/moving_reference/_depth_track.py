"""Feature-free rigid fitting of an own mesh to silhouettes and stereo points."""

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ._object import render, iou
from .camera import apply, project

SYMMETRIES = {'blue_cup': 'continuous', 'beige_cup': 'continuous', 'white_pot_lid': 'order 2'}
SETTINGS = dict(seed=31, surface_samples=8000, depth_samples=160, contour_samples=180,
                triangle_candidates=8, optimizer_width=320, outer_steps=3, max_nfev=20,
                diff_step=1e-4, silhouette_scale_px=3., depth_scale_m=.01,
                translation_prior_m=.05, rotation_prior_rad=.35, continuity_weight=.1,
                minimum_pixels=50, minimum_depth_points=6, minimum_iou=.30,
                maximum_depth_median_m=.015, maximum_depth_p90_m=.030)


def subsample(values, count):
    """Keep deterministic, uniformly indexed samples."""
    return values[np.linspace(0, len(values)-1, min(len(values), count)).astype(int)]


def mesh_axis(normals, areas):
    """Find the largest coherent unsigned cap normal of a procedural mesh."""
    normals = np.array(normals, float)
    dominant = np.argmax(abs(normals), axis=1)
    normals *= np.where(normals[np.arange(len(normals)), dominant] < 0, -1., 1.)[:, None]
    _, inverse = np.unique(np.round(normals, 6), axis=0, return_inverse=True)
    chosen = inverse == np.argmax(np.bincount(inverse, weights=areas))
    axis = np.average(normals[chosen], weights=np.asarray(areas)[chosen], axis=0)
    return axis / np.linalg.norm(axis)


def geometry(vertices, faces, name):
    """Build reusable sampled-surface and triangle lookup geometry."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    samples, ids = trimesh.sample.sample_surface(mesh, SETTINGS['surface_samples'], seed=31)
    axis = mesh_axis(mesh.face_normals, mesh.area_faces) if name in SYMMETRIES else np.array([0., 0., 1.])
    basis = np.linalg.svd(axis[None], full_matrices=True)[2][1:]
    return dict(vertices=np.asarray(vertices), faces=np.asarray(faces), triangles=mesh.triangles,
                samples=samples, sample_faces=ids, surface_tree=cKDTree(samples),
                silhouette_points=np.vstack([samples, vertices]), axis=axis, swing_basis=basis,
                symmetry=SYMMETRIES.get(name))


def closest_surface(points, model):
    """Use eight sampled-face candidates, then exact closest points on those triangles."""
    _, ids = model['surface_tree'].query(points, k=SETTINGS['triangle_candidates'])
    candidates = model['triangles'][model['sample_faces'][ids]]
    repeated = np.broadcast_to(points[:, None], candidates.shape[:2]+(3,))
    closest = trimesh.triangles.closest_point(candidates.reshape(-1, 3, 3), repeated.reshape(-1, 3))
    closest = closest.reshape(len(points), -1, 3)
    best = np.argmin(np.linalg.norm(closest-points[:, None], axis=2), axis=1)
    return closest[np.arange(len(points)), best]


def evidence(frame, k):
    """Prepare non-hand mask boundaries and finite masked stereo observations."""
    obj = frame['object_mask'].astype(bool)
    hands = cv2.dilate(frame['hands'].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    inner = cv2.erode(obj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    supported = inner & ~hands & frame['depth_valid'] & np.isfinite(frame['xyz']).all(2)
    cloud = frame['xyz'][supported]
    scale = SETTINGS['optimizer_width']/obj.shape[1]
    shape = (round(obj.shape[0]*scale), SETTINGS['optimizer_width'])
    small_obj = cv2.resize(obj.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    small_hands = cv2.resize(hands.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    edge = cv2.morphologyEx(small_obj.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    y, x = np.nonzero(edge & ~small_hands)
    pixels = subsample(np.c_[x, y].astype(float), SETTINGS['contour_samples'])
    scaled_k = k.copy(); scaled_k[:2] *= scale
    return dict(mask=obj, hands=hands, small_hands=small_hands, shape=shape, k=scaled_k,
                edge=edge & ~small_hands,
                pixels=pixels, cloud=subsample(cloud, SETTINGS['depth_samples']),
                stereo_pixels=len(cloud), visible_pixels=int((obj & ~hands).sum()))


def symmetry_degrees(relative, axis, symmetry):
    """Return directed axis angle or the minimum discrete-symmetry rotation angle."""
    rotation = Rotation.from_matrix(relative)
    if symmetry == 'continuous':
        moved = np.einsum('...ij,j->...i', relative, axis)
        return np.degrees(np.arccos(np.clip(moved@axis, -1., 1.)))
    angle = rotation.magnitude()
    if symmetry == 'order 2':
        alternate = rotation * Rotation.from_rotvec(np.pi*axis)
        angle = np.minimum(angle, alternate.magnitude())
    return np.degrees(angle)


def pose_delta(values, previous, model):
    """Compose a body-frame swing or SO(3) increment and camera-frame translation."""
    vector = values[:2]@model['swing_basis'] if model['symmetry'] == 'continuous' else values[:3]
    pose = previous.copy()
    pose[:3, :3] = previous[:3, :3]@Rotation.from_rotvec(vector).as_matrix()
    pose[:3, 3] += values[-3:]
    return pose


def contour_pairs(pose, model, observed):
    """Associate exact rendered boundaries with model surface and observed contours."""
    predicted = render(model['vertices'], model['faces'], pose, observed['k'], observed['shape'])
    edge = cv2.morphologyEx(predicted.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    y, x = np.nonzero(edge & ~observed['small_hands'])
    boundary = subsample(np.c_[x, y].astype(float), SETTINGS['contour_samples'])
    if not len(boundary) or not len(observed['pixels']):
        return np.empty((0, 3)), np.empty((0, 2))
    local = model['silhouette_points']
    camera_points = apply(pose, local)
    uv = project(camera_points, observed['k'])
    good = np.isfinite(uv).all(1) & (camera_points[:, 2] > .05)
    local, uv = local[good], uv[good]
    if not len(local):
        return np.empty((0, 3)), np.empty((0, 2))
    _, model_ids = cKDTree(uv).query(boundary)
    source = local[model_ids]
    _, forward = cKDTree(observed['pixels']).query(boundary)
    _, reverse = cKDTree(boundary).query(observed['pixels'])
    return np.vstack([source, source[reverse]]), np.vstack([observed['pixels'][forward], observed['pixels']])


def fit_pose(previous, model, observed):
    """Refine one previous-pose start with silhouette/depth and weak continuity terms."""
    configuration = model.get('configuration', {})
    pose = previous.copy()
    dimension = 5 if model['symmetry'] == 'continuous' else 6
    total_evaluations = 0
    for _ in range(SETTINGS['outer_steps']):
        points, target = contour_pairs(pose, model, observed)
        base = pose.copy()

        def residual(values):
            current = pose_delta(values, base, model)
            blocks = []
            if len(points):
                uv = project(apply(current, points), observed['k'])
                pixels = np.rint(uv).astype(int)
                inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < observed['shape'][1]) &
                          (pixels[:, 1] >= 0) & (pixels[:, 1] < observed['shape'][0]))
                occluded = np.zeros(len(points), bool)
                occluded[inside] = observed['small_hands'][pixels[inside, 1], pixels[inside, 0]]
                if configuration.get('distance_transform'):
                    distance = cv2.distanceTransform(
                        (~observed['edge']).astype(np.uint8), cv2.DIST_L2, 3)
                    forward = cv2.remap(distance, uv[:, 0].astype(np.float32)[:, None],
                        uv[:, 1].astype(np.float32)[:, None], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=40).ravel()
                    reverse = cKDTree(uv).query(observed['pixels'])[0]
                    forward[occluded] = 0.
                    blocks.append(np.r_[forward, reverse]/3/np.sqrt(len(forward)+len(reverse)))
                silhouette = (uv-target)/3/np.sqrt(target.size)
                silhouette[occluded] = 0.
                if not configuration.get('distance_transform'):
                    blocks.append(silhouette.ravel())
            local = apply(np.linalg.inv(current), observed['cloud'])
            nearest = closest_surface(local, model)
            depth = local-nearest
            trim = configuration.get('depth_trim_fraction', 1.)
            if trim < 1:
                distance = np.linalg.norm(depth, axis=1)
                depth[distance > np.quantile(distance, trim)] = 0.
            blocks.append((depth/.01/np.sqrt(local.size)).ravel())
            continuity = configuration.get('continuity_weight', .1)
            blocks.append(continuity*(current[:3, 3]-previous[:3, 3])/.05)
            vector = Rotation.from_matrix(previous[:3, :3].T@current[:3, :3]).as_rotvec()
            blocks.append(continuity*vector/.35)
            return np.concatenate(blocks)

        result = least_squares(residual, np.zeros(dimension), loss='soft_l1',
                               max_nfev=20, diff_step=1e-4, x_scale='jac',
                               f_scale=configuration.get('robust_scale', 1.))
        total_evaluations += result.nfev
        pose = pose_delta(result.x, base, model)
    if model['symmetry'] == 'order 2':
        other = pose.copy()
        other[:3, :3] = pose[:3, :3]@Rotation.from_rotvec(np.pi*model['axis']).as_matrix()
        if Rotation.from_matrix(previous[:3, :3].T@other[:3, :3]).magnitude() < Rotation.from_matrix(previous[:3, :3].T@pose[:3, :3]).magnitude():
            pose = other
    return pose, total_evaluations


def initial_starts(observed, world_T_camera, model):
    """Initialize without labels from a stereo centre and four planar PCA yaws."""
    cloud = apply(world_T_camera, observed['cloud'])
    centre = np.median(cloud, axis=0)
    _, vectors = np.linalg.eigh(np.cov(cloud[:, :2].T))
    yaw = np.arctan2(vectors[1, -1], vectors[0, -1])
    upright = Rotation.align_vectors(np.array([[0., 0., 1.]]), model['axis'][None])[0].as_matrix()
    starts = []
    for angle in yaw+np.arange(4)*np.pi/2:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('z', angle).as_matrix()@upright
        pose[:3, 3] = centre
        starts.append(np.linalg.inv(world_T_camera)@pose)
    return starts


def canonical_starts(observed, world_T_camera, model, canonical_rotation):
    """Use own-mesh canonical axes and the visible handle hole for pan initialization."""
    from .model_track import footprint
    vertices = model['vertices']@canonical_rotation
    try:
        xyz = np.full((*observed['mask'].shape, 3), np.nan)
        pixels = np.rint(project(observed['cloud'], observed['original_k'])).astype(int)
        xyz[pixels[:, 1], pixels[:, 0]] = observed['cloud']
        visible = np.isfinite(xyz).all(2)
        canonical, _ = footprint(observed['mask'], xyz, visible, observed['original_k'],
                                 world_T_camera, known_thickness=float(np.ptp(vertices[:, 2])))
        heading = Rotation.from_matrix(canonical[:3, :3]).as_euler('xyz')[2]
        centre = canonical[:3, 3]
    except (ValueError, IndexError, TypeError):
        cloud = apply(world_T_camera, observed['cloud'])
        centre = np.median(cloud, axis=0)
        _, vectors = np.linalg.eigh(np.cov(cloud[:, :2].T))
        heading = np.arctan2(vectors[1, -1], vectors[0, -1])
    starts = []
    for angle in heading+np.arange(4)*np.pi/2:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler('z', angle).as_matrix()@canonical_rotation.T
        pose[:3, 3] = centre
        starts.append(np.linalg.inv(world_T_camera)@pose)
    return starts


def fit_metrics(pose, model, observed, original_k):
    """Measure exact original-grid silhouette IoU and depth fit residuals."""
    predicted = render(model['vertices'], model['faces'], pose, original_k, observed['mask'].shape)
    local = apply(np.linalg.inv(pose), observed['cloud'])
    distance = np.linalg.norm(local-closest_surface(local, model), axis=1) if len(local) else np.array([])
    return dict(iou=iou(predicted, observed['mask'], observed['hands']),
                depth_median_m=float(np.median(distance)) if len(distance) else None,
                depth_p90_m=float(np.percentile(distance, 90)) if len(distance) else None,
                stereo_supported_pixels=observed['stereo_pixels'], visible_pixels=observed['visible_pixels'])


def pose_errors(prediction, truth, model):
    """Measure position and symmetry-aware orientation after one initial alignment."""
    alignment = truth[0]@np.linalg.inv(prediction[0])
    aligned = alignment[None]@prediction
    position = np.linalg.norm(aligned[:, :3, 3]-truth[:, :3, 3], axis=1)
    # Transport the own-mesh axis into the aligned initial label-body frame.
    axis_world = alignment[:3, :3]@prediction[0, :3, :3]@model['axis']
    axis_label = truth[0, :3, :3].T@axis_world
    relative = truth[:, :3, :3].transpose(0, 2, 1)@aligned[:, :3, :3]
    angle = symmetry_degrees(relative, axis_label, model['symmetry'])
    raw_position = np.linalg.norm(prediction[:, :3, 3]-truth[:, :3, 3], axis=1)
    raw_angle = symmetry_degrees(truth[:, :3, :3].transpose(0, 2, 1)@prediction[:, :3, :3], model['axis'], model['symmetry'])
    return position, angle, raw_position, raw_angle


def m1_errors(position, quaternion, truth_position, truth_quaternion, axis, symmetry):
    """Use M1's equal frame indices and its exact single first-pose alignment."""
    turn = Rotation.from_quat(truth_quaternion[0])*Rotation.from_quat(quaternion[0]).inv()
    aligned_position = turn.apply(position-position[0])+truth_position[0]
    aligned_quaternion = turn*Rotation.from_quat(quaternion)
    position_error = np.linalg.norm(aligned_position-truth_position, axis=1)
    relative = (Rotation.from_quat(truth_quaternion).inv()*aligned_quaternion).as_matrix()
    return position_error, symmetry_degrees(relative, axis, symmetry)
