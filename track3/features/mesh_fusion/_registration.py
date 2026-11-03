"""Sparse background-feature PnP and independent stereo correspondence checks."""

from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from track3.features.camera_motion.estimate import _features, _solve_pose
from track3.features.mesh_extents._geometry import camera_points
from track3.features.pose_insurance._registration import icp, kabsch


PNP = SimpleNamespace(minimum_inliers=6, matching_ratio=0.75, reprojection_limit_px=3.)
ICP = dict(translation_only_iters=0, rigid_iters=10, keep_fraction=0.8)


def region_box(points):
    low, high = np.percentile(points, [0.5, 99.5], axis=0)
    return np.array(np.meshgrid(*zip(low - 0.03, high + 0.03), indexing='ij')).reshape(3, -1).T


def background_mask(shape, intrinsics, pose, boxes, input_valid):
    """Exclude projected opening 3D foreground boxes and the lower hand-entry strip."""
    mask = input_valid.copy()
    mask[round(shape[0] * 0.8):] = False
    for box in boxes:
        camera = (box - pose[:3, 3]) @ pose[:3, :3]
        if (camera[:, 2] <= 0).any():
            continue
        pixel = camera @ intrinsics.T
        hull = cv2.convexHull(np.rint(pixel[:, :2] / pixel[:, 2:]).astype(np.int32))
        cv2.fillConvexPoly(mask.view(np.uint8), hull, 0)
    return mask


def features(image, stereo, intrinsics, static):
    valid = stereo['rgb_valid'] & np.isfinite(stereo['rgb_depth_m']) & static
    keypoints, descriptors = cv2.ORB.create(nfeatures=6000).detectAndCompute(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), static.astype(np.uint8) * 255)
    if descriptors is None or len(keypoints) < 6:
        raise ValueError('fewer than 6 background ORB features')
    pixels, good, points = _features(keypoints, stereo['rgb_depth_m'], valid, intrinsics, np)
    return dict(pixels=pixels[good], descriptors=descriptors[good], points=points[good])


def correspondences(reference, current):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(current['descriptors'], reference['descriptors'], k=2)
    matches = sorted((p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance),
                     key=lambda m: m.distance)
    unique, seen = [], set()
    for match in matches:
        if match.trainIdx not in seen:
            unique.append((match.queryIdx, match.trainIdx))
            seen.add(match.trainIdx)
    return np.asarray(unique, dtype=int).reshape(-1, 2)


def without_patches(mask, pixels):
    result = mask.astype(np.uint8)
    for pixel in np.rint(pixels).astype(int):
        cv2.circle(result, tuple(pixel), 8, 0, -1)
    return result.astype(bool)


def dense_background(stereo, intrinsics, static):
    points = camera_points(stereo['rgb_depth_m'], intrinsics)
    valid = static & stereo['rgb_valid'] & np.isfinite(points).all(-1)
    selected = points[valid][::2]
    if len(selected):
        _, indices = np.unique(np.floor(selected / 0.01).astype(np.int64), axis=0, return_index=True)
        selected = selected[np.sort(indices)]
    return selected[:12000]


def pnp(reference, current, intrinsics, pairs):
    """Hold out every fifth reference landmark before estimating the pose."""
    held = pairs[pairs[:, 1] % 5 == 0]
    train = np.arange(len(reference['points'])) % 5 != 0
    good = np.ones(len(current['points']), dtype=bool)
    good[held[:, 0]] = False
    result = _solve_pose(current['descriptors'], current['pixels'], good,
                         reference['points'][train], reference['descriptors'][train],
                         cv2.BFMatcher(cv2.NORM_HAMMING), intrinsics, PNP, cv2, np)
    return result, held


def refine(pose, reference_background, current_background):
    """The existing ICP transforms reference model points into current observations."""
    rotation = pose[:3, :3].T
    translation = -rotation @ pose[:3, 3]
    r, t, loss = icp(current_background, cKDTree(reference_background), reference_background,
                      rotation, translation, ICP)
    refined = np.eye(4)
    refined[:3, :3], refined[:3, 3] = r.T, -r.T @ t
    return refined, loss


def reprojection(pose, reference, current, pairs, intrinsics):
    xyz = (reference['points'][pairs[:, 1]] - pose[:3, 3]) @ pose[:3, :3]
    xy = xyz @ intrinsics.T
    return np.linalg.norm(xy[:, :2] / xy[:, 2:] - current['pixels'][pairs[:, 0]], axis=1)


def register(reference, image, stereo, intrinsics, boxes, input_valid):
    """Mask, solve, remask, refine on background, then assess unused correspondences."""
    cv2.setRNGSeed(31)
    pose = np.eye(4)
    for _ in range(2):
        static = background_mask(input_valid.shape, intrinsics, pose, boxes, input_valid)
        current = features(image, stereo, intrinsics, static)
        pairs = correspondences(reference['features'], current)
        if len(pairs) < 6:
            return None, dict(accepted=False, reasons=['fewer than 6 descriptor correspondences'])
        (pose, inliers, residual, _), held = pnp(reference['features'], current, intrinsics, pairs)
        if pose is None:
            return None, dict(accepted=False, pnp_inliers=inliers or 0,
                              pnp_reprojection_px=residual, reasons=['background PnP failed'])
    train_pairs = pairs[pairs[:, 1] % 5 != 0]
    initial_residual = reprojection(pose, reference['features'], current, train_pairs, intrinsics)
    consensus = train_pairs[initial_residual <= PNP.reprojection_limit_px]
    reserved = np.arange(len(reference['features']['pixels'])) % 5 == 0
    rmask = without_patches(reference['static'], reference['features']['pixels'][reserved])
    cmask = without_patches(static, current['pixels'][held[:, 0]])
    rpoints = dense_background(reference['stereo'], intrinsics, rmask)
    cpoints = dense_background(stereo, intrinsics, cmask)
    if min(len(rpoints), len(cpoints)) < 30:
        return None, dict(accepted=False, reasons=['fewer than 30 independent ICP background points'])
    pose, loss = refine(pose, rpoints, cpoints)
    final_residual = float(np.median(reprojection(pose, reference['features'], current, consensus, intrinsics)))
    observed = current['points'][held[:, 0]]
    expected = reference['features']['points'][held[:, 1]]
    errors = np.linalg.norm(observed @ pose[:3, :3].T + pose[:3, 3] - expected, axis=1)
    disagreement = float(np.median(errors)) if len(errors) else None
    # Paired Kabsch is descriptive only; it does not replace the fixed PnP + ICP estimate.
    metric_rotation = kabsch(reference['features']['points'][train_pairs[:, 1]],
                             current['points'][train_pairs[:, 0]])[0]
    reasons = []
    if inliers < 30:
        reasons.append('PnP inliers < 30')
    if not np.isfinite(final_residual) or final_residual > 1:
        reasons.append('median reprojection > 1 px')
    if len(errors) < 10:
        reasons.append('fewer than 10 independent background correspondences')
    if disagreement is None or disagreement > 0.003:
        reasons.append('independent background disagreement > 3 mm')
    return pose, dict(accepted=not reasons, reasons=reasons, pnp_inliers=int(inliers),
                      pnp_reprojection_px=float(residual), reprojection_px=final_residual,
                      independent_correspondences=len(errors), independent_disagreement_m=disagreement,
                      independent_p90_m=float(np.percentile(errors, 90)) if len(errors) else None,
                      icp_trimmed_residual_m=float(loss), icp_reference_points=len(rpoints),
                      icp_current_points=len(cpoints), descriptor_pairs=len(pairs),
                      paired_kabsch_rotation=metric_rotation.tolist(),
                      camera_centre_m=pose[:3, 3].tolist())
