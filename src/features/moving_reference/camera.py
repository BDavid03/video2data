"""Calibrated stereo, rigid estimation and independent image tracking on CPU."""

import cv2
import numpy as np


def rectification(calibration, width=960):
    left, right = (calibration['cameras'][key] for key in ('ego_cam_c', 'ego_cam_b'))
    size = tuple(left['resolution'])
    output = (width, round(size[1] * width / size[0]))
    transform, = [p['left_to_right'] for p in calibration['stereo_pairs'] if p['camera'] == 'ego']
    transform = np.asarray(transform, float)
    args = [np.asarray(c['intrinsic_matrix'], float) for c in (left, right)]
    distortions = [np.asarray(c['distortion_coefficients'], float) for c in (left, right)]
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        args[0], distortions[0], args[1], distortions[1], size,
        np.ascontiguousarray(transform[:3, :3]), np.ascontiguousarray(transform[:3, 3].reshape(3,1)),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=output)
    maps = [cv2.initUndistortRectifyMap(k, d, r, p[:, :3], output, cv2.CV_32FC1)
            for k, d, r, p in zip(args, distortions, (r1, r2), (p1, p2))]
    return maps, p1[:, :3], q, r1


def stereo(left, right, q):
    options = dict(numDisparities=128, blockSize=5, P1=8*25, P2=32*25,
                   disp12MaxDiff=1, uniquenessRatio=8, speckleWindowSize=80,
                   speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    dl = cv2.StereoSGBM_create(minDisparity=0, **options).compute(left, right).astype(float)/16
    dr = cv2.StereoSGBM_create(minDisparity=-128, **options).compute(right, left).astype(float)/16
    yy, xx = np.indices(dl.shape)
    xr = np.rint(xx-dl).astype(int)
    valid = (dl > 1) & (xr >= 0) & (xr < dl.shape[1])
    valid &= abs(dl + dr[yy, np.clip(xr, 0, dl.shape[1]-1)]) <= 1
    xyz = cv2.reprojectImageTo3D(dl.astype(np.float32), q).astype(float)
    valid &= np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > .1) & (xyz[..., 2] < 5)
    xyz[~valid] = np.nan
    return xyz, valid


def lift(pixels, xyz):
    pixels = np.asarray(pixels).reshape(-1, 2)
    ij = np.rint(pixels).astype(int)
    good = (ij[:, 0] >= 0) & (ij[:, 0] < xyz.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < xyz.shape[0])
    ij[:, 0] = np.clip(ij[:, 0], 0, xyz.shape[1]-1)
    ij[:, 1] = np.clip(ij[:, 1], 0, xyz.shape[0]-1)
    points = xyz[ij[:, 1], ij[:, 0]].copy()
    good &= np.isfinite(points).all(axis=1)
    return points, good


def apply(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def project(points, k):
    uv = points @ k.T
    return uv[:, :2] / uv[:, 2:3]


def kabsch(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-a).T @ (target-b))
    r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, b-r@a
    return pose


def rigid(source, target, threshold=.008, iterations=160):
    if len(source) < 6:
        return None, np.zeros(len(source), bool)
    rng = np.random.default_rng(31)
    best = np.zeros(len(source), bool)
    for _ in range(iterations):
        ids = rng.choice(len(source), 3, replace=False)
        if np.linalg.norm(np.cross(source[ids[1]]-source[ids[0]], source[ids[2]]-source[ids[0]])) < 1e-8:
            continue
        pose = kabsch(source[ids], target[ids])
        good = np.linalg.norm(apply(pose, source)-target, axis=1) < threshold
        if good.sum() > best.sum():
            best = good
    if best.sum() < 6:
        return None, best
    pose = kabsch(source[best], target[best])
    return pose, np.linalg.norm(apply(pose, source)-target, axis=1) < threshold


def plane(points, iterations=500):
    points = points[np.isfinite(points).all(axis=1)]
    rng = np.random.default_rng(31)
    if len(points) > 12000:
        points = points[rng.choice(len(points), 12000, replace=False)]
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        n /= max(np.linalg.norm(n), 1e-12)
        if n @ [0, -1, 0] < 0:
            n = -n
        good = abs((points-a) @ n) < .01
        if n @ [0, -1, 0] > np.cos(np.deg2rad(65)) and good.sum() > best.sum():
            best = good
    if best.sum() < 100:
        raise ValueError('Insufficient table plane support')
    p = points[best]
    _, _, vt = np.linalg.svd(p-p.mean(axis=0), full_matrices=False)
    n = vt[-1]
    if n @ [0, -1, 0] < 0:
        n = -n
    d = -np.median(p@n)
    return n, float(d), p


def define_world(xyz, static, object_mask):
    yy, xx = np.indices(static.shape)
    oy, ox = np.nonzero(object_mask)
    nearby = (abs(xx-np.median(ox)) < static.shape[1]*.3) & (yy > np.median(oy)-static.shape[0]*.1)
    n, d, points = plane(xyz[static & nearby])
    origin = -d*n
    heading = np.array([0., 0., 1.]) - n[2]*n
    heading /= np.linalg.norm(heading)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([heading, np.cross(n, heading), n])
    pose[:3, 3] = -pose[:3, :3] @ origin
    inlier = static & nearby & (abs(np.einsum('ijk,k->ij', xyz, n)+d) < .01)
    patch_normals = []
    for yhalf in (False, True):
        for xhalf in (False, True):
            patch = inlier & ((yy > np.median(oy)) == yhalf) & ((xx > np.median(ox)) == xhalf)
            if patch.sum() >= 100:
                _, _, vt = np.linalg.svd(xyz[patch]-xyz[patch].mean(axis=0), full_matrices=False)
                patch_normals.append(vt[-1] * np.sign(vt[-1]@n))
    angles = [np.rad2deg(np.arccos(np.clip(a@b, -1, 1)))
              for i, a in enumerate(patch_normals) for b in patch_normals[i+1:]]
    return pose, dict(normal_camera=n.tolist(), offset_camera_m=d,
                     residual_rms_m=float(np.sqrt(np.mean((points@n+d)**2))),
                     patch_normal_max_deg=float(max(angles)) if len(patch_normals)==4 else None,
                     patch_count=len(patch_normals), point_count=len(points), table_level_assumed=True)


def flow(previous, current, pixels, previous_mask=None, current_mask=None):
    pixels = np.asarray(pixels, np.float32).reshape(-1, 2)
    if not len(pixels):
        return pixels, np.zeros(0, bool)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, pixels[:, None], None, **options)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None, **options)
    nxt, back = nxt[:, 0], back[:, 0]
    good = status.ravel().astype(bool) & reverse.ravel().astype(bool)
    good &= np.linalg.norm(back-pixels, axis=1) <= .75
    for mask, uv in ((previous_mask, pixels), (current_mask, nxt)):
        if mask is not None:
            ij = np.rint(uv).astype(int)
            inside = (ij[:, 0]>=0) & (ij[:, 0]<mask.shape[1]) & (ij[:, 1]>=0) & (ij[:, 1]<mask.shape[0])
            good &= inside & mask[np.clip(ij[:, 1], 0, mask.shape[0]-1), np.clip(ij[:, 0], 0, mask.shape[1]-1)]
    return nxt, good


def corners(gray, mask, maximum=900, distance=3):
    points = cv2.goodFeaturesToTrack(gray, maximum, .005, distance, mask=mask.astype(np.uint8)*255, blockSize=3)
    return np.empty((0, 2), np.float32) if points is None else points[:, 0]


def pnp(world, pixels, k):
    if len(world) < 12:
        return None, 0
    ok, rv, tv, ids = cv2.solvePnPRansac(np.asarray(world, float), np.asarray(pixels, float), k, None,
        iterationsCount=200, reprojectionError=2, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or ids is None or len(ids)<12:
        return None, 0
    ids = ids.ravel()
    rv, tv = cv2.solvePnPRefineLM(np.asarray(world)[ids], np.asarray(pixels)[ids], k, None, rv, tv)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    return np.linalg.inv(pose), len(ids)


def matches(descriptors, map_descriptors, ratio=.75):
    if descriptors is None or len(map_descriptors)<2:
        return np.empty((0, 2), int)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, np.asarray(map_descriptors, np.uint8), k=2)
    candidates = sorted([a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<ratio*b.distance], key=lambda m:m.distance)
    result, used = [], set()
    for match in candidates:
        if match.trainIdx not in used:
            result.append((match.queryIdx, match.trainIdx)); used.add(match.trainIdx)
    return np.asarray(result, int).reshape(-1, 2)


def camera_start(gray, xyz, static, world_T_camera):
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=8)
    keys, descriptors = orb.detectAndCompute(gray, static.astype(np.uint8)*255)
    pixels = np.asarray([key.pt for key in keys])
    points, good = lift(pixels, xyz)
    ids = np.flatnonzero(good)
    # Round-robin each spatial cell before taking every fourth feature as hold-out.
    groups = [[] for _ in range(16)]
    for i in ids:
        cell = min(3, int(pixels[i,1]*4/gray.shape[0]))*4 + min(3, int(pixels[i,0]*4/gray.shape[1]))
        groups[cell].append(i)
    ordered = [group[j] for j in range(max(map(len,groups))) for group in groups if j<len(group)]
    # Taking a quarter *inside each cell* prevents all held-outs landing in one quadrant.
    held = np.asarray([i for group in groups for j,i in enumerate(group) if j%4==0], int)
    excluded = np.zeros(gray.shape, np.uint8)
    for u,v in pixels[held]:
        cv2.circle(excluded, (round(u),round(v)), 8, 1, -1)
    ij = np.rint(pixels).astype(int)
    fit = np.asarray([i for i in ordered if not excluded[ij[i,1],ij[i,0]]],int)
    state = dict(orb=orb, gray=gray, pixels=pixels[held].astype(np.float32),
                 held_world=apply(world_T_camera,points[held]),
                 reserved_world=apply(world_T_camera,points[held]), reserved_desc=descriptors[held],
                 map_world=apply(world_T_camera,points[fit]), map_desc=descriptors[fit],
                 pose=world_T_camera.copy())
    return state


def camera_step(state, gray, xyz, static, k):
    pixels, alive = flow(state['gray'], gray, state['pixels'], None, static)
    held_points, stereo_good = lift(pixels, xyz)
    alive &= stereo_good
    pixels, held_points, held_world = pixels[alive], held_points[alive], state['held_world'][alive]
    allowed = static.astype(np.uint8)*255
    reserve_camera = apply(np.linalg.inv(state['pose']),state['reserved_world'])
    reserve_uv = project(reserve_camera,k)
    for u,v in reserve_uv[np.isfinite(reserve_uv).all(axis=1) & (reserve_camera[:,2]>.1)]:
        if -10<u<gray.shape[1]+10 and -10<v<gray.shape[0]+10:
            cv2.circle(allowed,(round(u),round(v)),10,0,-1)
    keys, desc = state['orb'].detectAndCompute(gray, allowed)
    uv = np.asarray([key.pt for key in keys]).reshape(-1,2)
    lifted, good = lift(uv,xyz)
    fit_matches = matches(desc,state['map_desc'])
    if len(fit_matches):
        fit_matches=fit_matches[good[fit_matches[:,0]]]
        reserved_queries=set(matches(desc,state['reserved_desc'],.85)[:,0].tolist())
        fit_matches=np.asarray([pair for pair in fit_matches if int(pair[0]) not in reserved_queries],int).reshape(-1,2)
    pose, count = pnp(state['map_world'][fit_matches[:,1]], uv[fit_matches[:,0]], k)
    observed = pose is not None
    if pose is None:
        pose=state['pose'].copy()
    predicted = project(apply(np.linalg.inv(pose),held_world),k)
    residual = np.linalg.norm(predicted-pixels,axis=1)*640/gray.shape[1]
    world_residual = np.linalg.norm(apply(pose,held_points)-held_world,axis=1)
    regions = (pixels[:,0]>=gray.shape[1]/2).astype(int)+2*(pixels[:,1]>=gray.shape[0]/2)
    row = dict(observed=observed, inliers=count, heldout_count=len(pixels),
               heldout_regions=len(np.unique(regions)), heldout_reprojection_px640=residual.tolist(),
               heldout_world_residual_m=world_residual.tolist())
    # Reserve new features deterministically before they can enter either map.
    if observed and desc is not None:
        existing = apply(np.linalg.inv(pose),state['map_world'])
        occupied = np.zeros(gray.shape,np.uint8)
        for u,v in project(existing[existing[:,2]>.1],k):
            if -10<u<gray.shape[1]+10 and -10<v<gray.shape[0]+10:
                cv2.circle(occupied,(round(u),round(v)),8,1,-1)
        ij=np.rint(uv).astype(int)
        new=np.flatnonzero(good & ~occupied[np.clip(ij[:,1],0,gray.shape[0]-1),np.clip(ij[:,0],0,gray.shape[1]-1)].astype(bool))
        # Descriptor matches to reserved landmarks also forbid fitting/re-insertion.
        reserved=matches(desc,state['reserved_desc'],.85)
        forbidden=set(reserved[:,0].tolist())
        groups=[[] for _ in range(16)]
        for i in new:
            if int(i) not in forbidden:
                cell=min(3,int(uv[i,1]*4/gray.shape[0]))*4+min(3,int(uv[i,0]*4/gray.shape[1]))
                groups[cell].append(i)
        hold=np.asarray([i for group in groups for j,i in enumerate(group) if j%4==0],int)
        new_fit=np.asarray([i for group in groups for j,i in enumerate(group) if j%4!=0],int)
        # Exclude neighbourhoods of newly reserved points from the fitted map too.
        exclusion=np.zeros(gray.shape,np.uint8)
        for u,v in uv[hold]:
            cv2.circle(exclusion,(round(u),round(v)),8,1,-1)
        new_fit=new_fit[~exclusion[ij[new_fit,1],ij[new_fit,0]].astype(bool)]
        if len(state['map_world'])<20000:
            new_fit=new_fit[:20000-len(state['map_world'])]
            state['map_world']=np.concatenate([state['map_world'],apply(pose,lifted[new_fit])])
            state['map_desc']=np.concatenate([state['map_desc'],desc[new_fit]])
        if len(state['reserved_world'])<10000:
            hold=hold[:10000-len(state['reserved_world'])]
            new_world=apply(pose,lifted[hold])
            state['reserved_world']=np.concatenate([state['reserved_world'],new_world])
            state['reserved_desc']=np.concatenate([state['reserved_desc'],desc[hold]])
            pixels=np.concatenate([pixels,uv[hold].astype(np.float32)])
            held_world=np.concatenate([held_world,new_world])
    state.update(gray=gray,pixels=pixels,held_world=held_world,pose=pose)
    return pose, row
