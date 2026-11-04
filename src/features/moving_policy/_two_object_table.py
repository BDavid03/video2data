"""Measure the existing table-patch angle on the available disjoint patches."""

import numpy as np


def available_patch_angle(groups,normal,offset):
    normals=[];counts=[]
    for group in groups:
        points=np.asarray(group).reshape(-1,3)
        points=points[abs(points@normal+offset)<.01]
        counts.append(len(points))
        if len(points)<100:continue
        _,_,vt=np.linalg.svd(points-points.mean(0),full_matrices=False)
        normals.append(vt[-1]*np.sign(vt[-1]@normal))
    angles=[np.rad2deg(np.arccos(np.clip(a@b,-1,1))) for i,a in enumerate(normals) for b in normals[i+1:]]
    return dict(available_patches=len(normals),inlier_counts=counts,normals=np.asarray(normals).tolist(),
                maximum_angle_deg=float(max(angles)) if len(normals)>=3 else None)
